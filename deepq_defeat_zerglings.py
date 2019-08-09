import numpy as np
import os
import dill
import tempfile
import tensorflow as tf
import zipfile
from absl import flags
import baselines.deepq.utils as U
import baselines.common.tf_util as V
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from pysc2.lib import actions as sc2_actions
from pysc2.env import environment
from pysc2.lib import features
from pysc2.lib import actions
from common import common

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.selected.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_UNIT_ID = 1
_CONTROL_GROUP_SET = 1
_CONTROL_GROUP_RECALL = 0
_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]
UP, DOWN, LEFT, RIGHT = 'up', 'down', 'left', 'right'
FLAGS = flags.FLAGS

class ActWrapper(object):
    def __init__(self, act):
        self._act = act

    @staticmethod
    def load(path, act_params, num_cpu=16):
        with open(path, "rb") as f:
            model_data = dill.load(f)
        act = deepq.build_act(**act_params)
        sess = V.make_session(num_cpu=num_cpu)
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)
            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            U.load_state(os.path.join(td, "model"))
        return ActWrapper(act)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def save(self, path):
        with tempfile.TemporaryDirectory() as td:
            U.save_state(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            dill.dump((model_data), f)


def load(path, act_params, num_cpu=16):
    return ActWrapper.load(path, num_cpu=num_cpu, act_params=act_params)


def learn(env, q_func, num_actions, lr, max_timesteps, max_memory, epsilon_decay_rate, epsilon_min, epsilon_max,
          train_freq, batch_size=32, print_freq=1, checkpoint_freq=10000, target_network_update_freq=500,
          param_noise=False, param_noise_threshold=0.05, num_cpu=16, callback=None):
    sess = V.make_session(num_cpu=num_cpu)
    sess.__enter__()
    def make_obs_ph(name):
        return U.BatchInput((1, 64, 64), name=name)
    act, train, update_target, debug = deepq.build_train(make_obs_ph=make_obs_ph, q_func=q_func,
                                                         num_actions=num_actions,
                                                         optimizer=tf.train.AdamOptimizer(learning_rate=lr),
                                                         grad_norm_clipping=10)
    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': num_actions
    }
    replay_buffer = ReplayBuffer(max_memory)
    beta_schedule = None
    exploration = LinearSchedule(schedule_timesteps=int(epsilon_decay_rate * max_timesteps), initial_p=epsilon_max,
                                 final_p=epsilon_min)
    V.initialize()
    update_target()
    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    player_relative = obs[0].observation["feature_screen"][_PLAYER_RELATIVE]
    screen = player_relative
    screen = np.reshape(screen, (1, 64, 64))
    obs, xy_per_marine = common.init(env, obs)
    group_id = 0
    reset = True
    with tempfile.TemporaryDirectory() as td:
        model_save = False
        model_file = os.path.join(td, "model")
        for t in range(max_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                if param_noise_threshold >= 0.:
                    update_param_noise_threshold = param_noise_threshold
                else:
                    update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) /
                                                           float(num_actions))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True
            obs, screen, player = common.select_marine(env, obs)
            screen = np.reshape(screen, (1, 64, 64))
            action = act(np.array(screen)[None], update_eps=update_eps, **kwargs)[0]
            reset = False
            rew = 0
            new_action = None
            obs, new_action = common.marine_action(env, obs, player, action)
            army_count = env._obs[0].observation.player_common.army_count
            try:
                if army_count > 0 and _ATTACK_SCREEN in obs[0].observation["available_actions"]:
                    obs = env.step(actions=new_action)
                else:
                    new_action = [sc2_actions.FunctionCall(_NO_OP, [])]
                    obs = env.step(actions=new_action)
            except Exception as e:
                print(e)
            player_relative = obs[0].observation["feature_screen"][_PLAYER_RELATIVE]
            new_screen = player_relative
            new_screen = np.reshape(new_screen, (1, 64, 64))
            rew += obs[0].reward
            done = obs[0].step_type == environment.StepType.LAST
            selected = obs[0].observation["feature_screen"][_SELECTED]
            player_y, player_x = (selected == _PLAYER_FRIENDLY).nonzero()
            if len(player_y) > 0:
                player = [int(player_x.mean()), int(player_y.mean())]
            if len(player) == 2:
                if player[0] > 32:
                    new_screen = common.shift(LEFT, player[0] - 32, new_screen)
                elif player[0] < 32:
                    new_screen = common.shift(RIGHT, 32 - player[0], new_screen)
                if player[1] > 32:
                    new_screen = common.shift(UP, player[1] - 32, new_screen)
                elif player[1] < 32:
                    new_screen = common.shift(DOWN, 32 - player[1], new_screen)
            replay_buffer.add(screen, action, rew, new_screen, float(done))
            screen = new_screen
            episode_rewards[-1] += rew
            reward = episode_rewards[-1]
            if done:
                print("Episode Reward: %s" % episode_rewards[-1])
                obs = env.reset()
                player_relative = obs[0].observation["feature_screen"][_PLAYER_RELATIVE]
                screen = player_relative
                screen = np.reshape(screen, (1, 64, 64))
                group_list = common.init(env, obs)
                episode_rewards.append(0.0)
                reset = True
            if t % train_freq == 0:
                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                weights, batch_idxes = np.ones_like(rewards), None
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
            if t % target_network_update_freq == 0:
                update_target()
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("reward", reward)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()
            if checkpoint_freq is not None and num_episodes > 100 and t % checkpoint_freq == 0:
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving mdoel due to mean reward increase: {} -> {}".format(saved_mean_reward,
                                                                                               mean_100ep_reward))
                    U.save_state(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward
        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {]".format(saved_mean_reward))
            U.load_state(model_file)
    return ActWrapper(act)