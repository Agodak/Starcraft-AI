import sys
from absl import flags
from baselines import deepq
from pysc2.env import sc2_env
from pysc2.lib import actions, features
import os
import deepq_mineral_shards
import datetime
from common.vec_env.subproc_vec_env import SubprocVecEnv
from a2c.policies import CnnPolicy
from a2c import a2c
from baselines.logger import Logger, TensorBoardOutputFormat, HumanOutputFormat
import deepq_defeat_zerglings
import deepq_beacon

step_mul = 8
FLAGS = flags.FLAGS
flags.DEFINE_string("map", "FindAndDefeatZerglings", "Name of a map to use to play.")
flags.DEFINE_string("log", "tensorboard", "Logging type(stdout, tensorboard)")
flags.DEFINE_string("algorithm", "a2c", "RL algorithm to use.")
flags.DEFINE_integer("timesteps", 2000000, "Steps to train")
flags.DEFINE_float("epsilon_max", 1, "Starting value of epsilon")
flags.DEFINE_float("epsilon_min", 0, "Final value of epsilon")
flags.DEFINE_float("epsilon_decay_rate", 0.01, "Final value of epsilon")
flags.DEFINE_integer("train_frequency", 1, "After how many steps the weights are updated")
flags.DEFINE_integer("target_update_frequency", 500, "After how many steps the target network is updated")
flags.DEFINE_integer("max_memory", 10000, "Amount of experiences agent keeps in memory")
flags.DEFINE_float("lr", 0.00001, "Learning rate")
flags.DEFINE_integer("trajectory_training_steps", 40, "Number of forward steps before backpropagation is run.")
flags.DEFINE_float("value_gradient_strength", 0.5, "Strength of the critic in A2C")
flags.DEFINE_float("regularization_strength", 0.01, "Factor by which too dominant actions are penalized")
flags.DEFINE_string("mode", "run", "Mode")

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
max_mean_reward = 0
last_filename = ""
start_time = datetime.datetime.now().strftime("%m%d%H%M")


def main():
    FLAGS(sys.argv)
    print("algorithm : %s" % FLAGS.algorithm)
    print("timesteps : %s" % FLAGS.timesteps)
    print("lr : %s" % FLAGS.lr)
    logdir = "tensorboard"
    if FLAGS.algorithm == "deepq":
        logdir = "tensorboard/mineral/%s/%s_%s_lr%s%s" % (FLAGS.algorithm, FLAGS.timesteps, FLAGS.lr, start_time)
    elif FLAGS.algorithm == "a2c":
        logdir = "tensorboard/mineral/%s/%s/lr%s/%s" % (FLAGS.algorithm, FLAGS.timesteps, FLAGS.lr, start_time)
    if FLAGS.log == "tensorboard":
        Logger.DEFAULT = Logger.CURRENT = Logger(dir=None, output_formats=[TensorBoardOutputFormat(logdir)])
    elif FLAGS.log == "stdout":
        Logger.DEFAULT = Logger.CURRENT = Logger(dir=None, output_formats=[HumanOutputFormat(sys.stdout)])
    if FLAGS.mode == "train":
        if FLAGS.algorithm == "deepq":
            if FLAGS.map == "CollectMineralShards":
                with sc2_env.SC2Env(map_name="CollectMineralShards", step_mul=step_mul, visualize=True, agent_interface_format=features.AgentInterfaceFormat(feature_dimensions=features.Dimensions(screen=64, minimap=64), use_feature_units=True)) as env:
                    model = deepq.models.cnn_to_mlp(convs=[(16, 8, 4), (32, 4, 2)], hiddens=[256], dueling=True)
                    act_x, act_y = deepq_mineral_shards.learn(env, q_func=model, num_actions=16, lr=FLAGS.lr, max_timesteps=FLAGS.timesteps, max_memory=FLAGS.max_memory, epsilon_decay_rate=FLAGS.epsilon_decay_rate, epsilon_min=FLAGS.epsilon_min, epsilon_max=FLAGS.epsilon_max, train_freq=FLAGS.train_frequency, target_network_update_freq=FLAGS.target_update_frequency, callback=deepq_callback)
                    act_x.save("mineral_shards_x.pkl")
                    act_y.save("mineral_shards_y.pkl")
            elif FLAGS.map == "FindAndDefeatZerglings":
                with sc2_env.SC2Env(map_name="FindAndDefeatZerglings", step_mul=step_mul, visualize=True, agent_interface_format=features.AgentInterfaceFormat(feature_dimensions=features.Dimensions(screen=64, minimap=64), use_feature_units=True)) as env:
                    model = deepq.models.cnn_to_mlp(convs=[(16, 8, 4), (32, 4, 2)], hiddens=[256], dueling=True)
                    act = deepq_defeat_zerglings.learn(env, q_func=model, num_actions=16, lr=FLAGS.lr, max_timesteps=FLAGS.timesteps, max_memory=FLAGS.max_memory, epsilon_decay_rate=FLAGS.epsilon_decay_rate, epsilon_min=FLAGS.epsilon_min, epsilon_max=FLAGS.epsilon_max, train_freq=FLAGS.train_frequency, target_network_update_freq=FLAGS.target_update_frequency, callback=deepq_callback)
                    act.save("defeat_zerglings.pkl")
            elif FLAGS.map == "MoveToBeacon":
                with sc2_env.SC2Env(map_name="MoveToBeacon", step_mul=step_mul, visualize=True, agent_interface_format=features.AgentInterfaceFormat(feature_dimensions=features.Dimensions(screen=64, minimap=64), use_feature_units=True)) as env:
                    model = deepq.models.cnn_to_mlp(convs=[(16, 8, 4), (32, 4, 2)], hiddens=[256], dueling=True)
                    act_x, act_y = deepq_beacon.learn(env, q_func=model, num_actions=16, lr=FLAGS.lr, max_timesteps=FLAGS.timesteps, max_memory=FLAGS.max_memory, epsilon_decay_rate=FLAGS.epsilon_decay_rate, epsilon_min=FLAGS.epsilon_min, epsilon_max=FLAGS.epsilon_max, train_freq=FLAGS.train_frequency, target_network_update_freq=FLAGS.target_update_frequency, callback=deepq_callback)
                    act_x.save("beacon_x.pkl")
                    act_y.save("beacon_y.pkl")
        elif FLAGS.algorithm == "a2c":
            seed = 0
            env = SubprocVecEnv(FLAGS.map)
            policy_fn = CnnPolicy
            a2c.learn(policy_fn, env, seed, total_timesteps=FLAGS.timesteps, ent_coef=FLAGS.regularization_strength, nsteps=FLAGS.trajectory_training_steps, max_grad_norm=FLAGS.value_gradient_strength, callback=a2c_callback)
    elif FLAGS.mode == "run":
        if FLAGS.algorithm == "a2c":
            num_timesteps = 2000
            seed = 0
            env = SubprocVecEnv(FLAGS.map)
            policy_fn = CnnPolicy
            a2c.run(policy_fn, env, seed, total_timesteps=FLAGS.timesteps, ent_coef=FLAGS.regularization_strength, nsteps=FLAGS.trajectory_training_steps, max_grad_norm=FLAGS.value_gradient_strength, callback=a2c_callback)


def deepq_callback(locals, globals):
    global max_mean_reward, last_filename
    if 'done' in locals and locals['done'] == True:
        if 'mean_100ep_reward' in locals and locals['num_episodes'] >= 10 and locals['mean_100ep_reward'] > max_mean_reward:
            print("mean_100ep_reward : %s max_mean_reward : %s" % (locals['mean_100ep_reward'], max_mean_reward))
            if not os.path.exists(os.path.join(PROJ_DIR, 'models/deepq')):
                try:
                    os.mkdir(os.path.join(PROJ_DIR, 'models/'))
                except Exception as e:
                    print(str(e))
                try:
                    os.mkdir(os.path.join(PROJ_DIR, 'models/deepq'))
                except Exception as e:
                    print(str(e))
            if last_filename != "":
                os.remove(last_filename)
                print("delete last model file: %s" % last_filename)
            max_mean_reward = locals['mean_100ep_reward']
            if FLAGS.map == "CollectMineralShards":
                act_x = deepq_mineral_shards.ActWrapper(locals['act_x'])
                act_y = deepq_mineral_shards.ActWrapper(locals['act_y'])
                filename = os.path.join(PROJ_DIR, 'models/deepq/mineral_x_%s.pkl' % locals['mean_100ep_reward'])
                act_x.save(filename)
                filename = os.path.join(PROJ_DIR, 'models/deepq/mineral_y_%s.pkl' % locals['mean_100ep_reward'])
                act_y.save(filename)
            elif FLAGS.map == "FindAndDefeatZerglings":
                act = deepq_defeat_zerglings.ActWrapper(locals['act'])
                filename = os.path.join(PROJ_DIR, 'models/deepq/zergling_%s.pkl' % locals['mean_100ep_reward'])
                act.save(filename)
            elif FLAGS.map == "MoveToBeacon":
                act_x = deepq_mineral_shards.ActWrapper(locals['act_x'])
                act_y = deepq_mineral_shards.ActWrapper(locals['act_y'])
                filename = os.path.join(PROJ_DIR, 'models/deepq/beacon_x_%s.pkl' % locals['mean_100ep_reward'])
                act_x.save(filename)
                filename = os.path.join(PROJ_DIR, 'models/deepq/beacon_y_%s.pkl' % locals['mean_100ep_reward'])
                act_y.save(filename)
            print("Save best mean_100ep_reward model to %s" % filename)
            last_filename = filename

def a2c_callback(locals, globals):
    global max_mean_reward, last_filename
    if 'mean_100ep_reward' in locals and locals['num_episodes'] >= 10 and locals['mean_100ep_reward'] > max_mean_reward:
        print("mean_100ep_reward: %s max_mean_reward : %s" % (locals['mean_100ep_reward'], max_mean_reward))
        if not os.path.exists(os.path.join(PROJ_DIR, 'models/a2c/')):
            try:
                os.mkdir(os.path.join(PROJ_DIR, 'models/'))
            except Exception as e:
                print(str(e))
            try:
                os.mkdir(os.path.join(PROJ_DIR, 'models/a2c/'))
            except Exception as e:
                print(str(e))
        if last_filename != "":
            os.remove(last_filename)
            print("delete last model file : %s" % last_filename)
        max_mean_reward = locals['mean_100ep_reward']
        model = locals['model']
        filename = os.path.join(PROJ_DIR, 'models/a2c/%s_%s.pkl' % (FLAGS.map, locals['mean_100ep_reward']))
        model.save(filename)
        print("save best mean_100ep_reward model to %s" % filename)
        last_filename = filename

if __name__ == '__main__':
    main()