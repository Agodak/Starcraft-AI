import sys
from absl import flags
import baselines.deepq.utils as U
from baselines import deepq
from pysc2.env import environment, sc2_env
from pysc2.lib import actions
from pysc2.lib import features
import deepq_mineral_shards


_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]
step_mul = 16
steps = 400
FLAGS = flags.FLAGS
flags.DEFINE_string("map", "CollectMineralShards", "Name of a map to use to play.")

def main():
    FLAGS(sys.argv)
    if FLAGS.algorithm == 'deepq':
        with sc2_env.SC2Env(map_name="CollectMineralShards", step_mul=step_mul, visualize=True, game_steps_per_episode=steps * step_mul, agent_interface_format=features.AgentInterfaceFormat(feature_dimensions=features.Dimensions(screen=16, minimap=16), use_feature_units=True)) as env:
            model = deepq.models.cnn_to_mlp(convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], hiddens=[256], dueling=True)

            def make_obs_ph(name):
                return U.BatchInput((1, 64, 64), name=name)
            act_params = {
                'make_obs_ph': make_obs_ph,
                'q_func': model,
                'num_actions': 4,
            }
            act_x = deepq_mineral_shards.load("mineral_shards_x.pkl", act_params=act_params)
            act_y = deepq_mineral_shards.load("mineral_shards_y.pkl", act_params=act_params)
            act = [act_x, act_y]

if __name__ == '__main__':
    main()