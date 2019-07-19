import sys
import os
import datetime
from absl import flags
from baselines import deepq
from pysc2.env import sc2_env
from pysc2.lib import actions, features
import deepq_defeat_zerglings
from baselines.logger import Logger, TensorBoardOutputFormat, HumanOutputFormat

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

step_mul = 1
steps = 2000

FLAGS = flags.FLAGS
start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
flags.DEFINE_string("log", "tensorboard", "loggin type(stdout, tensorboard)")
flags.DEFINE_string("algorithm", "deepq", "RL algorithm to use.")
flags.DEFINE_integer("timesteps", 2000000, "Steps to train")
flags.DEFINE_float("exploration_fraction", 0.5, "Exploration Fraction")
flags.DEFINE_boolean("prioritized", True, "prioritized_replay")
flags.DEFINE_boolean("dueling", True, "dueling")
flags.DEFINE_float("lr", 0.001, "Learning rate")
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
max_mean_reward = 0
last_filename = ""


def main():
    FLAGS(sys.argv)
    logdir = "tensorboard"
    if FLAGS.algorithm == "deepq":
        logdir = "tensorboard/zergling/%s/%s_%s_prio%s_duel%s_lr%s/%s" % (FLAGS.algorithm, FLAGS.timesteps, FLAGS.exploration_fraction, FLAGS.prioritized, FLAGS.dueling, FLAGS.lr, start_time)
