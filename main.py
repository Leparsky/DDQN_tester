# https://github.com/germain-hug/Deep-RL-Keras
""" Deep Q-Learning for OpenAI Gym environment
"""

import os
import sys
import csv
#import gym
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from environment import Environment

# from A2C.a2c import A2C
# from A3C.a3c import A3C
from DDQN.ddqn import DDQN
# from DDPG.ddpg import DDPG

from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical

# from utils.atari_environment import AtariEnvironment
# from utils.continuous_environments import Environment
from utils.networks import get_session
from utils.csvwriter import WritetoCsvFile
# gym.logger.set_level(40)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    #
    parser.add_argument('--trainf', type=str, default='D:\PycharmProjects\RIZ8\SPFB.RTS-3.18(5M).csv', help="File For train")
    parser.add_argument('--evalf', type=str, default='D:\PycharmProjects\RIZ8\SPFB.RTS-6.18(5M).csv', help="File for evaluate")
    parser.add_argument('--usevol', dest='usevol', default=False, action='store_true', help="Add volume to environment")
    parser.add_argument('--type', type=str, default='DDQN', help="Algorithm to train from {A2C, A3C, DDQN, DDPG}")
    # parser.add_argument('--is_atari', dest='is_atari', action='store_true', help="Atari Environment")

    parser.add_argument('--history_win', type=int, default=90, help="Number history window")
    parser.add_argument('--with_PER', dest='with_per', action='store_true',
                        help="Use Prioritized Experience Replay (DDQN + PER)")
    parser.add_argument('--dueling', dest='dueling', action='store_true', help="Use a Dueling Architecture (DDQN)")
    parser.add_argument('--traineval', dest='traineval', action='store_true', help="train while eval")
    parser.add_argument('--allprices', dest='allprices', action='store_true',
                        help="use all stock prices open close high low")
    parser.add_argument('--allprices2', dest='allprices2', action='store_true',
                        help="use all stock prices open close high low [algoritm2]")

    parser.add_argument('--candlenum', dest='candlenum', action='store_true', help="use day weekday and candlenumber")

    #
    parser.add_argument('--nb_episodes', type=int, default=50, help="Number of training episodes")
    parser.add_argument('--batch_size', type=int, default=20, help="Batch size (experience replay)")
    # parser.add_argument('--consecutive_frames', type=int, default=4, help="Number of consecutive frames (action repeat)")
    # parser.add_argument('--training_interval', type=int, default=30, help="Network training frequency")
    # parser.add_argument('--n_threads', type=int, default=8, help="Number of threads (A3C)")
    #
    parser.add_argument('--gather_stats', dest='gather_stats', action='store_true',
                        help="Compute Average reward per episode (slower)")
    # parser.add_argument('--render', dest='render', action='store_true', help="Render environment while training")
    # parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4',help="OpenAI Gym Environment")
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')

    #
    parser.set_defaults(render=False)
    return parser.parse_args(args)


def main(args=None):
    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Check if a GPU ID was set
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_session(get_session())
    summary_writer = tf.summary.FileWriter(args.type + "/tensorboard_" + args.type)

    # Environment Initialization
    '''if(args.is_atari):
        # Atari Environment Wrapper
        env = AtariEnvironment(args)
        state_dim = env.get_state_size()
        action_dim = env.get_action_size()
    elif(args.type=="DDPG"):
        # Continuous Environments Wrapper
        env = Environment(gym.make(args.env), args.consecutive_frames)
        env.reset()
        state_dim = env.get_state_size()
        action_space = gym.make(args.env).action_space
        action_dim = action_space.high.shape[0]
        act_range = action_space.high
    else:'''

    # Standard Environments

    env = Environment(args)
    envtest = Environment(args)

    env.GetStockDataVecFN(args.trainf, False)
    envtest.GetStockDataVecFN(args.evalf, False)



    state_dim = env.get_state_size()
    action_dim = env.get_action_size()

    # Pick algorithm to train
    if (args.type == "DDQN"):
        algo = DDQN(action_dim, state_dim, args)
    # elif(args.type=="A2C"):
    #    algo = A2C(action_dim, state_dim, args.consecutive_frames)
    # elif(args.type=="A3C"):
    #    algo = A3C(action_dim, state_dim, args.consecutive_frames, is_atari=args.is_atari)
    # elif(args.type=="DDPG"):
    #    algo = DDPG(action_dim, state_dim, act_range, args.consecutive_frames)

    # Train
    stats = algo.train(env, args, summary_writer,envtest)
    # Assuming res is a list of lists
    WritetoCsvFile("logFile.csv",
                   ["stage", "file", "history_win", "usevol", "dueling", "traineval", "allprices", "allprices2", "candlenum", "maxProfit", "maxLOSS", "avgProfit", "avgLOSS", "maxdrop",
                    "Total profit", "TRADES"])
    WritetoCsvFile("logFile.csv", ["train",args.trainf, args.history_win, args.usevol, args.dueling, args.traineval, args.allprices, args.allprices2, args.candlenum] + stats)

    env.GetStockDataVecFN(args.evalf, False)
    # evaluate
    #stats = algo.evaluate(envtest, args, summary_writer, "./models/model_ep")
    # Assuming res is a list of lists
    #WritetoCsvFile("logFile.csv", ["stage", "file", "history_win", "usevol","dueling", "traineval", "allprices", "allprices2", "candlenum", "maxProfit", "maxLOSS", "avgProfit", "avgLOSS", "maxdrop", "Total profit", "TRADES"])
    #WritetoCsvFile("logFile.csv", ["evaluate", args.evalf, args.history_win, args.usevol, args.dueling, args.traineval, args.allprices, args.allprices2, args.candlenum] + stats )

if __name__ == "__main__":
    main()