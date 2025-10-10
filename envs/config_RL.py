import os
import shutil
from envs.config_SimPy import *

# RL algorithms
RL_ALGORITHM = "PPO"  # "DP", "DQN", "DDPG", "PPO", "SAC"
RL_EXPERIMENT = True
ACTION_SPACE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]