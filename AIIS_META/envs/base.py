from gym.core import Env
import numpy as np
import random
from envs.config_SimPy import *
from gym import spaces
import numpy as np
from envs.config_RL import * 
# import environment as env # for Meta-learning
import envs.environment as env #for baseline_test
from envs.log_SimPy import *
from envs.log_RL import *
from envs.scenarios import *


class MetaEnv(Env):
    """
    Wrapper around OpenAI gym environments, interface for meta learning
    """
    def __init__(self, env):
        super(MetaEnv, self).__init__()
        self.candidate_tasks = self.create_scenarios()
        self.env = env
        self.scenario = None
    
    def create_scenraios(self):
        '''
        return scenario
        '''
        raise NotImplementedError
    
    def sample_tasks(self, n_tasks):
        """
        Samples task of the meta-environment

        Args:
            n_tasks (int) : number of different meta-tasks needed

        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        print("all tasks reset")
        self.tasks = random.sample(self.candidate_tasks, n_tasks)
        self.candidate_tasks = create_scenarios()
        return self.tasks

    def set_task(self, task):
        #print("Task_Setted: ", task)
        """
        Sets the specified task to the current environment

        Args:
            task: task of the meta-learning environment
        """
        self.scenario = task

    def get_task(self):
        """
        Gets the task that the agent is performing in the current environment

        Returns:
            task: task of the meta-learning environment
        """
        return self.scenario

    def reset(self):
        #return state
        raise NotImplementedError

    def step(self, action):
        #return next_state, reward, done, self.cost_dict
        raise NotImplementedError
        

    def report_scalar(self, paths, rollouts):
        """
        Logs env-specific diagnostic information

        Args:
            paths (list) : list of all paths collected with this env during this iteration
            prefix (str) : prefix for logger
        """
        infos = {}

        rewards = 0
        for path in paths:
            rewards += sum(path['rewards'])

        average = rewards/(len(paths)*rollouts)
        infos['reward'] = average
        '''
        Write Here Additional Infos for logging
        '''
        return infos
 
    