# -*- coding: utf-8 -*-
from AIIS_META.Sampler.base import Sampler
from AIIS_META.Utils.vectorized_env_executor import MetaParallelEnvExecutor, MetaIterativeEnvExecutor
from AIIS_META.Utils import utils
from collections import OrderedDict

from pyprind import ProgBar
import numpy as np
import itertools
import time

class MetaSampler(Sampler):
    """
    Sampler for Meta-RL

    Args:
        env (meta_agent_search.envs.base.MetaEnv) : environment object
        agent (meta_agent_search.policies.base.agent) : agent object
        batch_size (int) : number of trajectories per task
        num_tasks (int) : number of meta tasks
        max_path_length (int) : max number of steps per trajectory
        envs_per_task (int) : number of envs to run vectorized for each task (influences the memory usage)
    """

    def __init__(
            self,
            env,
            agent,
            rollout_per_task,
            num_tasks,
            max_path_length,
            envs_per_task,
            parallel=False
            ):
        super(MetaSampler, self).__init__(env,
            agent,
            rollout_per_task,
            max_path_length)
        assert hasattr(env, 'set_task')
        self.envs_per_task = rollout_per_task if envs_per_task is None else envs_per_task
        self.num_tasks = num_tasks
        self.rollout_per_task = rollout_per_task
        self.total_samples = num_tasks * rollout_per_task * max_path_length
        print(self.total_samples)
        self.parallel = parallel
        self.total_timesteps_sampled = 0
        self.agent = agent  # old param

        # setup vectorized environment
        if self.parallel:
            self.vec_env = MetaParallelEnvExecutor(env, self.num_tasks, self.envs_per_task, self.max_path_length)
        else:
            self.vec_env = MetaIterativeEnvExecutor(env, self.num_tasks, self.envs_per_task, self.max_path_length)

        # 캐시: 액션 차원
        self._act_dim = int(np.prod(self.env.action_space.shape))
        self._num_envs = int(self.num_tasks * self.envs_per_task)

    def update_tasks(self):
        """
        Samples a new goal for each meta task
        """
        tasks = self.env.sample_tasks(self.num_tasks)
        assert len(tasks) == self.num_tasks
        self.vec_env.set_tasks(tasks)

    def obtain_samples(self, params_lst, post_update=False):
        """
        Collect batch_size trajectories from each task

        Args:
            log (boolean): whether to log sampling times
            agent_inforefix (str) : prefix for logger

        Returns: 
            (dict) : A dict of paths of size [num_tasks] x (batch_size) x [5] x (max_path_length)
        """

        # initial setup / preparation
        paths = OrderedDict()
        for i in range(self.num_tasks):
            paths[i] = []

        n_samples = 0
        running_paths = [_get_empty_running_paths_dict() for _ in range(self.vec_env.num_envs)]

        pbar = ProgBar(self.total_samples)
        agent_time, env_time = 0, 0

        # initial reset of envs
        obses = self.vec_env.reset()

        while n_samples < self.total_samples:
            
            # execute agent
            t = time.time()
            obs_per_task = np.split(np.asarray(obses), self.num_tasks)
            
            if post_update:
                actions, agent_info = self.policy.get_actions(obs_per_task, params=params_lst[n_samples//self.envs_per_task], post_update = True)
            else:
                actions, agent_info = self.agent.get_actions(obs_per_task, params=None)
            agent_time += time.time() - t

            # step environments
            t = time.time()
            actions = np.concatenate(actions.detach().cpu().numpy()) # stack meta batch
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
            
            env_time += time.time() - t

            #  stack agent_info and if no infos were provided (--> None) create empty dicts
            agent_info, env_infos = self._handle_info_dicts(agent_info, env_infos)
            
            new_samples = 0
            for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                    rewards, env_infos, agent_info,
                                                                                    dones):
                # append new samples to running paths
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_info"].append(agent_info)

                # if running path is done, add it to paths and empty the running path
                if done:
                    paths[idx // self.envs_per_task].append(dict(
                        observations=np.asarray(running_paths[idx]["observations"]),
                        actions=np.asarray(running_paths[idx]["actions"]),
                        rewards=np.asarray(running_paths[idx]["rewards"]),
                        env_infos=utils.stack_envs(running_paths[idx]["env_infos"]),
                        agent_info=utils.stack_agent_infos(running_paths[idx]["agent_info"]),))
                    new_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = _get_empty_running_paths_dict()

            pbar.update(new_samples)
            n_samples += new_samples
            obses = next_obses
        pbar.stop()

        self.total_timesteps_sampled += self.total_samples
        return paths

    def _handle_info_dicts(self, agent_info, env_infos):
        if not env_infos:
            env_infos = [dict() for _ in range(self.vec_env.num_envs)]
        if not agent_info:
            agent_info = [dict() for _ in range(self.vec_env.num_envs)]
        else:
            assert len(agent_info) == self.num_tasks
            assert len(agent_info[0]) == self.envs_per_task
            agent_info = sum(agent_info, [])  # stack agent_info

        assert len(agent_info) == self.num_tasks * self.envs_per_task == len(env_infos)
        return agent_info, env_infos
    
    def close(self):
        """
        vec_env가 프로세스를 쓰는 경우(Parallel) worker들을 정리한다.
        """
        if hasattr(self, "vec_env") and hasattr(self.vec_env, "close"):
            self.vec_env.close()

def _get_empty_running_paths_dict():
    return dict(observations=[], actions=[], rewards=[], env_infos=[], agent_info=[])

