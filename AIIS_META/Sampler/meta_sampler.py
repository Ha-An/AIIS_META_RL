# -*- coding: utf-8 -*-
from AIIS_META.Sampler.base import Sampler
from AIIS_META.Utils.vectorized_env_executor import MetaParallelEnvExecutor, MetaIterativeEnvExecutor
from AIIS_META.Utils import utils
from collections import OrderedDict

from pyprind import ProgBar
import numpy as np
import itertools
import torch

class MetaSampler(Sampler):
    """
    Sampler for Meta-RL

    Args:
        env (meta_policy_search.envs.base.MetaEnv) : environment object
        policy (meta_policy_search.policies.base.Policy) : agent object
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
            envs_per_task=None,
            parallel=False
            ):
        super(MetaSampler, self).__init__(env,
            agent,
            rollout_per_task,
            num_tasks,
            max_path_length,
            envs_per_task=None,
            parallel=False)
        assert hasattr(env, 'set_task')
        self.envs_per_task = rollout_per_task if envs_per_task is None else envs_per_task
        self.num_tasks = num_tasks
        self.rollout_per_task = rollout_per_task
        self.total_samples = num_tasks * rollout_per_task * max_path_length
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

    # -------------------- 내부 유틸 --------------------

    @staticmethod
    def _get_empty_running_paths_dict():
        return dict(observations=[], actions=[], rewards=[], env_infos=[], agent_infos=[], dones=[])

    @staticmethod
    def _to_env_action(a):
        """환경으로 보낼 안전한 numpy 액션."""
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy().astype(np.float32)
        return np.asarray(a, dtype=np.float32)

    def _split_per_task(self, batched_obs: np.ndarray):
        """(num_envs, obs_dim) -> list[num_tasks] of (envs_per_task, obs_dim)"""
        assert batched_obs.shape[0] == self._num_envs, f"expected {self._num_envs}, got {batched_obs.shape[0]}"
        return list(np.split(batched_obs, self.num_tasks, axis=0))

    def _flatten_infos(self, agent_infos, env_infos):
        """agent_infos: [num_tasks][envs_per_task] -> flat O(n)"""
        if not env_infos:
            env_infos = [dict() for _ in range(self.vec_env.num_envs)]
        else:
            assert len(env_infos) == self.vec_env.num_envs, f"env_infos len {len(env_infos)} != {self.vec_env.num_envs}"

        if not agent_infos:
            agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
        else:
            # 기대 형태: list[num_tasks][envs_per_task]
            assert len(agent_infos) == self.num_tasks
            assert len(agent_infos[0]) == self.envs_per_task
            agent_infos = [x for row in agent_infos for x in row]  # O(n)

        assert len(agent_infos) == self.vec_env.num_envs
        return agent_infos, env_infos

    def _maybe_load_task_params(self, agents, next_task_idx, post_update: bool):
        """post_update에서만 self.agent의 파라미터를 교체(객체 교체 X)."""
        if not post_update or agents is None:
            return
        if next_task_idx >= self.num_tasks:
            return
        sd = None
        # agents가 모델 리스트일 수도, state_dict 리스트일 수도 있음
        if isinstance(agents[next_task_idx], dict):
            sd = agents[next_task_idx]
        elif hasattr(agents[next_task_idx], "state_dict"):
            sd = agents[next_task_idx].state_dict()
        if sd is not None:
            with torch.no_grad():
                self.agent.load_state_dict(sd)

    # -------------------- 공개 API --------------------

    def obtain_samples(self, agents, post_update=False):
        """
        Collect batch_size trajectories from each task

        Args:
            log (boolean): whether to log sampling times
            log_prefix (str) : prefix for logger

        Returns:
            (dict) : A dict of paths of size [num_tasks] x (batch_size) x [5] x (max_path_length)
        """
        self.vec_env.set_tasks(self.env.tasks)

        # initial setup / preparation
        paths = OrderedDict((i, []) for i in range(self.num_tasks))
        n_samples = 0
        running_paths = [self._get_empty_running_paths_dict() for _ in range(self.vec_env.num_envs)]
        pbar = ProgBar(self.total_samples)

        # initial reset of envs (numpy 보장)
        obses = self.vec_env.reset()
        obses = np.asarray(obses, dtype=np.float32)

        rollout_counter = 0
        # post_update일 때 태스크 전환을 episode 완료 기준으로 수행
        current_task_idx = 0
        if post_update:
            self._maybe_load_task_params(agents, current_task_idx, post_update=True)

        while n_samples < self.total_samples:
            # --- 정책 입력: 태스크별로 분할 (numpy 유지) ---
            obs_per_task = self._split_per_task(obses)

            # --- 액션/정보 얻기 (정책 내부에서 텐서 변환, 출력은 numpy 권장) ---
            actions_nested, agent_infos_nested = self.agent.get_actions(obs_per_task)

            # 2) actions_nested → numpy로 (태스크 블록마다)
            if isinstance(actions_nested, list):
                actions_blocks = [utils.to_numpy(a, dtype=np.float32) for a in actions_nested]  # 각 (envs_per_task, act_dim)
                actions = utils.np.concatenate(actions_blocks, axis=0)                           # (num_envs, act_dim)
            else:
                actions = utils.to_numpy(actions_nested, dtype=np.float32).reshape(self._num_envs, self._act_dim)

            # 3) agent_infos_nested → numpy로 (구조 보존)
            #    기대 구조: list[num_tasks][envs_per_task] of dict
            if isinstance(agent_infos_nested, list):
                agent_infos_nested_np = []
                for row in agent_infos_nested:
                    agent_infos_nested_np.append([utils.dict_to_numpy(info, dtype=np.float32) for info in row])
            else:
                # 단일 dict인 경우 등
                agent_infos_nested_np = [[utils.dict_to_numpy(agent_infos_nested, dtype=np.float32)]]

            actions = actions.reshape(self._num_envs, self._act_dim)

            # --- env로 보낼 안전 리스트(np.ndarray) ---
            action_list = [self._to_env_action(a) for a in actions]

            # --- step ---
            next_obses, rewards, dones, env_infos = self.vec_env.step(action_list)
            next_obses = np.asarray(next_obses, dtype=np.float32)
            rewards    = np.asarray(rewards, dtype=np.float32).reshape(self._num_envs)
            dones      = np.asarray(dones, dtype=np.bool_).reshape(self._num_envs)

            # --- infos 평탄화 (O(n)) ---
            agent_infos, env_infos = self._flatten_infos(agent_infos_nested, env_infos)

            # --- 런닝 버퍼 적재 ---
            new_samples = 0
            for idx, observation, action, reward, env_info, agent_info, done in zip(
                itertools.count(), obses, actions, rewards, env_infos, agent_infos, dones
            ):
                rp = running_paths[idx]
                rp["observations"].append(observation)
                rp["actions"].append(action)
                rp["rewards"].append(reward)
                rp["env_infos"].append(env_info)
                rp["agent_infos"].append(agent_info)
                rp["dones"].append(done)

                # 에피소드 종료 시 path로 완성
                if done or len(rp["dones"]) >= self.max_path_length:
                    paths[idx // self.envs_per_task].append(dict(
                        observations=np.asarray(rp["observations"], dtype=np.float32),
                        actions=np.asarray(rp["actions"], dtype=np.float32),
                        rewards=np.asarray(rp["rewards"], dtype=np.float32),
                        env_infos=rp["env_infos"],
                        agent_infos=rp["agent_infos"],
                        dones=np.asarray(rp["dones"], dtype=np.bool_),
                    ))
                    new_samples += len(rp["rewards"])
                    running_paths[idx] = self._get_empty_running_paths_dict()
                    rollout_counter += 1

                    # post_update: 해당 태스크의 quota(rollout_per_task) 채우면 다음 태스크 파라미터 로드
                    if post_update and (rollout_counter % self.rollout_per_task == 0):
                        current_task_idx = rollout_counter // self.rollout_per_task
                        self._maybe_load_task_params(agents, current_task_idx, post_update=True)

            pbar.update(new_samples)
            n_samples += new_samples
            obses = next_obses

        pbar.stop()
        self.total_timesteps_sampled += self.total_samples
        return paths
