# sample_processor_refactored.py
from AIIS_META.Utils import utils
import numpy as np
import torch

class Sampler(object):
    """
    Sampler interface

    Args:
        env (gym.Env) : environment object
        agent : agent object
        batch_size (int) : number of trajectories per task
        max_path_length (int) : max number of steps per trajectory
    """

    def __init__(self,
            env,
            agent,
            rollout_per_task,
            meta_batch_size,
            max_path_length,
            envs_per_task=None,
            parallel=False):
        assert hasattr(env, 'reset') and hasattr(env, 'step')
        self.env = env
        self.agent = agent
        self.batch_size = meta_batch_size
        self.max_path_length = max_path_length

    def obtain_samples(self):
        """Collect batch_size trajectories -> List[Path]"""
        raise NotImplementedError


class SampleProcessor(object):
    """
    Sample processor
      - (옵션) 주어진 external advantages 사용
      - 아니면 baseline을 fit하고 GAE로 advantage 계산
      - normalize/shift 옵션 제공

    Args:
        baseline (Baseline or None) : reward baseline object (fit/predict 필요)
        discount (float) : gamma
        gae_lambda (float) : lambda
        normalize_adv (bool) : advantage 정규화
        positive_adv (bool) : advantage를 양수로 쉬프트
    """

    def __init__(
        self,
        baseline=None,
        discount=0.99,
        gae_lambda=1.0,
        normalize_adv=False,
        positive_adv=False,
        device = None,
    ):
        assert 0.0 <= discount <= 1.0, 'discount factor must be in [0,1]'
        assert 0.0 <= gae_lambda <= 1.0, 'gae_lambda must be in [0,1]'
        # baseline은 외부 adv를 쓰면 없어도 됨
        if baseline is not None:
            assert hasattr(baseline, 'fit') and hasattr(baseline, 'predict')
        self.baseline = baseline
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.normalize_adv = normalize_adv
        self.positive_adv = positive_adv
        self.device = device

    def process_samples(self, paths, log=False, log_prefix=''):
        """
        Processes sampled paths. This involves:
            - computing discounted rewards (returns)
            - fitting baseline estimator using the path returns and predicting the return baselines
            - estimating the advantages using GAE (+ advantage normalization id desired)
            - stacking the path data
            - logging statistics of the paths

        Args:
            paths (list): A list of paths of size (batch_size) x [5] x (max_path_length)
            log (boolean): indicates whether to log
            log_prefix (str): prefix for the logging keys

        Returns:
            (dict) : Processed sample data of size [7] x (batch_size x max_path_length)
        """
        assert type(paths) == list, 'paths must be a list'
        assert paths[0].keys() >= {'observations', 'actions', 'rewards'}
        assert self.baseline, 'baseline must be specified - use self.build_sample_processor(baseline_obj)'

        # fits baseline, compute advantages and stack path data
        samples_data, paths = self._compute_samples_data(paths)

        # 7) log statistics if desired
        self._log_path_stats(paths, log=log, log_prefix='')

        assert samples_data.keys() >= {'observations', 'actions', 'rewards', 'advantages', 'returns'}
        return samples_data
    
    def _compute_samples_data(self, paths):
        assert type(paths) == list

        # 1) compute discounted rewards (returns)
        for idx, path in enumerate(paths):
            path["returns"] = utils.discount_cumsum(path["rewards"], self.discount, device=self.device)

        # 2) fit baseline estimator using the path returns and predict the return baselines
        self.baseline.fit(paths, target_key="returns")
        all_path_baselines = [self.baseline.predict(path,self.device) for path in paths]

        paths = self._compute_advantages(paths, all_path_baselines)
        
        # 4) stack path data
        observations, actions, rewards, returns, advantages, env_infos, agent_infos = self._stack_path_data(paths)

        # 5) if desired normalize / shift advantages
        if self.normalize_adv:
            advantages = utils.normalize_advantages(advantages)
        if self.positive_adv:
            advantages = utils.shift_advantages_to_positive(advantages)
    
        # 6) create samples_data object
        samples_data = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            returns=returns,
            advantages=advantages,
            env_infos=env_infos,
            agent_infos=agent_infos,
        )

        return samples_data, paths

    
    def _stack_path_data(self, paths):
        observations = [path["observations"] for path in paths]
        actions = [path["actions"] for path in paths]
        rewards = [path["rewards"] for path in paths]
        returns = [path["returns"] for path in paths]
        advantages = [path["advantages"] for path in paths]
        env_infos = [utils.stack_tensor_dict_list(path["env_infos"]) for path in paths]
        agent_infos = [utils.stack_tensor_dict_list(path["agent_infos"]) for path in paths]

        return observations, actions, rewards, returns, advantages, env_infos, agent_infos
    
    def _compute_advantages(self, paths, all_path_baselines):
        assert len(paths) == len(all_path_baselines)
     
        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] + \
                     self.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = utils.discount_cumsum(
                deltas, self.discount * self.gae_lambda, device=self.device)

        return paths
