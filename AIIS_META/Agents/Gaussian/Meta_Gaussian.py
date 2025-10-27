# Meta_Gaussian.py  (수정본)
import torch
from torch.distributions.independent import Independent
from typing import List, Dict, Optional, Tuple
from AIIS_META.Utils.utils import *
from .Gaussian import GaussianAgent  # 기존 구현 사용
import torch.optim as optim

class MetaGaussianAgent(GaussianAgent):
    def __init__(self, num_tasks: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tasks = num_tasks
        self._pre_update_mode = True
        print("gaussian policy ready")

    def set_pre_update_mode(self, flag: bool = True):
        self._pre_update_mode = flag
        if flag:
            self.policies_params_vals = None

    # -----------------------------------------------------------
    # 핵심: 하나의 observation에 대해 out_dim 개수만큼의 mean/std/action/logp 정보를 반환
    # -----------------------------------------------------------

    def get_actions(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        obs: [B, obs_dim] or [obs_dim]
        returns: actions [B, out_dim], info dict {mean, log_std, logp}
        """
        dist =self.distribution(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()  # reparameterized sample (rsample) could be used; .sample() also ok

        logps = dist.log_prob(action)
        
        agent_info = [[dict(logp=logps[task_idx][rollout_idx]) for rollout_idx in range(len(logps[task_idx]))] for task_idx in range(self.num_tasks)]

        return action, agent_info

    def log_prob_by_params(self, obses,  actions, deterministic: bool = False):
        dist =self.distribution(obses)
        log_ps = dist.log_prob(actions)
    
        return log_ps
    
    def forward(self, obs, actions):
        # functional_call(agent, params, (obs, actions))에서 호출될 엔트리
        return self.get_outer_log_probs(obs, actions)