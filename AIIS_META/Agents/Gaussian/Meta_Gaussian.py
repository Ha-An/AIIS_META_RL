# Meta_Gaussian.py  (수정본)
import torch
from torch.distributions.independent import Independent
from typing import List, Dict, Optional, Tuple
from AIIS_META.Utils.utils import *
from .Gaussian import GaussianAgent  # 수정된 GaussianAgent 임포트
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

    @torch.no_grad()
    # 'params' 딕셔너리를 필수로 받습니다.
    def get_actions(self, obs: torch.Tensor, 
                  params: Dict[str, torch.Tensor], # <--- ★이 인자가 필수!
                  deterministic: bool = False, post_update = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        (Fully Functional) 제공된 'params' 딕셔너리를 사용하여 행동을 샘플링.
        """
        if post_update:
            # 1. 'params'를 'distribution'에 직접 전달
            current_params = params
        else:
            current_params = dict(self.named_parameters())

        dist = self.distribution(obs, params=current_params)
        
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        
        logp = dist.log_prob(action)
        
        agent_info = [[dict(logp=logp[task_idx][rollout_idx]) for rollout_idx in range(len(logp[task_idx]))] for task_idx in range(self.num_tasks)]
        return action, agent_info

    ### 2. log_prob_by_params의 'params'도 필수로 변경 ###
    def log_prob_by_params(self, obses, actions, 
                           params: Dict[str, torch.Tensor], 
                           deterministic: bool = False):
        """
        (Functional) 부모 클래스의 functional distribution을 호출합니다.
        반드시 'params' 딕셔너리를 사용해야 합니다.
        """
        # params 인자를 부모의 distribution 메서드로 전달
        dist = self.distribution(obses, params=params)
        log_ps = dist.log_prob(actions)
    
        return log_ps