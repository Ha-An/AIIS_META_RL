# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from torch.func import functional_call
from typing import Sequence, Type, List, Dict, Optional, Tuple, Callable
from AIIS_META.Utils.utils import *
from AIIS_META.Agents.base import BaseAgent
import time

class GaussianAgent(BaseAgent):
    """
    ... (주석 동일) ...
    """
    def __init__(self,
                 mlp,
                 gamma: float = 0.99,
                 learn_std: bool = True,
                 init_std: float = 1.0,
                 min_std: float = 1e-6,
                 state_dependent_std: bool = False,
                 has_value_fn: bool = False):
        super().__init__(mlp, gamma,
                         has_value_fn=has_value_fn)

        self.mlp = mlp
        self.gamma = gamma
        self.state_dependent_std = bool(state_dependent_std)
        if self.state_dependent_std:
            print("Warning: state_dependent_std=True 로직은 functional_call에 맞게 별도 수정이 필요할 수 있습니다.")

        self.min_log_std = torch.log(torch.tensor(min_std))
        init_log_std = float(torch.log(torch.tensor(init_std)))
        p = nn.Parameter(torch.full((self.mlp.output_dim,), init_log_std), requires_grad=True)
        self.register_parameter("log_std", p)
        self.log_std = p

    # --- [★추가★] ---
    def set_adapted_params(self, params: Optional[Dict[str, torch.Tensor]]):
        """
        (Stateful) get_actions가 사용할 adapted 파라미터를 
        모듈 내부에 저장합니다.
        
        Args:
            params (Dict): theta_prime에서 반환된 파라미터 딕셔너리.
                           None으로 설정하면 다시 기본 파라미터를 사용합니다.
        """
        self.adapted_params = params
    # --- [★추가★] ---

    # ---------------- build distribution (functional) ----------------
    # 이 함수는 'params'를 필수로 받습니다. (변경 없음)
    def distribution(self, obs: torch.Tensor,
                     params: Dict[str, torch.Tensor]) -> Independent:
        """
        (Functional) 'params' 딕셔너리를 *반드시* 사용하여 분포를 생성합니다.
        """
        device, dtype = module_device_dtype(self.mlp) 
        obs = torch.as_tensor(obs, device=device, dtype=dtype)

        mlp_params = {k.removeprefix('mlp.'): v 
                      for k, v in params.items() 
                      if k.startswith('mlp.')}
        mean = functional_call(self.mlp, mlp_params, (obs,))
        log_std_unclamped = params['log_std']
        
        log_std = torch.clamp(log_std_unclamped, min=self.min_log_std)
        std = torch.max(torch.exp(log_std), torch.exp(self.min_log_std))

        std = to_tensor(std, device=device)
        mean = to_tensor(mean, device=device)
        base = Normal(mean, std)
        return Independent(base, 1)

    # ---------------- act (Stateful) ----------------
    @torch.no_grad()
    # --- [★변경★] ---
    # 'params' 인자를 제거합니다.
    def get_actions(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        (Stateful) 모듈의 '현재' 파라미터를 사용하여 행동을 샘플링합니다.
        - self.adapted_params가 설정되어 있으면, 그것을 사용합니다. (Post-update)
        - None이면, self.named_parameters()를 사용합니다. (Pre-update)
        """
        
        # 1. 사용할 파라미터를 결정
        if self.adapted_params is not None:
            # Post-update 모드: 저장된 adapted 파라미터 사용
            current_params = self.adapted_params
        else:
            # Pre-update 모드: 모듈의 기본 파라미터 사용
            current_params = dict(self.named_parameters())
        # --- [★변경★] ---

        # 2. functional distribution 호출 (그래디언트 추적 안 함)
        dist = self.distribution(obs, params=current_params) 
        
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        
        mean = dist.mean
        log_std = torch.log(dist.base_dist.scale)
        logp = dist.log_prob(action)
        
        agent_info = [[dict(logp=logp[task_idx][rollout_idx]) for rollout_idx in range(len(logp[task_idx]))] for task_idx in range(self.num_tasks)]
        return action, agent_info
    
    # ---------------- log_prob (Functional) ----------------
    # 이 함수는 'params'를 필수로 받습니다. (변경 없음)
    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor, 
                 params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        (Functional) Loss 계산을 위해 'params'를 반드시 사용합니다.
        """
        dist = self.distribution(obs, params=params)
        logp = dist.log_prob(actions)
        
        return logp