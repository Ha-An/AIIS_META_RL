# -*- coding: utf-8 -*-
"""
Single-head diagonal Gaussian mlp (out_dim 기반)
- mean: backbone -> single linear head -> [B, out_dim]
- std:
    - 기본: state-independent log_std parameter of shape [out_dim]
    - 선택: state_dependent_std=True -> head outputs 2*out_dim (mean, log_std)
Implements common methods: forward, distribution, act, log_prob, functional forward/distribution.
"""
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from torch.func import functional_call
from typing import Sequence, Type, List, Dict, Optional, Tuple, Callable
from AIIS_META.Utils.utils import *
from AIIS_META.Agents.base import BaseAgent  # 사용 중인 base 인터페이스에 맞춰져 있어야 함

class GaussianAgent(BaseAgent): # mlp를 Agent로 변경해야함
    """
    out_dim 기준의 단일-head diagonal Gaussian policy.
    Args:
      obs_dim, out_dim: 차원
      hidden, activation: backbone
      learn_std: state-independent std를 학습할지 여부 (only used when state_dependent_std=False)
      init_std: 초기 std 값 (state-independent일 때)
      min_std: numerical 안정성 위한 최소 std
      state_dependent_std: True면 head가 2*out_dim 출력 (mean, log_std)
    """
    def __init__(self,
                 mlp,
                 gamma: float = 0.99,
                 learn_std: bool = True,
                 init_std: float = 1.0,
                 min_std: float = 1e-6,
                 state_dependent_std: bool = False,
                 has_value_fn: bool = False):
        super().__init__(mlp,gamma,
                         has_value_fn=has_value_fn)

        self.mlp = mlp
        self.gamma = gamma
        self.state_dependent_std = bool(state_dependent_std)
        self.min_log_std = float(torch.log(torch.tensor(min_std)))
        # state-independent log_std param (only used if not state_dependent_std)
        init_log_std = float(torch.log(torch.tensor(init_std)))
        if not self.state_dependent_std:
            if learn_std:
                self.log_std = nn.Parameter(torch.full((self.mlp.output_dim,), init_log_std))
            else:
                p = nn.Parameter(torch.full((self.mlp.output_dim,), init_log_std), requires_grad=False)
                # still register as parameter so it appears in state_dict
                self.register_parameter("log_std", p)
                self.log_std = p
        else:
            # placeholder for attribute existence (won't be used)
            self.register_parameter("log_std", None)

    # ---------------- build distribution (current params) ----------------
    def distribution(self, obs: torch.Tensor) -> Independent:
        """
        Returns a single Independent Normal distribution over out_dim.
        """
        device, dtype = module_device_dtype(self.mlp) 
        obs = torch.as_tensor(obs, device=device, dtype=dtype)

        mean = self.mlp(obs)
        log_std = torch.clamp(self.log_std, min=self.min_log_std)

        # expand to batch
        if mean.dim() == 2:
            log_std = log_std.unsqueeze(0).expand_as(mean)

        std = log_std.exp()

        std = to_tensor(std, device = device)
        mean = to_tensor(mean, device = device)
        base = Normal(mean, std)
        return Independent(base, 1)
    # ---------------- act ----------------
    @torch.no_grad()
    def get_actions(self, obs: torch.Tensor, deterministic: bool = False, need_probs: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        obs: [B, obs_dim] or [obs_dim]
        returns: actions [B, out_dim], info dict {mean, log_std, logp}
        """
        dist = self.distribution(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()  # reparameterized sample (rsample) could be used; .sample() also ok
        mean = dist.mean
        # reconstruct log_std: dist.base_dist.scale is std
        log_std = torch.log(dist.base_dist.scale)
        logp = dist.log_prob(action)
        
        info = {"mean": mean, "log_std": log_std, "logp": logp}

        return action, info
    
    # For original mlp
    def get_outer_log_probs(self, obs, actions):
        dist = self.distribution(obs)
        logp = dist.log_prob(actions)
        
        return logp

