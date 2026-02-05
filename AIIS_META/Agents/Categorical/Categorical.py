# -*- coding: utf-8 -*-
import torch
from torch.distributions.categorical import Categorical
from torch.func import functional_call
from typing import Dict, Tuple
from AIIS_META.Utils.utils import module_device_dtype, to_tensor
from AIIS_META.Agents.base import BaseAgent


class CategoricalAgent(BaseAgent):
    """
    Categorical Policy for discrete actions (functional version).
    Outputs logits for each action per dimension and samples discrete actions.
    """

    def __init__(self,
                 mlp,
                 action_dim: int,
                 num_actions: int,
                 gamma: float = 0.99):
        super().__init__(mlp, gamma)
        self.mlp = mlp
        self.gamma = gamma
        self.action_dim = int(action_dim)
        self.num_actions = int(num_actions)

    def distribution(self, obs: torch.Tensor,
                     params: Dict[str, torch.Tensor]) -> Categorical:
        device, dtype = module_device_dtype(self.mlp)
        obs = torch.as_tensor(obs, device=device, dtype=dtype)

        mlp_params = {
            k.removeprefix('mlp.'): v
            for k, v in params.items()
            if k.startswith('mlp.')
        }

        logits = functional_call(self.mlp, mlp_params, (obs,))
        logits = logits.view(*logits.shape[:-1], self.action_dim, self.num_actions)
        return Categorical(logits=logits)

    @torch.no_grad()
    def get_actions(self,
                    obs: torch.Tensor,
                    params: Dict[str, torch.Tensor],
                    deterministic: bool = False,
                    post_update: bool = False
                    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if post_update:
            current_params = params
        else:
            current_params = dict(self.named_parameters())

        dist = self.distribution(obs, params=current_params)
        if deterministic:
            action = torch.argmax(dist.logits, dim=-1)
        else:
            action = dist.sample()

        logp = dist.log_prob(action).sum(dim=-1)

        agent_info = [
            [
                dict(logp=logp[task_idx][rollout_idx])
                for rollout_idx in range(len(logp[task_idx]))
            ]
            for task_idx in range(self.num_tasks)
        ]

        return action, agent_info

    def log_prob(self,
                 obs: torch.Tensor,
                 actions: torch.Tensor,
                 params: Dict[str, torch.Tensor]) -> torch.Tensor:
        dist = self.distribution(obs, params=params)
        actions = to_tensor(actions, device=dist.logits.device).long()
        logp = dist.log_prob(actions).sum(dim=-1)
        return logp
