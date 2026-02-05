# -*- coding: utf-8 -*-
import torch
from typing import Dict, Optional, Tuple
from .Categorical import CategoricalAgent


class MetaCategoricalAgent(CategoricalAgent):
    """
    Meta-level categorical policy supporting parameter dictionaries per task.
    """

    def __init__(self, num_tasks: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tasks = num_tasks
        self._pre_update_mode = True
        print("Meta-Categorical policy ready")

    @torch.no_grad()
    def get_actions(self,
                    obs: torch.Tensor,
                    params: Optional[object],
                    deterministic: bool = False,
                    post_update: bool = False
                    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(params, (list, tuple)):
            actions = []
            logps = []
            for task_idx, task_params in enumerate(params):
                obs_task = obs[task_idx]
                dist = self.distribution(obs_task, params=task_params)
                if deterministic:
                    action = torch.argmax(dist.logits, dim=-1)
                else:
                    action = dist.sample()
                logp = dist.log_prob(action).sum(dim=-1)
                actions.append(action)
                logps.append(logp)

            action = torch.stack(actions, dim=0)
            logp = torch.stack(logps, dim=0)
        else:
            current_params = params if params is not None else dict(self.named_parameters())
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
            for task_idx in range(len(logp))
        ]

        return action, agent_info
