# base.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import Sequence, Type, List, Dict, Optional, Tuple
from collections import OrderedDict


class BaseAgent(nn.Module):
    """
    BaseAgent
    ----------
    A unified abstract interface for all policy classes (Gaussian, Categorical, etc.)
    expected by meta-RL algorithms such as ProMP, MAML, and Reptile.

    Core Design Principles:
      - Completely agnostic to the type of distribution (Gaussian, categorical, etc.)
      - The algorithm interacts only through:
          - log_prob(obs, actions, params)
          - get_actions(obs, params)
      - The sampler must record agent_infos['logp'] during data collection.

    Subclass Requirements:
      Each subclass (e.g., GaussianAgent) must implement:
        - get_actions(self, obs, params, deterministic=False, post_update=False)
            → returns (actions, agent_info)
            → agent_info must contain key 'logp'
        - log_prob(self, obs, actions, params)
            → returns tensor of shape [B] (log-prob per sample)
        - Optionally override forward() if needed.

    Utility:
      - BaseAgent can optionally include a backbone network (e.g., MLP)
        that can be reused or ignored by child classes.
      - Provides naming compatibility (get_action, get_actions_all_tasks, etc.)
        with older codebases.
    """

    def __init__(self,
                 mlp,
                 gamma: float = 0.99):
        """
        Args:
            mlp (nn.Module): Policy backbone network (e.g., MLP)
            gamma (float): Discount factor for RL algorithms
        """
        super().__init__()
        self.mlp = mlp
        self.gamma = gamma

    # ==========================================================
    # Minimal API — required by meta-learning algorithms
    # ==========================================================
    @torch.no_grad()
    def get_actions(self,
                    obs: torch.Tensor,
                    params: Dict[str, torch.Tensor],   # Functional parameter dict (required)
                    deterministic: bool = False,
                    post_update: bool = False
                    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Wrapper for sampling actions — must be overridden by subclasses.

        Args:
            obs (torch.Tensor): Input observation(s)
            params (Dict[str, torch.Tensor]): Parameter dictionary (task-adapted or meta)
            deterministic (bool): If True, selects mean action (no exploration)
            post_update (bool): Indicates post-inner-loop evaluation

        Returns:
            Tuple[
                torch.Tensor,  # Sampled actions
                Dict[str, torch.Tensor]  # agent_info including 'logp'
            ]

        Note:
            - In this base class, the function is abstract and raises NotImplementedError.
            - Subclasses (e.g., GaussianAgent) must implement the logic using their own distribution type.
        """
        raise NotImplementedError

    def log_prob(self,
                 obs: torch.Tensor,
                 actions: torch.Tensor,
                 params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute log-probabilities of given actions under the policy
        defined by the provided parameters.

        Args:
            obs (torch.Tensor): Observations, shape [B, obs_dim]
            actions (torch.Tensor): Actions, shape [B, act_dim]
            params (Dict[str, torch.Tensor]): Parameter dictionary

        Returns:
            torch.Tensor: Log-probabilities of shape [B]

        Notes:
            - Distribution type (Gaussian, Categorical, etc.) does not matter.
            - As long as this method correctly returns log-probabilities,
              the meta-learning algorithms will operate properly.
        """
        raise NotImplementedError
