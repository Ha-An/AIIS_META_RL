# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from torch.func import functional_call
from typing import Dict, Tuple
from AIIS_META.Utils.utils import *
from AIIS_META.Agents.base import BaseAgent


class GaussianAgent(BaseAgent):
    """
    Gaussian Policy for continuous control tasks (functional version)
    ----------------------------------------------------------------
    - Designed for meta-RL frameworks such as MAML and ProMP.
    - Can be evaluated with parameter dictionaries (functional) or
      as a standard PyTorch module (stateful).
    """

    def __init__(self,
                 mlp,
                 gamma: float = 0.99,
                 learn_std: bool = True,
                 init_std: float = 1.0,
                 min_std: float = 1e-6,
                 state_dependent_std: bool = False):
        """
        Args:
            mlp (nn.Module): Mean network (outputs action mean)
            gamma (float): Discount factor for RL
            learn_std (bool): Whether log_std is learnable
            init_std (float): Initial standard deviation
            min_std (float): Minimum allowed standard deviation
            state_dependent_std (bool): If True, std depends on state (not constant)
        """
        super().__init__(mlp, gamma)

        self.mlp = mlp
        self.gamma = gamma
        self.state_dependent_std = bool(state_dependent_std)

        if self.state_dependent_std:
            print("Warning: state_dependent_std=True may require separate handling "
                  "to remain compatible with functional_call.")

        # Numerical safety clamp for std
        self.min_log_std = torch.log(torch.tensor(min_std))
        init_log_std = float(torch.log(torch.tensor(init_std)))

        # log_std registered as parameter (per action dimension)
        p = nn.Parameter(torch.full((self.mlp.output_dim,), init_log_std),
                         requires_grad=learn_std)
        self.register_parameter("log_std", p)
        self.log_std = p

    # ==========================================================
    # Build Gaussian distribution (Functional)
    # ==========================================================
    def distribution(self, obs: torch.Tensor,
                     params: Dict[str, torch.Tensor]) -> Independent:
        """
        Construct the Gaussian distribution given a parameter dictionary.

        This method supports functional parameter usage — no internal
        weights are accessed directly; everything is read from 'params'.

        Args:
            obs (torch.Tensor): Observation tensor.
            params (Dict[str, torch.Tensor]): Parameter dictionary
                (keys must include 'mlp.*' and 'log_std').

        Returns:
            torch.distributions.Independent: Gaussian policy distribution.
        """
        device, dtype = module_device_dtype(self.mlp)
        obs = torch.as_tensor(obs, device=device, dtype=dtype)

        # Extract parameters that belong to the MLP submodule
        mlp_params = {
            k.removeprefix('mlp.'): v
            for k, v in params.items()
            if k.startswith('mlp.')
        }

        # Compute mean from the functional forward of the MLP
        mean = functional_call(self.mlp, mlp_params, (obs,))

        # Retrieve and clamp log_std for numerical stability
        log_std_unclamped = params['log_std']
        log_std = torch.clamp(log_std_unclamped, min=self.min_log_std)
        std = torch.max(torch.exp(log_std), torch.exp(self.min_log_std))

        # Ensure correct tensor type and device
        std = to_tensor(std, device=device)
        mean = to_tensor(mean, device=device)

        # Build multivariate Gaussian with independent dimensions
        base = Normal(mean, std)
        return Independent(base, 1)

    # ==========================================================
    # Action sampling (Fully Functional)
    # ==========================================================
    @torch.no_grad()
    def get_actions(self,
                    obs: torch.Tensor,
                    params: Dict[str, torch.Tensor],   # Functional parameter dictionary
                    deterministic: bool = False,
                    post_update: bool = False
                    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Sample actions using the provided parameter dictionary.

        This function supports both:
        - Pre-update sampling (using meta-parameters)
        - Post-update sampling (using adapted task-specific parameters)

        Args:
            obs (torch.Tensor): Observations from the environment.
            params (Dict[str, torch.Tensor]): Parameter dictionary for this evaluation.
            deterministic (bool): If True, use the mean (no sampling).
            post_update (bool): If True, indicates sampling after inner adaptation.

        Returns:
            Tuple[
                torch.Tensor,  # Sampled actions
                List[List[Dict[str, torch.Tensor]]]  # Agent info containing log-probs
            ]
        """
        # Select which parameters to use
        if post_update:
            # Use the provided (adapted) parameters
            current_params = params
        else:
            # Use the agent's current meta-parameters
            current_params = dict(self.named_parameters())

        # Build distribution from the selected parameter set
        dist = self.distribution(obs, params=current_params)

        # Sample or take mean depending on mode
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()  # Reparameterized sampling for differentiability

        # Compute log-probabilities
        logp = dist.log_prob(action)

        # Construct structured agent_info:
        # agent_info[task_idx][rollout_idx] = {"logp": log_prob_value}
        agent_info = [
            [
                dict(logp=logp[task_idx][rollout_idx])
                for rollout_idx in range(len(logp[task_idx]))
            ]
            for task_idx in range(self.num_tasks)
        ]

        return action, agent_info

    # ==========================================================
    # Log-probability evaluation (Functional)
    # ==========================================================
    def log_prob(self,
                 obs: torch.Tensor,
                 actions: torch.Tensor,
                 params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the log-probabilities of given actions under the policy
        specified by 'params'.

        Used in both inner and outer losses (functional, differentiable).

        Args:
            obs (torch.Tensor): Observations
            actions (torch.Tensor): Actions
            params (Dict[str, torch.Tensor]): Parameter dictionary

        Returns:
            torch.Tensor: Log-probability values
        """
        dist = self.distribution(obs, params=params)
        logp = dist.log_prob(actions)
        return logp
