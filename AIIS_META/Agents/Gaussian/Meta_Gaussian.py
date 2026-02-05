# Meta_Gaussian.py  (Revised)
import torch
from torch.distributions.independent import Independent
from typing import List, Dict, Optional, Tuple
from AIIS_META.Utils.utils import *
from .Gaussian import GaussianAgent  # Import the modified base GaussianAgent


class MetaGaussianAgent(GaussianAgent):
    """
    Meta-level Gaussian Policy for Meta-RL algorithms (e.g., ProMP, MAML)
    --------------------------------------------------------------------
    This class extends the standard GaussianAgent to support parameter dictionaries 
    (used in meta-learning contexts where parameters are functionally updated per task).
    """

    def __init__(self, num_tasks: int, *args, **kwargs):
        """
        Args:
            num_tasks (int): Number of parallel meta-tasks
            *args, **kwargs: Passed to base GaussianAgent constructor
        """
        super().__init__(*args, **kwargs)
        self.num_tasks = num_tasks
        self._pre_update_mode = True  # Flag for pre-/post-update policy usage
        print("Meta-Gaussian policy ready")

    @torch.no_grad()
    def get_actions(self,
                    obs: torch.Tensor,
                    params: Optional[object],   # dict or list/tuple of dicts
                    deterministic: bool = False,
                    post_update: bool = False
                    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Samples actions using a provided parameter dictionary (params).

        This method is designed to work in a fully functional way:
        instead of relying on the module’s internal parameters,
        it uses the given 'params' dictionary (used for inner/outer loop separation).

        Args:
            obs (torch.Tensor): Observations for each task or rollout batch.
            params (Dict[str, torch.Tensor]): Dictionary of parameters 
                (task-specific adapted parameters if post-update, 
                 otherwise current meta-parameters).
            deterministic (bool): If True, use the mean action (no exploration).
            post_update (bool): Whether to use adapted (post-inner-loop) parameters.

        Returns:
            Tuple[
                torch.Tensor,                    # Sampled actions tensor
                List[List[Dict[str, torch.Tensor]]]  # Agent info with log-probs
            ]
        """
        # If params is a list/tuple, apply per-task parameters.
        if isinstance(params, (list, tuple)):
            actions = []
            logps = []
            for task_idx, task_params in enumerate(params):
                obs_task = obs[task_idx]
                dist = self.distribution(obs_task, params=task_params)
                if deterministic:
                    action = dist.mean
                else:
                    action = dist.rsample()
                logp = dist.log_prob(action)
                actions.append(action)
                logps.append(logp)

            action = torch.stack(actions, dim=0)
            logp = torch.stack(logps, dim=0)
        else:
            # If params is None or a single dict, fall back to that.
            current_params = params if params is not None else dict(self.named_parameters())
            dist = self.distribution(obs, params=current_params)
            if deterministic:
                action = dist.mean
            else:
                action = dist.rsample()
            logp = dist.log_prob(action)

        # Build agent_info for each task and rollout
        agent_info = [
            [
                dict(logp=logp[task_idx][rollout_idx])
                for rollout_idx in range(len(logp[task_idx]))
            ]
            for task_idx in range(len(logp))
        ]

        return action, agent_info
