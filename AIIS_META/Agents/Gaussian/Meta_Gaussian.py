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
                    params: Dict[str, torch.Tensor],   # <--- Required: parameter dictionary
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
        # 1. Choose which parameters to use (pre-update vs post-update)
        if post_update:
            # Use the task-specific adapted parameter dictionary
            current_params = params
        else:
            # Use the policy's current (meta) parameters
            current_params = dict(self.named_parameters())

        # 2. Build the Gaussian distribution using current parameters
        dist = self.distribution(obs, params=current_params)

        # 3. Sample actions
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()  # reparameterized sample (keeps differentiability)

        # 4. Compute log-probability of sampled actions
        logp = dist.log_prob(action)

        # 5. Build agent_info for each task and rollout
        #    agent_info[task][rollout] = {"logp": <log_prob>}
        agent_info = [
            [
                dict(logp=logp[task_idx][rollout_idx])
                for rollout_idx in range(len(logp[task_idx]))
            ]
            for task_idx in range(self.num_tasks)
        ]

        return action, agent_info
