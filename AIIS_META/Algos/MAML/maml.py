# promp.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
import torch
from torch.utils.tensorboard import SummaryWriter
from .base import MAML_BASE
import AIIS_META.Utils.utils as utils
class VPG_MAML(MAML_BASE):
    """
    Proximal Meta-Policy Search (PyTorch)
      - Inner:  -(ratio * A).mean()
      - Outer:  PPO-clip + KL(old||new) penalty
      - KL(old||new) approx: E_old[ logp_old - logp_new ]
    """

    def __init__(self,
                 env: Any,      #Gym Environment
                 max_path_length: int,      # max path length
                 agent,     # agent nn.Module (returns logp)
                 alpha,
                 beta,
                 baseline,      # baseline(Cal Advantage)
                 tensor_log,        # Tensorboard_log
                 inner_grad_steps: int = 1,     # Inner_gradient_steps(inner adapts)
                 num_tasks: int = 4,        # Tasks
                 rollout_per_task: int = 5,     # Sampled paths from one task
                 outer_iters: int = 5,      # Outer learning steps
                 parallel: bool = False,        # Multi-processing Factor
                 clip_eps: float = 0.2,     # Clip epsilon for Promp
                 target_kl_diff: float = 0.01,      # Target KL
                 init_inner_kl_penalty: float = 1e-2,       # Start KL-Penalty (η)
                 adaptive_inner_kl_penalty: bool = False,       # Use KL-Penalty adaptive
                 anneal_factor: float = 1.0,    # 1.0 = fixed, <1.0 = decay
                 discount: float = 0.99,        # Gamma
                 gae_lambda: float = 1.0,       # lambda of GAE
                 normalize_adv: bool = True,        # Nomalizing Advantage
                 loss_type: str = "log_likelihood",
                 device: Optional[torch.device] = None,
                 action_log_interval: int = 100):
        # initial setting
        super().__init__(
            env, max_path_length, agent, alpha, beta, tensor_log, baseline,
            inner_grad_steps, num_tasks, rollout_per_task,
            outer_iters, parallel, clip_eps=clip_eps,
            init_inner_kl_penalty = init_inner_kl_penalty,
            discount=discount, gae_lambda=gae_lambda,
            normalize_adv=normalize_adv, device=device,
            action_log_interval=action_log_interval
        )
        self.clip_eps = float(clip_eps)
        self.target_kl_diff = float(target_kl_diff)
        self.adaptive_inner_kl_penalty = bool(adaptive_inner_kl_penalty)
        self.anneal_factor = float(anneal_factor)
        self.anneal_coeff = 1.0
        self.writer = SummaryWriter(log_dir=tensor_log)
        self.inner_kl_coeff = torch.full(
            (inner_grad_steps,), float(init_inner_kl_penalty),
            dtype=torch.float32, device=self.device
        )
        
        self._last_inner_kls = torch.zeros(
            inner_grad_steps, dtype=torch.float32, device=self.device
        )
        self.inner_kls = []
        self.alpha = alpha

        self.old_agent = None
        self.inner_loss_type = loss_type
    # ---------- surrogate ----------
    def _surrogate(self,
               logp_new: torch.Tensor,
               logp_old: torch.Tensor,
               advs: torch.Tensor, loss_type = "log_likelihood") -> torch.Tensor:
        """
        PPO/ProMP surrogate objective (vectorized)
        logp_* : tensor [...], summed log-prob per sample
        advs   : tensor [...], advantage per sample
        """
        # ensure tensors
        if isinstance(logp_new, (list, tuple)):
            logp_new = torch.stack(logp_new)
        if isinstance(logp_old, (list, tuple)):
            logp_old = torch.stack(logp_old).detach()
        if isinstance(advs, (list, tuple)):
            advs = torch.stack(advs).detach()
        if loss_type == 'likelihood_ratio':
            # Log-likelihood ratio
            ratios = torch.exp(logp_new - logp_old)
            surr = ratios.T * advs
        elif loss_type == 'log_likelihood':
            likelihood = logp_new - logp_old
            surr = likelihood.T*advs
            
        return -surr.mean()

   # ---------- Inner objective ----------
    def inner_obj(self, batchs: dict, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        ... (comments omitted) ...
        """
        surrs = []
        dev = next(self.agent.parameters()).device
        
        actions = utils.to_tensor(batchs["actions"], dev)
        obs = utils.to_tensor(batchs["observations"], dev)
        adv = utils.to_tensor(batchs["advantages"], dev)
        logp_old = batchs["agent_info"]["logp"] 
        
        logp_new = self.agent.log_prob(obs, actions, params=params)
        
        surrs = self._surrogate(logp_new=logp_new,
                                logp_old=logp_old,
                                advs=adv, loss_type=self.inner_loss_type)
        return surrs

    # ---------- Outer objective ----------
    def outer_obj(self,
                  adapted_params: Dict[str, torch.Tensor],
                  batch: Dict[str, Any],
                  ) -> torch.Tensor:
        """
        ... (comments omitted) ...
        """
        dev = next(self.agent.parameters()).device
        obs  = torch.as_tensor(batch["observations"], device=dev, dtype=torch.float32)
        acts = torch.as_tensor(batch["actions"],       device=dev, dtype=torch.float32)
        adv  = torch.as_tensor(batch["advantages"],     device=dev, dtype=torch.float32)
        
        logp_old = torch.as_tensor(batch["agent_info"]["logp"], device=dev, dtype=torch.float32).detach()

        logp_new = self.agent.log_prob(obs, acts, params=adapted_params)

        surr = self._surrogate(logp_new=logp_new,
                               logp_old=logp_old,
                               advs=adv, loss_type="log_likelihood")
        return surr


    def inner_loop(self, epoch) -> Tuple[List[Dict[str, torch.Tensor]], List]:
        """
        Returns: (adapted parameter dict list, final post-update trajectories)
        """
        
        self.old_params = {k: v.detach().clone() for k, v in self.agent.named_parameters()}
        
        current_params_theta = OrderedDict(self.agent.named_parameters())
        
        adapted_params_list = [OrderedDict(current_params_theta) for _ in range(self.num_tasks)]

        for step in range(self.inner_grad_steps):
            paths = self.sampler.obtain_samples(adapted_params_list, post_update=False) 
            paths = self.sample_processor.process_samples(paths)

            for task_id in range(self.num_tasks):
                batch = paths[task_id]
                
                new_adapted_params = self._theta_prime(
                    batch, 
                    params=adapted_params_list[task_id]
                )
                
                adapted_params_list[task_id] = new_adapted_params

        
        last_paths = self.sampler.obtain_samples(adapted_params_list, post_update=True)
        last_paths = self.sample_processor.process_samples(last_paths)
        
        reward_avg = [sum(path['rewards'])/(self.rollout_per_task*len(last_paths)) for path in last_paths]
        print(f"Epoch {epoch+1}: Reward: {sum(reward_avg)}")
        
        # 5. clip epsilon anneal
        self.anneal_coeff *= self.anneal_factor

        return adapted_params_list, last_paths
    
    def outer_loop(self,
                paths: List[Dict], # post_update_paths provided
                adapted_params_list: List[Dict[str, torch.Tensor]]): # final adapted params list
        
        for itr in range(self.outer_iters):
            loss_outs = []
            
            for task_id in range(self.num_tasks):
                batch = paths[task_id] # use post-update batch
                
                if itr == 0:
                    final_adapted_params = adapted_params_list[task_id]
                
                else:
                    final_adapted_params = self._theta_prime(batch, OrderedDict(self.agent.named_parameters()))
                
                
                loss = self.outer_obj(final_adapted_params, batch)
                loss_outs.append(loss)
                
            mean_loss_out = sum(loss_outs)/len(loss_outs)
            
            self.optimizer.zero_grad()
            mean_loss_out.backward() # gradients flow through inner_loop to base params
            self.optimizer.step()
    
    def _theta_prime(self, batch: dict, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            """
            ... (comments omitted) ...
            """
            surr = self.inner_obj(batch, params=params) 
            
            grads = torch.autograd.grad(
                surr,
                list(params.values()),
                create_graph=True
            )
            
            adapted_params = OrderedDict()
            
            for (name, p), g in zip(params.items(), grads):
                if g is None:
                    adapted_params[name] = p
                    continue
                
                step = self.inner_step_sizes[self._safe_key(name)]
                step = torch.clamp(step, min=0.0, max=self.inner_step_size_max)
                
                adapted_params[name] = p - step * g 

            return adapted_params   
    
    def close(self):
        """
        Clean up vec_env in MetaSampler (especially parallel workers).
        """
        if hasattr(self, "sampler") and hasattr(self.sampler, "close"):
            self.sampler.close()
