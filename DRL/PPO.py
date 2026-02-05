# PPO.py
# -*- coding: utf-8 -*-
"""
Proximal Policy Optimization (PPO) for Single-Task and Multi-Task Scenarios
===========================================================================
This implementation applies PPO to single tasks or multiple tasks without meta-learning.
It reuses components from the Meta-RL framework (environment, baseline, sampler) but
only optimizes the policy for a fixed or dynamically sampled task.

Key differences from ProMP:
- No inner-loop adaptation (no θ' computation)
- No outer-loop meta-optimization (no meta-parameter updates)
- Simple PPO gradient steps with task randomization option
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import AIIS_META.Utils.utils as utils
from AIIS_META.Agents.Gaussian.Meta_Gaussian import MetaGaussianAgent
from AIIS_META.Baselines.linear_baseline import LinearFeatureBaseline
from AIIS_META.Sampler.meta_sampler import MetaSampler
from AIIS_META.Sampler.meta_sample_processor import MetaSampleProcessor
from envs.promp_env import MetaEnv


class PPO:
    """
    Standard Proximal Policy Optimization (PPO) Algorithm.
    
    Features:
    - Supports single fixed task or randomized task sampling
    - Uses PPO-style clipped surrogate loss
    - Reuses Meta-RL framework components (environment, sampler, baseline)
    - Optional task randomization per epoch
    - TensorBoard logging support
    
    Architecture:
    - No inner-loop gradient steps
    - No meta-parameter optimization
    - Simple per-task training loop
    """
    
    def __init__(self,
                 env: Any,                      # Gym-compatible environment
                 max_path_length: int,          # Maximum trajectory length
                 agent: MetaGaussianAgent,      # Policy network
                 baseline: LinearFeatureBaseline, # Advantage baseline
                 tensor_log: str,               # TensorBoard log directory
                 num_tasks: int = 1,            # Number of parallel tasks (for sampling)
                 rollout_per_task: int = 5,     # Trajectories per task per epoch
                 beta: float = 0.0005,          # Learning rate (meta-RL compatible)
                 clip_eps: float = 0.3,         # PPO clipping epsilon
                 outer_iters: int = 5,          # PPO update iterations per epoch
                 discount: float = 0.99,        # Discount factor (γ)
                 gae_lambda: float = 1.0,       # GAE lambda parameter
                 normalize_adv: bool = True,    # Normalize advantages
                 parallel: bool = False,        # Use parallel sampling
                 fixed_task: Optional[Dict] = None,  # Fixed task parameters (e.g., demand_range, leadtime_range)
                 randomize_tasks: bool = True,  # Randomize tasks each epoch
                 device: Optional[torch.device] = None,
                 action_log_interval: int = 100):
        
        self.env = env
        self.agent = agent
        self.baseline = baseline
        self.max_path_length = max_path_length
        self.num_tasks = num_tasks
        self.rollout_per_task = rollout_per_task
        self.clip_eps = float(clip_eps)
        self.outer_iters = outer_iters
        self.discount = float(discount)
        self.gae_lambda = float(gae_lambda)
        self.normalize_adv = bool(normalize_adv)
        self.parallel = bool(parallel)
        self.fixed_task = fixed_task
        self.randomize_tasks = bool(randomize_tasks)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_log_interval = max(1, int(action_log_interval))
        
        # Move agent to device
        self.agent.to(self.device)
        
        # Optimizer (meta-RL compatible learning rate)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=beta)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=tensor_log)
        
        # Meta-RL compatible sampler
        self.sampler = MetaSampler(
            env=env,
            agent=agent,
            max_path_length=max_path_length,
            parallel=parallel,
            num_tasks=num_tasks,
            rollout_per_task=rollout_per_task,
            envs_per_task=None  # Use rollout_per_task as default
        )
        
        # Sample processor (from Meta-RL framework)
        self.sample_processor = MetaSampleProcessor(
            baseline=baseline,
            discount=discount,
            gae_lambda=gae_lambda,
            normalize_adv=normalize_adv
        )
        
        # Task configuration
        self._setup_task_distribution()
    
    def _setup_task_distribution(self):
        """
        Configure task distribution for sampling.
        
        If fixed_task is provided, it will be used for all epochs.
        Otherwise, tasks will be randomly sampled from the environment's task distribution.
        """
        if self.fixed_task is not None:
            print(f"✓ Using fixed task: {self.fixed_task}")
            self.randomize_tasks = False
        elif self.randomize_tasks:
            print(f"✓ Task randomization enabled (new tasks each epoch)")
        else:
            print(f"✓ Task randomization disabled (same tasks throughout)")
    
    # ==========================================================
    # PPO Surrogate Loss
    # ==========================================================
    def _ppo_loss(self,
                  logp_new: torch.Tensor,
                  logp_old: torch.Tensor,
                  advantages: torch.Tensor) -> torch.Tensor:
        """
        Compute PPO-clipped surrogate loss.
        
        Args:
            logp_new: log-probabilities under current policy
            logp_old: log-probabilities under old (initial) policy
            advantages: advantage estimates (GAE)
        
        Returns:
            PPO loss (scalar, to be minimized)
        """
        if isinstance(logp_new, (list, tuple)):
            logp_new = torch.stack(logp_new)
        if isinstance(logp_old, (list, tuple)):
            logp_old = torch.stack(logp_old).detach()
        if isinstance(advantages, (list, tuple)):
            advantages = torch.stack(advantages).detach()
        
        # Likelihood ratio: π(a|s) / π_old(a|s)
        ratios = torch.exp(logp_new - logp_old)
        
        # PPO-style clipping
        clipped_ratios = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        
        # min(ratio·A, clip(ratio)·A)
        surr = torch.min(ratios.T * advantages, clipped_ratios.T * advantages)
        
        # Return negative because we minimize
        return -surr.mean()
    
    # ==========================================================
    # Batch Data Preparation
    # ==========================================================
    def _prepare_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare batch data for PPO loss computation.
        
        Returns:
            (observations, actions, advantages, logp_old)
        """
        dev = self.device
        obs = utils.to_tensor(batch["observations"], dev)
        acts = utils.to_tensor(batch["actions"], dev)
        adv = utils.to_tensor(batch["advantages"], dev)
        logp_old = batch["agent_info"]["logp"]  # Already tensor list
        
        return obs, acts, adv, logp_old
    
    # ==========================================================
    # Trajectory Sampling
    # ==========================================================
    def _collect_trajectories(self) -> List[Dict]:
        """
        Collect trajectories using the current policy.
        
        Uses the meta-sampler to maintain compatibility with Meta-RL framework.
        
        Returns:
            List of processed trajectory batches (one per task)
        """
        # Sampler expects a parameter dict list (even though we're not adapting)
        params_list = [OrderedDict(self.agent.named_parameters()) for _ in range(self.num_tasks)]
        
        # Obtain raw samples
        paths = self.sampler.obtain_samples(params_list, post_update=False)
        
        # Process samples (compute advantages, returns, etc.)
        processed_paths = self.sample_processor.process_samples(paths)
        
        return processed_paths
    
    # ==========================================================
    # PPO Update Step
    # ==========================================================
    def _ppo_update_step(self, batch: Dict[str, Any]) -> float:
        """
        Perform a single PPO gradient step.
        
        Args:
            batch: Trajectory batch with observations, actions, advantages, logp_old
        
        Returns:
            Loss value
        """
        obs, acts, advs, logp_old = self._prepare_batch(batch)
        
        # Get current agent parameters
        current_params = dict(self.agent.named_parameters())
        
        # Compute new log-probabilities
        logp_new = self.agent.log_prob(obs, acts, params=current_params)
        
        # Compute PPO loss
        loss = self._ppo_loss(logp_new, logp_old, advs)
        
        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return float(loss.item())
    
    # ==========================================================
    # Training Loop
    # ==========================================================
    def learn(self, epochs: int, callback: Optional[callable] = None):
        """
        Main training loop for PPO.
        
        Args:
            epochs: Number of training epochs
            callback: Optional callback function called after training with reward history
        
        Training dynamics:
        - If fixed_task: Use the same task throughout training
        - If randomize_tasks: Sample new tasks each epoch
        - Per epoch: Collect trajectories → PPO updates (outer_iters times)
        """
        reward_history = []
        
        for epoch in range(epochs):
            # ===== 1) Task Selection =====
            if self.randomize_tasks and self.fixed_task is None:
                # Resample tasks each epoch (meta-RL sampler handles this internally)
                self.sampler.update_tasks()
            elif self.fixed_task is not None:
                # Use fixed task for all environments
                # Create a list with the fixed task repeated for each task slot
                fixed_tasks = [self.fixed_task] * self.sampler.num_tasks
                self.sampler.vec_env.set_tasks(fixed_tasks)
            
            # ===== 2) Trajectory Collection =====
            paths = self._collect_trajectories()
            
            # ===== 3) Logging =====
            logging_infos = self.env.report_scalar(paths, self.rollout_per_task)
            
            # TensorBoard logging
            for key in logging_infos.keys():
                if isinstance(logging_infos[key], (int, float, np.generic, torch.Tensor)):
                    self.writer.add_scalar(f"{key}", logging_infos[key], global_step=epoch)
                elif isinstance(logging_infos[key], dict):
                    self.writer.add_scalars(f"{key}", logging_infos[key], global_step=epoch)

            # Action distribution (aggregate over all tasks)
            if epoch % self.action_log_interval == 0:
                try:
                    actions = np.concatenate([p["actions"] for p in paths], axis=0)
                except Exception:
                    actions = None
                if actions is not None and actions.size > 0:
                    actions_flat = actions.reshape(-1)
                    self.writer.add_histogram("PPO/Actions", actions_flat, global_step=epoch)
            
            # Extract reward for history
            reward_value = None
            for k in ["AverageReturn", "AverageReward", "Return", "Reward", "reward"]:
                if k in logging_infos:
                    reward_value = logging_infos[k]
                    break
            
            if reward_value is None:
                for v in logging_infos.values():
                    if isinstance(v, (int, float, np.generic, torch.Tensor)):
                        reward_value = v
                        break
            
            if reward_value is not None:
                if isinstance(reward_value, torch.Tensor):
                    reward_value = reward_value.item()
                reward_history.append(float(reward_value))
            
            # ===== 4) PPO Optimization =====
            print(f"Epoch {epoch + 1}: PPO Update Start (outer_iters={self.outer_iters})")
            
            # Perform multiple PPO update steps on the same batch (like meta-RL's outer_iters)
            epoch_losses = []
            for ppo_iter in range(self.outer_iters):
                iter_losses = []
                
                # Update on each task's batch
                for task_id in range(self.num_tasks):
                    batch = paths[task_id]
                    loss = self._ppo_update_step(batch)
                    iter_losses.append(loss)
                
                mean_loss = np.mean(iter_losses)
                epoch_losses.append(mean_loss)
                
                # Log per iteration
                self.writer.add_scalar(
                    "PPO/Loss",
                    mean_loss,
                    global_step=epoch * self.outer_iters + ppo_iter
                )
            
            # Log epoch statistics
            mean_epoch_loss = np.mean(epoch_losses)
            print(f"  Epoch {epoch + 1} - Mean PPO Loss: {mean_epoch_loss:.4f}")
            self.writer.add_scalar("Epoch/PPO_Loss", mean_epoch_loss, global_step=epoch)
        
        # ===== Callback =====
        if callback is not None:
            callback(reward_history)
        
        self.writer.flush()
        print("\nTraining completed!")
    
    def save_model(self, path: str):
        """Save agent parameters to disk."""
        torch.save(self.agent.state_dict(), path)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path: str):
        """Load agent parameters from disk."""
        self.agent.load_state_dict(torch.load(path))
        print(f"✓ Model loaded from {path}")
    
    def close(self):
        """Clean up resources (parallel workers, etc.)."""
        if hasattr(self, "sampler") and hasattr(self.sampler, "close"):
            self.sampler.close()
        self.writer.close()
