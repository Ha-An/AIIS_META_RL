# promp.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
import torch
from torch.utils.tensorboard import SummaryWriter
from .base import MAML_BASE
import AIIS_META.Utils.utils as utils


class ProMP(MAML_BASE):
    """
    Proximal Meta-Policy Search (ProMP) - PyTorch Implementation
    ------------------------------------------------------------
    - Inner loop:  -(ratio * A).mean()  (standard PPO-style surrogate)
    - Outer loop:  PPO-clip + λ * KL(old||new)
    - KL divergence approximation:  E_old[ logp_old - logp_new ]

    ProMP extends first-order MAML (FOMAML) by applying a proximal regularization
    (PPO-style clipping and KL penalty) between the pre- and post-adapted policies.
    """

    def __init__(self,
                 env: Any,                     # Gym-compatible environment
                 max_path_length: int,         # Maximum trajectory length
                 agent,                        # Policy network (nn.Module)
                 alpha,                        # Inner loop learning rate
                 beta,                         # Outer loop learning rate
                 baseline,                     # Baseline (advantage estimation)
                 tensor_log,                   # TensorBoard log directory
                 inner_grad_steps: int = 1,    # Number of inner updates
                 num_tasks: int = 4,           # Number of meta-tasks per iteration
                 rollout_per_task: int = 5,    # Rollouts collected per task
                 outer_iters: int = 5,         # Outer optimization steps
                 parallel: bool = False,       # Multiprocessing flag
                 clip_eps: float = 0.2,        # PPO clipping epsilon
                 target_kl_diff: float = 0.01, # Target KL divergence for adaptation
                 init_inner_kl_penalty: float = 1e-2,  # Initial KL penalty coefficient
                 adaptive_inner_kl_penalty: bool = False, # Adaptive λ scheduling
                 anneal_factor: float = 1.0,   # ε annealing factor (<1.0 = decay)
                 discount: float = 0.99,       # Discount factor (γ)
                 gae_lambda: float = 1.0,      # GAE λ parameter
                 normalize_adv: bool = True,   # Normalize advantages
                 trainable_learning_rate=True, # Allow inner α_i to be learnable
                 device: Optional[torch.device] = None):

        # Initialize base MAML components
        super().__init__(
            env, max_path_length, agent, alpha, beta, tensor_log, baseline,
            inner_grad_steps, num_tasks, rollout_per_task,
            outer_iters, parallel, clip_eps=clip_eps,
            init_inner_kl_penalty=init_inner_kl_penalty,
            discount=discount, gae_lambda=gae_lambda,
            normalize_adv=normalize_adv,
            trainable_learning_rate=trainable_learning_rate,
            device=device
        )

        # ---------- Algorithm hyperparameters ----------
        self.clip_eps = float(clip_eps)
        self.target_kl_diff = float(target_kl_diff)
        self.adaptive_inner_kl_penalty = bool(adaptive_inner_kl_penalty)
        self.anneal_factor = float(anneal_factor)
        self.anneal_coeff = 1.0
        self.writer = SummaryWriter(log_dir=tensor_log)

        # KL penalty coefficient (λ) for each inner step
        self.inner_kl_coeff = torch.full(
            (inner_grad_steps,), float(init_inner_kl_penalty),
            dtype=torch.float32, device=self.device
        )

        self.alpha = alpha
        self.old_agent = None  # Reference to pre-adaptation policy (for KL)

    # ==========================================================
    # Surrogate objective (PPO/ProMP-style)
    # ==========================================================
    def _surrogate(self,
                   logp_new: torch.Tensor,
                   logp_old: torch.Tensor,
                   advs: torch.Tensor,
                   clip: bool = False) -> torch.Tensor:
        """
        Compute the surrogate loss:
        - If clip=False → standard policy gradient objective.
        - If clip=True  → PPO-style clipped surrogate.

        logp_* : (tensor) log-probabilities per sample
        advs   : (tensor) advantages per sample
        """
        if isinstance(logp_new, (list, tuple)):
            logp_new = torch.stack(logp_new)
        if isinstance(logp_old, (list, tuple)):
            logp_old = torch.stack(logp_old).detach()
        if isinstance(advs, (list, tuple)):
            advs = torch.stack(advs).detach()

        ratios = torch.exp(logp_new - logp_old)  # likelihood ratio π_new / π_old

        if clip:
            # PPO-style clipping
            clipped_ratios = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
            surr = torch.min(ratios.T * advs, clipped_ratios.T * advs)
        else:
            # Vanilla (unclipped) surrogate
            surr = ratios.T * advs

        # Note: loss is negated because we minimize
        return -surr.mean()

    # ==========================================================
    # KL divergence approximation
    # ==========================================================
    @staticmethod
    def _kl_from_logps(logp_old, logp_new):
        """
        Approximate KL(old || new) ≈ E_old[logp_old - logp_new].
        Assumes logp_* are valid log-probabilities.
        """
        kl = (logp_old.detach() - logp_new).mean()
        return kl

    # ==========================================================
    # Inner objective (task-specific loss)
    # ==========================================================
    def inner_obj(self, batchs: dict, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Computes the inner-loop objective for adaptation.
        This is typically a standard PPO surrogate without clipping.
        """
        dev = next(self.agent.parameters()).device
        actions = utils.to_tensor(batchs["actions"], dev)
        obs = utils.to_tensor(batchs["observations"], dev)
        adv = utils.to_tensor(batchs["advantages"], dev)
        logp_old = batchs["agent_info"]["logp"]

        # Use policy forward with custom parameter dictionary (no functional_call)
        logp_new = self.agent.log_prob(obs, actions, params=params)

        surr = self._surrogate(
            logp_new=logp_new,
            logp_old=logp_old,
            advs=adv,
            clip=False
        )
        return surr

    # ==========================================================
    # Outer objective (meta loss)
    # ==========================================================
    def outer_obj(self,
                  adapted_params: Dict[str, torch.Tensor],
                  batch: Dict[str, Any]) -> torch.Tensor:
        """
        Computes the outer-loop meta-objective (PPO-clipped loss + optional KL penalty).
        Evaluated using the final adapted parameters.
        """
        dev = next(self.agent.parameters()).device
        obs = torch.as_tensor(batch["observations"], device=dev, dtype=torch.float32)
        acts = torch.as_tensor(batch["actions"], device=dev, dtype=torch.float32)
        adv = torch.as_tensor(batch["advantages"], device=dev, dtype=torch.float32)
        logp_old = torch.as_tensor(batch["agent_info"]["logp"], device=dev, dtype=torch.float32).detach()

        # Evaluate new log-probabilities with adapted parameters
        logp_new = self.agent.log_prob(obs, acts, params=adapted_params)

        # PPO-clip surrogate
        surr = self._surrogate(logp_new=logp_new,
                               logp_old=logp_old,
                               advs=adv,
                               clip=True)

        # Optional: add KL penalty
        kl_penalty = self._kl_from_logps(logp_old, logp_new).mean()
        return surr + self.inner_kl_coeff * kl_penalty

    # ==========================================================
    # Inner adaptation step (θ' = θ - α∇θJ)
    # ==========================================================
    def _theta_prime(self, batch: dict, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Performs one inner adaptation step:
        θ' = θ - α_i * ∇_θ L_inner(θ)
        Returns a new parameter dictionary preserving computation graph.
        """
        surr = self.inner_obj(batch, params=params)

        grads = torch.autograd.grad(
            surr,
            self.agent.parameters(),
            create_graph=True
        )

        adapted_params = OrderedDict()
        for (name, p), g in zip(self.agent.named_parameters(), grads):
            if g is None:
                # Skip parameters without gradients
                adapted_params[name] = p
                continue
            step = self.inner_step_sizes[self._safe_key(name)]
            adapted_params[name] = p - step * g

        return adapted_params

    # ==========================================================
    # Inner loop (task-wise adaptation)
    # ==========================================================
    def inner_loop(self, epoch) -> Tuple[List[Dict[str, torch.Tensor]], List]:
        """
        Inner adaptation across all tasks.
        Returns:
          - adapted_params_list : list of final adapted parameter dictionaries (per task)
          - last_paths : processed post-update trajectories
        """
        # 1. Store old (pre-update) parameters for KL estimation (detached)
        self.old_params = {k: v.detach().clone() for k, v in self.agent.named_parameters()}

        # 2. Record initial (meta) parameters for gradient flow
        current_params_theta = OrderedDict(self.agent.named_parameters())

        # 3. Initialize per-task parameter copies (graph-connected)
        adapted_params_list = [OrderedDict(current_params_theta) for _ in range(self.num_tasks)]

        # 4. Perform inner adaptation steps
        for step in range(self.inner_grad_steps):
            # 4a. Collect trajectories using current parameters
            paths = self.sampler.obtain_samples(adapted_params_list, post_update=False)
            paths = self.sample_processor.process_samples(paths)

            # 4b. Update task-specific parameters (θ → θ')
            for task_id in range(self.num_tasks):
                batch = paths[task_id]
                new_adapted_params = self._theta_prime(batch, params=adapted_params_list[task_id])
                adapted_params_list[task_id] = new_adapted_params

        # 5. Post-update sampling with final adapted parameters
        last_paths = self.sampler.obtain_samples(adapted_params_list, post_update=True)
        last_paths = self.sample_processor.process_samples(last_paths)

        # 6. Anneal clipping epsilon if applicable
        self.anneal_coeff *= self.anneal_factor
        reward_avg = [sum(path['rewards'])/(self.rollout_per_task*len(last_paths)) for path in last_paths]
        print(f"Epoch {epoch+1}: Reward: {sum(reward_avg)}")
        # 7. Return adapted parameters and post-update data
        return adapted_params_list, last_paths

    # ==========================================================
    # Outer loop (meta-optimization)
    # ==========================================================
    def outer_loop(self,
                   paths: List[Dict],
                   adapted_params_list: List[Dict[str, torch.Tensor]]):
        """
        Outer optimization loop over meta-parameters (θ).
        Performs multiple PPO-style gradient steps per meta-iteration.
        """
        for itr in range(self.outer_iters):
            loss_outs = []

            for task_id in range(self.num_tasks):
                batch = paths[task_id]

                # Use the final adapted parameters from the inner loop
                if itr == 0:
                    final_adapted_params = adapted_params_list[task_id]
                else:
                    # Optionally recompute adaptation if needed for stability
                    final_adapted_params = self._theta_prime(
                        batch, OrderedDict(self.agent.named_parameters())
                    )

                # Compute outer loss with final adapted parameters
                loss = self.outer_obj(final_adapted_params, batch)
                loss_outs.append(loss)

            mean_loss_out = sum(loss_outs) / len(loss_outs)

            # 2) Backpropagate through inner updates into meta-parameters θ
            self.optimizer.zero_grad()
            mean_loss_out.backward()   # Gradients flow through inner steps
            self.optimizer.step()

    # ==========================================================
    # Adaptive KL coefficient update
    # ==========================================================
    def _adapt_inner_kl_coeff(self, inner_kls: torch.Tensor):
        """
        Adjusts the KL penalty coefficient (λ) based on KL magnitude.
        - If KL < target/1.5 → halve λ (too small, less regularization)
        - If KL > target*1.5 → double λ (too large, more penalty)
        """
        new_coeff = self.inner_kl_coeff.clone()
        low, high = self.target_kl_diff / 1.5, self.target_kl_diff * 1.5
        for i, kl in enumerate(inner_kls):
            v = float(kl.item())
            if v < low:
                new_coeff[i] = new_coeff[i] * 0.5
            elif v > high:
                new_coeff[i] = new_coeff[i] * 2.0
        self.inner_kl_coeff = new_coeff

    def close(self):
        """
        MetaSampler 안의 vec_env를 정리(특히 parallel worker 종료)한다.
        """
        if hasattr(self, "sampler") and hasattr(self.sampler, "close"):
            self.sampler.close()
