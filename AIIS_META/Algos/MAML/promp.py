# promp.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import copy
from typing import Dict, List, Tuple, Optional, Any

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from .base import MAML_BASE


class ProMP(MAML_BASE):
    """
    Proximal Meta-Policy Search (PyTorch)
      - Inner:  -(ratio * A).mean()
      - Outer:  PPO-clip + λ * KL(old||new)   (λ는 step별 계수 평균)
      - KL(old||new) 추정: E_old[ logp_old - logp_new ]
    """

    def __init__(self,
                 env: Any,      #Gym Environment
                 max_path_length: int,      # max path length
                 agent,     # agent nn.Module (get_outer_actions: logp 반환)
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
                 anneal_factor: float = 1.0,    # 1.0이면 고정, <1.0이면 점감
                 discount: float = 0.99,        # Gamma
                 gae_lambda: float = 1.0,       # lambda of GAE
                 normalize_adv: bool = True,        # Nomalizing Advantage
                 device: Optional[torch.device] = None):
        # initial setting
        super().__init__(
            env, max_path_length, agent, alpha, beta, tensor_log, baseline,
            inner_grad_steps, num_tasks, rollout_per_task,
            outer_iters, parallel, clip_eps=clip_eps,
            init_inner_kl_penalty = init_inner_kl_penalty,
            discount=discount, gae_lambda=gae_lambda,
            normalize_adv=normalize_adv, device=device
        )
        self.clip_eps = float(clip_eps)
        self.target_kl_diff = float(target_kl_diff)
        self.adaptive_inner_kl_penalty = bool(adaptive_inner_kl_penalty)
        self.anneal_factor = float(anneal_factor)
        self.anneal_coeff = 1.0
        self.writer = SummaryWriter(log_dir=tensor_log)
        # step별 KL penalty 계수/최근 KL
        self.inner_kl_coeff = torch.full(
            (inner_grad_steps,), float(init_inner_kl_penalty),
            dtype=torch.float32, device=self.device
        )
        
        self._last_inner_kls = torch.zeros(
            inner_grad_steps, dtype=torch.float32, device=self.device
        )
        self.inner_kls = []
        self.alpha = alpha
        self.optimizers = []
        # inner 적응용 에이전트 복사본(태스크별)
        self.adapted_agents = [copy.deepcopy(agent) for _ in range(num_tasks)]
        self.old_agent = None

    # ---------- surrogate ----------
    def _surrogate(self,
               logp_new: torch.Tensor,
               logp_old: torch.Tensor,
               advs: torch.Tensor,
               clip: bool = False) -> torch.Tensor:
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
        # Log-likelihood ratio
        ratios = torch.exp(logp_new - logp_old)

        if clip:
            # PPO-style clipped surrogate
            clipped_ratios = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
            surr = torch.min(ratios.T * advs, clipped_ratios.T * advs)
        else:
            # unclipped (vanilla policy gradient)
            surr = ratios.T * advs
        return -surr.mean()


    # ---------- KL (샘플 근사) ----------
    @staticmethod
    def _kl_from_logps(logp_old, logp_new):
        """
        KL(old || new) ≈ E_old[logp_old - logp_new]
        logp_* 는 정확한 log 확률값(log p(x))임을 전제로 함.
        """
        kl = (torch.stack(logp_old) - logp_new).mean()
        return kl

    # ---------- Inner objective ----------
    def inner_obj(self, new_agent, batchs: dict) -> torch.Tensor:
        """
        new_agents: inner_model
        -(ratio * A).mean()  (클립 없음)
        rollout 간 mean으로 집계하여 스케일 불변성 유지
        """
        surrs = []
        dev = next(self.agent.parameters()).device
        # num of rollouts
        actions = self._to_tensor(batchs["actions"], dev, torch.float32)
        obs = self._to_tensor(batchs["observations"], dev, torch.float32)
        adv = self._to_tensor(batchs["advantages"], dev, torch.float32)
        logp_old = batchs["agent_info"]["logp"]   # list[t] of tensors or tensor
        logp_new = new_agent.get_outer_log_probs(obs, actions) # Cal log_probabilty
        self._surrogate(logp_new=logp_new,
                                        logp_old=logp_old,
                                        advs=adv,
                                        clip=False)
        self.inner_kls.append(self._kl_from_logps(logp_old, logp_new))
        return surrs

    # ---------- Outer objective ----------
    def outer_obj(self, batchs) -> torch.Tensor:
        """
        각 rollout에 대해 PPO-clip surrogate를 계산해 평균
        + λ * KL(old||new) (λ는 step별 계수 평균)
        """
        dev = next(self.agent.parameters()).device
        actions = self._to_tensor(batchs["actions"], dev, torch.float32)
        obs = self._to_tensor(batchs["observations"], dev, torch.float32)
        adv = self._to_tensor(batchs["advantages"], dev, torch.float32)
        logp_new = self.agent.get_outer_log_probs(obs, actions)
        logp_old = batchs["agent_info"]["logp"]   # list[t] of tensors or tensor
        surr_loss = self._surrogate(logp_new=logp_new,
                                        logp_old=logp_old,
                                        advs=adv,
                                        clip=True)

        # KL은 페널티이므로 + 로 더한다
        return surr_loss + self._last_inner_kls.detach() * self.inner_kl_coeff

    # ---------- Inner loop (태스크별 적응 + KL 모니터링/anneal) ----------
    def inner_loop(self,
                   base_state_dict: Optional[Dict[str, torch.Tensor]] = None
                   ) -> Tuple[List[Dict[str, torch.Tensor]], List]:
        """
        반환: (적응된 에이전트 리스트, 마지막(post_update) 수집 경로들)
        """
        # 태스크별 적응용 에이전트
        self.adapted_agents = [copy.deepcopy(self.agent) for _ in range(self.num_tasks)]
        self.old_agent = copy.deepcopy(self.agent)
        #self.optimizers = [optim.Adam(self.adapted_agents[i].parameters(), lr=self.alpha) for i in range(len(self.adapted_agents))]

        for step in range(self.inner_grad_steps + 1):
            if step == self.inner_grad_steps:
                # post-update 수집
                last_paths = self.sampler.obtain_samples(self.adapted_agents, post_update=True)
                last_paths = self.sample_processor.process_samples(last_paths)

                # KL 계수 적응
                self._last_inner_kls = sum(self.inner_kls)/len(self.inner_kls)
                if self.adaptive_inner_kl_penalty and self.inner_grad_steps > 0:
                    self._adapt_inner_kl_coeff(self._last_inner_kls)

                # clip epsilon anneal
                self.anneal_coeff *= self.anneal_factor

                return self.adapted_agents, last_paths
            else:
                self.sampler.agent = self.agent # old parameter 적용
                last_paths = self.sampler.obtain_samples(self.adapted_agents, post_update = False)  # list[task] of paths
                last_paths = self.sample_processor.process_samples(last_paths)
                print("log_p:", len(last_paths[0]["agent_info"]["logp"]))
                # 3) 태스크별 inner 업데이트 + 이번 step KL 측정
                for task_id in range(self.num_tasks):
                    print(f"Inner: {task_id+1}/{self.num_tasks}")
                    batch = last_paths[task_id]
                    # (a) 현재 파라미터로 inner objective 계산
                    loss_in = self.inner_obj(self.adapted_agents[task_id], batch)
                    
                    # (c) 그 파라미터들에 대한 grad 계산 (Outer에서 접근 가능해야 하기 때문에 Create_Graph True)
                    grads = torch.autograd.grad(
                        loss_in,
                        self.adapted_agents[task_id].parameters(),
                        create_graph=True,
                        allow_unused=True
                    )
                    # (d) 한 스텝 업데이트: θ' = θ + α * ∇θ (pseudo-code Line 8)
                    with torch.no_grad():
                        for (name, p), g in zip(self.agent.named_parameters(), grads):
                            if g is None:
                                continue
                            step = self.inner_step_sizes[self._safe_key(name)]
                            # θ' = θ - α_i ⊙ ∇θ  (ProMP inner update)
                            p.add_(-step * g)

    # ---------- KL penalty coeff update ----------
    def _adapt_inner_kl_coeff(self, inner_kls: torch.Tensor):
        """
        목표 KL 대비 작/크면 λ를 ÷2 / ×2 로 조절 (완만한 스케줄이 필요하면 여기서 조절)
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
