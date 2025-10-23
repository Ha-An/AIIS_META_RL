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
        self.alpha = alpha
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

        # inner 적응용 에이전트 복사본(태스크별)
        self.dummy_agents = [copy.deepcopy(agent) for _ in range(num_tasks)]

    # ---------- surrogate ----------
    def _surrogate(self,
                   logp_new: torch.Tensor,
                   logp_old: torch.Tensor,
                   advs: torch.Tensor,
                   clip: bool = False) -> torch.Tensor:
        """
        logp_*: [...], 액션 차원까지 합쳐진 샘플별 log-prob 형태를 가정
        advs  : [...] (샘플별 advantage)
        """
        if isinstance(logp_new, (list, tuple)):
            logp_new = torch.stack(logp_new, dim=0)
        if isinstance(logp_old, (list, tuple)):
            logp_old = torch.stack(logp_old, dim=0)
        if isinstance(advs, (list, tuple)):
            advs = torch.stack(advs, dim=0)

        # Cal Log-Likelihood Ratio
        # 분모(old)와 adv는 stop-grad
        logp_old = logp_old.detach()
        advs = advs.detach()
        # 입실론 세팅
        eps = self.clip_eps * self.anneal_coeff
        #delta = (logp_new - logp_old).clamp(-20.0, 20.0)   # 수치 안전
        delta = (logp_new - logp_old)
        ratio = torch.exp(delta)

        # If Need clipping
        if clip:
            surr_loss = -(ratio.clamp(1-eps, 1+eps)*advs).mean()

        # Do not need clipping
        else:
            surr_loss = -(ratio * advs).mean()

        return surr_loss

    # ---------- KL (샘플 근사) ----------
    @staticmethod
    def _kl_from_logps(logp_old: torch.Tensor, logp_new: torch.Tensor) -> torch.Tensor:
        """
        KL(old||new) ≈ E_old[ logp_old - logp_new ]
        """
        return (logp_old.detach() - logp_new).mean()

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
        n_rollouts = len(batchs["actions"]) 
        for idx in range(n_rollouts):
            actions = self._to_tensor(batchs["actions"][idx], dev, torch.float32)
            obs = self._to_tensor(batchs["observations"][idx], dev, torch.float32)
            adv = self._to_tensor(batchs["advantages"][idx], dev, torch.float32)
            logp_old = batchs["agent_infos"][idx]["logp"]   # list[t] of tensors or tensor
            logp_new = new_agent.get_outer_log_probs(obs, actions) # Cal log_probabilty
            surrs.append(self._surrogate(logp_new=logp_new,
                                         logp_old=logp_old,
                                         advs=adv,
                                         clip=False))
        return torch.stack(surrs).mean()

    # ---------- Outer objective ----------
    def outer_obj(self, batchs) -> torch.Tensor:
        """
        각 rollout에 대해 PPO-clip surrogate를 계산해 평균
        + λ * KL(old||new) (λ는 step별 계수 평균)
        """
        surrs, kl_list = [], []
        dev = next(self.agent.parameters()).device
        n_rollouts = len(batchs["actions"])

        for idx in range(n_rollouts):
            actions = self._to_tensor(batchs["actions"][idx], dev, torch.float32)
            obs = self._to_tensor(batchs["observations"][idx], dev, torch.float32)
            adv = self._to_tensor(batchs["advantages"][idx], dev, torch.float32)

            logp_new = self.agent.get_outer_log_probs(obs, actions)
            logp_old = torch.stack(batchs["agent_infos"][idx]["logp"])

            surrs.append(self._surrogate(logp_new=logp_new,
                                         logp_old=logp_old,
                                         advs=adv,
                                         clip=True))

        surr_loss = torch.stack(surrs).mean() 

        # KL은 페널티이므로 + 로 더한다
        return surr_loss + self._last_inner_kls 

    # ---------- Inner loop (태스크별 적응 + KL 모니터링/anneal) ----------
    def inner_loop(self,
                   base_state_dict: Optional[Dict[str, torch.Tensor]] = None
                   ) -> Tuple[List[Dict[str, torch.Tensor]], List]:
        """
        반환: (적응된 에이전트 리스트, 마지막(post_update) 수집 경로들)
        """
        # 태스크별 적응용 에이전트
        adapted_agents = [copy.deepcopy(self.agent) for _ in range(self.num_tasks)]

        inner_kls_per_step = torch.zeros(self.inner_grad_steps,
                                         dtype=torch.float32,
                                         device=self.device)
        for step in range(self.inner_grad_steps + 1):
            if step == self.inner_grad_steps:
                # post-update 수집
                last_paths = self.sampler.obtain_samples(adapted_agents, post_update=True)
                last_paths = self.sample_processor.process_samples(last_paths)

                # KL 계수 적응
                self._last_inner_kls = inner_kls_per_step.detach()
                if self.adaptive_inner_kl_penalty and self.inner_grad_steps > 0:
                    self._adapt_inner_kl_coeff(self._last_inner_kls)

                # clip epsilon anneal
                self.anneal_coeff *= self.anneal_factor

                return adapted_agents, last_paths
            else:
                step_kls = []
                self.sampler.agent = self.agent # old parameter 적용
                last_paths = self.sampler.obtain_samples(adapted_agents, post_update = False)  # list[task] of paths
                last_paths = self.sample_processor.process_samples(last_paths)

                # 3) 태스크별 inner 업데이트 + 이번 step KL 측정
                for task_id in range(self.num_tasks):
                    inner_optimizer = optim.Adam(self.dummy_agents[task_id].parameters(), lr = self.alpha)
                    batch = last_paths[task_id]
                    # (a) 현재 파라미터 로드 후 inner loss/grad
                    loss_in = self.inner_obj(self.dummy_agents[task_id], batch)
                    inner_optimizer.zero_grad()
                    loss_in.backward()
                    inner_optimizer.step()

                # (c) KL(old||new) 측정: old=수집 당시 logp, new=업데이트된 파라미터로 재평가
                with torch.no_grad():
                    # rollout 평균으로 KL 집계
                    kls_this_task = []
                    n_rollouts = len(batch["actions"])
                    dev = next(self.agent.parameters()).device
                    for ridx in range(n_rollouts):
                        obs = self._to_tensor(batch["observations"][ridx], dev, torch.float32)
                        acts = self._to_tensor(batch["actions"][ridx], dev, torch.float32)
                        logp_old = torch.stack(batch["agent_infos"][ridx]["logp"]).detach()
                        logp_new = adapted_agents[task_id].get_outer_log_probs(obs, acts)
                        kls_this_task.append(self._kl_from_logps(logp_old, logp_new))
                    step_kls.append(torch.stack(kls_this_task).mean())

            inner_kls_per_step[step] = torch.stack(step_kls).mean()

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
