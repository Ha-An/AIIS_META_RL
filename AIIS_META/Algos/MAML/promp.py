# promp.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import copy
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from .base import MAML_BASE

import higher
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

        # inner 적응용 에이전트 복사본(태스크별)
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
        kl = (logp_old.detach() - logp_new).mean()
        return kl

   # ---------- Inner objective ----------
    # def inner_obj(self, batchs: dict) -> torch.Tensor: # [변경 전]
    def inner_obj(self, batchs: dict, params: Dict[str, torch.Tensor]) -> torch.Tensor: # [변경 후]
        """
        ... (주석 동일) ...
        """
        surrs = []
        dev = next(self.agent.parameters()).device
        
        actions = self._to_tensor(batchs["actions"], dev, torch.float32)
        obs = self._to_tensor(batchs["observations"], dev, torch.float32)
        adv = self._to_tensor(batchs["advantages"], dev, torch.float32)
        logp_old = batchs["agent_info"]["logp"] 
        
        # [변경!] self.agent.get_outer_log_probs 호출을
        # functional 'log_prob' 호출로 변경
        # logp_new = self.agent.get_outer_log_probs(obs, actions) # [변경 전]
        logp_new = self.agent.log_prob(obs, actions, params=params) # [변경 후]
        
        surrs = self._surrogate(logp_new=logp_new,
                                logp_old=logp_old,
                                advs=adv,
                                clip=False)
        return surrs

    # ---------- Outer objective ----------
    def outer_obj(self,
                  # adapted_agent,               # [변경 전]
                  adapted_params: Dict[str, torch.Tensor], # [변경 후]
                  batch: Dict[str, Any],
                  ) -> torch.Tensor:
        """
        ... (주석 동일) ...
        """
        dev = next(self.agent.parameters()).device
        obs  = torch.as_tensor(batch["observations"], device=dev, dtype=torch.float32)
        acts = torch.as_tensor(batch["actions"],       device=dev, dtype=torch.float32)
        adv  = torch.as_tensor(batch["advantages"],     device=dev, dtype=torch.float32)
        
        logp_old = torch.as_tensor(batch["agent_info"]["logp"], device=dev, dtype=torch.float32).detach()

        # [변경!] adapted_agent 모듈 호출을 functional 'log_prob' 호출로 변경
        # logp_new = adapted_agent.get_outer_log_probs(obs, acts) # [변경 전]
        logp_new = self.agent.log_prob(obs, acts, params=adapted_params, post_update = True) # [변경 후]

        surr = self._surrogate(logp_new=logp_new,
                               logp_old=logp_old,
                               advs=adv,
                               clip=True)
        
        # [변경!] self.old_agent (모듈) -> self.old_params (딕셔너리) 사용
        # kl_term_logp_old = self.old_agent.get_outer_log_probs(obs,acts) # [변경 전]
        with torch.no_grad(): # KL 계산시 old_params 쪽으로는 그래디언트 불필요
            #kl_term_logp_old = self.agent.log_prob(obs, acts, params=self.old_params).detach() # [변경 후]
            kl_term_logp_old = self.old_agent.log_prob(obs, acts, params=self.old_params).detach() # [변경 후]
        return surr + self.inner_kl_coeff * self._kl_from_logps(kl_term_logp_old, logp_new).mean()

    
    # ---------- Inner loop (태스크별 적응 + KL 모니터링/anneal) ----------
    # ---------- Inner loop (태스크별 적응 + KL 모니터링/anneal) ----------
    def inner_loop(self, epoch) -> Tuple[List[Dict[str, torch.Tensor]], List]:
        """
        반환: (적응된 파라미터 딕셔너리 리스트, 마지막(post_update) 수집 경로들)
        """
        
        # 1. (ProMP용) KL(old||new) 계산을 위한 'old_params' (그래프 분리 O - 정상)
        self.old_params = {k: v.detach().clone() for k, v in self.agent.named_parameters()}
        
        # 2. (MAML용) 2차-그래디언트 계산의 '시작점'이 될 원본 파라미터(θ)
        #    [★수정★] .detach().clone()을 절대 사용하면 안 됩니다.
        current_params_theta = OrderedDict(self.agent.named_parameters())
        
        # 3. [★수정★] 'adapted_params_list'를 'current_params_theta' (그래프 연결됨)로 초기화
        adapted_params_list = [OrderedDict(current_params_theta) for _ in range(self.num_tasks)]

        # 4. Inner-loop (파라미터 교체)
        for step in range(self.inner_grad_steps):
            
            # 4a. 현재 adapted_params_list (θ, θ', ...)를 사용하여 샘플 수집
            #     (step=0일 때는 원본 파라미터(θ)로 샘플링)
            paths = self.sampler.obtain_samples(adapted_params_list, post_update=False) 
            paths = self.sample_processor.process_samples(paths)

            # 4b. 태스크별 inner-step
            for task_id in range(self.num_tasks):
                batch = paths[task_id]
                
                # 4c. [정상 동작]
                #    step=0일 때, 'params'는 그래프에 연결된 'current_params_theta'가 됨
                new_adapted_params = self.theta_prime(
                    batch, 
                    params=adapted_params_list[task_id]
                )
                
                adapted_params_list[task_id] = new_adapted_params

        # --- Inner loop 종료 ---
        
        # 3. [변경!] Post-update 샘플링 (루프 밖에서 최종 adapted_params 사용)
        last_paths = self.sampler.obtain_samples(adapted_params_list, post_update=True)
        last_paths = self.sample_processor.process_samples(last_paths)
        
        # 4. 리포트
        reward, cost_dict, _ = self.env.report_scalar(last_paths)
        self.writer.add_scalar("Reward", reward, global_step=epoch)
        self.writer.add_scalars("Costs",cost_dict, global_step=epoch)
        print(f"Epochs: {epoch+1}'s Reward:{reward}")
        
        # 5. clip epsilon anneal
        self.anneal_coeff *= self.anneal_factor

        # 6. 최종 파라미터 딕셔너리 리스트와 샘플 반환
        return adapted_params_list, last_paths
                    

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
