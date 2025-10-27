# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import copy
from contextlib import contextmanager
import torch.optim as optim
from AIIS_META.Sampler.meta_sampler import MetaSampler as sampler
from AIIS_META.Sampler.meta_sample_processor import MetaSampleProcessor
from torch.utils.tensorboard import SummaryWriter
class MAML_BASE(nn.Module):
    """
    MAML 틀 (FO-MAML):
      - inner_loop: meta-parameter로부터 태스크별 적응 파라미터 생성(1차 근사)
      - outer_loop: 적응 파라미터들에서의 outer objective grad를 모아 meta-parameter 업데이트
      - learn: inner/outer 오케스트레이션
    """
    def __init__(self,
                 env,
                 max_path_length,
                 agent,
                 alpha,
                 beta,
                 tensor_log,
                 baseline = None,
                 inner_grad_steps: int = 1,
                 num_tasks: int = 4,
                 rollout_per_task: int = 5,
                 outer_iters: int = 5,
                 parallel: bool = False,
                 clip_eps: float = 0.2,
                 init_inner_kl_penalty: float = 1e-2,
                 discount: float = 0.99,
                 gae_lambda: float = 1.0,
                 normalize_adv = True,
                 device = torch.device('cuda')):
        super().__init__()
        self.agent = agent
        self.optimizer = optim.Adam(self.agent.parameters(), lr = beta)
        self.env = env
        self.inner_grad_steps = inner_grad_steps
        self.num_tasks = num_tasks
        self.outer_iters = outer_iters
        self.rollout_per_task = rollout_per_task
        self.clip_eps = clip_eps
        self.inner_kl_coeff = torch.full((inner_grad_steps,), init_inner_kl_penalty, dtype=torch.float32, device=device)
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.adapted_agents = None
        self.sample_processor = MetaSampleProcessor(baseline=baseline,discount=discount ,gae_lambda= gae_lambda,normalize_adv=normalize_adv)
        self.writer = SummaryWriter(log_dir=tensor_log)
        self.sampler = sampler(self.env,
                               self.agent,
                               self.rollout_per_task,
                               self.num_tasks,
                               max_path_length,
                               envs_per_task=None,
                               parallel=parallel)
        

    # -------------------- 훅(오버라이드 지점) --------------------
    def inner_obj(self, batch: dict, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """예: -(ratio * adv).mean()  (최소화 기준)"""
        raise NotImplementedError

    def outer_obj(self, params: Dict[str, torch.Tensor],  batch: dict) -> torch.Tensor:
        """예: PPO-clip + (필요 시) KL penalty 포함 (최소화 기준)"""
        raise NotImplementedError

    def step_kl(self, batch: dict, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """(옵션) KL(old||new) 측정/패널티용"""
        raise NotImplementedError
    @staticmethod
    def _safe_key(name: str) -> str:
        # ParameterDict 등록 시 키에 '.' 쓰면 안 됨
        return name.replace('.', '__')
    
    def _create_step_size_tensors(self) -> None:
        """
        각 파라미터와 동일 shape의 step size 텐서를 생성.
        - trainable=True  -> nn.ParameterDict 에 등록 (메타 최적화 시 함께 학습 가능)
        - trainable=False -> 일반 텐서 dict로 보관 (requires_grad=False)
        """

        pdict = nn.ParameterDict()
        for name, p in self.agent.named_parameters():
            key = self._safe_key(name)
            init = torch.full_like(p, fill_value=self.alpha, device=self.device)
            pdict[key] = nn.Parameter(init)  # learnable α_i
        self.inner_step_sizes = pdict

    # -------------------- 내부 유틸 --------------------

    def _to_tensor(self, x, device, dtype=torch.float32):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(x, device=device, dtype=dtype)
    
    def theta_prime(self, batch):
        adapted_agent = copy.deepcopy(self.agent)
        surr = self.inner_obj(batch)
        # (c) 그 파라미터들에 대한 grad 계산 (Outer에서 접근 가능해야 하기 때문에 Create_Graph True)
        grads = torch.autograd.grad(
            surr,
            self.agent.parameters(),
            create_graph=True,
            allow_unused=False
        )
        for (name, p), g in zip(adapted_agent.named_parameters(), grads):
            if g is None:
                continue
            step = self.inner_step_sizes[self._safe_key(name)]
            # θ' = θ - α_i ⊙ ∇θ  (ProMP inner update)
            p = p -step * g

        return adapted_agent
    # -------------------- 학습 루프(오케스트레이션) --------------------
    def learn(self, epochs: int):
        """
        1) inner_loop -> adapted_state_dicts, last_paths
        2) (반복) outer_loop -> meta update
        """
        self._create_step_size_tensors()
        for epoch in range(epochs):
            self.sampler.update_tasks()
            adapted_agents, paths = self.inner_loop(epoch)
            print("Outer Learning Start")
            self.outer_loop(paths,adapted_agents)
   
    # -------------------- inner / outer --------------------
    def inner_loop(self,
                   ) -> Tuple[List[Dict[str, torch.Tensor]], List]:
        
        raise NotImplementedError
    
    def outer_loop(self,
                   paths, adapted_agents):  # inner_loop에서 마지막으로 수집된 경로(또는 새로 수집 가능)):
        for itr in range(self.outer_iters):
            loss_outs = []
            for task_id in range(self.num_tasks):
                batch = paths[task_id]
                if itr !=0:
                    adapted_agents[task_id] = self.theta_prime(batch)
                else:
                    pass
                loss_outs.append(self.outer_obj(adapted_agents[task_id], batch))
            mean_loss_out = sum(loss_outs)/len(loss_outs)
            # 2) meta-파라미터 기준으로 grad 계산 (create_graph=False, retain_graph=False)
            #    ※ meta 그라디언트를 원하면 inputs는 self.agent.parameters() 여야 합니다.
            self.optimizer.zero_grad()
            mean_loss_out.backward()
            self.optimizer.step()