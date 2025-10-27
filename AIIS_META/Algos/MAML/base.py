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
from collections import OrderedDict
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
        self._create_step_size_tensors()

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
    
    

    def theta_prime(self, batch: dict, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        ... (주석 동일) ...
        """
        surr = self.inner_obj(batch, params=params) 
        
        grads = torch.autograd.grad(
            surr,
            params.values(),
            create_graph=True
        )
        
        adapted_params = OrderedDict()
        
        for (name, p), g in zip(params.items(), grads):
            if g is None:
                adapted_params[name] = p
                continue
            
            # [★핵심 수정★]
            # self.alpha (float) 대신 learnable step size 딕셔너리를 사용
            # step = self.alpha # [변경 전]
            step = self.inner_step_sizes[self._safe_key(name)] # [변경 후]
            
            adapted_params[name] = p - step * g 

        return adapted_params
    
    # -------------------- 학습 루프(오케스트레이션) --------------------
    def learn(self, epochs: int):
        """
        ... (주석 동일) ...
        """
        # [변경!] __init__으로 이동
        # self._create_step_size_tensors() 
        
        for epoch in range(epochs):
            self.sampler.update_tasks()
            
            # [변경!] 변수명 변경 (이제 'params 딕셔너리 리스트'가 반환됨)
            # adapted_agents, paths = self.inner_loop(epoch) # [변경 전]
            adapted_params_list, paths = self.inner_loop(epoch) # [변경 후]
            
            print("Outer Learning Start")
            
            # [변경!] 수정된 변수명 전달
            # self.outer_loop(paths, adapted_agents) # [변경 전]
            self.outer_loop(paths, adapted_params_list) # [변경 후]
   
    # -------------------- inner / outer --------------------
    def inner_loop(self,
                   ) -> Tuple[List[Dict[str, torch.Tensor]], List]:
        
        raise NotImplementedError
    
    def outer_loop(self,
                   paths: List[Dict], # 'post_update_paths'가 전달됨
                   adapted_params_list: List[Dict[str, torch.Tensor]]): # '최종' 파라미터 리스트
        
        # 'outer_iters'는 PPO의 에포크처럼
        # '동일한' post-update 배치(paths)에 대해 메타-옵티마이저를 여러 번 스텝하는 용도
        for itr in range(self.outer_iters):
            loss_outs = []
            
            for task_id in range(self.num_tasks):
                batch = paths[task_id] # post-update 배치 사용
                
                # [★핵심 수정★]
                # inner_loop가 반환한 '최종' adapted params를 가져옴
                if itr == 0:
                    final_adapted_params = adapted_params_list[task_id]
                
                else:
                    final_adapted_params = self.theta_prime(batch, OrderedDict(self.agent.named_parameters()))
                
                # [★핵심 제거★]
                # 'if itr != 0:' 블록 전체를 제거합니다.
                # inner-step은 'inner_loop'에서 이미 완료되었습니다.
                
                # '최종' 파라미터로 outer_obj 계산
                loss = self.outer_obj(final_adapted_params, batch)
                loss_outs.append(loss)
                
            mean_loss_out = sum(loss_outs)/len(loss_outs)
            
            # 2) meta-파라미터(self.agent.parameters)에 대해 미분
            self.optimizer.zero_grad()
            mean_loss_out.backward() # (그래프가 inner_loop를 거쳐 원본 파라미터까지 연결됨)
            self.optimizer.step()