# base.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import Sequence, Type, List, Dict, Optional, Tuple
from collections import OrderedDict

class BaseAgent(nn.Module):
  """
  알고리즘(ProMP 등)이 기대하는 최소 정책 API를 한 클래스에 통합.

  핵심 원칙
    - 분포 타입(가우시안/카테고리/혼합 등)에는 전혀 의존하지 않음.
    - 알고리즘은 오직 log_prob(obs, actions, params)만 호출.
    - 샘플러는 act(obs) 호출 시 agent_infos['logp'](샘플 시점의 log_prob)를 반드시 기록.

  서브클래스가 구현해야 할 것
    - act(self, obs, deterministic=False) -> (actions, agent_infos[필수:'logp'])
    - log_prob(self, obs, actions, params=None) -> [B] 텐서
    - (선택) _functional_forward(self, obs, params): inner-loop로 적응한 params로 forward 하고 싶을 때
    - (필요 시) forward(...) 재정의 (분포/출력 방식이 다르면)

  편의 기능
    - 기본 MLP 백본(self.net)을 옵션으로 제공(build_backbone=True). 필요 없으면 무시하거나 False로 끄면 됨.
    - get_action / get_actions_all_tasks: 기존 코드와의 이름 호환용 래퍼(선택 구현).
  """

  def __init__(self,
               mlp,
               optimizer,
                gamma: float = 0.99,
                has_value_fn: bool = False,
                need_probs: bool = False):
      super().__init__()
      self.mlp = mlp
      self.optimizer = optimizer
      self.gamma = gamma
      self.has_value_fn = has_value_fn  # [NEW]

  # ---------------------- 최소 API (알고리즘이 의존) ----------------------
  torch.no_grad()
  def get_actions(self,
                  observation: torch.Tensor,
                  task: int = 0,
                  deterministic: bool = False, need_probs: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    기존 코드 호환을 위해 제공하는 래퍼. 내부적으로 act(...)를 호출한다.
    - task 인자는 시그니처 호환용일 뿐, 기본 Basemlp에서는 사용하지 않는다.
    """
    raise NotImplementedError
  
  def log_prob(self,
                obs: torch.Tensor,
                actions: torch.Tensor,) -> torch.Tensor:
      """
      Args:
        obs: [B, obs_dim]
        actions: [B, out_dim]
      Returns:
        [B] 텐서 (각 샘플의 log_prob)
      Note:
        - 분포 타입에 상관없이 로그확률만 정확히 반환하면 알고리즘 쪽은 그대로 작동한다.
      """
      raise NotImplementedError