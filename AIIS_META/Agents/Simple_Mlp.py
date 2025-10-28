from typing import Sequence, Type, List, Dict, Optional, Tuple
from collections import OrderedDict
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
  """
  가장 단순한 MLP:
    - hidden_sizes 길이만큼 Linear 쌓기
    - hidden 활성함수는 전 레이어 공통 적용
    - 출력층 별도 Linear + (옵션) 출력 활성함수
  """
  def __init__(self,
                input_dim: int,
                output_dim: int,
                hidden_layers: list,
                ):
      super().__init__()
      self.input_dim = input_dim
      self.output_dim = output_dim
      self.layers = [nn.Linear(self.input_dim, hidden_layers[0]), nn.ReLU() ]
      od = OrderedDict()
      
      last = input_dim
      
      for layer_id in range(len(hidden_layers)):
          od[f"fc{layer_id}"]  = nn.Linear(last, hidden_layers[layer_id])
          od[f"act{layer_id}"] = nn.Tanh() 
          last = hidden_layers[layer_id]
      od[f"{len(hidden_layers)}"] = nn.Linear(last, output_dim)
      self.model =  nn.Sequential(od)

  def value_function(self,
                      obs: torch.Tensor,
                      params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
      """
      A2C/PPO 등 critic이 있는 정책에서만 구현.
      없는 경우 NotImplementedError 발생.
      """
      if not self.has_value_fn:
          raise NotImplementedError("Policy has no value_function.")
      raise NotImplementedError("Subclass with baseline must implement this.")

  def forward(self, state: torch.Tensor) -> torch.Tensor:
      output = self.model(state)
      return output