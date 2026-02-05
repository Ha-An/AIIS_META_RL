from typing import Sequence, Type, List, Dict, Optional, Tuple
from collections import OrderedDict
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
  """
  Simplest MLP:
    - Stack Linear layers for each hidden size
    - Use the same activation for all hidden layers
    - Final output layer is a separate Linear (optional output activation)
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
      Implemented only for policies with a critic (e.g., A2C/PPO).
      Raises NotImplementedError if not available.
      """
      if not self.has_value_fn:
          raise NotImplementedError("Policy has no value_function.")
      raise NotImplementedError("Subclass with baseline must implement this.")

  def forward(self, state: torch.Tensor) -> torch.Tensor:
      output = self.model(state)
      return output