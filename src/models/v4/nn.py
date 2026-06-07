# I will recreate the following classes from PyTorch:
# - Linear
# - Tanh
# - BatchNorm1d
# I will try to use similar namings as them.

import torch
import torch.nn.functional as F

class Linear:
    def __init__(self, fan_in: int, fan_out: int, has_bias = True):
        self.w = torch.randn(fan_in, fan_out) / torch.tensor(fan_in, dtype=torch.float)**0.5 # Kaiming initialization
        if has_bias:
            self.b = torch.zeros(fan_out)
        else:
            self.b = None
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.w
        if self.b is not None:
            y += self.b
        return y
    
    def paremeters(self) -> list[torch.Tensor]:
        return [self.w] + ([] if self.b is None else [self.b])

class Tanh:
    def __call__(self, x: torch.Tensor):
        return torch.tanh(x)
    def parameters(self) -> list[torch.Tensor]:
        return []