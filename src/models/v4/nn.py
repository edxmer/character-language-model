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

class BatchNorm1d:
    '''
    Normalizes the inputs' standard deviation.
    
    Batch normalization layer based on the Ioffe et al., 2015 paper.
    '''
    def __init__(self, features: int, eps = 1e-5, momentum = 0.05):
        
        self.gamma = torch.randn(features)
        self.beta  = torch.zeros(features)
        
        self.eps = eps
        self.momentum = momentum
        self.training = True
        
        self.running_mean = torch.ones(features)
        self.running_var  = torch.ones(features)
        
    def __call__(self, x: torch.Tensor):
        
        # shape of x: (batch_size, features), if training,
        #             (features) else
        
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0)
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        y = self.gamma * normalized + self.beta # scale and shift
        
        return y
    
    def parameters(self) -> list[torch.Tensor]:
        return [self.gamma, self.beta]