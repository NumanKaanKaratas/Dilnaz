import torch
from torch import nn


class DilRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
        self.reduction_dtype: torch.dtype | None = torch.float32

    def set_reduction_dtype(self, dtype: torch.dtype | None):
        self.reduction_dtype = dtype
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normed = hidden_states if self.reduction_dtype is None else hidden_states.to(self.reduction_dtype)
        variance = normed.pow(2).mean(dim=-1, keepdim=True)
        normed = normed * torch.rsqrt(variance + self.eps)
        return (normed * (1.0 + self.weight.to(normed.dtype))).to(input_dtype)
