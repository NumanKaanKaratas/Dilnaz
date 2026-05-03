from __future__ import annotations

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


class PartialRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, partial_rotary_factor: float, rope_theta: float):
        super().__init__()
        partial_dim = int(head_dim * partial_rotary_factor)
        partial_dim = max(2, partial_dim - partial_dim % 2)
        if partial_dim > head_dim:
            raise ValueError("partial rotary dimension cannot exceed head_dim")
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, partial_dim, 2, dtype=torch.float32) / partial_dim)
        )
        self.head_dim = head_dim
        self.partial_dim = partial_dim
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.partial_dim == 0:
            return query_states, key_states
        freqs = torch.einsum(
            "bt,d->btd",
            position_ids.to(dtype=self.inv_freq.dtype),
            self.inv_freq,
        )
        cos = freqs.cos().repeat_interleave(2, dim=-1).unsqueeze(2)
        sin = freqs.sin().repeat_interleave(2, dim=-1).unsqueeze(2)

        query_rot = query_states[..., : self.partial_dim]
        query_pass = query_states[..., self.partial_dim :]
        key_rot = key_states[..., : self.partial_dim]
        key_pass = key_states[..., self.partial_dim :]
        query_rot = query_rot * cos + rotate_half(query_rot) * sin
        key_rot = key_rot * cos + rotate_half(key_rot) * sin
        return torch.cat((query_rot, query_pass), dim=-1), torch.cat((key_rot, key_pass), dim=-1)
