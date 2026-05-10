from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GatedFeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class SparseMoEFeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        shared_intermediate_size: int,
        expert_intermediate_size: int,
        num_experts: int,
        top_k: int,
    ):
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be > 0")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        self.num_experts = num_experts
        self.top_k = top_k
        self.shared = GatedFeedForward(hidden_size, shared_intermediate_size)
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            GatedFeedForward(hidden_size, expert_intermediate_size)
            for _ in range(num_experts)
        )
        self.register_buffer("_uniform_usage", torch.full((num_experts,), 1.0 / num_experts), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        flat_states = hidden_states.reshape(batch_size * sequence_length, hidden_size)
        router_logits = self.router(flat_states)
        router_probs = F.softmax(router_logits.float(), dim=-1).to(hidden_states.dtype)
        top_weights, top_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(top_weights.dtype).eps)

        routed = torch.zeros_like(flat_states)
        for expert_idx, expert in enumerate(self.experts):
            selected = top_indices.eq(expert_idx)
            selected_token_idx, selected_route_idx = selected.nonzero(as_tuple=True)
            if selected_token_idx.numel() == 0:
                continue
            expert_input = flat_states.index_select(0, selected_token_idx)
            expert_output = expert(expert_input)
            route_weight = top_weights[selected_token_idx, selected_route_idx].unsqueeze(-1)
            routed.index_add_(0, selected_token_idx, expert_output * route_weight)

        usage = router_probs.float().mean(dim=0)
        load = F.one_hot(top_indices.reshape(-1), self.num_experts).float().mean(dim=0)
        uniform = self._uniform_usage.to(device=usage.device, dtype=usage.dtype)
        balance_loss = (
            usage * (usage.clamp_min(1e-8).log() - uniform.log())
        ).sum() + (
            load * (load.clamp_min(1e-8).log() - uniform.log())
        ).sum()
        output = self.shared(hidden_states) + routed.reshape(batch_size, sequence_length, hidden_size)
        return output, balance_loss, usage.detach()
