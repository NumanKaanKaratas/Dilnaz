from __future__ import annotations

import math

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
        self.expert_gate_weight = nn.Parameter(torch.empty(num_experts, expert_intermediate_size, hidden_size))
        self.expert_up_weight = nn.Parameter(torch.empty(num_experts, expert_intermediate_size, hidden_size))
        self.expert_down_weight = nn.Parameter(torch.empty(num_experts, hidden_size, expert_intermediate_size))
        self.register_buffer("_uniform_usage", torch.full((num_experts,), 1.0 / num_experts), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in (self.expert_gate_weight, self.expert_up_weight):
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.expert_down_weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        flat_states = hidden_states.reshape(batch_size * sequence_length, hidden_size)
        router_logits = self.router(flat_states)
        router_probs = F.softmax(router_logits.float(), dim=-1).to(hidden_states.dtype)
        top_weights, top_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(top_weights.dtype).eps)

        token_indices = torch.arange(flat_states.shape[0], device=flat_states.device).unsqueeze(1).expand_as(top_indices)
        token_indices = token_indices.reshape(-1)
        selected_experts = top_indices.reshape(-1)
        selected_weights = top_weights.reshape(-1)
        sorted_order = torch.argsort(selected_experts)
        selected_experts = selected_experts.index_select(0, sorted_order)
        token_indices = token_indices.index_select(0, sorted_order)
        selected_weights = selected_weights.index_select(0, sorted_order)

        routed = flat_states.new_zeros(flat_states.shape)
        expert_counts = torch.bincount(selected_experts, minlength=self.num_experts)
        expert_offsets = expert_counts.cumsum(dim=0)
        start = 0
        for expert_idx, end_tensor in enumerate(expert_offsets):
            end = int(end_tensor.item())
            if end == start:
                continue
            current_tokens = token_indices[start:end]
            expert_input = flat_states.index_select(0, current_tokens)
            gate = F.linear(expert_input, self.expert_gate_weight[expert_idx])
            up = F.linear(expert_input, self.expert_up_weight[expert_idx])
            expert_hidden = F.silu(gate) * up
            expert_output = F.linear(expert_hidden, self.expert_down_weight[expert_idx])
            expert_output = expert_output * selected_weights[start:end].unsqueeze(-1)
            routed.index_add_(0, current_tokens, expert_output)
            start = end

        usage = router_probs.float().mean(dim=0)
        load = expert_counts.to(dtype=torch.float32) / flat_states.shape[0]
        uniform = self._uniform_usage.to(device=usage.device, dtype=usage.dtype)
        balance_loss = (
            usage * (usage.clamp_min(1e-8).log() - uniform.log())
        ).sum() + (
            load * (load.clamp_min(1e-8).log() - uniform.log())
        ).sum()
        output = self.shared(hidden_states) + routed.reshape(batch_size, sequence_length, hidden_size)
        return output, balance_loss, usage.detach()
