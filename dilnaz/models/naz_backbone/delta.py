from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from fla.modules import FusedRMSNormGated
from fla.modules.convolution import ShortConvolution
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla.ops.gated_delta_rule.naive import naive_chunk_gated_delta_rule, naive_recurrent_gated_delta_rule
from torch import nn

from .cache import NazBackboneLayerCache


class SemanticDeltaMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.linear_key_head_dim != config.linear_value_head_dim:
            raise ValueError("linear key/value head dims must match in SemanticDeltaMixer")
        if config.linear_num_value_heads % config.linear_num_key_heads != 0:
            raise ValueError("linear_num_value_heads must be divisible by linear_num_key_heads")
        self.hidden_size = config.hidden_size
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.head_dim = config.linear_key_head_dim
        self.key_dim = self.num_key_heads * self.head_dim
        self.value_dim = self.num_value_heads * self.head_dim
        if self.value_dim != config.hidden_size:
            raise ValueError("linear_num_value_heads * linear_value_head_dim must equal hidden_size")
        self.conv_kernel_size = config.linear_conv_kernel_size
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.activation = config.hidden_act
        self.rms_norm_eps = config.rms_norm_eps

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_value_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_value_heads, bias=False)
        self.conv1d = ShortConvolution(
            self.conv_dim,
            self.conv_kernel_size,
            bias=False,
            activation=self.activation,
            backend="triton",
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_value_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_value_heads).uniform_(0.0, 16.0).clamp_min(1e-4)))
        self.norm = FusedRMSNormGated(self.head_dim, eps=self.rms_norm_eps, activation=self.activation)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def _torch_causal_conv(
        self,
        mixed_qkv: torch.Tensor,
        cache: Optional[NazBackboneLayerCache],
        use_cache: bool,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = mixed_qkv.shape
        if use_cache and cache is not None and cache.conv_state is not None:
            prefix = cache.conv_state.transpose(1, 2)[:, -(self.conv_kernel_size - 1) :]
        else:
            prefix = mixed_qkv.new_zeros(batch_size, self.conv_kernel_size - 1, self.conv_dim)
        combined = torch.cat((prefix, mixed_qkv), dim=1)
        conved = F.conv1d(
            combined.transpose(1, 2),
            self.conv1d.weight,
            self.conv1d.bias,
            groups=self.conv_dim,
        ).transpose(1, 2)
        conved = F.silu(conved) if self.activation in {"silu", "swish"} else conved
        if use_cache and cache is not None:
            new_state = combined[:, -self.conv_kernel_size :]
            if new_state.shape[1] < self.conv_kernel_size:
                pad = mixed_qkv.new_zeros(batch_size, self.conv_kernel_size - new_state.shape[1], self.conv_dim)
                new_state = torch.cat((pad, new_state), dim=1)
            cache.conv_state = new_state.transpose(1, 2).contiguous()
        return conved[:, -sequence_length:]

    def _causal_conv(
        self,
        mixed_qkv: torch.Tensor,
        cache: Optional[NazBackboneLayerCache],
        use_cache: bool,
    ) -> torch.Tensor:
        if mixed_qkv.is_cuda:
            conv_state = cache.conv_state if use_cache and cache is not None else None
            mixed_qkv, new_conv_state = self.conv1d(
                mixed_qkv,
                cache=conv_state,
                output_final_state=use_cache and cache is not None,
            )
            if use_cache and cache is not None:
                cache.conv_state = new_conv_state
            return mixed_qkv
        return self._torch_causal_conv(mixed_qkv, cache, use_cache)

    def _gated_rms_norm(self, core_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if core_output.is_cuda:
            return self.norm(core_output, gate)
        gate = F.silu(gate) if self.activation in {"silu", "swish"} else torch.sigmoid(gate)
        output = core_output * gate
        normed = output.float() * torch.rsqrt(output.float().pow(2).mean(dim=-1, keepdim=True) + self.rms_norm_eps)
        return (normed * self.norm.weight.float()).to(core_output.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[NazBackboneLayerCache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        batch_size, sequence_length, _ = hidden_states.shape
        has_cached_state = use_cache and cache is not None and cache.delta_state is not None

        mixed_qkv = self._causal_conv(self.in_proj_qkv(hidden_states), cache, use_cache)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.view(batch_size, sequence_length, self.num_key_heads, self.head_dim)
        key = key.view(batch_size, sequence_length, self.num_key_heads, self.head_dim)
        value = value.view(batch_size, sequence_length, self.num_value_heads, self.head_dim)

        gate = self.in_proj_z(hidden_states).view(batch_size, sequence_length, self.num_value_heads, self.head_dim)
        beta = self.in_proj_b(hidden_states).sigmoid()
        decay = -self.A_log.float().exp() * F.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias)
        initial_state = cache.delta_state if has_cached_state else None
        output_final_state = use_cache and cache is not None

        if hidden_states.is_cuda:
            if has_cached_state and sequence_length == 1:
                core_output, final_state = fused_recurrent_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=decay,
                    beta=beta,
                    initial_state=initial_state,
                    output_final_state=output_final_state,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                core_output, final_state = chunk_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=decay,
                    beta=beta,
                    initial_state=initial_state,
                    output_final_state=output_final_state,
                    use_qk_l2norm_in_kernel=True,
                )
        else:
            query = F.normalize(query, p=2.0, dim=-1)
            key = F.normalize(key, p=2.0, dim=-1)
            if has_cached_state and sequence_length == 1:
                core_output, final_state = naive_recurrent_gated_delta_rule(
                    query,
                    key,
                    value,
                    beta=beta,
                    g=decay,
                    initial_state=initial_state,
                    output_final_state=output_final_state,
                )
            else:
                core_output, final_state = naive_chunk_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=decay,
                    beta=beta,
                    initial_state=initial_state,
                    output_final_state=output_final_state,
                )

        if output_final_state:
            cache.delta_state = final_state

        core_output = core_output.reshape(-1, self.head_dim)
        gate = gate.reshape(-1, self.head_dim)
        core_output = self._gated_rms_norm(core_output, gate)
        core_output = core_output.reshape(batch_size, sequence_length, self.value_dim)
        return self.out_proj(core_output)
