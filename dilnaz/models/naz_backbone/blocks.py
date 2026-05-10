from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .attention import SemanticGlobalAttention
from .cache import NazBackboneLayerCache
from .delta import SemanticDeltaMixer
from .feedforward import GatedFeedForward, SparseMoEFeedForward
from .normalization import ZeroCenteredRMSNorm


class NazHybridBlock(nn.Module):
    def __init__(self, config, layer_type: str, layer_idx: int):
        super().__init__()
        if layer_type not in {"delta", "global"}:
            raise ValueError(f"unsupported NAZ layer_type={layer_type}")
        self.layer_type = layer_type
        self.input_norm = ZeroCenteredRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mixer_norm = ZeroCenteredRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mixer = (
            SemanticGlobalAttention(config)
            if layer_type == "global"
            else SemanticDeltaMixer(config)
        )
        moe_start = max(config.num_hidden_layers - config.moe_layers, 0)
        self.uses_moe = layer_idx >= moe_start
        self.feedforward = (
            SparseMoEFeedForward(
                hidden_size=config.hidden_size,
                shared_intermediate_size=config.intermediate_size,
                expert_intermediate_size=config.moe_expert_intermediate_size,
                num_experts=config.moe_num_experts,
                top_k=config.moe_top_k,
            )
            if self.uses_moe
            else GatedFeedForward(config.hidden_size, config.intermediate_size)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.LongTensor,
        cache: Optional[NazBackboneLayerCache] = None,
        use_cache: bool = False,
        cache_position: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        if self.layer_type == "global":
            hidden_states = self.mixer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache=cache,
                use_cache=use_cache,
                cache_position=cache_position,
            )
        else:
            hidden_states = self.mixer(
                hidden_states,
                attention_mask=attention_mask,
                cache=cache,
                use_cache=use_cache,
            )
        hidden_states = residual + hidden_states
        feedforward_input = self.post_mixer_norm(hidden_states)
        if self.uses_moe:
            feedforward_output, moe_balance_loss, moe_usage = self.feedforward(feedforward_input)
        else:
            feedforward_output = self.feedforward(feedforward_input)
            moe_balance_loss = hidden_states.new_zeros(())
            moe_usage = None
        hidden_states = hidden_states + feedforward_output
        return hidden_states, moe_balance_loss, moe_usage
