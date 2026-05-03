from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .attention import SemanticGlobalAttention
from .cache import NazBackboneLayerCache
from .delta import SemanticDeltaMixer
from .feedforward import GatedFeedForward
from .normalization import ZeroCenteredRMSNorm


class NazHybridBlock(nn.Module):
    def __init__(self, config, layer_type: str):
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
        self.feedforward = GatedFeedForward(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.LongTensor,
        cache: Optional[NazBackboneLayerCache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        if self.layer_type == "global":
            hidden_states = self.mixer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache=cache,
                use_cache=use_cache,
            )
        else:
            hidden_states = self.mixer(hidden_states, cache=cache, use_cache=use_cache)
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.feedforward(self.post_mixer_norm(hidden_states))
        return hidden_states
