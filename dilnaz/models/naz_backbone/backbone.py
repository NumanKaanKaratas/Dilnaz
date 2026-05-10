from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .blocks import NazHybridBlock
from .cache import NazBackboneCache
from .normalization import ZeroCenteredRMSNorm


@dataclass
class NazBackboneOutput:
    last_hidden_state: torch.Tensor
    past_key_values: Optional[NazBackboneCache] = None


class NazSemanticBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.full_attention_interval <= 0:
            raise ValueError("full_attention_interval must be > 0")
        if config.num_attention_heads % config.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if config.hidden_size != config.num_attention_heads * config.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if config.partial_rotary_factor <= 0.0 or config.partial_rotary_factor > 1.0:
            raise ValueError("partial_rotary_factor must be in (0, 1]")

        self.layer_types = tuple(
            "global" if (idx + 1) % config.full_attention_interval == 0 else "delta"
            for idx in range(config.num_hidden_layers)
        )
        self.layers = nn.ModuleList(
            [NazHybridBlock(config, layer_type) for layer_type in self.layer_types]
        )
        self.norm = ZeroCenteredRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[NazBackboneCache] = None,
        use_cache: bool = False,
        max_cache_length: Optional[int] = None,
    ) -> NazBackboneOutput:
        batch_size, sequence_length, _ = inputs_embeds.shape
        if use_cache and past_key_values is None:
            past_key_values = NazBackboneCache.empty(
                len(self.layers),
                batch_size=batch_size,
                max_cache_length=max_cache_length,
                num_key_value_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
                global_layer_indices=tuple(
                    idx for idx, layer_type in enumerate(self.layer_types) if layer_type == "global"
                ),
            )
        if position_ids is None:
            start = past_key_values.position if past_key_values is not None else 0
            position_ids = torch.arange(
                start,
                start + sequence_length,
                device=inputs_embeds.device,
            ).reshape(1, sequence_length)
            position_ids = position_ids.expand(batch_size, sequence_length)

        hidden_states = inputs_embeds
        cache_position = past_key_values.position if past_key_values is not None else 0
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = past_key_values.layers[layer_idx] if past_key_values is not None else None
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache=layer_cache,
                use_cache=use_cache,
                cache_position=cache_position,
            )
        hidden_states = self.norm(hidden_states)
        if use_cache and past_key_values is not None:
            past_key_values.position += sequence_length
        return NazBackboneOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )
