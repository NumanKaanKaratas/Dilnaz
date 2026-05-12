from typing import Optional

import torch
from torch import nn

from .backbone import NazBackboneCache, NazSemanticBackbone
from .configuration import NazConfig
from .dynamics_head import SemanticDynamicsMixtureHead


class NazStudentCore(nn.Module):
    def __init__(self, config: NazConfig):
        super().__init__()
        self.semantic_embed_proj = nn.Sequential(
            nn.Linear(config.latent_size, 2 * config.hidden_size),
            nn.SiLU(),
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=1e-6),
        )
        self.backbone = NazSemanticBackbone(config)
        self.semantic_head = SemanticDynamicsMixtureHead(config)

    def embed_semantic_states(self, semantic_states: torch.Tensor) -> torch.Tensor:
        return self.semantic_embed_proj(semantic_states)

    def forward(
        self,
        semantic_states: torch.Tensor,
        unit_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[NazBackboneCache] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs_embeds = self.embed_semantic_states(semantic_states)
        output = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return output.last_hidden_state, output.moe_balance_loss, output.moe_usage
