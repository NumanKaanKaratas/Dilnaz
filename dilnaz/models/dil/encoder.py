import torch
from torch import nn

from dilnaz.surface import PackedSurface
from dilnaz.surface.ops import scatter_softmax_by_unit

from ..common.norms import DilRMSNorm
from .layers import DilPackedConvSwiGLUBlock
from .sequence_blocks import DilUnitContextBackbone


class DilPackedSurfaceStem(nn.Module):
    def __init__(self, config):
        super().__init__()
        intermediate_size = config.hidden_size * config.byte_conv_expansion
        self.layers = nn.ModuleList(
            [
                DilPackedConvSwiGLUBlock(
                    config.hidden_size,
                    intermediate_size,
                    config.byte_conv_kernel_size,
                    config.rms_norm_eps,
                    config.mlp_bias,
                    config.dil_dropout,
                )
                for _ in range(config.byte_conv_layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor, surface: PackedSurface) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, surface.unit_ids, surface.mask)
        return hidden_states


class DilEncoderCore(nn.Module):
    def __init__(self, config, token_embeddings: nn.Embedding):
        super().__init__()
        self.latent_size = config.latent_size
        self.max_sequence_units = config.max_sequence_units

        object.__setattr__(self, "embed_tokens", token_embeddings)
        self.surface_stem = DilPackedSurfaceStem(config)
        self.pool_norm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pool_score = nn.Linear(config.hidden_size, 1, bias=False)
        self.pool_value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.unit_backbone = DilUnitContextBackbone(config)
        self.hidden_to_semantic = nn.Linear(config.hidden_size, config.latent_size)
        self.norm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def pool_unit_states(self, hidden_states: torch.Tensor, surface: PackedSurface) -> torch.Tensor:
        scores = self.pool_score(self.pool_norm(hidden_states)).squeeze(-1)
        attention = scatter_softmax_by_unit(scores, surface.unit_ids, surface.unit_count, surface.mask)
        values = self.pool_value(hidden_states)
        output = values.new_zeros((values.shape[0], surface.unit_count, values.shape[-1]))
        index = surface.unit_ids.clamp_min(0).unsqueeze(-1).expand(-1, -1, values.shape[-1])
        output.scatter_add_(1, index, values * attention.unsqueeze(-1))
        return output * surface.unit_mask.unsqueeze(-1).to(output.dtype)

    def forward(
        self,
        surface: PackedSurface,
        output_hidden_states: bool = False,
    ) -> torch.Tensor:
        if not isinstance(surface, PackedSurface):
            raise TypeError("DilEncoderCore.forward expects PackedSurface")
        if surface.unit_count > self.max_sequence_units:
            raise ValueError(f"packed surface unit_count {surface.unit_count} exceeds max_sequence_units={self.max_sequence_units}")

        hidden_states = self.embed_tokens(surface.ids)
        hidden_states = hidden_states * surface.mask.unsqueeze(-1).to(hidden_states.dtype)
        hidden_states = self.surface_stem(hidden_states, surface)

        token_states = self.pool_unit_states(hidden_states, surface)
        hidden_states, layer_vectors = self.unit_backbone(
            token_states,
            surface.unit_mask,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = self.norm(hidden_states)
        semantic = self.hidden_to_semantic(hidden_states)
        if output_hidden_states:
            return semantic, layer_vectors
        return semantic
