import torch
from torch import nn

from dilnaz.surface import PackedSurface
from dilnaz.surface.ops import scatter_softmax_by_unit

from ..common.norms import DilRMSNorm
from ..common.latents import compose_factorized_latent, normalize_semantic_latents
from .layers import DilPackedConvSwiGLUBlock, DilLayer


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


def dil_context_attention_heads(hidden_size: int) -> int:
    return next(heads for heads in (8, 4, 2, 1) if hidden_size % heads == 0)


class DilUnitContextBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = dil_context_attention_heads(config.hidden_size)
        self.head_dim = config.hidden_size // self.num_heads
        self.attn_norm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(config.dil_dropout)
        self.mlp = nn.Sequential(
            DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps),
            nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias),
            nn.SiLU(),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias),
            nn.Dropout(config.dil_dropout),
        )

    def forward(self, hidden_states: torch.Tensor, unit_mask: torch.Tensor) -> torch.Tensor:
        batch_size, unit_count, _ = hidden_states.shape
        normed = self.attn_norm(hidden_states)
        query = self.q_proj(normed).view(batch_size, unit_count, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(normed).view(batch_size, unit_count, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(normed).view(batch_size, unit_count, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim**0.5)
        key_mask = unit_mask.view(batch_size, 1, 1, unit_count)
        safe_key_mask = torch.where(
            key_mask.any(dim=-1, keepdim=True),
            key_mask,
            torch.ones_like(key_mask),
        )
        scores = scores.masked_fill(~safe_key_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores.float(), dim=-1).to(scores.dtype).masked_fill(~safe_key_mask, 0.0)
        attention = attention * key_mask.any(dim=-1, keepdim=True).to(attention.dtype)
        context = torch.matmul(attention, value).transpose(1, 2).reshape(batch_size, unit_count, self.hidden_size)
        hidden_states = hidden_states + self.attn_dropout(self.o_proj(context))
        hidden_states = hidden_states + self.mlp(hidden_states)
        return hidden_states * unit_mask.unsqueeze(-1).to(hidden_states.dtype)


class DilEncoderCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context_size = config.context_size
        self.target_index = config.target_index
        self.latent_size = config.latent_size
        self.semantic_latent_size = config.semantic_latent_size
        self.surface_latent_size = config.surface_latent_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.surface_stem = DilPackedSurfaceStem(config)
        self.encoder_layers = nn.ModuleList([DilLayer(config) for _ in range(config.num_encoder_layers)])
        self.num_stage_layers = config.num_encoder_layers // 2
        self.pool_norm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pool_score = nn.Linear(config.hidden_size, 1, bias=False)
        self.pool_value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.context_offset_embeddings = nn.Embedding(config.context_size, config.hidden_size)
        self.context_blocks = nn.ModuleList([DilUnitContextBlock(config) for _ in range(config.encoder_context_layers)])
        self.semantic_head = nn.Linear(config.hidden_size, config.semantic_latent_size)
        self.surface_head = nn.Linear(config.hidden_size, config.surface_latent_size)
        self.norm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def pool_unit_states(self, hidden_states: torch.Tensor, surface: PackedSurface) -> torch.Tensor:
        scores = self.pool_score(self.pool_norm(hidden_states)).squeeze(-1)
        attention = scatter_softmax_by_unit(scores, surface.unit_ids, surface.unit_count, surface.mask)
        values = self.pool_value(hidden_states)
        output = values.new_zeros((values.shape[0], surface.unit_count, values.shape[-1]))
        index = surface.unit_ids.clamp_min(0).unsqueeze(-1).expand(-1, -1, values.shape[-1])
        output.scatter_add_(1, index, values * attention.unsqueeze(-1))
        return output * surface.unit_mask.unsqueeze(-1).to(output.dtype)

    def pooled_target_vector(self, hidden_states: torch.Tensor, surface: PackedSurface) -> torch.Tensor:
        token_states = self.pool_unit_states(hidden_states, surface)
        target_idx = min(self.target_index, token_states.shape[1] - 1)
        return token_states[:, target_idx]

    def factorized_latent(self, context_states: torch.Tensor, surface_states: torch.Tensor) -> torch.Tensor:
        semantic = normalize_semantic_latents(self.semantic_head(context_states))
        surface = self.surface_head(surface_states.detach())
        return compose_factorized_latent(semantic, surface)

    def forward(
        self,
        surface: PackedSurface,
        output_hidden_states: bool = False,
        return_all: bool = False,
    ) -> torch.Tensor:
        if not isinstance(surface, PackedSurface):
            raise TypeError("DilEncoderCore.forward expects PackedSurface")

        hidden_states = self.embed_tokens(surface.ids)
        hidden_states = hidden_states * surface.mask.unsqueeze(-1).to(hidden_states.dtype)
        hidden_states = self.surface_stem(hidden_states, surface)

        layer_vectors = [] if output_hidden_states else None
        for layer_idx in range(self.num_stage_layers):
            hidden_states = self.encoder_layers[layer_idx](hidden_states)
            hidden_states = hidden_states * surface.mask.unsqueeze(-1).to(hidden_states.dtype)
            if output_hidden_states:
                layer_vectors.append(self.pooled_target_vector(hidden_states, surface))

        token_states = self.pool_unit_states(hidden_states, surface)
        surface_states = token_states
        if return_all or token_states.shape[1] != self.context_size:
            hidden_states = self.norm(token_states)
            for block in self.context_blocks:
                hidden_states = block(hidden_states, surface.unit_mask)
            semantic = self.factorized_latent(hidden_states, surface_states)
            if output_hidden_states:
                return semantic, tuple(layer_vectors)
            return semantic

        offsets = torch.arange(self.context_size, device=token_states.device)
        token_states = self.norm(token_states + self.context_offset_embeddings(offsets).unsqueeze(0))
        token_mask = surface.unit_mask
        hidden_states = token_states
        for block in self.context_blocks:
            hidden_states = block(hidden_states, token_mask)
        hidden_states = hidden_states[:, self.target_index]
        surface_states = surface_states[:, self.target_index]

        for layer_idx in range(self.num_stage_layers):
            encoder_idx = self.num_stage_layers + layer_idx
            hidden_states = self.encoder_layers[encoder_idx](hidden_states)
            if output_hidden_states:
                layer_vectors.append(hidden_states)

        hidden_states = self.norm(hidden_states)
        semantic = self.factorized_latent(hidden_states, surface_states)
        if output_hidden_states:
            return semantic, tuple(layer_vectors)
        return semantic
