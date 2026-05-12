import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel

from .configuration_dil import DilConfig


@dataclass
class DilOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    semantic: Optional[torch.FloatTensor] = None
    distill_loss: Optional[torch.FloatTensor] = None
    writer_loss: Optional[torch.FloatTensor] = None
    writer_token_loss: Optional[torch.FloatTensor] = None
    writer_commit_loss: Optional[torch.FloatTensor] = None
    mean_geometry_loss: Optional[torch.FloatTensor] = None
    variance_loss: Optional[torch.FloatTensor] = None
    byte_acc: Optional[torch.FloatTensor] = None
    token_exact: Optional[torch.FloatTensor] = None
    stop_acc: Optional[torch.FloatTensor] = None


@dataclass
class DilWriterOutput(ModelOutput):
    token_logits: Optional[torch.FloatTensor] = None
    state_valid_logits: Optional[torch.FloatTensor] = None
    emit_logits: Optional[torch.FloatTensor] = None


def semantic_unit_latents(latents: torch.Tensor) -> torch.Tensor:
    raw = latents.float()
    norm = raw.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(raw)
    fallback[..., 0] = 1.0
    return torch.where(norm > 1e-6, raw / norm.clamp_min(1e-6), fallback)


def normalize_semantic_latents(latents: torch.Tensor) -> torch.Tensor:
    scale = math.sqrt(latents.shape[-1])
    return semantic_unit_latents(latents).to(latents.dtype) * scale


def angular_noise_like(latents: torch.Tensor, min_cos: torch.Tensor, max_cos: torch.Tensor) -> torch.Tensor:
    unit = semantic_unit_latents(latents)
    noise = torch.randn_like(unit)
    noise = noise - (noise * unit).sum(dim=-1, keepdim=True) * unit
    noise = F.normalize(noise, dim=-1, eps=1e-6)
    cos = torch.empty_like(min_cos).uniform_(0.0, 1.0)
    cos = min_cos + cos * (max_cos - min_cos)
    sin = torch.sqrt((1.0 - cos.square()).clamp_min(0.0))
    return (cos.unsqueeze(-1) * unit + sin.unsqueeze(-1) * noise) * math.sqrt(latents.shape[-1])


class DilRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (hidden_states * (1.0 + self.weight.float())).to(input_dtype)


class DilGatedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class DilLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = DilGatedMLP(config)
        self.layernorm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class DilConvSwiGLUBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, kernel_size: int, eps: float, bias: bool, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.norm = DilRMSNorm(hidden_size, eps=eps)
        self.depthwise = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_size,
            bias=bias,
        )
        self.gate_proj = nn.Conv1d(hidden_size, intermediate_size, kernel_size=1, bias=bias)
        self.up_proj = nn.Conv1d(hidden_size, intermediate_size, kernel_size=1, bias=bias)
        self.down_proj = nn.Conv1d(intermediate_size, hidden_size, kernel_size=1, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        if mask is not None:
            hidden_states = hidden_states * mask.unsqueeze(-1).to(hidden_states.dtype)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.depthwise(hidden_states)
        hidden_states = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        hidden_states = self.down_proj(hidden_states).transpose(1, 2)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        if mask is not None:
            hidden_states = hidden_states * mask.unsqueeze(-1).to(hidden_states.dtype)
        return hidden_states


class DilAdaLNModulation(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size * 3)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if condition.dim() == hidden_states.dim() - 1:
            condition = condition.unsqueeze(-2)
        shift, scale, gate = self.proj(condition).chunk(3, dim=-1)
        return hidden_states * (1.0 + scale) + shift, gate


class DilAdaLNConvSwiGLUBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, kernel_size: int, eps: float, bias: bool, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.norm = DilRMSNorm(hidden_size, eps=eps)
        self.adaln = DilAdaLNModulation(hidden_size)
        self.depthwise = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_size,
            bias=bias,
        )
        self.gate_proj = nn.Conv1d(hidden_size, intermediate_size, kernel_size=1, bias=bias)
        self.up_proj = nn.Conv1d(hidden_size, intermediate_size, kernel_size=1, bias=bias)
        self.down_proj = nn.Conv1d(intermediate_size, hidden_size, kernel_size=1, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, condition: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states, residual_gate = self.adaln(self.norm(hidden_states), condition)
        if mask is not None:
            hidden_states = hidden_states * mask.unsqueeze(-1).to(hidden_states.dtype)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.depthwise(hidden_states)
        hidden_states = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        hidden_states = self.down_proj(hidden_states).transpose(1, 2)
        hidden_states = residual + self.dropout(hidden_states) * residual_gate
        if mask is not None:
            hidden_states = hidden_states * mask.unsqueeze(-1).to(hidden_states.dtype)
        return hidden_states


class DilByteStateCrossAttention(nn.Module):
    def __init__(self, hidden_size: int, heads: int, eps: float, dropout: float):
        super().__init__()
        self.query_norm = DilRMSNorm(hidden_size, eps=eps)
        self.state_norm = DilRMSNorm(hidden_size, eps=eps)
        self.adaln = DilAdaLNModulation(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition: torch.Tensor,
        state_hidden: torch.Tensor,
        state_mask: torch.Tensor,
        byte_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        safe_state_mask = state_mask.bool().clone()
        has_state = safe_state_mask.any(dim=1)
        empty_rows = ~has_state
        if empty_rows.any():
            safe_state_mask[empty_rows, 0] = True
        query, residual_gate = self.adaln(self.query_norm(hidden_states), condition)
        key_value = self.state_norm(state_hidden) * safe_state_mask.unsqueeze(-1).to(state_hidden.dtype)
        attn_output, _ = self.attn(
            query,
            key_value,
            key_value,
            key_padding_mask=~safe_state_mask,
            need_weights=False,
        )
        attn_output = attn_output * has_state.view(-1, 1, 1).to(attn_output.dtype)
        if byte_mask is not None:
            attn_output = attn_output * byte_mask.unsqueeze(-1).to(attn_output.dtype)
        hidden_states = hidden_states + self.dropout(attn_output) * residual_gate
        if byte_mask is not None:
            hidden_states = hidden_states * byte_mask.unsqueeze(-1).to(hidden_states.dtype)
        return hidden_states


class DilWriterWordMixerBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int, eps: float, bias: bool, dropout: float):
        super().__init__()
        self.attn_norm = DilRMSNorm(hidden_size, eps=eps)
        self.attn_adaln = DilAdaLNModulation(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = DilRMSNorm(hidden_size, eps=eps)
        self.ffn_adaln = DilAdaLNModulation(hidden_size)
        intermediate_size = hidden_size * 4
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition: torch.Tensor,
        window_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if window_mask is not None:
            safe_mask = window_mask.bool().clone()
            empty_rows = ~safe_mask.any(dim=1)
            if empty_rows.any():
                safe_mask[empty_rows, 0] = True
            hidden_states = hidden_states * safe_mask.unsqueeze(-1).to(hidden_states.dtype)
            key_padding_mask = ~safe_mask
        else:
            safe_mask = None
            key_padding_mask = None

        residual = hidden_states
        attn_input, attn_gate = self.attn_adaln(self.attn_norm(hidden_states), condition)
        attn_output, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden_states = residual + self.attn_dropout(attn_output) * attn_gate

        residual = hidden_states
        hidden_states, ffn_gate = self.ffn_adaln(self.ffn_norm(hidden_states), condition)
        hidden_states = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        hidden_states = residual + self.ffn_dropout(self.down_proj(hidden_states)) * ffn_gate
        if safe_mask is not None:
            hidden_states = hidden_states * safe_mask.unsqueeze(-1).to(hidden_states.dtype)
        return hidden_states


class DilByteConvStem(nn.Module):
    def __init__(self, config: DilConfig):
        super().__init__()
        intermediate_size = config.hidden_size * config.byte_conv_expansion
        self.layers = nn.ModuleList(
            [
                DilConvSwiGLUBlock(
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

    def forward(self, hidden_states: torch.Tensor, word_masks: torch.Tensor) -> torch.Tensor:
        batch_size, context_size, byte_width, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size * context_size, byte_width, hidden_size)
        masks = word_masks.reshape(batch_size * context_size, byte_width)
        for layer in self.layers:
            hidden_states = layer(hidden_states, masks)
        return hidden_states.reshape(batch_size, context_size, byte_width, hidden_size)


def dil_context_attention_heads(hidden_size: int) -> int:
    return next(heads for heads in (8, 4, 2, 1) if hidden_size % heads == 0)


class DilEncoderCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_word_bytes = config.max_word_bytes
        self.context_size = config.context_size
        self.target_index = config.target_index
        self.latent_size = config.latent_size
        self.context_attention_heads = dil_context_attention_heads(config.hidden_size)
        self.context_head_dim = config.hidden_size // self.context_attention_heads

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.byte_stem = DilByteConvStem(config)
        self.encoder_layers = nn.ModuleList([DilLayer(config) for _ in range(config.num_encoder_layers)])
        self.num_stage_layers = config.num_encoder_layers // 2
        self.pool_norm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pool_score = nn.Linear(config.hidden_size, 1, bias=False)
        self.pool_value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.context_offset_embeddings = nn.Embedding(config.context_size, config.hidden_size)
        self.context_norm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.target_norm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.context_q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.context_k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.context_v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.context_out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.context_gate = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.hidden_to_semantic = nn.Linear(config.hidden_size, config.latent_size)
        self.norm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        indices = torch.arange(config.context_size)
        self.register_buffer("context_indices", indices[indices != self.target_index], persistent=False)

    def pooled_target_vector(self, hidden_states: torch.Tensor, word_masks: torch.Tensor) -> torch.Tensor:
        target_states = hidden_states[:, self.target_index]
        target_masks = word_masks[:, self.target_index].unsqueeze(-1).to(target_states.dtype)
        denom = target_masks.sum(dim=1).clamp_min(1.0)
        return (target_states * target_masks).sum(dim=1) / denom

    def pool_token_states(self, hidden_states: torch.Tensor, word_masks: torch.Tensor) -> torch.Tensor:
        scores = self.pool_score(self.pool_norm(hidden_states)).squeeze(-1)
        scores = scores.masked_fill(~word_masks, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        attention = attention * word_masks.to(attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(attention.dtype).tiny)
        values = self.pool_value(hidden_states)
        return (values * attention.unsqueeze(-1)).sum(dim=2)

    def target_conditioned_by_context(self, token_states: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        target_state = token_states[:, self.target_index]
        context_states = token_states.index_select(1, self.context_indices).detach()
        context_mask = token_mask.index_select(1, self.context_indices)
        if context_states.shape[1] == 0:
            return target_state

        batch_size = target_state.shape[0]
        query = self.context_q_proj(self.target_norm(target_state)).reshape(
            batch_size,
            self.context_attention_heads,
            self.context_head_dim,
        )
        keys = self.context_k_proj(self.context_norm(context_states)).reshape(
            batch_size,
            context_states.shape[1],
            self.context_attention_heads,
            self.context_head_dim,
        )
        values = self.context_v_proj(context_states).reshape(
            batch_size,
            context_states.shape[1],
            self.context_attention_heads,
            self.context_head_dim,
        )
        scores = torch.einsum("bhd,bchd->bhc", query, keys) / (self.context_head_dim**0.5)
        context_mask = context_mask.unsqueeze(1)
        scores = scores.masked_fill(~context_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores.float(), dim=-1).to(scores.dtype).masked_fill(~context_mask, 0.0)
        context_delta = torch.einsum("bhc,bchd->bhd", attention, values).reshape(batch_size, -1)
        context_delta = self.context_out_proj(context_delta)
        gate_input = torch.cat(
            [target_state, context_delta, target_state * context_delta, target_state - context_delta],
            dim=-1,
        )
        gate = torch.sigmoid(self.context_gate(gate_input))
        return target_state + gate * context_delta

    def forward(
        self,
        input_ids: torch.LongTensor,
        word_masks: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> torch.Tensor:
        if input_ids.dim() == 2:
            input_ids = input_ids.unsqueeze(1)
            word_masks = word_masks.unsqueeze(1)
        batch_size, context_size, byte_width = input_ids.shape
        if context_size != self.context_size:
            raise ValueError(f"input_ids context width {context_size} != context_size {self.context_size}")
        if byte_width != self.max_word_bytes:
            raise ValueError(f"input_ids width {byte_width} != max_word_bytes {self.max_word_bytes}")

        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states * word_masks.unsqueeze(-1).to(hidden_states.dtype)
        hidden_states = self.byte_stem(hidden_states, word_masks)

        layer_vectors = [] if output_hidden_states else None
        for layer_idx in range(self.num_stage_layers):
            hidden_states = self.encoder_layers[layer_idx](hidden_states)
            if output_hidden_states:
                layer_vectors.append(self.pooled_target_vector(hidden_states, word_masks))

        token_states = self.pool_token_states(hidden_states, word_masks)
        offsets = torch.arange(context_size, device=token_states.device)
        token_states = token_states + self.context_offset_embeddings(offsets).unsqueeze(0)
        token_mask = word_masks.any(dim=-1)
        hidden_states = self.target_conditioned_by_context(token_states, token_mask)

        for layer_idx in range(self.num_stage_layers):
            encoder_idx = self.num_stage_layers + layer_idx
            hidden_states = self.encoder_layers[encoder_idx](hidden_states)
            if output_hidden_states:
                layer_vectors.append(hidden_states)

        hidden_states = self.norm(hidden_states)
        semantic = self.hidden_to_semantic(hidden_states)
        if output_hidden_states:
            return semantic, tuple(layer_vectors)
        return semantic


class DilConditionalWriter(nn.Module):
    STATE_EMPTY = 0
    STATE_DRAFT = 1
    STATE_KNOWN = 2
    ZONE_LEFT = 0
    ZONE_ACTIVE = 1
    ZONE_RIGHT = 2

    def __init__(self, config: DilConfig):
        super().__init__()
        self.max_word_bytes = config.max_word_bytes
        self.writer_max_positions = config.writer_max_positions
        self.pad_token_id = config.pad_token_id
        self.writer_stop_token_id = config.writer_stop_token_id
        self.writer_vocab_size = config.writer_vocab_size
        self.writer_state_vocab_size = config.writer_state_vocab_size
        self.writer_empty_token_id = config.writer_empty_token_id
        self.max_window_size = config.writer_max_window_size
        self.hidden_size = config.hidden_size
        self.refinement_step_scale = config.initializer_range
        self.writer_refinement_steps = config.writer_refinement_steps
        self.use_step_embedding = config.writer_use_step_embedding
        self.max_position_age = config.writer_max_position_age
        self.gradient_checkpointing = config.writer_gradient_checkpointing
        self.commit_temperature = config.writer_commit_temperature
        self.semantic_proj = nn.Linear(config.latent_size, config.hidden_size)
        self.future_latent_proj = nn.Linear(config.latent_size, config.hidden_size, bias=False)
        self.future_query_norm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.future_key_norm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.future_query_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.future_key_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.future_value_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.future_out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.future_gate = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.future_horizon_embeddings = nn.Embedding(config.writer_max_window_size, config.hidden_size)
        self.state_token_embeddings = nn.Embedding(config.writer_state_vocab_size, config.hidden_size)
        self.state_kind_embeddings = nn.Embedding(3, config.hidden_size)
        self.frozen_embeddings = nn.Embedding(2, config.hidden_size)
        self.zone_embeddings = nn.Embedding(3, config.hidden_size)
        self.position_age_embeddings = nn.Embedding(config.writer_max_position_age + 1, config.hidden_size)
        self.word_position_embeddings = nn.Embedding(config.writer_max_window_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.writer_max_positions, config.hidden_size)
        self.state_quality_proj = nn.Linear(4, config.hidden_size, bias=False)
        self.condition_norm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.condition_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.word_mixers = nn.ModuleList(
            [
                DilWriterWordMixerBlock(
                    config.hidden_size,
                    config.writer_word_attention_heads,
                    config.rms_norm_eps,
                    config.mlp_bias,
                    config.writer_dropout,
                )
                for _ in range(config.writer_word_mixer_layers)
            ]
        )
        intermediate_size = config.hidden_size * config.writer_conv_expansion
        self.byte_state_cross_attention = DilByteStateCrossAttention(
            config.hidden_size,
            config.writer_word_attention_heads,
            config.rms_norm_eps,
            config.writer_dropout,
        )
        self.blocks = nn.ModuleList(
            [
                DilAdaLNConvSwiGLUBlock(
                    config.hidden_size,
                    intermediate_size,
                    config.writer_conv_kernel_size,
                    config.rms_norm_eps,
                    config.mlp_bias,
                    config.writer_dropout,
                )
                for _ in range(config.writer_num_layers)
            ]
        )
        self.final_norm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.token_head = nn.Linear(config.hidden_size, config.writer_vocab_size, bias=False)
        self.state_valid_head = nn.Linear(config.hidden_size, 1, bias=True)
        self.emit_head = nn.Linear(config.hidden_size, 1, bias=True)
        self.dropout = nn.Dropout(config.writer_dropout)

    def _canonical_semantic(self, semantic: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if semantic.shape[-1] == 0:
            raise ValueError("semantic last dimension must be non-empty")
        if semantic.dim() == 2:
            return semantic.unsqueeze(1), True
        if semantic.dim() != 3:
            raise ValueError("writer semantic must be shaped [batch, latent] or [batch, window, latent]")
        return semantic, False

    def _canonical_window_arg(
        self,
        value: Optional[torch.Tensor],
        batch_size: int,
        window_size: int,
        dtype: torch.dtype,
        device: torch.device,
        fill_value: int | bool,
    ) -> torch.Tensor:
        if value is None:
            return torch.full((batch_size, window_size, self.writer_max_positions), fill_value, dtype=dtype, device=device)
        if value.dim() == 2:
            value = value.unsqueeze(1)
        if value.shape != (batch_size, window_size, self.writer_max_positions):
            raise ValueError(
                f"writer state argument must be shaped {(batch_size, window_size, self.writer_max_positions)}, "
                f"got {tuple(value.shape)}"
            )
        return value.to(device=device, dtype=dtype)

    def _canonical_zone_ids(
        self,
        zone_ids: Optional[torch.Tensor],
        batch_size: int,
        window_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if zone_ids is None:
            return torch.full((batch_size, window_size), self.ZONE_ACTIVE, dtype=torch.long, device=device)
        if zone_ids.dim() == 1:
            zone_ids = zone_ids.unsqueeze(0).expand(batch_size, -1)
        if zone_ids.shape != (batch_size, window_size):
            raise ValueError(f"zone_ids must be shaped {(batch_size, window_size)}, got {tuple(zone_ids.shape)}")
        return zone_ids.to(device=device, dtype=torch.long)

    def _canonical_window_mask(
        self,
        window_mask: Optional[torch.Tensor],
        batch_size: int,
        window_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if window_mask is None:
            return torch.ones((batch_size, window_size), dtype=torch.bool, device=device)
        if window_mask.dim() == 1:
            window_mask = window_mask.unsqueeze(0).expand(batch_size, -1)
        if window_mask.shape != (batch_size, window_size):
            raise ValueError(f"window_mask must be shaped {(batch_size, window_size)}, got {tuple(window_mask.shape)}")
        return window_mask.to(device=device, dtype=torch.bool)

    def _canonical_position_age(
        self,
        position_age: Optional[torch.Tensor],
        batch_size: int,
        window_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if position_age is None:
            return torch.zeros((batch_size, window_size), dtype=torch.long, device=device)
        if position_age.dim() == 1:
            position_age = position_age.unsqueeze(0).expand(batch_size, -1)
        if position_age.shape != (batch_size, window_size):
            raise ValueError(f"position_age must be shaped {(batch_size, window_size)}, got {tuple(position_age.shape)}")
        return position_age.to(device=device, dtype=torch.long).clamp(0, self.max_position_age)

    def _state_embeddings(
        self,
        surface_state: torch.Tensor,
        surface_state_mask: Optional[torch.Tensor],
        frozen_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_token = surface_state.ge(0) & surface_state.lt(self.writer_vocab_size)
        input_ids = torch.where(valid_token, surface_state, torch.full_like(surface_state, self.writer_empty_token_id))
        if surface_state_mask is None:
            state_kind = torch.where(
                valid_token,
                torch.full_like(surface_state, self.STATE_DRAFT),
                torch.full_like(surface_state, self.STATE_EMPTY),
            )
        else:
            if surface_state_mask.shape != surface_state.shape:
                raise ValueError("surface_state_mask must match surface_state")
            state_kind = surface_state_mask.to(device=surface_state.device, dtype=torch.long).clamp(0, 2)
        state_kind = torch.where(frozen_mask, torch.full_like(state_kind, self.STATE_KNOWN), state_kind)
        state_hidden = (
            self.state_token_embeddings(input_ids)
            + self.state_kind_embeddings(state_kind)
            + self.frozen_embeddings(frozen_mask.to(dtype=torch.long))
        )
        state_present = valid_token & state_kind.gt(0)
        denom = torch.full(
            state_present.shape[:2],
            float(self.writer_max_positions),
            device=state_hidden.device,
            dtype=state_hidden.dtype,
        )
        empty_ratio = (~state_present).sum(dim=2).to(state_hidden.dtype) / denom
        draft_ratio = (state_present & state_kind.eq(self.STATE_DRAFT)).sum(dim=2).to(state_hidden.dtype) / denom
        known_ratio = (state_present & state_kind.eq(self.STATE_KNOWN)).sum(dim=2).to(state_hidden.dtype) / denom
        frozen_ratio = (state_present & frozen_mask).sum(dim=2).to(state_hidden.dtype) / denom
        state_quality = torch.stack((empty_ratio, draft_ratio, known_ratio, frozen_ratio), dim=-1)
        return state_hidden, state_present, state_kind, state_quality

    def _refinement_step_embedding(self, refinement_step: int | torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        step = torch.as_tensor(refinement_step, device=device, dtype=torch.float32).reshape(())
        frequencies = torch.arange(0, self.hidden_size, 2, device=device, dtype=torch.float32)
        angles = step * torch.exp(-math.log(10000.0) * frequencies / max(self.hidden_size, 1))
        embedding = torch.zeros((self.hidden_size,), device=device, dtype=torch.float32)
        embedding[0::2] = torch.sin(angles)
        embedding[1::2] = torch.cos(angles[: embedding[1::2].shape[0]])
        return (embedding * self.refinement_step_scale).to(dtype=dtype)

    def _future_attention_summary(
        self,
        future_latents: Optional[torch.Tensor],
        query_hidden: torch.Tensor,
        window_mask: torch.Tensor,
        latent_size: int,
    ) -> torch.Tensor:
        batch_size, window_size, hidden_size = query_hidden.shape
        if future_latents is None:
            return query_hidden.new_zeros((batch_size, window_size, hidden_size))
        if future_latents.dim() == 3:
            future_latents = future_latents.unsqueeze(2)
        if future_latents.shape[:2] != (batch_size, window_size) or future_latents.shape[-1] != latent_size:
            raise ValueError("future_latents must be shaped [batch, window, horizons, latent]")
        horizons = future_latents.shape[2]
        if horizons == 0:
            return query_hidden.new_zeros((batch_size, window_size, hidden_size))
        if horizons > self.future_horizon_embeddings.num_embeddings:
            raise ValueError("future_latents horizon count exceeds writer_max_window_size")
        horizon_ids = torch.arange(horizons, device=future_latents.device)
        future_hidden = self.future_latent_proj(future_latents) + self.future_horizon_embeddings(horizon_ids).view(
            1,
            1,
            horizons,
            hidden_size,
        )
        future_valid = future_latents.float().norm(dim=-1).gt(1e-6) & window_mask.unsqueeze(-1)
        safe_valid = future_valid.clone()
        has_future = safe_valid.any(dim=2)
        empty_rows = ~has_future
        if empty_rows.any():
            safe_valid[empty_rows, 0] = True
        query = self.future_query_proj(self.future_query_norm(query_hidden)).unsqueeze(2)
        keys = self.future_key_proj(self.future_key_norm(future_hidden))
        values = self.future_value_proj(future_hidden)
        scores = (query * keys).sum(dim=-1) / math.sqrt(hidden_size)
        scores = scores.masked_fill(~safe_valid, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores.float(), dim=-1).to(scores.dtype).masked_fill(~safe_valid, 0.0)
        summary = (attention.unsqueeze(-1) * values).sum(dim=2)
        summary = self.future_out_proj(summary) * has_future.unsqueeze(-1).to(summary.dtype)
        gate = torch.sigmoid(self.future_gate(torch.cat((query_hidden, summary), dim=-1)))
        return gate * summary

    def _writer_condition(
        self,
        semantic_window: torch.Tensor,
        state_quality: torch.Tensor,
        zone_ids: torch.Tensor,
        position_age: torch.Tensor,
        window_mask: torch.Tensor,
        future_latents: Optional[torch.Tensor],
        refinement_step: Optional[int | torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, window_size, latent_size = semantic_window.shape
        semantic_hidden = self.semantic_proj(semantic_window.reshape(batch_size * window_size, latent_size)).reshape(
            batch_size,
            window_size,
            -1,
        )
        condition_core = (
            semantic_hidden
            + self.zone_embeddings(zone_ids)
            + self.position_age_embeddings(position_age)
            + self.state_quality_proj(state_quality)
        )
        if refinement_step is not None and self.use_step_embedding:
            condition_core = condition_core + self._refinement_step_embedding(
                refinement_step,
                semantic_hidden.device,
                semantic_hidden.dtype,
            ).view(
                1,
                1,
                -1,
            )
        condition = condition_core + self._future_attention_summary(future_latents, condition_core, window_mask, latent_size)
        condition = self.condition_proj(self.condition_norm(condition))
        word_positions = torch.arange(window_size, device=semantic_window.device)
        word_hidden = semantic_hidden + self.word_position_embeddings(word_positions).unsqueeze(0)
        return word_hidden, condition

    def transition(
        self,
        semantic: torch.Tensor,
        surface_state: Optional[torch.Tensor] = None,
        surface_state_mask: Optional[torch.Tensor] = None,
        frozen_mask: Optional[torch.Tensor] = None,
        zone_ids: Optional[torch.Tensor] = None,
        window_mask: Optional[torch.Tensor] = None,
        future_latents: Optional[torch.Tensor] = None,
        refinement_step: Optional[int | torch.Tensor] = None,
        position_age: Optional[torch.Tensor] = None,
    ) -> DilWriterOutput:
        semantic_window, squeeze_window = self._canonical_semantic(semantic)
        batch_size, window_size, latent_size = semantic_window.shape
        if window_size > self.max_window_size:
            raise ValueError(f"writer window size {window_size} exceeds writer_max_window_size={self.max_window_size}")
        device = semantic_window.device
        surface_state = self._canonical_window_arg(
            surface_state,
            batch_size,
            window_size,
            torch.long,
            device,
            -100,
        )
        frozen_mask = self._canonical_window_arg(
            frozen_mask,
            batch_size,
            window_size,
            torch.bool,
            device,
            False,
        )
        if surface_state_mask is not None:
            surface_state_mask = self._canonical_window_arg(
                surface_state_mask,
                batch_size,
                window_size,
                torch.long,
                device,
                self.STATE_EMPTY,
            )
        zone_ids = self._canonical_zone_ids(zone_ids, batch_size, window_size, device)
        window_mask = self._canonical_window_mask(window_mask, batch_size, window_size, device)
        position_age = self._canonical_position_age(position_age, batch_size, window_size, device)

        state_hidden, state_present, _, state_quality = self._state_embeddings(surface_state, surface_state_mask, frozen_mask)
        word_hidden, condition = self._writer_condition(
            semantic_window,
            state_quality,
            zone_ids,
            position_age,
            window_mask,
            future_latents,
            refinement_step,
        )

        word_hidden = self.dropout(word_hidden)
        for mixer in self.word_mixers:
            if self.gradient_checkpointing and self.training and word_hidden.requires_grad:
                word_hidden = checkpoint(mixer, word_hidden, condition, window_mask, use_reentrant=False)
            else:
                word_hidden = mixer(word_hidden, condition, window_mask)

        positions = torch.arange(self.writer_max_positions, device=semantic.device)
        hidden_states = word_hidden.unsqueeze(2) + self.position_embeddings(positions).view(1, 1, -1, word_hidden.shape[-1])
        hidden_states = hidden_states + state_hidden
        hidden_states = hidden_states.reshape(batch_size * window_size, self.writer_max_positions, -1)
        state_hidden = state_hidden.reshape(batch_size * window_size, self.writer_max_positions, -1)
        state_present = state_present.reshape(batch_size * window_size, self.writer_max_positions)
        byte_condition = condition.reshape(batch_size * window_size, -1)
        hidden_states = self.dropout(hidden_states)
        byte_mask = window_mask.unsqueeze(-1).expand(-1, -1, self.writer_max_positions).reshape(
            batch_size * window_size,
            self.writer_max_positions,
        )
        if self.gradient_checkpointing and self.training and hidden_states.requires_grad:
            hidden_states = checkpoint(
                self.byte_state_cross_attention,
                hidden_states,
                byte_condition,
                state_hidden,
                state_present,
                byte_mask,
                use_reentrant=False,
            )
        else:
            hidden_states = self.byte_state_cross_attention(hidden_states, byte_condition, state_hidden, state_present, byte_mask)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and hidden_states.requires_grad:
                hidden_states = checkpoint(block, hidden_states, byte_condition, byte_mask, use_reentrant=False)
            else:
                hidden_states = block(hidden_states, byte_condition, byte_mask)
        hidden_states = self.final_norm(hidden_states)
        token_logits = self.token_head(hidden_states).reshape(
            batch_size,
            window_size,
            self.writer_max_positions,
            self.writer_vocab_size,
        )
        state_valid_logits = self.state_valid_head(hidden_states).squeeze(-1).reshape(
            batch_size,
            window_size,
            self.writer_max_positions,
        )
        emit_logits = self.emit_head(hidden_states).squeeze(-1).reshape(
            batch_size,
            window_size,
            self.writer_max_positions,
        )
        if squeeze_window:
            token_logits = token_logits.squeeze(1)
            state_valid_logits = state_valid_logits.squeeze(1)
            emit_logits = emit_logits.squeeze(1)
        return DilWriterOutput(
            token_logits=token_logits,
            state_valid_logits=state_valid_logits,
            emit_logits=emit_logits,
        )

    def forward(
        self,
        semantic: torch.Tensor,
        surface_state: Optional[torch.Tensor] = None,
        surface_state_mask: Optional[torch.Tensor] = None,
        frozen_mask: Optional[torch.Tensor] = None,
        zone_ids: Optional[torch.Tensor] = None,
        window_mask: Optional[torch.Tensor] = None,
        future_latents: Optional[torch.Tensor] = None,
        refinement_step: Optional[int | torch.Tensor] = None,
        position_age: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.transition(
            semantic,
            surface_state=surface_state,
            surface_state_mask=surface_state_mask,
            frozen_mask=frozen_mask,
            zone_ids=zone_ids,
            window_mask=window_mask,
            future_latents=future_latents,
            refinement_step=refinement_step,
            position_age=position_age,
        ).token_logits

    @torch.no_grad()
    def _decode_generated(self, generated: torch.Tensor) -> tuple[torch.LongTensor, torch.Tensor, torch.LongTensor]:
        stop_hits = generated.eq(self.writer_stop_token_id)
        has_stop = stop_hits.any(dim=-1)
        first_stop = stop_hits.float().argmax(dim=-1).long()
        fallback = torch.full_like(first_stop, self.max_word_bytes)
        lengths = torch.where(has_stop, first_stop, fallback).clamp_max(self.max_word_bytes)
        generated = generated[..., : self.max_word_bytes]
        mask = torch.arange(self.max_word_bytes, device=generated.device).view(
            *([1] * (generated.dim() - 1)),
            self.max_word_bytes,
        ) < lengths.unsqueeze(-1)
        return generated.masked_fill(~mask, self.pad_token_id), mask, lengths

    @torch.no_grad()
    def generate(self, semantic: torch.Tensor) -> tuple[torch.LongTensor, torch.Tensor, torch.LongTensor]:
        token_logits = self.forward(semantic)
        generated = token_logits.argmax(dim=-1)
        return self._decode_generated(generated)

    @torch.no_grad()
    def generate_window(
        self,
        semantic: torch.Tensor,
        surface_state: Optional[torch.Tensor] = None,
        surface_state_mask: Optional[torch.Tensor] = None,
        frozen_mask: Optional[torch.Tensor] = None,
        zone_ids: Optional[torch.Tensor] = None,
        window_mask: Optional[torch.Tensor] = None,
        future_latents: Optional[torch.Tensor] = None,
        refinement_steps: Optional[int] = None,
        position_age: Optional[torch.Tensor] = None,
    ) -> tuple[torch.LongTensor, torch.Tensor, torch.LongTensor, torch.Tensor]:
        output = self.transition(
            semantic,
            surface_state=surface_state,
            surface_state_mask=surface_state_mask,
            frozen_mask=frozen_mask,
            zone_ids=zone_ids,
            window_mask=window_mask,
            future_latents=future_latents,
            position_age=position_age,
        )
        steps = int(self.writer_refinement_steps if refinement_steps is None else refinement_steps)
        if steps <= 0:
            raise ValueError("refinement_steps must be > 0")
        surface_state_base = surface_state
        surface_state_mask_base = surface_state_mask
        frozen_mask_base = frozen_mask
        for step_idx in range(1, steps):
            generated = output.token_logits.argmax(dim=-1)
            next_state = generated
            next_mask = torch.full_like(generated, self.STATE_DRAFT)
            if surface_state_base is not None:
                base_state = self._canonical_window_arg(
                    surface_state_base,
                    generated.shape[0],
                    1 if generated.dim() == 2 else generated.shape[1],
                    torch.long,
                    generated.device,
                    -100,
                )
                if frozen_mask_base is None:
                    frozen = torch.zeros_like(generated, dtype=torch.bool)
                else:
                    frozen = self._canonical_window_arg(
                        frozen_mask_base,
                        generated.shape[0],
                        1 if generated.dim() == 2 else generated.shape[1],
                        torch.bool,
                        generated.device,
                        False,
                    )
                next_state = torch.where(frozen, base_state, next_state)
                if surface_state_mask_base is not None:
                    base_mask = self._canonical_window_arg(
                        surface_state_mask_base,
                        generated.shape[0],
                        1 if generated.dim() == 2 else generated.shape[1],
                        torch.long,
                        generated.device,
                        self.STATE_EMPTY,
                    )
                    next_mask = torch.where(frozen, base_mask, next_mask)
                else:
                    next_mask = torch.where(
                        frozen,
                        torch.full_like(next_mask, self.STATE_KNOWN),
                        next_mask,
                    )
            output = self.transition(
                semantic,
                surface_state=next_state,
                surface_state_mask=next_mask,
                frozen_mask=frozen_mask,
                zone_ids=zone_ids,
                window_mask=window_mask,
                future_latents=future_latents,
                refinement_step=step_idx,
                position_age=position_age,
            )
        generated = output.token_logits.argmax(dim=-1)
        token_ids, token_mask, lengths = self._decode_generated(generated)
        commit_scores = torch.sigmoid(output.emit_logits.float() / float(self.commit_temperature))
        return token_ids, token_mask, lengths, commit_scores


class Dil(PreTrainedModel):
    config_class = DilConfig

    def __init__(self, config):
        super().__init__(config)
        if config.checkpoint_format_version != 24:
            raise ValueError("DIL block diffusion writer checkpoints require checkpoint_format_version=24")
        if config.pad_token_id >= config.vocab_size:
            raise ValueError("pad_token_id must be inside the tokenizer vocabulary")
        if config.eos_token_id >= config.vocab_size:
            raise ValueError("eos_token_id must be inside the tokenizer vocabulary")
        if config.decoder_start_token_id >= config.vocab_size:
            raise ValueError("decoder_start_token_id must be inside the tokenizer vocabulary")
        if config.writer_stop_token_id != config.vocab_size or config.writer_vocab_size != config.vocab_size + 1:
            raise ValueError("Writer stop token contract must be writer_stop_token_id=vocab_size")
        if config.writer_max_positions != config.max_word_bytes + 1:
            raise ValueError("Writer max positions must be max_word_bytes + 1")

        self.encoder = DilEncoderCore(config)
        self.writer = DilConditionalWriter(config)
        self.dil_dropout = config.dil_dropout
        self.distillation_weight = config.distillation_weight
        self.mean_geometry_weight = config.mean_geometry_weight
        self.variance_weight = config.variance_weight
        self.writer_loss_weight = config.writer_loss_weight

        self.post_init()

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv1d):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, value):
        self.encoder.embed_tokens = value

    def encode(self, input_ids: torch.LongTensor, word_masks: torch.Tensor, output_hidden_states: bool = False):
        compiled_forward = getattr(self, "_compiled_encoder_forward", None)
        if compiled_forward is not None:
            encoded = compiled_forward(input_ids, word_masks, output_hidden_states)
        else:
            encoded = self.encoder(input_ids=input_ids, word_masks=word_masks, output_hidden_states=output_hidden_states)
        if output_hidden_states:
            semantic, layer_vectors = encoded
            return normalize_semantic_latents(semantic), layer_vectors
        return normalize_semantic_latents(encoded)

    def writer_outputs(self, semantic: torch.Tensor, **writer_kwargs) -> torch.Tensor:
        compiled_forward = getattr(self, "_compiled_writer_forward", None)
        if compiled_forward is not None and not writer_kwargs:
            return compiled_forward(semantic)
        return self.writer(semantic, **writer_kwargs)

    def _transition_output(self, semantic: torch.Tensor, **writer_kwargs) -> DilWriterOutput:
        compiled_forward = getattr(self, "_compiled_transition_forward", None)
        if compiled_forward is not None:
            output = compiled_forward(semantic, **writer_kwargs)
        else:
            output = self.writer.transition(semantic, **writer_kwargs)
        if isinstance(output, DilWriterOutput):
            return output
        token_logits, state_valid_logits, emit_logits = output
        return DilWriterOutput(
            token_logits=token_logits,
            state_valid_logits=state_valid_logits,
            emit_logits=emit_logits,
        )

    def _refined_transition_outputs(
        self,
        semantic: torch.Tensor,
        refinement_steps: int,
        return_step0: bool = False,
        **writer_kwargs,
    ):
        if refinement_steps <= 0:
            raise ValueError("refinement_steps must be > 0")
        output = self._transition_output(semantic, **writer_kwargs)
        step0_output = output
        if refinement_steps == 1:
            return (output, step0_output) if return_step0 else output
        surface_state = writer_kwargs.get("surface_state")
        surface_state_mask = writer_kwargs.get("surface_state_mask")
        frozen_mask = writer_kwargs.get("frozen_mask")
        for step_idx in range(1, refinement_steps):
            generated = output.token_logits.argmax(dim=-1)
            next_state = generated
            next_mask = torch.full_like(generated, self.writer.STATE_DRAFT)
            if surface_state is not None:
                base_state = self.writer._canonical_window_arg(
                    surface_state,
                    generated.shape[0],
                    1 if generated.dim() == 2 else generated.shape[1],
                    torch.long,
                    generated.device,
                    -100,
                )
                if frozen_mask is None:
                    frozen = torch.zeros_like(generated, dtype=torch.bool)
                else:
                    frozen = self.writer._canonical_window_arg(
                        frozen_mask,
                        generated.shape[0],
                        1 if generated.dim() == 2 else generated.shape[1],
                        torch.bool,
                        generated.device,
                        False,
                    )
                next_state = torch.where(frozen, base_state, next_state)
                if surface_state_mask is not None:
                    base_mask = self.writer._canonical_window_arg(
                        surface_state_mask,
                        generated.shape[0],
                        1 if generated.dim() == 2 else generated.shape[1],
                        torch.long,
                        generated.device,
                        self.writer.STATE_EMPTY,
                    )
                    next_mask = torch.where(frozen, base_mask, next_mask)
                else:
                    next_mask = torch.where(
                        frozen,
                        torch.full_like(next_mask, self.writer.STATE_KNOWN),
                        next_mask,
                    )
            writer_kwargs = {
                **writer_kwargs,
                "surface_state": next_state,
                "surface_state_mask": next_mask,
                "refinement_step": step_idx,
            }
            output = self._transition_output(semantic, **writer_kwargs)
        return (output, step0_output) if return_step0 else output

    def writer_transition_outputs(self, semantic: torch.Tensor, **writer_kwargs) -> DilWriterOutput:
        return_step0 = bool(writer_kwargs.pop("return_step0", False))
        refinement_steps = int(writer_kwargs.pop("refinement_steps", self.config.writer_refinement_steps))
        return self._refined_transition_outputs(semantic, refinement_steps, return_step0=return_step0, **writer_kwargs)

    def set_compiled_forwards(self, encoder_forward=None, writer_forward=None, transition_forward=None):
        object.__setattr__(self, "_compiled_encoder_forward", encoder_forward)
        object.__setattr__(self, "_compiled_writer_forward", writer_forward)
        object.__setattr__(self, "_compiled_transition_forward", transition_forward)

    def geometry_loss(
        self,
        model_vectors: torch.Tensor,
        teacher_vectors: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            model_vectors = model_vectors[mask]
            teacher_vectors = teacher_vectors[mask]
        if model_vectors.shape[0] < 2:
            return model_vectors.new_zeros(())

        model_sim = F.normalize(model_vectors, dim=-1) @ F.normalize(model_vectors, dim=-1).T
        teacher_sim = F.normalize(teacher_vectors, dim=-1) @ F.normalize(teacher_vectors, dim=-1).T
        off_diagonal = ~torch.eye(model_sim.shape[0], dtype=torch.bool, device=model_sim.device)
        return F.mse_loss(model_sim[off_diagonal], teacher_sim[off_diagonal])

    def variance_regularizer(self, model_vectors: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            model_vectors = model_vectors[mask]
        if model_vectors.shape[0] < 2:
            return model_vectors.new_zeros(())
        std = torch.sqrt(model_vectors.float().var(dim=0, unbiased=False) + 1e-4)
        return F.relu(1.0 - std).mean()

    def writer_metrics(self, logits: torch.Tensor, labels: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = labels.ne(-100)
        predictions = logits.argmax(dim=-1)
        byte_valid = valid & labels.ne(self.config.writer_stop_token_id)
        stop_valid = labels.eq(self.config.writer_stop_token_id)
        byte_acc = (predictions.eq(labels) & byte_valid).sum().float() / byte_valid.sum().clamp_min(1).float()
        stop_acc = (predictions.eq(labels) & stop_valid).sum().float() / stop_valid.sum().clamp_min(1).float()
        row_valid = valid.any(dim=-1)
        exact = ((predictions.eq(labels) | ~valid).all(dim=-1) & row_valid).sum().float()
        token_exact = exact / row_valid.sum().clamp_min(1).float()
        return byte_acc, token_exact, stop_acc

    def writer_training_semantic(
        self,
        semantic: torch.Tensor,
        training_step: int | None,
        zone_ids: Optional[torch.Tensor] = None,
        window_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.training:
            return semantic
        if training_step is None or training_step <= self.config.writer_noise_warmup_steps:
            return semantic

        prefix_shape = semantic.shape[:-1]
        if zone_ids is not None and zone_ids.shape != prefix_shape:
            raise ValueError("zone_ids must match semantic prefix shape for writer noise")
        if window_mask is not None and window_mask.shape != prefix_shape:
            raise ValueError("window_mask must match semantic prefix shape for writer noise")
        ratio = semantic.new_tensor(
            [
                self.config.writer_noise_clean_ratio,
                self.config.writer_noise_easy_ratio,
                self.config.writer_noise_mid_ratio,
                self.config.writer_noise_hard_ratio,
            ],
            dtype=torch.float32,
        )
        cumulative = (ratio / ratio.sum()).cumsum(dim=0)
        draw = torch.rand(prefix_shape, device=semantic.device)
        bucket = torch.zeros(prefix_shape, device=semantic.device, dtype=torch.long)
        bucket = bucket.masked_fill(draw.ge(cumulative[0]) & draw.lt(cumulative[1]), 1)
        bucket = bucket.masked_fill(draw.ge(cumulative[1]) & draw.lt(cumulative[2]), 2)
        bucket = bucket.masked_fill(draw.ge(cumulative[2]), 3)
        if zone_ids is not None and self.config.writer_use_zone_noise:
            zone_ids = zone_ids.to(device=semantic.device, dtype=torch.long)
            left = zone_ids.eq(DilConditionalWriter.ZONE_LEFT)
            right = zone_ids.eq(DilConditionalWriter.ZONE_RIGHT)
            bucket = torch.where(left, (bucket - 1).clamp_min(0), bucket)
            bucket = torch.where(right, (bucket + 1).clamp_max(3), bucket)
        easy = bucket.eq(1)
        mid = bucket.eq(2)
        hard = bucket.eq(3)
        if window_mask is not None:
            valid_noise = window_mask.to(device=semantic.device, dtype=torch.bool)
            easy = easy & valid_noise
            mid = mid & valid_noise
            hard = hard & valid_noise
        noised = semantic.float().clone()
        ranges = (
            (easy, self.config.writer_noise_easy_min_cos, self.config.writer_noise_easy_max_cos),
            (mid, self.config.writer_noise_mid_min_cos, self.config.writer_noise_mid_max_cos),
            (hard, self.config.writer_noise_hard_min_cos, self.config.writer_noise_hard_max_cos),
        )
        for mask, min_cos, max_cos in ranges:
            if mask.any():
                min_tensor = torch.full(mask.shape, min_cos, device=semantic.device, dtype=torch.float32)[mask]
                max_tensor = torch.full(mask.shape, max_cos, device=semantic.device, dtype=torch.float32)[mask]
                noised[mask] = angular_noise_like(noised[mask], min_tensor, max_tensor)
        return noised.to(semantic.dtype)

    def writer_loss_and_metrics(
        self,
        semantic: torch.Tensor,
        labels: torch.LongTensor,
        training_step: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        semantic = self.writer_training_semantic(semantic, training_step)
        token_logits = self.writer_outputs(semantic)
        token_loss = F.cross_entropy(
            token_logits.reshape(-1, self.config.writer_vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
        )
        byte_acc, token_exact, stop_acc = self.writer_metrics(token_logits, labels)
        return token_loss, token_loss, byte_acc, token_exact, stop_acc

    def writer_transition_loss_and_metrics(
        self,
        semantic: torch.Tensor,
        labels: torch.LongTensor,
        surface_state: torch.LongTensor,
        surface_state_mask: Optional[torch.Tensor],
        frozen_mask: torch.Tensor,
        zone_ids: torch.LongTensor,
        window_mask: torch.Tensor,
        future_latents: Optional[torch.Tensor] = None,
        position_age: Optional[torch.Tensor] = None,
        training_refinement_step: Optional[int] = None,
        training_step: int | None = None,
        return_metrics: bool = False,
    ):
        labels = labels.to(semantic.device)
        surface_state = surface_state.to(semantic.device)
        frozen_mask = frozen_mask.to(semantic.device, dtype=torch.bool)
        zone_ids = zone_ids.to(semantic.device, dtype=torch.long)
        window_mask = window_mask.to(semantic.device, dtype=torch.bool)
        if surface_state_mask is not None:
            surface_state_mask = surface_state_mask.to(semantic.device, dtype=torch.long)
        if future_latents is not None:
            future_latents = future_latents.to(semantic.device)
        if position_age is not None:
            position_age = position_age.to(semantic.device, dtype=torch.long)
        semantic = self.writer_training_semantic(
            semantic,
            training_step,
            zone_ids=zone_ids,
            window_mask=window_mask,
        )
        output = self.writer_transition_outputs(
            semantic,
            surface_state=surface_state,
            surface_state_mask=surface_state_mask,
            frozen_mask=frozen_mask,
            zone_ids=zone_ids,
            window_mask=window_mask,
            future_latents=future_latents,
            position_age=position_age,
            refinement_step=training_refinement_step,
            refinement_steps=1 if training_refinement_step is not None else self.config.writer_refinement_steps,
            return_step0=return_metrics,
        )
        if return_metrics:
            output, step0_output = output
        else:
            step0_output = None
        logits = output.token_logits
        state_valid_logits = output.state_valid_logits
        emit_logits = output.emit_logits
        valid = labels.ne(-100) & window_mask.unsqueeze(-1)
        left = zone_ids.eq(DilConditionalWriter.ZONE_LEFT).unsqueeze(-1) & valid
        active = zone_ids.eq(DilConditionalWriter.ZONE_ACTIVE).unsqueeze(-1) & valid
        right = zone_ids.eq(DilConditionalWriter.ZONE_RIGHT).unsqueeze(-1) & valid
        token_weights = active.to(logits.dtype) + right.to(logits.dtype) * self.config.writer_right_guard_loss_weight
        per_token_loss = F.cross_entropy(
            logits.reshape(-1, self.config.writer_vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(labels)
        token_loss = (per_token_loss * token_weights).sum() / token_weights.sum().clamp_min(1.0)
        active_weights = active.to(logits.dtype)
        right_weights = right.to(logits.dtype)
        active_token_loss = (per_token_loss * active_weights).sum() / active_weights.sum().clamp_min(1.0)
        right_guard_token_loss = (per_token_loss * right_weights).sum() / right_weights.sum().clamp_min(1.0)
        left_weights = left.to(logits.dtype)
        left_loss = (per_token_loss * left_weights).sum() / left_weights.sum().clamp_min(1.0)

        state_matches = surface_state.eq(labels) & valid
        state_present = surface_state_mask.gt(0) if surface_state_mask is not None else surface_state.ge(0)
        state_valid_scope = (left | active) & valid & state_present
        state_valid_target = state_matches & state_valid_scope
        state_valid_weights = state_valid_scope.to(logits.dtype)
        state_valid_per_pos = F.binary_cross_entropy_with_logits(
            state_valid_logits.float(),
            state_valid_target.to(dtype=torch.float32),
            reduction="none",
        ).to(logits.dtype)
        state_valid_loss = (state_valid_per_pos * state_valid_weights).sum() / state_valid_weights.sum().clamp_min(1.0)

        predictions = logits.argmax(dim=-1)
        output_matches = predictions.eq(labels) & valid
        emit_scope = (left | active) & valid
        emit_target = output_matches & emit_scope
        emit_weights = emit_scope.to(logits.dtype)
        emit_per_pos = F.binary_cross_entropy_with_logits(
            emit_logits.float(),
            emit_target.to(dtype=torch.float32),
            reduction="none",
        ).to(logits.dtype)
        emit_loss = (emit_per_pos * emit_weights).sum() / emit_weights.sum().clamp_min(1.0)
        commit_loss = state_valid_loss + emit_loss
        loss = (
            token_loss
            + left_loss * self.config.writer_left_consistency_weight
            + commit_loss * self.config.writer_commit_loss_weight
        )
        metric_labels = labels.masked_fill(~active, -100)
        byte_acc, token_exact, stop_acc = self.writer_metrics(logits, metric_labels)
        if return_metrics:
            right_metric_labels = labels.masked_fill(~right, -100)
            right_byte_acc, right_token_exact, right_stop_acc = self.writer_metrics(logits, right_metric_labels)
            commit_scope = emit_scope
            commit_scores = torch.sigmoid(emit_logits.float() / float(self.config.writer_commit_temperature))
            commit_pred = commit_scores.ge(float(self.config.writer_commit_threshold)) & commit_scope
            commit_positive = emit_target & commit_scope
            true_positive = (commit_pred & commit_positive).sum().float()
            predicted_positive = commit_pred.sum().float()
            actual_positive = commit_positive.sum().float()
            commit_precision = true_positive / predicted_positive.clamp_min(1.0)
            commit_recall = true_positive / actual_positive.clamp_min(1.0)
            commit_f1 = 2.0 * commit_precision * commit_recall / (commit_precision + commit_recall).clamp_min(1e-6)
            false_commit_rate = (commit_pred & ~commit_positive).sum().float() / predicted_positive.clamp_min(1.0)
            mean_commit_score = (
                (commit_scores * commit_scope.to(commit_scores.dtype)).sum()
                / commit_scope.sum().clamp_min(1).to(commit_scores.dtype)
            )
            step0_labels = labels.masked_fill(~active, -100)
            step0_byte_acc, step0_token_exact, step0_stop_acc = self.writer_metrics(step0_output.token_logits, step0_labels)
            return {
                "loss": loss,
                "token_loss": token_loss,
                "active_token_loss": active_token_loss,
                "right_guard_token_loss": right_guard_token_loss,
                "left_consistency_loss": left_loss,
                "commit_loss": commit_loss,
                "state_valid_loss": state_valid_loss,
                "emit_loss": emit_loss,
                "byte_acc": byte_acc,
                "token_exact": token_exact,
                "stop_acc": stop_acc,
                "right_guard_byte_acc": right_byte_acc,
                "right_guard_token_exact": right_token_exact,
                "right_guard_stop_acc": right_stop_acc,
                "commit_precision": commit_precision,
                "commit_recall": commit_recall,
                "commit_f1": commit_f1,
                "false_commit_rate": false_commit_rate,
                "mean_commit_score": mean_commit_score,
                "step0_byte_acc": step0_byte_acc,
                "step0_token_exact": step0_token_exact,
                "step0_stop_acc": step0_stop_acc,
                "stepT_byte_acc": byte_acc,
                "stepT_token_exact": token_exact,
                "stepT_stop_acc": stop_acc,
                "emit_calibration_logits": emit_logits.detach().float()[commit_scope],
                "emit_calibration_targets": commit_positive.detach()[commit_scope].float(),
            }
        return loss, token_loss, commit_loss, byte_acc, token_exact, stop_acc

    def forward(
        self,
        input_ids: torch.LongTensor,
        word_masks: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        teacher_layers: Optional[torch.Tensor] = None,
        teacher_mask: Optional[torch.Tensor] = None,
        training_step: Optional[int] = None,
    ) -> DilOutput:
        encoder_masks = word_masks
        if self.training and self.dil_dropout > 0:
            keep = torch.rand_like(word_masks.float()) >= self.dil_dropout
            encoder_masks = word_masks * keep.to(word_masks.dtype)

        semantic = self.encode(input_ids=input_ids, word_masks=encoder_masks)
        semantic = semantic.float()
        loss = semantic.new_zeros(())
        distill_loss = semantic.new_zeros(())
        mean_geometry_loss = semantic.new_zeros(())
        variance_loss = semantic.new_zeros(())

        if teacher_layers is not None:
            teacher_layers = teacher_layers.to(semantic.device, dtype=torch.float32)
            if teacher_mask is None:
                teacher_mask = torch.ones(teacher_layers.shape[0], dtype=torch.bool, device=semantic.device)
            else:
                teacher_mask = teacher_mask.to(semantic.device, dtype=torch.bool)
            mean_geometry_loss = self.geometry_loss(semantic, teacher_layers[:, -1], teacher_mask)
            variance_loss = self.variance_regularizer(semantic, teacher_mask)
            distill_loss = (
                mean_geometry_loss * self.mean_geometry_weight
                + variance_loss * self.variance_weight
            )
            loss = loss + distill_loss * self.distillation_weight

        writer_loss = semantic.new_zeros(())
        byte_acc = semantic.new_zeros(())
        token_exact = semantic.new_zeros(())
        writer_token_loss = semantic.new_zeros(())
        writer_commit_loss = semantic.new_zeros(())
        stop_acc = semantic.new_zeros(())
        if labels is not None and self.writer_loss_weight > 0.0:
            writer_semantic = semantic.detach()
            labels = labels.to(semantic.device)
            writer_loss, writer_token_loss, byte_acc, token_exact, stop_acc = self.writer_loss_and_metrics(
                writer_semantic,
                labels,
                training_step,
            )
            loss = loss + writer_loss * self.writer_loss_weight

        return DilOutput(
            loss=loss,
            semantic=semantic,
            distill_loss=distill_loss,
            writer_loss=writer_loss,
            writer_token_loss=writer_token_loss,
            writer_commit_loss=writer_commit_loss,
            mean_geometry_loss=mean_geometry_loss,
            variance_loss=variance_loss,
            byte_acc=byte_acc,
            token_exact=token_exact,
            stop_acc=stop_acc,
        )

    @torch.no_grad()
    def decode_semantic(self, semantic: torch.Tensor) -> tuple[torch.LongTensor, torch.Tensor, torch.LongTensor]:
        return self.writer.generate(semantic)

    @torch.no_grad()
    def decode_semantic_window(self, semantic: torch.Tensor, **writer_kwargs):
        output = self.writer_transition_outputs(semantic, **writer_kwargs)
        generated = output.token_logits.argmax(dim=-1)
        token_ids, token_mask, lengths = self.writer._decode_generated(generated)
        commit_scores = torch.sigmoid(output.emit_logits.float() / float(self.config.writer_commit_temperature))
        return token_ids, token_mask, lengths, commit_scores
