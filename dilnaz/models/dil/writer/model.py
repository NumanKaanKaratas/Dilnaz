import math
from typing import Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from dilnaz.surface import (
    PackedSurface,
    PackedSurfaceState,
    empty_surface_state,
    generated_unit_tensors,
    gather_unit_values,
    merge_frozen_state,
    writer_query_from_lengths,
)
from dilnaz.surface.state import STATE_DRAFT, STATE_EMPTY, STATE_KNOWN

from ..configuration import DilConfig
from ...common.norms import DilRMSNorm
from .blocks import DilAdaLNConvSwiGLUBlock, DilByteStateCrossAttention, DilWriterWordMixerBlock
from .outputs import DilWriterOutput


class DilConditionalWriter(nn.Module):
    STATE_EMPTY = STATE_EMPTY
    STATE_DRAFT = STATE_DRAFT
    STATE_KNOWN = STATE_KNOWN
    ZONE_LEFT = 0
    ZONE_ACTIVE = 1
    ZONE_RIGHT = 2

    def __init__(self, config: DilConfig):
        super().__init__()
        self.pad_token_id = config.pad_token_id
        self.writer_stop_token_id = config.writer_stop_token_id
        self.writer_vocab_size = config.writer_vocab_size
        self.writer_state_vocab_size = config.writer_state_vocab_size
        self.writer_empty_token_id = config.writer_empty_token_id
        self.max_surface_pieces_per_unit = config.max_surface_pieces_per_unit
        self.surface_bucket_sizes = tuple(config.surface_bucket_sizes)
        self.max_window_size = config.writer_max_window_size
        self.hidden_size = config.hidden_size
        self.refinement_step_scale = config.initializer_range
        self.writer_refinement_steps = config.writer_refinement_steps
        self.use_step_embedding = config.writer_use_step_embedding
        self.max_position_age = config.writer_max_position_age
        self.gradient_checkpointing = config.writer_gradient_checkpointing
        self.commit_temperature = config.writer_commit_temperature
        self.last_decode_missing_stop_count = 0
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
        self.position_embeddings = nn.Embedding(config.max_surface_pieces_per_unit + 1, config.hidden_size)
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

    def _default_query(self, window_mask: torch.Tensor) -> PackedSurface:
        max_writer_length = self.max_surface_pieces_per_unit + 1
        lengths = torch.full_like(window_mask, max_writer_length, dtype=torch.long)
        lengths = torch.where(window_mask, lengths, torch.zeros_like(lengths))
        return writer_query_from_lengths(
            lengths,
            pad_token_id=self.pad_token_id,
            surface_bucket_sizes=self.surface_bucket_sizes,
        )

    def _validate_surface(self, surface: PackedSurface, batch_size: int, window_size: int, device: torch.device) -> PackedSurface:
        surface = surface.to(device)
        if surface.ids.shape[0] != batch_size or surface.unit_count != window_size:
            raise ValueError("writer packed surface must share semantic batch and window dimensions")
        if int(surface.pos_in_unit.max().detach().cpu()) > self.max_surface_pieces_per_unit:
            raise ValueError("writer packed surface position exceeds max_surface_pieces_per_unit")
        return surface

    def _validate_state(self, state: PackedSurfaceState, batch_size: int, window_size: int, device: torch.device) -> PackedSurfaceState:
        state = state.to(device)
        if state.ids.shape[0] != batch_size or state.unit_count != window_size:
            raise ValueError("writer packed state must share semantic batch and window dimensions")
        return state

    def _missing_stop_mask(self, generated: torch.Tensor, query: PackedSurface) -> torch.BoolTensor:
        stop_hits = generated.eq(self.writer_stop_token_id) & query.mask
        counts = torch.zeros_like(query.unit_lengths, dtype=torch.long)
        counts.scatter_add_(1, query.unit_ids, stop_hits.to(dtype=torch.long))
        return query.unit_mask & counts.eq(0)

    def _state_embeddings(self, state: PackedSurfaceState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_token = state.mask & state.ids.ge(0) & state.ids.lt(self.writer_vocab_size)
        input_ids = torch.where(valid_token, state.ids, torch.full_like(state.ids, self.writer_empty_token_id))
        state_kind = state.state_kind.clamp(0, 2)
        state_kind = torch.where(state.frozen, torch.full_like(state_kind, self.STATE_KNOWN), state_kind)
        state_hidden = (
            self.state_token_embeddings(input_ids)
            + self.state_kind_embeddings(state_kind)
            + self.frozen_embeddings(state.frozen.to(dtype=torch.long))
        )
        state_present = valid_token & state_kind.gt(0)
        batch_size, unit_count = state.unit_lengths.shape
        denom = state.unit_lengths.clamp_min(1).to(state_hidden.dtype)
        index = state.unit_ids.clamp_min(0)
        present_counts = state_hidden.new_zeros((batch_size, unit_count))
        draft_counts = state_hidden.new_zeros((batch_size, unit_count))
        known_counts = state_hidden.new_zeros((batch_size, unit_count))
        frozen_counts = state_hidden.new_zeros((batch_size, unit_count))
        present_counts.scatter_add_(1, index, state_present.to(state_hidden.dtype))
        draft_counts.scatter_add_(1, index, (state_present & state_kind.eq(self.STATE_DRAFT)).to(state_hidden.dtype))
        known_counts.scatter_add_(1, index, (state_present & state_kind.eq(self.STATE_KNOWN)).to(state_hidden.dtype))
        frozen_counts.scatter_add_(1, index, (state_present & state.frozen).to(state_hidden.dtype))
        empty_ratio = (denom - present_counts).clamp_min(0.0) / denom
        state_quality = torch.stack(
            (
                empty_ratio,
                draft_counts / denom,
                known_counts / denom,
                frozen_counts / denom,
            ),
            dim=-1,
        )
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
            ).view(1, 1, -1)
        condition = condition_core + self._future_attention_summary(future_latents, condition_core, window_mask, latent_size)
        condition = self.condition_proj(self.condition_norm(condition))
        word_positions = torch.arange(window_size, device=semantic_window.device)
        word_hidden = semantic_hidden + condition + self.word_position_embeddings(word_positions).unsqueeze(0)
        return word_hidden, condition

    def transition(
        self,
        semantic: torch.Tensor,
        query_surface: Optional[PackedSurface] = None,
        surface_state: Optional[PackedSurfaceState] = None,
        zone_ids: Optional[torch.Tensor] = None,
        window_mask: Optional[torch.Tensor] = None,
        future_latents: Optional[torch.Tensor] = None,
        refinement_step: Optional[int | torch.Tensor] = None,
        position_age: Optional[torch.Tensor] = None,
    ) -> DilWriterOutput:
        semantic_window, _ = self._canonical_semantic(semantic)
        batch_size, window_size, _ = semantic_window.shape
        if window_size > self.max_window_size:
            raise ValueError(f"writer window size {window_size} exceeds writer_max_window_size={self.max_window_size}")
        device = semantic_window.device
        zone_ids = self._canonical_zone_ids(zone_ids, batch_size, window_size, device)
        window_mask = self._canonical_window_mask(window_mask, batch_size, window_size, device)
        position_age = self._canonical_position_age(position_age, batch_size, window_size, device)
        query_surface = self._default_query(window_mask) if query_surface is None else query_surface
        query_surface = self._validate_surface(query_surface, batch_size, window_size, device)
        surface_state = empty_surface_state(query_surface, self.writer_empty_token_id) if surface_state is None else surface_state
        surface_state = self._validate_state(surface_state, batch_size, window_size, device)

        state_hidden, state_present, _, state_quality = self._state_embeddings(surface_state)
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

        byte_condition = gather_unit_values(condition, query_surface.unit_ids)
        hidden_states = gather_unit_values(word_hidden, query_surface.unit_ids)
        pos_ids = query_surface.pos_in_unit.clamp_max(self.max_surface_pieces_per_unit)
        hidden_states = hidden_states + self.position_embeddings(pos_ids)
        hidden_states = self.dropout(hidden_states)
        byte_mask = query_surface.mask & window_mask.gather(1, query_surface.unit_ids.clamp_max(window_size - 1))
        state_mask = state_present & surface_state.mask & window_mask.gather(1, surface_state.unit_ids.clamp_max(window_size - 1))

        if self.gradient_checkpointing and self.training and hidden_states.requires_grad:
            hidden_states = checkpoint(
                self.byte_state_cross_attention,
                hidden_states,
                byte_condition,
                state_hidden,
                state_mask,
                query_surface.unit_ids,
                surface_state.unit_ids,
                byte_mask,
                use_reentrant=False,
            )
        else:
            hidden_states = self.byte_state_cross_attention(
                hidden_states,
                byte_condition,
                state_hidden,
                state_mask,
                query_surface.unit_ids,
                surface_state.unit_ids,
                byte_mask,
            )
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and hidden_states.requires_grad:
                hidden_states = checkpoint(block, hidden_states, byte_condition, query_surface.unit_ids, byte_mask, use_reentrant=False)
            else:
                hidden_states = block(hidden_states, byte_condition, query_surface.unit_ids, byte_mask)
        hidden_states = self.final_norm(hidden_states)
        token_logits = self.token_head(hidden_states)
        state_valid_logits = self.state_valid_head(hidden_states).squeeze(-1)
        emit_logits = self.emit_head(hidden_states).squeeze(-1)
        return DilWriterOutput(
            token_logits=token_logits,
            state_valid_logits=state_valid_logits,
            emit_logits=emit_logits,
            query_surface=query_surface,
        )

    def forward(
        self,
        semantic: torch.Tensor,
        query_surface: Optional[PackedSurface] = None,
        surface_state: Optional[PackedSurfaceState] = None,
        zone_ids: Optional[torch.Tensor] = None,
        window_mask: Optional[torch.Tensor] = None,
        future_latents: Optional[torch.Tensor] = None,
        refinement_step: Optional[int | torch.Tensor] = None,
        position_age: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.transition(
            semantic,
            query_surface=query_surface,
            surface_state=surface_state,
            zone_ids=zone_ids,
            window_mask=window_mask,
            future_latents=future_latents,
            refinement_step=refinement_step,
            position_age=position_age,
        ).token_logits

    @torch.no_grad()
    def generate(self, semantic: torch.Tensor) -> tuple[torch.LongTensor, torch.Tensor, torch.LongTensor]:
        token_ids, token_mask, lengths, _ = self.generate_window(semantic)
        if token_ids.shape[1] == 1:
            return token_ids[:, 0], token_mask[:, 0], lengths[:, 0]
        return token_ids, token_mask, lengths

    @torch.no_grad()
    def generate_window(
        self,
        semantic: torch.Tensor,
        surface_state: Optional[PackedSurfaceState] = None,
        zone_ids: Optional[torch.Tensor] = None,
        window_mask: Optional[torch.Tensor] = None,
        future_latents: Optional[torch.Tensor] = None,
        refinement_steps: Optional[int] = None,
        position_age: Optional[torch.Tensor] = None,
    ) -> tuple[torch.LongTensor, torch.Tensor, torch.LongTensor, torch.Tensor]:
        semantic_window, _ = self._canonical_semantic(semantic)
        batch_size, window_size, _ = semantic_window.shape
        device = semantic_window.device
        window_mask = self._canonical_window_mask(window_mask, batch_size, window_size, device)
        query = self._default_query(window_mask)
        steps = int(self.writer_refinement_steps if refinement_steps is None else refinement_steps)
        if steps <= 0:
            raise ValueError("refinement_steps must be > 0")
        for step_idx in range(steps):
            output = self.transition(
                semantic_window,
                query_surface=query,
                surface_state=surface_state,
                zone_ids=zone_ids,
                window_mask=window_mask,
                future_latents=future_latents,
                refinement_step=step_idx if step_idx > 0 else None,
                position_age=position_age,
            )
            if step_idx + 1 < steps:
                surface_state = merge_frozen_state(
                    empty_surface_state(query, self.writer_empty_token_id),
                    output.token_logits.argmax(dim=-1),
                    query.mask,
                )
        generated = output.token_logits.argmax(dim=-1)
        missing_stop = self._missing_stop_mask(generated, query)
        self.last_decode_missing_stop_count = int(missing_stop.sum().detach().cpu().item())
        token_ids, token_mask, lengths = generated_unit_tensors(
            generated,
            query,
            stop_token_id=self.writer_stop_token_id,
            pad_token_id=self.pad_token_id,
        )
        packed_commit_scores = torch.sigmoid(output.emit_logits.float() / float(self.commit_temperature))
        commit_width = int(query.unit_lengths.max().detach().cpu().item())
        commit_scores = packed_commit_scores.new_zeros((query.batch_size, query.unit_count, max(commit_width, 1)))
        for row_idx in range(query.batch_size):
            for unit_idx in range(query.unit_count):
                width = int(query.unit_lengths[row_idx, unit_idx].detach().cpu())
                if width <= 0:
                    continue
                start = int(query.unit_offsets[row_idx, unit_idx].detach().cpu())
                commit_scores[row_idx, unit_idx, :width] = packed_commit_scores[row_idx, start : start + width]
        return token_ids, token_mask, lengths, commit_scores
