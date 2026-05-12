import math
from typing import Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..configuration import DilConfig
from ...common.norms import DilRMSNorm
from .blocks import DilAdaLNConvSwiGLUBlock, DilByteStateCrossAttention, DilWriterWordMixerBlock
from .outputs import DilWriterOutput


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
