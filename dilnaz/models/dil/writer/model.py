import math
from typing import Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from dilnaz.surface import PackedSurface, gather_unit_values, writer_query_from_lengths

from ..configuration import DilConfig
from ...common.norms import DilRMSNorm
from .blocks import DilCausalAdaLNConvSwiGLUBlock, DilWriterWordMixerBlock
from .outputs import DilWriterGeneration, DilWriterOutput


class DilConditionalWriter(nn.Module):
    ZONE_LEFT = 0
    ZONE_ACTIVE = 1
    ZONE_RIGHT = 2

    def __init__(self, config: DilConfig):
        super().__init__()
        self.pad_token_id = config.pad_token_id
        self.writer_stop_token_id = config.writer_stop_token_id
        self.writer_vocab_size = config.writer_vocab_size
        self.writer_bos_token_id = config.writer_bos_token_id
        self.writer_empty_token_id = config.writer_empty_token_id
        self.writer_input_vocab_size = config.writer_input_vocab_size
        self.max_surface_pieces_per_unit = config.max_surface_pieces_per_unit
        self.surface_bucket_sizes = tuple(config.surface_bucket_sizes)
        self.max_window_size = config.writer_max_window_size
        self.hidden_size = config.hidden_size
        self.max_position_age = config.writer_max_position_age
        self.gradient_checkpointing = config.writer_gradient_checkpointing
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
        self.surface_input_embeddings = nn.Embedding(config.writer_input_vocab_size, config.hidden_size)
        self.zone_embeddings = nn.Embedding(3, config.hidden_size)
        self.position_age_embeddings = nn.Embedding(config.writer_max_position_age + 1, config.hidden_size)
        self.word_position_embeddings = nn.Embedding(config.writer_max_window_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_surface_pieces_per_unit + 1, config.hidden_size)
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
        self.blocks = nn.ModuleList(
            [
                DilCausalAdaLNConvSwiGLUBlock(
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

    def _bos_query(self, window_mask: torch.Tensor) -> PackedSurface:
        lengths = torch.where(window_mask, torch.ones_like(window_mask, dtype=torch.long), torch.zeros_like(window_mask, dtype=torch.long))
        query = writer_query_from_lengths(
            lengths,
            pad_token_id=self.writer_empty_token_id,
            surface_bucket_sizes=self.surface_bucket_sizes,
        )
        ids = query.ids.clone()
        ids[query.mask] = self.writer_bos_token_id
        return PackedSurface(
            ids=ids,
            mask=query.mask,
            unit_ids=query.unit_ids,
            pos_in_unit=query.pos_in_unit,
            unit_lengths=query.unit_lengths,
            unit_offsets=query.unit_offsets,
            unit_mask=query.unit_mask,
        )

    def _validate_query(self, query: PackedSurface, batch_size: int, window_size: int, device: torch.device) -> PackedSurface:
        query = query.to(device)
        if query.ids.shape[0] != batch_size or query.unit_count != window_size:
            raise ValueError("writer query must share semantic batch and window dimensions")
        if int(query.pos_in_unit.max().detach().cpu()) > self.max_surface_pieces_per_unit:
            raise ValueError("writer query position exceeds max_surface_pieces_per_unit")
        return query

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
        zone_ids: torch.Tensor,
        position_age: torch.Tensor,
        window_mask: torch.Tensor,
        future_latents: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, window_size, latent_size = semantic_window.shape
        semantic_hidden = self.semantic_proj(semantic_window.reshape(batch_size * window_size, latent_size)).reshape(
            batch_size,
            window_size,
            -1,
        )
        condition_core = semantic_hidden + self.zone_embeddings(zone_ids) + self.position_age_embeddings(position_age)
        condition = condition_core + self._future_attention_summary(future_latents, condition_core, window_mask, latent_size)
        condition = self.condition_proj(self.condition_norm(condition))
        word_positions = torch.arange(window_size, device=semantic_window.device)
        word_hidden = semantic_hidden + condition + self.word_position_embeddings(word_positions).unsqueeze(0)
        return word_hidden, condition

    def transition(
        self,
        semantic: torch.Tensor,
        query_surface: Optional[PackedSurface] = None,
        zone_ids: Optional[torch.Tensor] = None,
        window_mask: Optional[torch.Tensor] = None,
        future_latents: Optional[torch.Tensor] = None,
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
        query_surface = self._bos_query(window_mask) if query_surface is None else query_surface
        query_surface = self._validate_query(query_surface, batch_size, window_size, device)

        word_hidden, condition = self._writer_condition(
            semantic_window,
            zone_ids,
            position_age,
            window_mask,
            future_latents,
        )
        word_hidden = self.dropout(word_hidden)
        for mixer in self.word_mixers:
            if self.gradient_checkpointing and self.training and word_hidden.requires_grad:
                word_hidden = checkpoint(mixer, word_hidden, condition, window_mask, use_reentrant=False)
            else:
                word_hidden = mixer(word_hidden, condition, window_mask)

        byte_condition = gather_unit_values(condition, query_surface.unit_ids)
        surface_input_ids = torch.where(
            query_surface.mask & query_surface.ids.ge(0) & query_surface.ids.lt(self.writer_input_vocab_size),
            query_surface.ids,
            torch.full_like(query_surface.ids, self.writer_empty_token_id),
        )
        hidden_states = self.surface_input_embeddings(surface_input_ids)
        hidden_states = hidden_states + gather_unit_values(word_hidden, query_surface.unit_ids)
        pos_ids = query_surface.pos_in_unit.clamp_max(self.max_surface_pieces_per_unit)
        hidden_states = hidden_states + self.position_embeddings(pos_ids)
        hidden_states = self.dropout(hidden_states)
        byte_mask = query_surface.mask & window_mask.gather(1, query_surface.unit_ids.clamp_max(window_size - 1))
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and hidden_states.requires_grad:
                hidden_states = checkpoint(block, hidden_states, byte_condition, query_surface.unit_ids, byte_mask, use_reentrant=False)
            else:
                hidden_states = block(hidden_states, byte_condition, query_surface.unit_ids, byte_mask)
        token_logits = self.token_head(self.final_norm(hidden_states))
        return DilWriterOutput(token_logits=token_logits, query_surface=query_surface)

    def forward(
        self,
        semantic: torch.Tensor,
        query_surface: Optional[PackedSurface] = None,
        zone_ids: Optional[torch.Tensor] = None,
        window_mask: Optional[torch.Tensor] = None,
        future_latents: Optional[torch.Tensor] = None,
        position_age: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.transition(
            semantic,
            query_surface=query_surface,
            zone_ids=zone_ids,
            window_mask=window_mask,
            future_latents=future_latents,
            position_age=position_age,
        ).token_logits

    def _dense_query(self, ids: torch.LongTensor, mask: torch.BoolTensor, window_mask: torch.BoolTensor) -> PackedSurface:
        batch_size, unit_count, width_per_unit = ids.shape
        device = ids.device
        flat_ids = ids.reshape(batch_size, unit_count * width_per_unit)
        flat_mask = mask.reshape(batch_size, unit_count * width_per_unit)
        base_positions = torch.arange(unit_count * width_per_unit, device=device)
        unit_ids = (base_positions // width_per_unit).view(1, -1).expand(batch_size, -1)
        pos_in_unit = (base_positions % width_per_unit).view(1, -1).expand(batch_size, -1)
        offsets = torch.arange(unit_count + 1, device=device, dtype=torch.long).view(1, -1).expand(batch_size, -1) * width_per_unit
        unit_lengths = torch.where(
            window_mask,
            torch.full((batch_size, unit_count), width_per_unit, dtype=torch.long, device=device),
            torch.zeros((batch_size, unit_count), dtype=torch.long, device=device),
        )
        return PackedSurface(
            ids=flat_ids,
            mask=flat_mask,
            unit_ids=unit_ids,
            pos_in_unit=pos_in_unit,
            unit_lengths=unit_lengths,
            unit_offsets=offsets,
            unit_mask=window_mask,
        )

    @torch.no_grad()
    def generate(self, semantic: torch.Tensor) -> DilWriterGeneration:
        output = self.generate_window(semantic)
        if output.token_ids.shape[1] == 1:
            return DilWriterGeneration(
                token_ids=output.token_ids[:, 0],
                token_mask=output.token_mask[:, 0],
                lengths=output.lengths[:, 0],
                stopped=output.stopped[:, 0],
            )
        return output

    @torch.no_grad()
    def generate_window(
        self,
        semantic: torch.Tensor,
        zone_ids: Optional[torch.Tensor] = None,
        window_mask: Optional[torch.Tensor] = None,
        future_latents: Optional[torch.Tensor] = None,
        position_age: Optional[torch.Tensor] = None,
    ) -> DilWriterGeneration:
        semantic_window, _ = self._canonical_semantic(semantic)
        batch_size, window_size, _ = semantic_window.shape
        device = semantic_window.device
        window_mask = self._canonical_window_mask(window_mask, batch_size, window_size, device)
        zone_ids = self._canonical_zone_ids(zone_ids, batch_size, window_size, device)
        position_age = self._canonical_position_age(position_age, batch_size, window_size, device)
        width = self.max_surface_pieces_per_unit + 1
        input_ids = torch.full((batch_size, window_size, width), self.writer_empty_token_id, dtype=torch.long, device=device)
        input_mask = torch.zeros((batch_size, window_size, width), dtype=torch.bool, device=device)
        input_ids[:, :, 0] = self.writer_bos_token_id
        input_mask[:, :, 0] = window_mask
        token_ids = torch.full((batch_size, window_size, self.max_surface_pieces_per_unit), self.pad_token_id, dtype=torch.long, device=device)
        token_mask = torch.zeros_like(token_ids, dtype=torch.bool)
        lengths = torch.zeros((batch_size, window_size), dtype=torch.long, device=device)
        stopped = torch.zeros((batch_size, window_size), dtype=torch.bool, device=device)
        finished = ~window_mask
        for step_idx in range(width):
            query = self._dense_query(input_ids, input_mask, window_mask)
            logits = self.transition(
                semantic_window,
                query_surface=query,
                zone_ids=zone_ids,
                window_mask=window_mask,
                future_latents=future_latents,
                position_age=position_age,
            ).token_logits.reshape(batch_size, window_size, width, self.writer_vocab_size)
            next_ids = logits[:, :, step_idx].argmax(dim=-1)
            active = ~finished & window_mask
            stop_now = active & next_ids.eq(self.writer_stop_token_id)
            stopped = stopped | stop_now
            finished = finished | stop_now
            if step_idx < self.max_surface_pieces_per_unit:
                emit = active & ~stop_now
                token_ids[:, :, step_idx] = torch.where(emit, next_ids, token_ids[:, :, step_idx])
                token_mask[:, :, step_idx] = emit
                lengths = lengths + emit.to(dtype=torch.long)
                input_ids[:, :, step_idx + 1] = torch.where(emit, next_ids, input_ids[:, :, step_idx + 1])
                input_mask[:, :, step_idx + 1] = emit
            if bool(finished.all().detach().cpu()):
                break
        return DilWriterGeneration(
            token_ids=token_ids,
            token_mask=token_mask,
            lengths=lengths,
            stopped=stopped,
        )
