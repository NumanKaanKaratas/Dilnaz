from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from dilnaz.surface import PackedSurface, writer_query_from_lengths

from ..configuration import DilConfig
from ...common.norms import DilRMSNorm
from .blocks import DilCausalAdaLNConvSwiGLUBlock
from .outputs import DilWriterGeneration, DilWriterOutput


class DilConditionalWriter(nn.Module):
    def __init__(self, config: DilConfig, token_embeddings: nn.Embedding):
        super().__init__()
        object.__setattr__(self, "token_embeddings", token_embeddings)
        self.pad_token_id = config.pad_token_id
        self.start_token_id = config.decoder_start_token_id
        self.stop_token_id = config.eos_token_id
        self.vocab_size = config.vocab_size
        self.max_surface_pieces_per_unit = config.max_surface_pieces_per_unit
        self.surface_bucket_sizes = tuple(config.surface_bucket_sizes)
        self.gradient_checkpointing = config.writer_gradient_checkpointing
        self.semantic_proj = nn.Linear(config.latent_size, config.hidden_size)
        self.condition_norm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.condition_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.position_embeddings = nn.Embedding(config.max_surface_pieces_per_unit + 1, config.hidden_size)
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
        self.dropout = nn.Dropout(config.writer_dropout)

    def _canonical_semantic(self, semantic: torch.Tensor) -> torch.Tensor:
        if semantic.dim() != 2:
            raise ValueError("writer semantic must be shaped [batch, latent]")
        if semantic.shape[-1] == 0:
            raise ValueError("writer semantic last dimension must be non-empty")
        return semantic

    def _unit_query(self, unit_lengths: torch.LongTensor) -> PackedSurface:
        return writer_query_from_lengths(
            unit_lengths,
            pad_token_id=self.pad_token_id,
            surface_bucket_sizes=self.surface_bucket_sizes,
        )

    def _bos_query(self, batch_size: int, device: torch.device) -> PackedSurface:
        lengths = torch.ones((batch_size, 1), dtype=torch.long, device=device)
        query = self._unit_query(lengths)
        ids = query.ids.clone()
        ids[query.mask] = self.start_token_id
        return PackedSurface(
            ids=ids,
            mask=query.mask,
            unit_ids=query.unit_ids,
            pos_in_unit=query.pos_in_unit,
            unit_lengths=query.unit_lengths,
            unit_offsets=query.unit_offsets,
            unit_mask=query.unit_mask,
        )

    def _validate_query(self, query: PackedSurface, batch_size: int, device: torch.device) -> PackedSurface:
        query = query.to(device)
        if query.batch_size != batch_size or query.unit_count != 1:
            raise ValueError("writer query must be shaped as one surface unit per semantic row")
        if query.ids.shape != query.mask.shape or query.ids.shape != query.unit_ids.shape or query.ids.shape != query.pos_in_unit.shape:
            raise ValueError("writer query packed tensors must share shape")
        return query

    def transition(self, semantic: torch.Tensor, query_surface: Optional[PackedSurface] = None) -> DilWriterOutput:
        semantic = self._canonical_semantic(semantic)
        batch_size = semantic.shape[0]
        device = semantic.device
        query_surface = self._bos_query(batch_size, device) if query_surface is None else self._validate_query(query_surface, batch_size, device)
        condition = self.condition_proj(self.condition_norm(self.semantic_proj(semantic)))

        hidden_states = self.token_embeddings(query_surface.ids)
        hidden_states = hidden_states + condition.unsqueeze(1)
        hidden_states = hidden_states + self.position_embeddings(query_surface.pos_in_unit.clamp_max(self.max_surface_pieces_per_unit))
        hidden_states = self.dropout(hidden_states)
        mask = query_surface.mask
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and hidden_states.requires_grad:
                hidden_states = checkpoint(block, hidden_states, condition, query_surface.unit_ids, mask, use_reentrant=False)
            else:
                hidden_states = block(hidden_states, condition, query_surface.unit_ids, mask)
        token_logits = F.linear(self.final_norm(hidden_states), self.token_embeddings.weight)
        return DilWriterOutput(token_logits=token_logits, query_surface=query_surface)

    def forward(self, semantic: torch.Tensor, query_surface: Optional[PackedSurface] = None) -> torch.Tensor:
        return self.transition(semantic, query_surface=query_surface).token_logits

    def _dense_query(self, ids: torch.LongTensor, mask: torch.BoolTensor) -> PackedSurface:
        batch_size, width = ids.shape
        device = ids.device
        unit_ids = torch.zeros((batch_size, width), dtype=torch.long, device=device)
        pos_in_unit = torch.arange(width, device=device).view(1, -1).expand(batch_size, -1)
        offsets = torch.zeros((batch_size, 2), dtype=torch.long, device=device)
        offsets[:, 1] = width
        unit_lengths = torch.full((batch_size, 1), width, dtype=torch.long, device=device)
        unit_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
        return PackedSurface(
            ids=ids,
            mask=mask,
            unit_ids=unit_ids,
            pos_in_unit=pos_in_unit,
            unit_lengths=unit_lengths,
            unit_offsets=offsets,
            unit_mask=unit_mask,
        )

    @torch.no_grad()
    def generate(self, semantic: torch.Tensor) -> DilWriterGeneration:
        semantic = self._canonical_semantic(semantic)
        batch_size = semantic.shape[0]
        device = semantic.device
        width = self.max_surface_pieces_per_unit + 1
        input_ids = torch.full((batch_size, width), self.pad_token_id, dtype=torch.long, device=device)
        input_mask = torch.zeros((batch_size, width), dtype=torch.bool, device=device)
        input_ids[:, 0] = self.start_token_id
        input_mask[:, 0] = True
        token_ids = torch.full((batch_size, self.max_surface_pieces_per_unit), self.pad_token_id, dtype=torch.long, device=device)
        token_mask = torch.zeros_like(token_ids, dtype=torch.bool)
        lengths = torch.zeros((batch_size,), dtype=torch.long, device=device)
        stopped = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        finished = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        for step_idx in range(width):
            query = self._dense_query(input_ids, input_mask)
            logits = self.transition(semantic, query_surface=query).token_logits.reshape(batch_size, width, self.vocab_size)
            next_ids = logits[:, step_idx].argmax(dim=-1)
            active = ~finished
            stop_now = active & next_ids.eq(self.stop_token_id)
            stopped = stopped | stop_now
            finished = finished | stop_now
            if step_idx < self.max_surface_pieces_per_unit:
                emit = active & ~stop_now
                token_ids[:, step_idx] = torch.where(emit, next_ids, token_ids[:, step_idx])
                token_mask[:, step_idx] = emit
                lengths = lengths + emit.to(dtype=torch.long)
                input_ids[:, step_idx + 1] = torch.where(emit, next_ids, input_ids[:, step_idx + 1])
                input_mask[:, step_idx + 1] = emit
            if bool(finished.all().detach().cpu()):
                break
        return DilWriterGeneration(
            token_ids=token_ids,
            token_mask=token_mask,
            lengths=lengths,
            stopped=stopped,
        )
