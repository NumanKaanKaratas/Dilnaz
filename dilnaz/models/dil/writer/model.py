import torch
from torch import nn

from dilnaz.surface import PackedSurface, writer_query_from_lengths

from ..configuration import DilConfig
from ...common.norms import DilRMSNorm
from .blocks import DilCausalConvSwiGLUBlock
from .outputs import DilWriterGeneration, DilWriterOutput


class DilConditionalWriter(nn.Module):
    def __init__(self, config: DilConfig):
        super().__init__()
        self.pad_token_id = config.pad_token_id
        self.start_token_id = config.decoder_start_token_id
        self.stop_token_id = config.writer_stop_token_id
        self.vocab_size = config.writer_vocab_size
        self.max_surface_pieces_per_unit = config.max_surface_pieces_per_unit
        self.surface_bucket_sizes = tuple(config.surface_bucket_sizes)

        self.token_embeddings = nn.Embedding(config.writer_vocab_size, config.hidden_size)
        self.semantic_proj = nn.Linear(config.latent_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_surface_pieces_per_unit + 1, config.hidden_size)
        intermediate_size = config.hidden_size * config.writer_conv_expansion
        self.blocks = nn.ModuleList(
            [
                DilCausalConvSwiGLUBlock(
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

    def forward(self, semantic: torch.Tensor, query_surface: PackedSurface) -> torch.Tensor:
        if semantic.dim() == 2:
            semantic = semantic.unsqueeze(1)
        if semantic.dim() != 3:
            raise ValueError("writer semantic must be shaped [batch, latent] or [batch, units, latent]")
        unit_states = self.semantic_proj(semantic)
        unit_index = query_surface.unit_ids.unsqueeze(-1).expand(-1, -1, unit_states.shape[-1])
        hidden_states = self.token_embeddings(query_surface.ids)
        hidden_states = hidden_states + torch.gather(unit_states, 1, unit_index)
        hidden_states = hidden_states + self.position_embeddings(query_surface.pos_in_unit.clamp_max(self.max_surface_pieces_per_unit))
        hidden_states = self.dropout(hidden_states)
        mask = query_surface.mask
        for block in self.blocks:
            hidden_states = block(hidden_states, query_surface.unit_ids, mask)
        return self.token_head(self.final_norm(hidden_states))

    def transition(self, semantic: torch.Tensor, query_surface: PackedSurface) -> DilWriterOutput:
        return DilWriterOutput(token_logits=self.forward(semantic, query_surface))

    def _unit_query(self, unit_lengths: torch.LongTensor) -> PackedSurface:
        return writer_query_from_lengths(
            unit_lengths,
            pad_token_id=self.pad_token_id,
            surface_bucket_sizes=self.surface_bucket_sizes,
        )

    def _prefix_query(self, ids: torch.LongTensor) -> PackedSurface:
        batch_size, width = ids.shape
        device = ids.device
        mask = torch.ones((batch_size, width), dtype=torch.bool, device=device)
        unit_ids = torch.zeros((batch_size, width), dtype=torch.long, device=device)
        pos_in_unit = torch.arange(width, device=device).view(1, -1).expand(batch_size, -1)
        unit_lengths = torch.full((batch_size, 1), width, dtype=torch.long, device=device)
        unit_offsets = torch.zeros((batch_size, 2), dtype=torch.long, device=device)
        unit_offsets[:, 1] = width
        unit_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
        return PackedSurface(
            ids=ids,
            mask=mask,
            unit_ids=unit_ids,
            pos_in_unit=pos_in_unit,
            unit_lengths=unit_lengths,
            unit_offsets=unit_offsets,
            unit_mask=unit_mask,
        )

    @torch.no_grad()
    def generate(self, semantic: torch.Tensor) -> DilWriterGeneration:
        batch_size = semantic.shape[0]
        device = semantic.device
        prefix = torch.full(
            (batch_size, self.max_surface_pieces_per_unit + 1),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        prefix[:, 0] = self.start_token_id
        token_ids = torch.full(
            (batch_size, self.max_surface_pieces_per_unit),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        mask = torch.zeros_like(token_ids, dtype=torch.bool)
        lengths = torch.zeros((batch_size,), dtype=torch.long, device=device)
        stopped = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        for step_idx in range(self.max_surface_pieces_per_unit):
            query = self._prefix_query(prefix[:, : step_idx + 1])
            logits = self.forward(semantic, query)
            next_ids = logits[:, step_idx].argmax(dim=-1)
            active = ~stopped
            stop_now = active & next_ids.eq(self.stop_token_id)
            emit = active & ~stop_now
            token_ids[emit, step_idx] = next_ids[emit]
            mask[emit, step_idx] = True
            lengths = torch.where(stop_now, torch.full_like(lengths, step_idx), lengths)
            stopped = stopped | stop_now
            if step_idx + 1 <= self.max_surface_pieces_per_unit:
                prefix[emit, step_idx + 1] = next_ids[emit]
            if bool(stopped.all()):
                break
        lengths = torch.where(stopped, lengths, torch.full_like(lengths, self.max_surface_pieces_per_unit))
        return DilWriterGeneration(
            token_ids=token_ids,
            token_mask=mask,
            lengths=lengths,
            stopped=stopped,
        )
