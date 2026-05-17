import torch
import torch.nn.functional as F
from torch import nn

from dilnaz.surface import PackedSurface, writer_query_from_lengths

from ...common.latents import split_factorized_latent
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
        self.encoder_vocab_size = config.vocab_size
        self.semantic_latent_size = config.semantic_latent_size
        self.surface_latent_size = config.surface_latent_size
        self.max_surface_pieces_per_unit = config.max_surface_pieces_per_unit
        self.surface_bucket_sizes = tuple(config.surface_bucket_sizes)

        self.token_embeddings = nn.Embedding(config.writer_vocab_size, config.hidden_size)
        self.encoder_prior_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.encoder_prior_gate = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.semantic_proj = nn.Linear(config.semantic_latent_size, config.hidden_size)
        self.surface_proj = nn.Linear(config.surface_latent_size, config.hidden_size)
        self.surface_gate = nn.Linear(config.semantic_latent_size, config.hidden_size)
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

    def token_condition(
        self,
        token_ids: torch.LongTensor,
        encoder_embedding_weight: torch.Tensor,
    ) -> torch.Tensor:
        writer_embed = self.token_embeddings(token_ids)
        encoder_ids = token_ids.clamp_max(self.encoder_vocab_size - 1)
        encoder_prior = F.embedding(encoder_ids, encoder_embedding_weight.detach()).to(writer_embed.dtype)
        prior_mask = token_ids.lt(self.encoder_vocab_size).unsqueeze(-1)
        encoder_prior = encoder_prior * prior_mask.to(encoder_prior.dtype)
        prior_delta = self.encoder_prior_proj(encoder_prior)
        gate = torch.sigmoid(self.encoder_prior_gate(torch.cat([writer_embed, encoder_prior], dim=-1)))
        return writer_embed + gate * prior_delta

    def forward(
        self,
        semantic: torch.Tensor,
        query_surface: PackedSurface,
        encoder_embedding_weight: torch.Tensor,
    ) -> torch.Tensor:
        if semantic.dim() == 2:
            semantic = semantic.unsqueeze(1)
        if semantic.dim() != 3:
            raise ValueError("writer semantic must be shaped [batch, latent] or [batch, units, latent]")
        semantic_part, surface_part = split_factorized_latent(
            semantic,
            self.semantic_latent_size,
            self.surface_latent_size,
        )
        unit_states = self.semantic_proj(semantic_part) + torch.sigmoid(
            self.surface_gate(semantic_part)
        ) * self.surface_proj(surface_part)
        unit_index = query_surface.unit_ids.unsqueeze(-1).expand(-1, -1, unit_states.shape[-1])
        hidden_states = self.token_condition(query_surface.ids, encoder_embedding_weight)
        hidden_states = hidden_states + torch.gather(unit_states, 1, unit_index)
        hidden_states = hidden_states + self.position_embeddings(query_surface.pos_in_unit.clamp_max(self.max_surface_pieces_per_unit))
        hidden_states = self.dropout(hidden_states)
        mask = query_surface.mask
        for block in self.blocks:
            hidden_states = block(hidden_states, query_surface.unit_ids, mask)
        return self.token_head(self.final_norm(hidden_states))

    def transition(
        self,
        semantic: torch.Tensor,
        query_surface: PackedSurface,
        encoder_embedding_weight: torch.Tensor,
    ) -> DilWriterOutput:
        return DilWriterOutput(token_logits=self.forward(semantic, query_surface, encoder_embedding_weight))

    def unit_condition(self, semantic: torch.Tensor) -> torch.Tensor:
        if semantic.dim() != 2:
            raise ValueError("writer unit_condition expects [batch, latent]")
        semantic_part, surface_part = split_factorized_latent(
            semantic,
            self.semantic_latent_size,
            self.surface_latent_size,
        )
        return self.semantic_proj(semantic_part) + torch.sigmoid(
            self.surface_gate(semantic_part)
        ) * self.surface_proj(surface_part)

    def step(
        self,
        semantic: torch.Tensor,
        token_ids: torch.LongTensor,
        positions: torch.LongTensor,
        caches: list[torch.Tensor | None],
        encoder_embedding_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        hidden_state = self.token_condition(token_ids, encoder_embedding_weight) + self.unit_condition(semantic)
        hidden_state = hidden_state + self.position_embeddings(positions.clamp_max(self.max_surface_pieces_per_unit))
        hidden_state = self.dropout(hidden_state)
        next_caches: list[torch.Tensor | None] = []
        for block, cache in zip(self.blocks, caches):
            hidden_state, cache = block.step(hidden_state, cache)
            next_caches.append(cache)
        logits = self.token_head(self.final_norm(hidden_state))
        return logits, next_caches

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
    def generate(
        self,
        semantic: torch.Tensor,
        encoder_embedding_weight: torch.Tensor,
    ) -> DilWriterGeneration:
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
        caches: list[torch.Tensor | None] = [None for _ in self.blocks]
        current_ids = prefix[:, 0]
        for step_idx in range(self.max_surface_pieces_per_unit):
            positions = torch.full((batch_size,), step_idx, dtype=torch.long, device=device)
            logits, caches = self.step(semantic, current_ids, positions, caches, encoder_embedding_weight)
            next_ids = logits.argmax(dim=-1)
            active = ~stopped
            stop_now = active & next_ids.eq(self.stop_token_id)
            emit = active & ~stop_now
            token_ids[emit, step_idx] = next_ids[emit]
            mask[emit, step_idx] = True
            lengths = torch.where(stop_now, torch.full_like(lengths, step_idx), lengths)
            stopped = stopped | stop_now
            current_ids = torch.where(emit, next_ids, self.pad_token_id)
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
