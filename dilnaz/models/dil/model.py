from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_utils import PreTrainedModel

from dilnaz.surface import PackedSurface, PackedWriterTarget

from ..common.latents import angular_noise_like, normalize_semantic_latents
from .configuration import DilConfig
from .encoder import DilEncoderCore
from .layers import DilPackedDepthwiseConv
from .outputs import DilOutput
from .writer import DilConditionalWriter, DilWriterGeneration, DilWriterOutput
from .writer.blocks import DilPackedCausalDepthwiseConv


class Dil(PreTrainedModel):
    config_class = DilConfig

    def __init__(self, config):
        super().__init__(config)
        if config.checkpoint_format_version != 29:
            raise ValueError("DIL sequence semantic encoder checkpoints require checkpoint_format_version=29")
        if config.pad_token_id >= config.vocab_size:
            raise ValueError("pad_token_id must be inside the tokenizer vocabulary")
        if config.eos_token_id >= config.vocab_size:
            raise ValueError("eos_token_id must be inside the tokenizer vocabulary")
        if config.decoder_start_token_id >= config.vocab_size:
            raise ValueError("decoder_start_token_id must be inside the tokenizer vocabulary")
        if config.writer_stop_token_id != config.vocab_size or config.writer_vocab_size != config.vocab_size + 1:
            raise ValueError("Writer stop token contract must be writer_stop_token_id=vocab_size")
        if config.writer_bos_token_id != config.vocab_size + 1 or config.writer_empty_token_id != config.vocab_size + 2:
            raise ValueError("Writer input-only token contract must be BOS=vocab_size+1 and EMPTY=vocab_size+2")

        self.encoder = DilEncoderCore(config)
        self.writer = DilConditionalWriter(config)
        self.dil_dropout = config.dil_dropout
        self.distillation_weight = config.distillation_weight
        self.mean_geometry_weight = config.mean_geometry_weight
        self.variance_weight = config.variance_weight
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
        elif isinstance(module, (DilPackedDepthwiseConv, DilPackedCausalDepthwiseConv)):
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

    def encode(self, surface: PackedSurface, output_hidden_states: bool = False):
        compiled_forward = getattr(self, "_compiled_encoder_forward", None)
        encoded = (
            compiled_forward(surface, output_hidden_states)
            if compiled_forward is not None
            else self.encoder(surface=surface, output_hidden_states=output_hidden_states)
        )
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
        output = (
            compiled_forward(semantic, **writer_kwargs)
            if compiled_forward is not None
            else self.writer.transition(semantic, **writer_kwargs)
        )
        if isinstance(output, DilWriterOutput):
            return output
        return DilWriterOutput(token_logits=output)

    def writer_transition_outputs(self, semantic: torch.Tensor, **writer_kwargs) -> DilWriterOutput:
        return self._transition_output(semantic, **writer_kwargs)

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

    def sequence_geometry_loss(
        self,
        semantic: torch.Tensor,
        teacher_vectors: torch.Tensor,
        teacher_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if semantic.dim() == 2:
            return self.geometry_loss(semantic, teacher_vectors, teacher_mask)
        if semantic.dim() != 3:
            raise ValueError("semantic must be shaped [batch, latent] or [batch, units, latent]")
        if teacher_vectors.dim() != 3:
            raise ValueError("sequence teacher vectors must be shaped [batch, units, teacher_dim]")
        if teacher_vectors.shape[:2] != semantic.shape[:2]:
            raise ValueError("sequence teacher vectors must match semantic batch/unit dimensions")
        if teacher_mask is None:
            teacher_mask = torch.ones(semantic.shape[:2], dtype=torch.bool, device=semantic.device)
        else:
            teacher_mask = teacher_mask.to(semantic.device, dtype=torch.bool)
        return self.geometry_loss(semantic[teacher_mask], teacher_vectors[teacher_mask], None)

    def variance_regularizer(self, model_vectors: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            model_vectors = model_vectors[mask]
        if model_vectors.shape[0] < 2:
            return model_vectors.new_zeros(())
        std = torch.sqrt(model_vectors.float().var(dim=0, unbiased=False) + 1e-4)
        return F.relu(1.0 - std).mean()

    def writer_metrics(
        self,
        logits: torch.Tensor,
        target: PackedWriterTarget,
        valid_override: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = target.labels.to(logits.device)
        valid = target.label_mask.to(logits.device) if valid_override is None else valid_override.to(logits.device)
        predictions = logits.argmax(dim=-1)
        byte_valid = valid & labels.ne(self.config.writer_stop_token_id)
        stop_valid = valid & labels.eq(self.config.writer_stop_token_id)
        byte_acc = (predictions.eq(labels) & byte_valid).sum().float() / byte_valid.sum().clamp_min(1).float()
        stop_acc = (predictions.eq(labels) & stop_valid).sum().float() / stop_valid.sum().clamp_min(1).float()
        mismatch = (predictions.ne(labels) & valid).to(dtype=torch.long)
        unit_bad = torch.zeros_like(target.true_lengths.to(logits.device), dtype=torch.long)
        unit_bad.scatter_add_(1, target.query.unit_ids.to(logits.device), mismatch)
        unit_valid = target.true_lengths.to(logits.device).gt(0)
        token_exact = (unit_bad.eq(0) & unit_valid).sum().float() / unit_valid.sum().clamp_min(1).float()
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
        target: PackedWriterTarget,
        training_step: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target = target.to(semantic.device)
        semantic = self.writer_training_semantic(semantic, training_step)
        output = self.writer_transition_outputs(semantic, query_surface=target.query)
        token_loss = F.cross_entropy(
            output.token_logits.reshape(-1, self.config.writer_vocab_size),
            target.labels.reshape(-1),
            ignore_index=-100,
        )
        byte_acc, token_exact, stop_acc = self.writer_metrics(output.token_logits, target)
        return token_loss, token_loss, byte_acc, token_exact, stop_acc

    def writer_transition_loss_and_metrics(
        self,
        semantic: torch.Tensor,
        target: PackedWriterTarget,
        zone_ids: torch.LongTensor,
        window_mask: torch.Tensor,
        future_latents: Optional[torch.Tensor] = None,
        position_age: Optional[torch.Tensor] = None,
        training_step: int | None = None,
        return_metrics: bool = False,
    ):
        target = target.to(semantic.device)
        zone_ids = zone_ids.to(semantic.device, dtype=torch.long)
        window_mask = window_mask.to(semantic.device, dtype=torch.bool)
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
            query_surface=target.query,
            zone_ids=zone_ids,
            window_mask=window_mask,
            future_latents=future_latents,
            position_age=position_age,
        )
        logits = output.token_logits
        labels = target.labels
        zone_per_pos = zone_ids.gather(1, target.query.unit_ids.clamp_max(zone_ids.shape[1] - 1))
        window_per_pos = window_mask.gather(1, target.query.unit_ids.clamp_max(window_mask.shape[1] - 1))
        valid = target.label_mask & window_per_pos
        left = zone_per_pos.eq(DilConditionalWriter.ZONE_LEFT) & valid
        active = zone_per_pos.eq(DilConditionalWriter.ZONE_ACTIVE) & valid
        right = zone_per_pos.eq(DilConditionalWriter.ZONE_RIGHT) & valid
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
        loss = (
            token_loss
            + left_loss * self.config.writer_left_consistency_weight
        )
        byte_acc, token_exact, stop_acc = self.writer_metrics(logits, target, active)
        if return_metrics:
            right_byte_acc, right_token_exact, right_stop_acc = self.writer_metrics(logits, target, right)
            return {
                "loss": loss,
                "token_loss": token_loss,
                "active_token_loss": active_token_loss,
                "right_guard_token_loss": right_guard_token_loss,
                "left_consistency_loss": left_loss,
                "byte_acc": byte_acc,
                "token_exact": token_exact,
                "stop_acc": stop_acc,
                "right_guard_byte_acc": right_byte_acc,
                "right_guard_token_exact": right_token_exact,
                "right_guard_stop_acc": right_stop_acc,
                "stepT_byte_acc": byte_acc,
                "stepT_token_exact": token_exact,
                "stepT_stop_acc": stop_acc,
            }
        return loss, token_loss, byte_acc, token_exact, stop_acc

    def forward(
        self,
        surface: PackedSurface,
        teacher_layers: Optional[torch.Tensor] = None,
        teacher_mask: Optional[torch.Tensor] = None,
        training_step: Optional[int] = None,
    ) -> DilOutput:
        encoder_surface = surface
        if self.training and self.dil_dropout > 0:
            mask_keep = torch.rand_like(surface.mask.float()) >= self.dil_dropout
            encoder_surface = PackedSurface(
                ids=surface.ids,
                mask=surface.mask & mask_keep,
                unit_ids=surface.unit_ids,
                pos_in_unit=surface.pos_in_unit,
                unit_lengths=surface.unit_lengths,
                unit_offsets=surface.unit_offsets,
                unit_mask=surface.unit_mask,
            )

        semantic = self.encode(surface=encoder_surface)
        semantic = semantic.float()
        loss = semantic.new_zeros(())
        distill_loss = semantic.new_zeros(())
        mean_geometry_loss = semantic.new_zeros(())
        variance_loss = semantic.new_zeros(())

        if teacher_layers is not None:
            teacher_layers = teacher_layers.to(semantic.device, dtype=torch.float32)
            if teacher_mask is None:
                teacher_mask = torch.ones(teacher_layers.shape[:-2], dtype=torch.bool, device=semantic.device)
            else:
                teacher_mask = teacher_mask.to(semantic.device, dtype=torch.bool)
            mean_geometry_loss = self.sequence_geometry_loss(semantic, teacher_layers[..., -1, :], teacher_mask)
            variance_loss = self.variance_regularizer(semantic, teacher_mask)
            distill_loss = (
                mean_geometry_loss * self.mean_geometry_weight
                + variance_loss * self.variance_weight
            )
            loss = loss + distill_loss * self.distillation_weight

        return DilOutput(
            loss=loss,
            semantic=semantic,
            distill_loss=distill_loss,
            mean_geometry_loss=mean_geometry_loss,
            variance_loss=variance_loss,
        )

    @torch.no_grad()
    def decode_semantic(self, semantic: torch.Tensor) -> DilWriterGeneration:
        return self.writer.generate(semantic)

    @torch.no_grad()
    def decode_semantic_window(self, semantic: torch.Tensor, **writer_kwargs):
        return self.writer.generate_window(semantic, **writer_kwargs)
