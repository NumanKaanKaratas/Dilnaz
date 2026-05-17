from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_utils import PreTrainedModel

from dilnaz.surface import PackedSurface, PackedWriterTarget

from ..common.latents import compose_factorized_latent, normalize_factorized_latents, split_factorized_latent
from .configuration import DilConfig
from .encoder import DilEncoderCore
from .layers import DilPackedDepthwiseConv
from .outputs import DilOutput
from .writer import DilConditionalWriter, DilWriterGeneration
from .writer.blocks import DilPackedCausalDepthwiseConv


class Dil(PreTrainedModel):
    config_class = DilConfig

    def __init__(self, config):
        super().__init__(config)
        if config.checkpoint_format_version != 31:
            raise ValueError("DIL factorized latent v2 checkpoints require checkpoint_format_version=31")
        if config.pad_token_id >= config.vocab_size:
            raise ValueError("pad_token_id must be inside the tokenizer vocabulary")
        if config.eos_token_id >= config.vocab_size:
            raise ValueError("eos_token_id must be inside the tokenizer vocabulary")
        if config.decoder_start_token_id >= config.vocab_size:
            raise ValueError("decoder_start_token_id must be inside the tokenizer vocabulary")
        if config.writer_stop_token_id != config.vocab_size or config.writer_vocab_size != config.vocab_size + 1:
            raise ValueError("Writer stop token contract must be writer_stop_token_id=vocab_size")

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

    def encode(self, surface: PackedSurface, output_hidden_states: bool = False, return_all: bool = False):
        compiled_forward = getattr(self, "_compiled_encoder_forward", None)
        encoded = (
            compiled_forward(surface, output_hidden_states, return_all)
            if compiled_forward is not None
            else self.encoder(surface=surface, output_hidden_states=output_hidden_states, return_all=return_all)
        )
        if output_hidden_states:
            latent, layer_vectors = encoded
            return normalize_factorized_latents(
                latent,
                self.config.semantic_latent_size,
                self.config.surface_latent_size,
            ), layer_vectors
        return normalize_factorized_latents(
            encoded,
            self.config.semantic_latent_size,
            self.config.surface_latent_size,
        )

    def split_latent(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return split_factorized_latent(
            latents,
            self.config.semantic_latent_size,
            self.config.surface_latent_size,
        )

    def writer_outputs(self, semantic: torch.Tensor, query_surface: PackedSurface) -> torch.Tensor:
        compiled_forward = getattr(self, "_compiled_writer_forward", None)
        if compiled_forward is not None:
            return compiled_forward(semantic, query_surface)
        return self.writer(semantic, query_surface)

    def set_compiled_forwards(self, encoder_forward=None, writer_forward=None):
        object.__setattr__(self, "_compiled_encoder_forward", encoder_forward)
        object.__setattr__(self, "_compiled_writer_forward", writer_forward)

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

    def writer_metrics(
        self,
        logits: torch.Tensor,
        target: PackedWriterTarget,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = target.labels.to(logits.device)
        valid = target.label_mask.to(logits.device)
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

    def writer_loss_and_metrics(
        self,
        semantic: torch.Tensor,
        target: PackedWriterTarget,
        return_metrics: bool = False,
    ):
        target = target.to(semantic.device)
        output = self.writer_outputs(semantic, query_surface=target.query)
        token_loss = F.cross_entropy(
            output.reshape(-1, self.config.writer_vocab_size),
            target.labels.reshape(-1),
            ignore_index=-100,
        )
        byte_acc, token_exact, stop_acc = self.writer_metrics(output, target)
        if return_metrics:
            return {
                "loss": token_loss,
                "token_loss": token_loss,
                "byte_acc": byte_acc,
                "token_exact": token_exact,
                "stop_acc": stop_acc,
            }
        return token_loss, token_loss, byte_acc, token_exact, stop_acc

    def forward(
        self,
        surface: PackedSurface,
        labels: Optional[PackedWriterTarget] = None,
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
        semantic_part, surface_part = self.split_latent(semantic)
        loss = semantic.new_zeros(())
        distill_loss = semantic.new_zeros(())
        semantic_loss = semantic.new_zeros(())
        semantic_cos = semantic.new_zeros(())
        mean_geometry_loss = semantic.new_zeros(())
        variance_loss = semantic.new_zeros(())
        surface_loss = semantic.new_zeros(())
        surface_norm = surface_part.float().norm(dim=-1).mean().to(semantic.dtype)

        if teacher_layers is not None:
            teacher_layers = teacher_layers.to(semantic.device, dtype=torch.float32)
            if teacher_mask is None:
                teacher_mask = torch.ones(teacher_layers.shape[:-2], dtype=torch.bool, device=semantic.device)
            else:
                teacher_mask = teacher_mask.to(semantic.device, dtype=torch.bool)
            teacher_target = teacher_layers[..., -1, :]
            mean_geometry_loss = self.geometry_loss(semantic_part, teacher_target, teacher_mask)
            variance_loss = self.variance_regularizer(semantic_part, teacher_mask)
            semantic_loss = (
                mean_geometry_loss * self.mean_geometry_weight
                + variance_loss * self.variance_weight
            )
            distill_loss = semantic_loss
            if teacher_target.shape[-1] == semantic_part.shape[-1]:
                active_cos = F.cosine_similarity(semantic_part.float(), teacher_target.float(), dim=-1)
                cos_mask = teacher_mask.to(active_cos.dtype)
                semantic_cos = (active_cos * cos_mask).sum() / cos_mask.sum().clamp_min(1.0)
            loss = loss + distill_loss * self.distillation_weight

        writer_loss = semantic.new_zeros(())
        byte_acc = semantic.new_zeros(())
        token_exact = semantic.new_zeros(())
        writer_token_loss = semantic.new_zeros(())
        stop_acc = semantic.new_zeros(())
        if labels is not None and self.writer_loss_weight > 0.0:
            writer_semantic = compose_factorized_latent(semantic_part.detach(), surface_part)
            writer_loss, writer_token_loss, byte_acc, token_exact, stop_acc = self.writer_loss_and_metrics(
                writer_semantic,
                labels,
            )
            surface_loss = writer_loss
            loss = loss + surface_loss * self.writer_loss_weight

        return DilOutput(
            loss=loss,
            semantic=semantic,
            distill_loss=distill_loss,
            semantic_loss=semantic_loss,
            semantic_cos=semantic_cos,
            surface_loss=surface_loss,
            surface_norm=surface_norm,
            writer_loss=writer_loss,
            writer_token_loss=writer_token_loss,
            mean_geometry_loss=mean_geometry_loss,
            variance_loss=variance_loss,
            byte_acc=byte_acc,
            token_exact=token_exact,
            stop_acc=stop_acc,
        )

    @torch.no_grad()
    def decode_semantic(self, semantic: torch.Tensor) -> DilWriterGeneration:
        return self.writer.generate(semantic)
