from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_utils import PreTrainedModel

from ..common.latents import angular_noise_like, normalize_semantic_latents
from .configuration import DilConfig
from .encoder import DilEncoderCore
from .outputs import DilOutput
from .writer import DilConditionalWriter, DilWriterOutput


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
