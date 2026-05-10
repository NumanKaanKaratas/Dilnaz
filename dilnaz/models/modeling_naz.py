from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel

from .configuration_dil import DilConfig
from .configuration_naz import NazConfig
from .modeling_dil import Dil
from .naz_backbone import NazBackboneCache, NazSemanticBackbone


@dataclass
class NazOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    reconstruction_loss: Optional[torch.FloatTensor] = None
    mse_loss: Optional[torch.FloatTensor] = None
    mse_mean: Optional[torch.FloatTensor] = None
    cosine_loss: Optional[torch.FloatTensor] = None
    latent_cos: Optional[torch.FloatTensor] = None
    latent_predictions: Optional[torch.FloatTensor] = None
    predicted_latents: Optional[torch.FloatTensor] = None
    target_latents: Optional[torch.FloatTensor] = None
    num_targets: Optional[torch.LongTensor] = None


@dataclass
class NazGenerationOutput(ModelOutput):
    prompt_latents: Optional[torch.FloatTensor] = None
    generated_latents: Optional[torch.FloatTensor] = None


@dataclass
class NazGenerationStep(ModelOutput):
    latent: Optional[torch.FloatTensor] = None
    latent_cos_to_previous: Optional[torch.FloatTensor] = None
    should_stop: Optional[torch.Tensor] = None


class NazFinalLayer(nn.Module):
    def __init__(self, model_channels: int, out_channels: int):
        super().__init__()
        self.in_ln = nn.LayerNorm(model_channels, eps=1e-6)
        self.linears = nn.Sequential(
            nn.Linear(model_channels, model_channels, bias=True),
            nn.SiLU(),
            nn.Linear(model_channels, out_channels, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linears(self.in_ln(x))


class NazLatentNormalizer(nn.Module):
    def __init__(self, latent_size: int, epsilon: float):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("mean", torch.zeros(latent_size), persistent=True)
        self.register_buffer("scale", torch.ones(latent_size), persistent=True)

    @torch.no_grad()
    def fit(self, latents: torch.Tensor):
        flat = latents.reshape(-1, latents.shape[-1]).float()
        self.mean.copy_(flat.mean(dim=0))
        self.scale.copy_(flat.std(dim=0, unbiased=False).clamp_min(self.epsilon))

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        return (latents - self.mean.to(latents.device, latents.dtype)) / self.scale.to(latents.device, latents.dtype)

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        return latents * self.scale.to(latents.device, latents.dtype) + self.mean.to(latents.device, latents.dtype)


class NazStudentCore(nn.Module):
    def __init__(self, config: NazConfig):
        super().__init__()
        self.semantic_embed_proj = nn.Sequential(
            nn.Linear(config.latent_size, 2 * config.hidden_size),
            nn.SiLU(),
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=1e-6),
        )
        self.backbone = NazSemanticBackbone(config)
        self.latent_head = NazFinalLayer(config.hidden_size, config.latent_size)

    def embed_semantic_states(self, semantic_states: torch.Tensor) -> torch.Tensor:
        return self.semantic_embed_proj(semantic_states)

    def forward(
        self,
        semantic_states: torch.Tensor,
        unit_mask: torch.Tensor,
        past_key_values: Optional[NazBackboneCache] = None,
        use_cache: bool = False,
    ):
        inputs_embeds = self.embed_semantic_states(semantic_states)
        return self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=unit_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        ).last_hidden_state


class Naz(PreTrainedModel):
    config_class = NazConfig

    def __init__(self, config: NazConfig):
        super().__init__(config)
        if config.dil_path is None:
            raise ValueError("NazConfig.dil_path is required")

        self.student_core = NazStudentCore(config)
        self.latent_normalizer = NazLatentNormalizer(config.latent_size, config.normalizer_epsilon)
        self.max_word_bytes = config.max_word_bytes
        self.pad_token_id = config.pad_token_id
        self.reconstruction_loss_weight = config.reconstruction_loss_weight
        self.repetition_cos_threshold = config.repetition_cos_threshold
        self.min_new_tokens = config.min_new_tokens

        self.dil_config = DilConfig.from_pretrained(config.dil_path)
        self._validate_dil_config(config, self.dil_config)
        self.dil_model = self._load_dil(Path(config.dil_path), self.dil_config)

    def train(self, mode: bool = True):
        super().train(mode)
        self.dil_model.eval()
        return self

    @property
    def transformer(self):
        return self.student_core.backbone

    @property
    def latent_head(self):
        return self.student_core.latent_head

    def _validate_dil_config(self, config: NazConfig, dil_config: DilConfig):
        if config.latent_size != dil_config.latent_size:
            raise ValueError(
                f"Naz latent_size={config.latent_size} does not match Dil latent_size={dil_config.latent_size}"
            )
        if config.max_word_bytes != dil_config.max_word_bytes:
            raise ValueError(
                f"Naz max_word_bytes={config.max_word_bytes} does not match Dil max_word_bytes={dil_config.max_word_bytes}"
            )
        if config.vocab_size != dil_config.vocab_size:
            raise ValueError(
                f"Naz vocab_size={config.vocab_size} does not match Dil vocab_size={dil_config.vocab_size}"
            )
        if config.pad_token_id != dil_config.pad_token_id:
            raise ValueError(
                f"Naz pad_token_id={config.pad_token_id} does not match Dil pad_token_id={dil_config.pad_token_id}"
            )

    def _load_dil(self, dil_path: Path, dil_config: DilConfig):
        model = Dil(dil_config)
        checkpoint = torch.load(dil_path / "checkpoint.pt", map_location="cpu", weights_only=False)
        if checkpoint["format_version"] != dil_config.checkpoint_format_version:
            raise ValueError(f"unsupported Dil checkpoint format_version={checkpoint.get('format_version')}")
        model.load_state_dict(checkpoint["model_state_dict"])
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        return model

    def trainable_state_dict(self):
        return {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith("dil_model.")
        }

    def load_trainable_state_dict(self, state_dict):
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        non_dil_missing = [key for key in missing if not key.startswith("dil_model.")]
        if unexpected:
            raise ValueError(f"unexpected Naz checkpoint keys: {unexpected}")
        if non_dil_missing:
            raise ValueError(f"missing Naz checkpoint keys: {non_dil_missing}")

    def set_compiled_student_forward(self, compiled_forward=None):
        object.__setattr__(self, "_student_core_forward", compiled_forward)

    def get_input_embeddings(self):
        return None

    def set_input_embeddings(self, value):
        raise ValueError("Naz uses semantic inputs_embeds and has no token embedding table")

    def validate_byte_inputs(self, input_ids: torch.LongTensor):
        batch_size, sequence_length, byte_width = input_ids.shape
        if byte_width != self.max_word_bytes:
            raise ValueError(f"input_ids byte width {byte_width} != max_word_bytes {self.max_word_bytes}")
        return batch_size, sequence_length

    @torch.no_grad()
    def dil_context_inputs(
        self,
        target_input_ids: torch.LongTensor,
        target_word_masks: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> tuple[torch.LongTensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, byte_width = target_input_ids.shape
        context_radius = self.dil_config.context_radius
        context_size = self.dil_config.context_size
        context_ids = torch.full(
            (batch_size, sequence_length, context_size, byte_width),
            self.pad_token_id,
            dtype=target_input_ids.dtype,
            device=target_input_ids.device,
        )
        context_masks = torch.zeros(
            (batch_size, sequence_length, context_size, byte_width),
            dtype=target_word_masks.dtype,
            device=target_word_masks.device,
        )
        for context_idx, offset in enumerate(range(-context_radius, 1)):
            if offset < 0:
                dst = slice(-offset, sequence_length)
                src = slice(0, sequence_length + offset)
            else:
                dst = slice(0, sequence_length)
                src = slice(0, sequence_length)
            context_ids[:, dst, context_idx] = target_input_ids[:, src]
            context_masks[:, dst, context_idx] = target_word_masks[:, src]
        context_masks = context_masks & unit_mask.unsqueeze(-1).unsqueeze(-1)
        active = unit_mask.reshape(-1)
        return (
            context_ids.reshape(-1, context_size, byte_width)[active],
            context_masks.reshape(-1, context_size, byte_width)[active],
            active,
        )

    @torch.no_grad()
    def raw_latent_distribution(
        self,
        input_ids: torch.LongTensor,
        word_masks: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context_ids, context_masks, _ = self.dil_context_inputs(input_ids, word_masks, unit_mask)
        latent_states = self.dil_model.encode(input_ids=context_ids, word_masks=context_masks)
        mean = latent_states.float().clone()
        log_std = torch.zeros_like(mean)
        return mean, log_std

    def latent_distribution(
        self,
        input_ids: torch.LongTensor,
        word_masks: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.raw_latent_distribution(input_ids, word_masks, unit_mask)
        return self.normalize_latents(mean), log_std

    target_distribution = latent_distribution

    def semantic_states(
        self,
        input_ids: torch.LongTensor,
        word_masks: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length = self.validate_byte_inputs(input_ids)
        mean, _ = self.latent_distribution(input_ids, word_masks, unit_mask)
        active = unit_mask.reshape(-1)
        semantic_states = torch.zeros(
            (batch_size * sequence_length, self.config.latent_size),
            dtype=mean.dtype,
            device=mean.device,
        )
        semantic_states[active] = mean
        return semantic_states.reshape(batch_size, sequence_length, -1)

    def semantic_embeddings(
        self,
        input_ids: torch.LongTensor,
        word_masks: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.student_core.embed_semantic_states(
            self.semantic_states(input_ids, word_masks, unit_mask)
        )

    def _student_hidden(
        self,
        semantic_states: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor:
        compiled_forward = getattr(self, "_student_core_forward", None)
        if compiled_forward is not None:
            return compiled_forward(semantic_states, unit_mask)
        return self.student_core(semantic_states, unit_mask)

    def predict_semantic_latents(
        self,
        semantic_states: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self._student_hidden(semantic_states, unit_mask)
        return self.latent_head(hidden_states).float()

    def predict_latents(
        self,
        input_ids: torch.LongTensor,
        word_masks: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.predict_semantic_latents(
            self.semantic_states(input_ids, word_masks, unit_mask),
            unit_mask,
        )

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.latent_normalizer.normalize(latents.float())

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.latent_normalizer.denormalize(latents.float())

    def lcm_mse_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mse_per_target = F.mse_loss(predicted.float(), target.float(), reduction="none").sum(dim=-1)
        return mse_per_target.sum(), mse_per_target

    def forward_semantic(
        self,
        semantic_states: torch.Tensor,
        target_latents: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> NazOutput:
        predicted_full = self.predict_semantic_latents(semantic_states, unit_mask)
        active_target = target_latents if target_latents.dim() == 2 else target_latents[unit_mask]
        active_predicted = predicted_full[unit_mask]

        if active_target.shape[0] == 0:
            zero = predicted_full.new_zeros(())
            return NazOutput(
                loss=zero,
                reconstruction_loss=zero,
                mse_loss=zero,
                mse_mean=zero,
                cosine_loss=zero,
                latent_cos=zero,
                latent_predictions=predicted_full,
                predicted_latents=active_predicted,
                target_latents=active_target,
                num_targets=torch.zeros((), dtype=torch.long, device=predicted_full.device),
            )

        reconstruction_loss, mse_per_target = self.lcm_mse_loss(active_predicted, active_target)
        mse_loss = reconstruction_loss
        mse_mean = mse_per_target.mean()
        raw_predicted = self.denormalize_latents(active_predicted)
        raw_target = self.denormalize_latents(active_target)
        cosine = F.cosine_similarity(raw_predicted.float(), raw_target.float(), dim=-1)
        latent_cos = cosine.mean()
        cosine_loss = (1.0 - cosine).mean()
        loss = self.reconstruction_loss_weight * reconstruction_loss
        return NazOutput(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            mse_loss=mse_loss,
            mse_mean=mse_mean,
            cosine_loss=cosine_loss,
            latent_cos=latent_cos,
            latent_predictions=predicted_full,
            predicted_latents=active_predicted,
            target_latents=active_target,
            num_targets=torch.tensor(active_target.shape[0], dtype=torch.long, device=predicted_full.device),
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        word_masks: torch.Tensor,
        target_input_ids: torch.LongTensor,
        target_word_masks: torch.Tensor,
        unit_mask: torch.Tensor,
        training_step: Optional[int] = None,
    ) -> NazOutput:
        del training_step
        target_latents, _ = self.target_distribution(target_input_ids, target_word_masks, unit_mask)
        return self.forward_semantic(
            self.semantic_states(input_ids, word_masks, unit_mask),
            target_latents,
            unit_mask,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        word_masks: torch.Tensor,
        unit_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 16,
        min_new_tokens: Optional[int] = None,
        repetition_cos_threshold: Optional[float] = None,
    ) -> NazGenerationOutput:
        prompt_model_latents = self.semantic_states(
            input_ids,
            word_masks,
            unit_mask if unit_mask is not None else word_masks.any(dim=-1),
        )
        generated = [
            step.latent
            for step in self._generate_stream_from_semantic_states(
                semantic_states=prompt_model_latents,
                unit_mask=unit_mask if unit_mask is not None else word_masks.any(dim=-1),
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                repetition_cos_threshold=repetition_cos_threshold,
            )
        ]
        prompt_latents = self.denormalize_latents(prompt_model_latents)
        generated_latents = torch.stack(generated, dim=1) if generated else prompt_latents.new_empty(
            prompt_latents.shape[0],
            0,
            prompt_latents.shape[-1],
        )
        return NazGenerationOutput(prompt_latents=prompt_latents, generated_latents=generated_latents)

    def _generate_stream_from_semantic_states(
        self,
        semantic_states: torch.Tensor,
        unit_mask: torch.Tensor,
        max_new_tokens: int = 16,
        min_new_tokens: Optional[int] = None,
        repetition_cos_threshold: Optional[float] = None,
    ):
        min_new_tokens = self.min_new_tokens if min_new_tokens is None else min_new_tokens
        repetition_cos_threshold = self.repetition_cos_threshold if repetition_cos_threshold is None else repetition_cos_threshold
        if min_new_tokens < 0:
            raise ValueError("min_new_tokens must be >= 0")

        previous_model_latent = semantic_states[:, -1]
        current_input = semantic_states
        current_mask = unit_mask
        past_key_values = None
        max_cache_length = semantic_states.shape[1] + max_new_tokens

        for generated_idx in range(max_new_tokens):
            inputs_embeds = self.student_core.embed_semantic_states(current_input)
            outputs = self.transformer(
                inputs_embeds=inputs_embeds,
                attention_mask=current_mask,
                past_key_values=past_key_values,
                use_cache=True,
                max_cache_length=max_cache_length if past_key_values is None else None,
            )
            past_key_values = outputs.past_key_values
            model_latent = self.latent_head(outputs.last_hidden_state[:, -1]).float()
            repeated = F.cosine_similarity(previous_model_latent.float(), model_latent.float(), dim=-1).ge(repetition_cos_threshold)
            latent = self.denormalize_latents(model_latent)
            should_stop = repeated & torch.full_like(repeated, generated_idx + 1 >= min_new_tokens)
            yield NazGenerationStep(
                latent=latent,
                latent_cos_to_previous=F.cosine_similarity(previous_model_latent.float(), model_latent.float(), dim=-1),
                should_stop=should_stop,
            )
            previous_model_latent = model_latent
            current_input = model_latent.unsqueeze(1)
            current_mask = torch.ones((model_latent.shape[0], 1), dtype=torch.bool, device=model_latent.device)
            if bool(should_stop.all().detach().cpu()):
                break

    @torch.no_grad()
    def generate_stream(
        self,
        input_ids: torch.LongTensor,
        word_masks: torch.Tensor,
        unit_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 16,
        min_new_tokens: Optional[int] = None,
        repetition_cos_threshold: Optional[float] = None,
    ):
        self.eval()
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        if input_ids.dim() != 3 or word_masks.shape != input_ids.shape:
            raise ValueError("input_ids and word_masks must be shaped [batch, units, bytes]")
        unit_mask = unit_mask if unit_mask is not None else word_masks.any(dim=-1)
        if unit_mask.shape != input_ids.shape[:2]:
            raise ValueError("unit_mask must be shaped [batch, units]")
        if not bool(unit_mask.all().detach().cpu()):
            raise ValueError("Naz.generate_stream expects packed prompts without unit padding")

        yield from self._generate_stream_from_semantic_states(
            semantic_states=self.semantic_states(input_ids, word_masks, unit_mask),
            unit_mask=unit_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            repetition_cos_threshold=repetition_cos_threshold,
        )
