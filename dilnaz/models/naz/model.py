import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_utils import PreTrainedModel

from dilnaz.surface import PackedSurface

from ..common.latents import angular_noise_like
from ..dil import Dil, DilConfig
from .configuration import NazConfig
from .outputs import NazDynamicsOutput, NazGenerationOutput, NazGenerationStep, NazOutput
from .student import NazStudentCore

_DEFAULT_ATTENTION_MASK = object()


class Naz(PreTrainedModel):
    config_class = NazConfig

    def __init__(self, config: NazConfig):
        super().__init__(config)
        if config.dil_path is None:
            raise ValueError("NazConfig.dil_path is required")

        self.student_core = NazStudentCore(config)
        self.pad_token_id = config.pad_token_id
        self.reconstruction_loss_weight = config.reconstruction_loss_weight
        self.repetition_cos_threshold = config.repetition_cos_threshold
        self.min_new_tokens = config.min_new_tokens
        if config.mtp_horizons <= 0:
            raise ValueError("NazConfig.mtp_horizons must be > 0")
        if len(config.mtp_loss_weights) != config.mtp_horizons:
            raise ValueError("NazConfig.mtp_loss_weights length must equal mtp_horizons")
        self.mixture_sigma_min = float(config.mixture_sigma_min)
        self.mixture_sigma_max = float(config.mixture_sigma_max)
        sigma_ratio = (float(config.mixture_sigma) - self.mixture_sigma_min) / (
            self.mixture_sigma_max - self.mixture_sigma_min
        )
        sigma_logit = math.log(sigma_ratio / (1.0 - sigma_ratio))
        self.mixture_sigma_logit = nn.Parameter(torch.full((config.mtp_horizons,), sigma_logit))
        self.usage_balance_weight = config.usage_balance_weight
        self.router_responsibility_weight = config.router_responsibility_weight
        self.moe_balance_weight = config.moe_balance_weight

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
    def semantic_head(self):
        return self.student_core.semantic_head

    @property
    def mixture_sigma(self) -> torch.Tensor:
        return self.mixture_sigma_min + (
            self.mixture_sigma_max - self.mixture_sigma_min
        ) * torch.sigmoid(self.mixture_sigma_logit)

    def _validate_dil_config(self, config: NazConfig, dil_config: DilConfig):
        if config.latent_size != dil_config.latent_size:
            raise ValueError(
                f"Naz latent_size={config.latent_size} does not match Dil latent_size={dil_config.latent_size}"
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
        supported_versions = {dil_config.checkpoint_format_version, 30}
        if checkpoint["format_version"] not in supported_versions:
            raise ValueError(f"unsupported Dil checkpoint format_version={checkpoint.get('format_version')}")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
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

    @torch.no_grad()
    def encode_sequence_latents(
        self,
        surface: PackedSurface,
        unit_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        unit_mask = surface.unit_mask if unit_mask is None else unit_mask.to(surface.ids.device, dtype=torch.bool)
        windowed_surface = self._build_windowed_surface(surface, unit_mask)
        semantic_states = self.dil_model.encode(surface=windowed_surface)
        batch_size, unit_count = unit_mask.shape
        if semantic_states.dim() == 3 and semantic_states.shape[1] == 1:
            semantic_states = semantic_states.squeeze(1)
        if semantic_states.shape[0] != batch_size * unit_count:
            raise ValueError(
                f"DIL encoder produced {semantic_states.shape[0]} latents; expected {batch_size * unit_count}"
            )
        semantic_states = semantic_states.view(batch_size, unit_count, -1)
        return semantic_states * unit_mask.unsqueeze(-1).to(semantic_states.dtype)

    def _build_windowed_surface(
        self,
        surface: PackedSurface,
        unit_mask: torch.Tensor,
    ) -> PackedSurface:
        from dilnaz.surface import pack_token_units

        batch_size, unit_count = unit_mask.shape
        context_radius = self.dil_config.context_radius
        device = surface.ids.device
        ids_cpu = surface.ids.detach().cpu()
        offsets_cpu = surface.unit_offsets.detach().cpu()
        lengths_cpu = surface.unit_lengths.detach().cpu()
        unit_mask_cpu = unit_mask.detach().cpu()
        rows: list[list[list[int]]] = []
        for b in range(batch_size):
            unit_pieces: list[list[int]] = []
            for u in range(unit_count):
                start = int(offsets_cpu[b, u])
                length = int(lengths_cpu[b, u])
                unit_pieces.append(ids_cpu[b, start : start + length].tolist())
            for u in range(unit_count):
                window: list[list[int]] = []
                for offset in range(-context_radius, context_radius + 1):
                    neighbor = u + offset
                    if (
                        0 <= neighbor < unit_count
                        and bool(unit_mask_cpu[b, neighbor])
                    ):
                        window.append(unit_pieces[neighbor])
                    else:
                        window.append([])
                rows.append(window)
        return pack_token_units(
            rows,
            pad_token_id=self.dil_config.pad_token_id,
            bucket_sizes=self.dil_config.surface_bucket_sizes,
            max_pieces_per_unit=self.dil_config.max_surface_pieces_per_unit,
            device=device,
        )

    def embed_sequence_latents(
        self,
        surface: PackedSurface,
        unit_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.student_core.embed_semantic_states(
            self.encode_sequence_latents(surface, unit_mask)
        )

    def _student_output(
        self,
        semantic_states: torch.Tensor,
        unit_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        compiled_forward = getattr(self, "_student_core_forward", None)
        if compiled_forward is not None:
            return compiled_forward(semantic_states, unit_mask, attention_mask)
        return self.student_core(semantic_states, unit_mask, attention_mask=attention_mask)

    def predict_semantic_latents(
        self,
        semantic_states: torch.Tensor,
        unit_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states, _, _ = self._student_output(semantic_states, unit_mask, attention_mask=attention_mask)
        return self.semantic_head(hidden_states).selected_latents[:, :, 0]

    def predict_semantic_dynamics(
        self,
        semantic_states: torch.Tensor,
        unit_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> NazDynamicsOutput:
        hidden_states, _, _ = self._student_output(semantic_states, unit_mask, attention_mask=attention_mask)
        return self.semantic_head(hidden_states)

    def jitter_semantic_states_for_training(self, semantic_states: torch.Tensor, unit_mask: torch.Tensor) -> torch.Tensor:
        if not self.training or self.config.naz_input_jitter_prob <= 0.0:
            return semantic_states
        jitter_mask = unit_mask.bool() & torch.rand(unit_mask.shape, device=semantic_states.device).lt(
            self.config.naz_input_jitter_prob
        )
        if not jitter_mask.any():
            return semantic_states
        noised = semantic_states.float().clone()
        min_cos = torch.full(
            jitter_mask.shape,
            self.config.naz_input_jitter_min_cos,
            device=semantic_states.device,
            dtype=torch.float32,
        )[jitter_mask]
        max_cos = torch.full(
            jitter_mask.shape,
            self.config.naz_input_jitter_max_cos,
            device=semantic_states.device,
            dtype=torch.float32,
        )[jitter_mask]
        noised[jitter_mask] = angular_noise_like(noised[jitter_mask], min_cos, max_cos)
        return noised.to(semantic_states.dtype)

    def predict_next_latents_from_surface(
        self,
        surface: PackedSurface,
        unit_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        unit_mask = surface.unit_mask if unit_mask is None else unit_mask
        return self.predict_semantic_latents(
            self.encode_sequence_latents(surface, unit_mask),
            unit_mask,
        )

    def lcm_mse_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mse_per_target = F.mse_loss(predicted.float(), target.float(), reduction="none").sum(dim=-1)
        return mse_per_target.sum(), mse_per_target

    def horizon_loss_weights(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(self.config.mtp_loss_weights, device=device, dtype=dtype)

    def semantic_mixture_losses(
        self,
        dynamics: NazDynamicsOutput,
        target_latents: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        candidates = dynamics.candidate_latents.float()
        logits = dynamics.router_logits.float()
        target = target_latents.float()
        mask = target_mask.bool()
        weights = self.horizon_loss_weights(target.device, target.dtype).view(1, 1, -1)
        weighted_mask = mask.to(target.dtype) * weights
        normalizer = weighted_mask.sum().clamp_min(1.0)

        sq_dist = (candidates - target.unsqueeze(3)).square().sum(dim=-1)
        sigma = self.mixture_sigma.to(device=target.device, dtype=target.dtype).view(1, 1, -1, 1)
        sigma_sq = sigma.square()
        log_prob = -0.5 * sq_dist / sigma_sq
        log_prob = log_prob - 0.5 * self.config.latent_size * torch.log(
            target.new_tensor(2.0 * torch.pi) * sigma_sq
        )
        log_pi = F.log_softmax(logits, dim=-1)
        nll_per_target = -torch.logsumexp(log_pi + log_prob, dim=-1)
        mixture_nll = (nll_per_target * weighted_mask).sum() / normalizer

        responsibilities = F.softmax(log_pi.detach() + log_prob.detach(), dim=-1)
        responsibility_per_target = -(responsibilities * log_pi).sum(dim=-1)
        responsibility_loss = (responsibility_per_target * weighted_mask).sum() / normalizer

        probs = F.softmax(logits, dim=-1)
        usage_denominator = mask.to(target.dtype).sum(dim=(0, 1), keepdim=False).clamp_min(1.0).unsqueeze(-1)
        usage = (probs * mask.unsqueeze(-1).to(probs.dtype)).sum(dim=(0, 1)) / usage_denominator
        uniform_log = torch.log(target.new_tensor(1.0 / self.config.num_semantic_candidates))
        usage_balance_loss = (
            usage * (usage.clamp_min(1e-8).log() - uniform_log)
        ).sum(dim=-1).mean()

        entropy = (-(probs * probs.clamp_min(1e-8).log()).sum(dim=-1) * weighted_mask).sum() / normalizer
        selected = dynamics.selected_latents.float()
        chosen_sq_dist = (selected - target).square().sum(dim=-1)
        min_sq_dist = sq_dist.min(dim=-1).values
        chosen_mse = (chosen_sq_dist * weighted_mask).sum() / normalizer
        min_mse = (min_sq_dist * weighted_mask).sum() / normalizer

        return {
            "mixture_nll": mixture_nll,
            "responsibility_loss": responsibility_loss,
            "usage_balance_loss": usage_balance_loss,
            "router_entropy": entropy,
            "chosen_mse": chosen_mse,
            "min_mse": min_mse,
            "candidate_usage": usage.detach(),
            "sq_dist": sq_dist,
            "chosen_sq_dist": chosen_sq_dist,
            "weighted_mask": weighted_mask,
            "normalizer": normalizer,
        }

    def forward_semantic(
        self,
        semantic_states: torch.Tensor,
        target_latents: torch.Tensor,
        unit_mask: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        attention_mask=_DEFAULT_ATTENTION_MASK,
    ) -> NazOutput:
        student_attention_mask = unit_mask if attention_mask is _DEFAULT_ATTENTION_MASK else attention_mask
        hidden_states, moe_balance_loss, moe_usage = self._student_output(
            self.jitter_semantic_states_for_training(semantic_states, unit_mask),
            unit_mask,
            attention_mask=student_attention_mask,
        )
        dynamics = self.semantic_head(hidden_states)
        if target_mask is None:
            target_mask = unit_mask.unsqueeze(-1).expand(
                *unit_mask.shape,
                self.config.mtp_horizons,
            )
        if target_latents.dim() != 4:
            raise ValueError("target_latents must be shaped [batch, sequence, horizons, latent_size]")
        if target_mask.shape != target_latents.shape[:3]:
            raise ValueError("target_mask must be shaped [batch, sequence, horizons]")

        losses = self.semantic_mixture_losses(dynamics, target_latents, target_mask)
        mixture_nll = losses["mixture_nll"]
        responsibility_loss = losses["responsibility_loss"]
        usage_balance_loss = losses["usage_balance_loss"]
        moe_balance_loss = moe_balance_loss.to(mixture_nll.device)
        reconstruction_loss = mixture_nll
        mse_loss = losses["chosen_mse"] * losses["normalizer"]
        mse_mean = losses["chosen_mse"]
        active_predicted = dynamics.selected_latents[target_mask]
        active_target = target_latents[target_mask]
        cosine = F.cosine_similarity(dynamics.selected_latents.float(), target_latents.float(), dim=-1)
        weighted_mask = losses["weighted_mask"]
        normalizer = losses["normalizer"]
        latent_cos = (cosine * weighted_mask).sum() / normalizer
        cosine_loss = ((1.0 - cosine) * weighted_mask).sum() / normalizer
        loss = self.reconstruction_loss_weight * (
            mixture_nll
            + self.router_responsibility_weight * responsibility_loss
            + self.usage_balance_weight * usage_balance_loss
            + self.moe_balance_weight * moe_balance_loss
        )
        return NazOutput(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            mse_loss=mse_loss,
            mse_mean=mse_mean,
            mixture_nll=mixture_nll,
            responsibility_loss=responsibility_loss,
            usage_balance_loss=usage_balance_loss,
            moe_balance_loss=moe_balance_loss,
            min_mse=losses["min_mse"],
            chosen_mse=losses["chosen_mse"],
            router_entropy=losses["router_entropy"],
            candidate_usage=losses["candidate_usage"],
            moe_usage=moe_usage.to(mixture_nll.device),
            cosine_loss=cosine_loss,
            latent_cos=latent_cos,
            latent_predictions=dynamics.selected_latents[:, :, 0],
            predicted_latents=active_predicted,
            target_latents=active_target,
            num_targets=target_mask.to(dtype=torch.long).sum(),
        )

    def forward(
        self,
        semantic_states: torch.Tensor,
        target_latents: torch.Tensor,
        unit_mask: torch.Tensor,
        target_mask: torch.Tensor,
        attention_mask=_DEFAULT_ATTENTION_MASK,
        training_step: Optional[int] = None,
    ) -> NazOutput:
        del training_step
        return self.forward_semantic(
            semantic_states,
            target_latents,
            unit_mask,
            target_mask,
            attention_mask=attention_mask,
        )

    @torch.no_grad()
    def generate(
        self,
        surface: PackedSurface,
        unit_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 16,
        min_new_tokens: Optional[int] = None,
        repetition_cos_threshold: Optional[float] = None,
    ) -> NazGenerationOutput:
        was_training = self.training
        self.eval()
        try:
            unit_mask = surface.unit_mask if unit_mask is None else unit_mask
            prompt_model_latents = self.encode_sequence_latents(
                surface,
                unit_mask,
            )
            generated = [
                step.latent
                for step in self._generate_stream_from_semantic_states(
                    semantic_states=prompt_model_latents,
                    unit_mask=unit_mask,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    repetition_cos_threshold=repetition_cos_threshold,
                )
            ]
            prompt_latents = prompt_model_latents
            generated_latents = torch.stack(generated, dim=1) if generated else prompt_latents.new_empty(
                prompt_latents.shape[0],
                0,
                prompt_latents.shape[-1],
            )
            return NazGenerationOutput(prompt_latents=prompt_latents, generated_latents=generated_latents)
        finally:
            if was_training:
                self.train()

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
            dynamics = self.semantic_head(outputs.last_hidden_state[:, -1:])
            model_latent = dynamics.selected_latents[:, 0, 0]
            candidate_index = dynamics.selected_indices[:, 0, 0]
            repeated = F.cosine_similarity(previous_model_latent, model_latent, dim=-1).ge(repetition_cos_threshold)
            should_stop = repeated & torch.full_like(repeated, generated_idx + 1 >= min_new_tokens)
            yield NazGenerationStep(
                latent=model_latent,
                latent_cos_to_previous=F.cosine_similarity(previous_model_latent, model_latent, dim=-1),
                should_stop=should_stop,
                candidate_index=candidate_index,
            )
            previous_model_latent = model_latent
            current_input = model_latent.unsqueeze(1)
            current_mask = torch.ones((model_latent.shape[0], 1), dtype=torch.bool, device=model_latent.device)
            if bool(should_stop.all().detach().cpu()):
                break

    @torch.no_grad()
    def generate_stream(
        self,
        surface: PackedSurface,
        unit_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 16,
        min_new_tokens: Optional[int] = None,
        repetition_cos_threshold: Optional[float] = None,
        prompt_latents: Optional[torch.Tensor] = None,
    ):
        was_training = self.training
        self.eval()
        try:
            if max_new_tokens <= 0:
                raise ValueError("max_new_tokens must be > 0")
            unit_mask = surface.unit_mask if unit_mask is None else unit_mask.to(surface.ids.device, dtype=torch.bool)
            if unit_mask.shape != surface.unit_lengths.shape:
                raise ValueError("unit_mask must be shaped [batch, units]")
            if not bool(unit_mask.all().detach().cpu()):
                raise ValueError("Naz.generate_stream expects packed prompts without unit padding")
            if prompt_latents is not None:
                if prompt_latents.dim() != 3 or prompt_latents.shape[:2] != unit_mask.shape:
                    raise ValueError("prompt_latents must be shaped [batch, units, latent_size]")
                if prompt_latents.shape[-1] != self.config.latent_size:
                    raise ValueError("prompt_latents last dimension must equal config.latent_size")
                semantic_states = prompt_latents
            else:
                semantic_states = self.encode_sequence_latents(surface, unit_mask)

            yield from self._generate_stream_from_semantic_states(
                semantic_states=semantic_states,
                unit_mask=unit_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                repetition_cos_threshold=repetition_cos_threshold,
            )
        finally:
            if was_training:
                self.train()
