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
from .modeling_dil import Dil, angular_noise_like, normalize_semantic_latents
from .naz_backbone import NazBackboneCache, NazSemanticBackbone


@dataclass
class NazOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    reconstruction_loss: Optional[torch.FloatTensor] = None
    mse_loss: Optional[torch.FloatTensor] = None
    mse_mean: Optional[torch.FloatTensor] = None
    mixture_nll: Optional[torch.FloatTensor] = None
    responsibility_loss: Optional[torch.FloatTensor] = None
    usage_balance_loss: Optional[torch.FloatTensor] = None
    moe_balance_loss: Optional[torch.FloatTensor] = None
    min_mse: Optional[torch.FloatTensor] = None
    chosen_mse: Optional[torch.FloatTensor] = None
    router_entropy: Optional[torch.FloatTensor] = None
    candidate_usage: Optional[torch.FloatTensor] = None
    moe_usage: Optional[torch.FloatTensor] = None
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
    candidate_index: Optional[torch.LongTensor] = None


@dataclass
class NazDynamicsOutput(ModelOutput):
    candidate_latents: Optional[torch.FloatTensor] = None
    router_logits: Optional[torch.FloatTensor] = None
    selected_latents: Optional[torch.FloatTensor] = None
    selected_indices: Optional[torch.LongTensor] = None


class SemanticDynamicsMixtureHead(nn.Module):
    def __init__(self, config: NazConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.latent_size = config.latent_size
        self.num_candidates = config.num_semantic_candidates
        self.horizons = config.mtp_horizons
        expert_size = config.hidden_size
        self.in_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.shared = nn.Sequential(
            nn.Linear(config.hidden_size, 2 * config.hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(2 * config.hidden_size, config.hidden_size, bias=True),
            nn.SiLU(),
        )
        self.base = nn.Linear(config.hidden_size, self.horizons * self.latent_size, bias=True)
        self.expert_up = nn.Linear(config.hidden_size, self.num_candidates * expert_size, bias=True)
        self.expert_gate = nn.Linear(config.hidden_size, self.num_candidates * expert_size, bias=True)
        self.expert_down = nn.Parameter(
            torch.empty(self.num_candidates, expert_size, self.horizons * self.latent_size)
        )
        self.expert_down_bias = nn.Parameter(torch.zeros(self.num_candidates, self.horizons * self.latent_size))
        self.router = nn.Linear(config.hidden_size, self.horizons * self.num_candidates, bias=True)
        self.offset_gate = nn.Linear(config.hidden_size, self.horizons * self.num_candidates, bias=True)
        self.reset_parameters(config.initializer_range)

    def reset_parameters(self, initializer_range: float) -> None:
        nn.init.normal_(self.expert_down, mean=0.0, std=initializer_range)
        nn.init.normal_(self.router.weight, mean=0.0, std=initializer_range)
        nn.init.zeros_(self.router.bias)

    def forward(self, hidden_states: torch.Tensor) -> NazDynamicsOutput:
        batch_size, sequence_length, _ = hidden_states.shape
        shared = self.shared(self.in_norm(hidden_states))
        base = self.base(shared).view(batch_size, sequence_length, self.horizons, self.latent_size)
        expert_up = self.expert_up(shared).view(batch_size, sequence_length, self.num_candidates, -1)
        expert_gate = self.expert_gate(shared).view(batch_size, sequence_length, self.num_candidates, -1)
        expert_hidden = F.silu(expert_gate) * expert_up
        offsets = torch.einsum("btke,keh->btkh", expert_hidden, self.expert_down)
        offsets = offsets + self.expert_down_bias.view(1, 1, self.num_candidates, -1)
        offsets = offsets.view(
            batch_size,
            sequence_length,
            self.num_candidates,
            self.horizons,
            self.latent_size,
        ).permute(0, 1, 3, 2, 4)
        offset_gate = torch.sigmoid(
            self.offset_gate(shared).view(batch_size, sequence_length, self.horizons, self.num_candidates)
        ).unsqueeze(-1)
        candidate_latents = normalize_semantic_latents(base.unsqueeze(3) + offset_gate * offsets)
        router_logits = self.router(shared).view(batch_size, sequence_length, self.horizons, self.num_candidates)
        selected_indices = router_logits.argmax(dim=-1)
        gather_index = selected_indices.unsqueeze(-1).unsqueeze(-1).expand(
            batch_size,
            sequence_length,
            self.horizons,
            1,
            self.latent_size,
        )
        selected_latents = candidate_latents.gather(dim=3, index=gather_index).squeeze(3)
        return NazDynamicsOutput(
            candidate_latents=candidate_latents.float(),
            router_logits=router_logits.float(),
            selected_latents=selected_latents.float(),
            selected_indices=selected_indices,
        )


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
        self.semantic_head = SemanticDynamicsMixtureHead(config)
        self.last_moe_balance_loss = torch.zeros(())
        self.last_moe_usage = torch.empty(0)

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
        output = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=unit_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        self.last_moe_balance_loss = output.moe_balance_loss
        self.last_moe_usage = output.moe_usage
        return output.last_hidden_state


class Naz(PreTrainedModel):
    config_class = NazConfig

    def __init__(self, config: NazConfig):
        super().__init__(config)
        if config.dil_path is None:
            raise ValueError("NazConfig.dil_path is required")

        self.student_core = NazStudentCore(config)
        self.max_word_bytes = config.max_word_bytes
        self.pad_token_id = config.pad_token_id
        self.reconstruction_loss_weight = config.reconstruction_loss_weight
        self.repetition_cos_threshold = config.repetition_cos_threshold
        self.min_new_tokens = config.min_new_tokens
        self.mixture_sigma = config.mixture_sigma
        self.usage_balance_weight = config.usage_balance_weight
        self.router_responsibility_weight = config.router_responsibility_weight
        self.moe_balance_weight = config.moe_balance_weight
        if config.mtp_horizons <= 0:
            raise ValueError("NazConfig.mtp_horizons must be > 0")
        if len(config.mtp_loss_weights) != config.mtp_horizons:
            raise ValueError("NazConfig.mtp_loss_weights length must equal mtp_horizons")

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
        for context_idx, offset in enumerate(range(-context_radius, context_radius + 1)):
            if offset < 0:
                if -offset >= sequence_length:
                    continue
                dst = slice(-offset, sequence_length)
                src = slice(0, sequence_length + offset)
            elif offset > 0:
                if offset >= sequence_length:
                    continue
                dst = slice(0, sequence_length - offset)
                src = slice(offset, sequence_length)
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
    def latent_distribution(
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

    target_distribution = latent_distribution

    def target_horizon_distribution(
        self,
        target_input_ids: torch.LongTensor,
        target_word_masks: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        if target_input_ids.dim() != 4:
            raise ValueError("target_input_ids must be shaped [batch, sequence, horizons, bytes]")
        if target_word_masks.shape != target_input_ids.shape:
            raise ValueError("target_word_masks must match target_input_ids")
        if target_mask.shape != target_input_ids.shape[:3]:
            raise ValueError("target_mask must be shaped [batch, sequence, horizons]")
        horizon_latents = [
            self.semantic_states(
                target_input_ids[:, :, horizon_idx],
                target_word_masks[:, :, horizon_idx],
                target_mask[:, :, horizon_idx],
            )
            for horizon_idx in range(target_input_ids.shape[2])
        ]
        return torch.stack(horizon_latents, dim=2)

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
        return self.semantic_head(hidden_states).selected_latents[:, :, 0].float()

    def predict_semantic_dynamics(
        self,
        semantic_states: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> NazDynamicsOutput:
        hidden_states = self._student_hidden(semantic_states, unit_mask)
        return self.semantic_head(hidden_states)

    def training_semantic_states(self, semantic_states: torch.Tensor, unit_mask: torch.Tensor) -> torch.Tensor:
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
        log_prob = -0.5 * sq_dist / (self.mixture_sigma * self.mixture_sigma)
        log_prob = log_prob - 0.5 * self.config.latent_size * torch.log(
            target.new_tensor(2.0 * torch.pi * self.mixture_sigma * self.mixture_sigma)
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
    ) -> NazOutput:
        dynamics = self.predict_semantic_dynamics(self.training_semantic_states(semantic_states, unit_mask), unit_mask)
        if target_mask is None:
            target_mask = unit_mask.unsqueeze(-1).expand(
                *unit_mask.shape,
                self.config.mtp_horizons,
            )
        if target_latents.dim() != 4:
            raise ValueError("target_latents must be shaped [batch, sequence, horizons, latent_size]")
        if target_mask.shape != target_latents.shape[:3]:
            raise ValueError("target_mask must be shaped [batch, sequence, horizons]")
        active_count = target_mask.sum()

        if int(active_count.detach().cpu()) == 0:
            zero = dynamics.selected_latents.new_zeros(())
            return NazOutput(
                loss=zero,
                reconstruction_loss=zero,
                mse_loss=zero,
                mse_mean=zero,
                mixture_nll=zero,
                responsibility_loss=zero,
                usage_balance_loss=zero,
                moe_balance_loss=zero,
                min_mse=zero,
                chosen_mse=zero,
                router_entropy=zero,
                candidate_usage=dynamics.router_logits.new_zeros(
                    self.config.mtp_horizons,
                    self.config.num_semantic_candidates,
                ),
                moe_usage=self.student_core.last_moe_usage.to(dynamics.router_logits.device),
                cosine_loss=zero,
                latent_cos=zero,
                latent_predictions=dynamics.selected_latents[:, :, 0],
                predicted_latents=dynamics.selected_latents[target_mask],
                target_latents=target_latents[target_mask],
                num_targets=torch.zeros((), dtype=torch.long, device=dynamics.router_logits.device),
            )

        losses = self.semantic_mixture_losses(dynamics, target_latents, target_mask)
        mixture_nll = losses["mixture_nll"]
        responsibility_loss = losses["responsibility_loss"]
        usage_balance_loss = losses["usage_balance_loss"]
        moe_balance_loss = self.student_core.last_moe_balance_loss.to(mixture_nll.device)
        reconstruction_loss = mixture_nll
        mse_loss = losses["chosen_mse"] * losses["normalizer"]
        mse_mean = losses["chosen_mse"]
        active_predicted = dynamics.selected_latents[target_mask]
        active_target = target_latents[target_mask]
        cosine = F.cosine_similarity(active_predicted.float(), active_target.float(), dim=-1)
        latent_cos = cosine.mean()
        cosine_loss = (1.0 - cosine).mean()
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
            moe_usage=self.student_core.last_moe_usage.to(mixture_nll.device),
            cosine_loss=cosine_loss,
            latent_cos=latent_cos,
            latent_predictions=dynamics.selected_latents[:, :, 0],
            predicted_latents=active_predicted,
            target_latents=active_target,
            num_targets=torch.tensor(active_target.shape[0], dtype=torch.long, device=dynamics.router_logits.device),
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        word_masks: torch.Tensor,
        target_input_ids: torch.LongTensor,
        target_word_masks: torch.Tensor,
        unit_mask: torch.Tensor,
        target_mask: torch.Tensor,
        training_step: Optional[int] = None,
    ) -> NazOutput:
        del training_step
        target_latents = self.target_horizon_distribution(target_input_ids, target_word_masks, target_mask)
        return self.forward_semantic(
            self.semantic_states(input_ids, word_masks, unit_mask),
            target_latents,
            unit_mask,
            target_mask,
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
        prompt_latents = prompt_model_latents
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
            dynamics = self.semantic_head(outputs.last_hidden_state[:, -1:])
            model_latent = dynamics.selected_latents[:, 0, 0].float()
            candidate_index = dynamics.selected_indices[:, 0, 0]
            repeated = F.cosine_similarity(previous_model_latent.float(), model_latent.float(), dim=-1).ge(repetition_cos_threshold)
            should_stop = repeated & torch.full_like(repeated, generated_idx + 1 >= min_new_tokens)
            yield NazGenerationStep(
                latent=model_latent,
                latent_cos_to_previous=F.cosine_similarity(previous_model_latent.float(), model_latent.float(), dim=-1),
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
