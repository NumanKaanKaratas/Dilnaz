from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_dil import DilConfig
from .configuration_naz import NazConfig
from .modeling_dil import Dil
from .naz_backbone import NazSemanticBackbone


@dataclass
class NazOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    energy: Optional[torch.FloatTensor] = None
    latent_cos: Optional[torch.FloatTensor] = None
    log_std_loss: Optional[torch.FloatTensor] = None
    latent_predictions: Optional[torch.FloatTensor] = None
    predicted_mean: Optional[torch.FloatTensor] = None
    predicted_log_std: Optional[torch.FloatTensor] = None
    target_mean: Optional[torch.FloatTensor] = None
    target_log_std: Optional[torch.FloatTensor] = None


@dataclass
class NazGenerationOutput(ModelOutput):
    sequences: Optional[torch.LongTensor] = None
    word_masks: Optional[torch.Tensor] = None
    unit_mask: Optional[torch.Tensor] = None
    generated_lengths: Optional[torch.LongTensor] = None
    generated_mean: Optional[torch.FloatTensor] = None
    generated_log_std: Optional[torch.FloatTensor] = None
    generated_raw_mean: Optional[torch.FloatTensor] = None
    roundtrip_cosine: Optional[torch.FloatTensor] = None


class NazMLPBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.linears = nn.Sequential(
            nn.Linear(2 * channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, 2 * channels, bias=True),
        )
        self.gate_act = nn.SiLU()
        self.down_proj = nn.Linear(channels, channels, bias=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        hidden = self.linears(torch.cat((self.in_ln(x), y), dim=-1))
        gate_proj, up_proj = torch.chunk(hidden, 2, dim=-1)
        return x + self.down_proj(self.gate_act(gate_proj) * up_proj)


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


class NazLatentGenerator(nn.Module):
    def __init__(self, config: NazConfig):
        super().__init__()
        self.latent_size = config.latent_size
        self.log_std_min = config.pred_log_std_min
        self.log_std_max = config.pred_log_std_max
        self.state_embd = nn.Linear(config.hidden_size, config.hidden_size)
        self.hidden_embd = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm_hidden = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm_state = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.mlp_blocks = nn.ModuleList(
            [NazMLPBlock(config.hidden_size) for _ in range(config.num_mlp_layers)]
        )
        self.final_layer = NazFinalLayer(config.hidden_size, config.latent_size * 2)

    def initialize_weights(self):
        nn.init.constant_(self.final_layer.linears[-1].weight, 0)
        nn.init.constant_(self.final_layer.linears[-1].bias, 0)

    def sample_distribution(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        condition = self.norm_hidden(self.hidden_embd(hidden_states))
        states = self.norm_state(self.state_embd(hidden_states))

        for block in self.mlp_blocks:
            states = block(states, condition)

        mean, log_std = torch.chunk(self.final_layer(states), 2, dim=-1)
        return mean, log_std.clamp(self.log_std_min, self.log_std_max)

    def sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.sample_distribution(hidden_states)
        eps = torch.randn_like(mean)
        return mean + eps * torch.exp(log_std)


class NazStudentCore(nn.Module):
    def __init__(self, config: NazConfig):
        super().__init__()
        self.semantic_embed_proj = nn.Sequential(
            nn.Linear(config.latent_size * 2, 2 * config.hidden_size),
            nn.SiLU(),
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=1e-6),
        )
        self.backbone = NazSemanticBackbone(config)
        self.generative_head = NazLatentGenerator(config)

    def initialize_weights(self):
        self.generative_head.initialize_weights()

    def embed_semantic_states(self, semantic_states: torch.Tensor) -> torch.Tensor:
        return self.semantic_embed_proj(semantic_states)

    def embed_distribution(self, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        return self.embed_semantic_states(torch.cat((mean, log_std), dim=-1))

    def forward(
        self,
        semantic_states: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs_embeds = self.embed_semantic_states(semantic_states)
        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=unit_mask,
            use_cache=False,
        )
        return self.generative_head.sample_distribution(outputs.last_hidden_state)


class Naz(PreTrainedModel):
    config_class = NazConfig

    def __init__(self, config: NazConfig):
        super().__init__(config)
        if config.dil_path is None:
            raise ValueError("NazConfig.dil_path is required")
        if config.num_samples < 2:
            raise ValueError("num_samples must be >= 2 for energy score")
        if config.semantic_feedback != "mean":
            raise ValueError("Naz semantic_feedback must be 'mean'")

        self.student_core = NazStudentCore(config)
        self.beta = config.beta
        self.num_samples = config.num_samples
        self.energy_target_samples = config.energy_target_samples
        self.max_word_bytes = config.max_word_bytes
        self.pad_token_id = config.pad_token_id
        self.log_std_loss_weight = config.log_std_loss_weight
        self.decode_chunk_size = config.decode_chunk_size

        self.student_core.initialize_weights()
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
    def generative_head(self):
        return self.student_core.generative_head

    def _validate_dil_config(
        self,
        config: NazConfig,
        dil_config: DilConfig,
    ):
        if config.latent_size != dil_config.latent_size:
            raise ValueError(
                f"Naz latent_size={config.latent_size} does not match "
                f"Dil latent_size={dil_config.latent_size}"
            )
        if config.max_word_bytes != dil_config.max_word_bytes:
            raise ValueError(
                f"Naz max_word_bytes={config.max_word_bytes} does not match "
                f"Dil max_word_bytes={dil_config.max_word_bytes}"
            )
        if config.vocab_size != dil_config.vocab_size:
            raise ValueError(
                f"Naz vocab_size={config.vocab_size} does not match "
                f"Dil vocab_size={dil_config.vocab_size}"
            )
        if config.pad_token_id != dil_config.pad_token_id:
            raise ValueError(
                f"Naz pad_token_id={config.pad_token_id} does not match "
                f"Dil pad_token_id={dil_config.pad_token_id}"
            )

    def _load_dil(self, dil_path: Path, dil_config: DilConfig):
        model = Dil(dil_config)
        checkpoint = torch.load(
            dil_path / "checkpoint.pt",
            map_location="cpu",
            weights_only=False,
        )
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

    def distance(self, x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(x_1 - x_2, ord=2, dim=-1).pow(self.beta)

    def energy_score(
        self,
        predictions: torch.Tensor,
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> torch.Tensor:
        n_x = predictions.shape[0]
        x_i = predictions.unsqueeze(1)
        x_j = predictions.unsqueeze(0)
        distance_x = self.distance(x_i, x_j).sum(dim=(0, 1)) / (n_x * (n_x - 1))

        std = torch.exp(log_std)
        eps = torch.randn(
            (self.energy_target_samples, *mean.shape),
            dtype=mean.dtype,
            device=mean.device,
        )
        targets = mean.unsqueeze(0) + eps * std.unsqueeze(0)
        distance_y = self.distance(
            predictions.reshape(n_x, 1, *predictions.shape[1:]),
            targets.reshape(1, self.energy_target_samples, *targets.shape[1:]),
        ).mean(dim=(0, 1))

        return distance_x - distance_y * 2

    def validate_byte_inputs(self, input_ids: torch.LongTensor):
        batch_size, sequence_length, byte_width = input_ids.shape
        if byte_width != self.max_word_bytes:
            raise ValueError(
                f"input_ids byte width {byte_width} != max_word_bytes {self.max_word_bytes}"
            )
        return batch_size, sequence_length

    def hidden_states(
        self,
        input_ids: torch.LongTensor,
        word_masks: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor:
        semantic_states = self.semantic_states(input_ids, word_masks, unit_mask)
        inputs_embeds = self.student_core.embed_semantic_states(semantic_states)
        return self.student_core.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=unit_mask,
            use_cache=False,
        ).last_hidden_state

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
                dst = slice(-offset, sequence_length)
                src = slice(0, sequence_length + offset)
            elif offset > 0:
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
        context_ids, context_masks, _ = self.dil_context_inputs(
            input_ids,
            word_masks,
            unit_mask,
        )
        latent_states = self.dil_model.encode(
            input_ids=context_ids,
            word_masks=context_masks,
        )
        mean, log_std = torch.chunk(latent_states, 2, dim=-1)
        mean, log_std = self.dil_model.normalize_distribution(mean.float(), log_std.float())
        return mean.clone(), log_std.clone()

    def semantic_states(
        self,
        input_ids: torch.LongTensor,
        word_masks: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length = self.validate_byte_inputs(input_ids)
        mean, log_std = self.latent_distribution(input_ids, word_masks, unit_mask)
        active = unit_mask.reshape(-1)
        semantic_states = torch.zeros(
            (batch_size * sequence_length, self.config.latent_size * 2),
            dtype=mean.dtype,
            device=mean.device,
        )
        semantic_states[active] = torch.cat((mean, log_std), dim=-1)
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

    target_distribution = latent_distribution

    def _student_forward(
        self,
        semantic_states: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        compiled_forward = getattr(self, "_student_core_forward", None)
        if compiled_forward is not None:
            mean, log_std = compiled_forward(semantic_states, unit_mask)
        else:
            mean, log_std = self.student_core(semantic_states, unit_mask)
        return self.dil_model.guard_normalized_distribution(mean, log_std)

    def predict_distribution(
        self,
        input_ids: torch.LongTensor,
        word_masks: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._student_forward(
            self.semantic_states(input_ids, word_masks, unit_mask),
            unit_mask,
        )

    def predict_semantic_distribution(
        self,
        semantic_states: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self._student_forward(semantic_states, unit_mask)
        return mean[unit_mask], log_std[unit_mask]

    def sample_from_distribution(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        sample_count: int,
    ) -> torch.Tensor:
        eps = torch.randn(
            (sample_count, *mean.shape),
            dtype=mean.dtype,
            device=mean.device,
        )
        return mean.unsqueeze(0) + eps * torch.exp(log_std).unsqueeze(0)

    def sample_latents(
        self,
        input_ids: torch.LongTensor,
        word_masks: torch.Tensor,
        unit_mask: torch.Tensor,
        num_samples: Optional[int] = None,
    ) -> torch.Tensor:
        sample_count = num_samples or self.num_samples
        mean, log_std = self.predict_distribution(input_ids, word_masks, unit_mask)
        return self.sample_from_distribution(mean, log_std, sample_count).float()

    def forward_semantic(
        self,
        semantic_states: torch.Tensor,
        target_mean: torch.Tensor,
        target_log_std: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> NazOutput:
        predicted_mean, predicted_log_std = self._student_forward(
            semantic_states,
            unit_mask,
        )
        predicted_mean = predicted_mean[unit_mask]
        predicted_log_std = predicted_log_std[unit_mask]
        active_target_mean = target_mean if target_mean.dim() == 2 else target_mean[unit_mask]
        active_target_log_std = target_log_std if target_log_std.dim() == 2 else target_log_std[unit_mask]
        latent_predictions = self.sample_from_distribution(
            predicted_mean,
            predicted_log_std,
            self.num_samples,
        ).float()
        energy = self.energy_score(
            latent_predictions,
            active_target_mean,
            active_target_log_std,
        ).mean()
        log_std_loss = F.smooth_l1_loss(predicted_log_std, active_target_log_std)
        loss = -energy + self.log_std_loss_weight * log_std_loss
        latent_cos = F.cosine_similarity(
            predicted_mean,
            active_target_mean,
            dim=-1,
        ).mean()
        return NazOutput(
            loss=loss,
            energy=energy,
            latent_cos=latent_cos,
            log_std_loss=log_std_loss,
            latent_predictions=latent_predictions,
            predicted_mean=predicted_mean,
            predicted_log_std=predicted_log_std,
            target_mean=active_target_mean,
            target_log_std=active_target_log_std,
        )

    def decode_latent_tokens(
        self,
        latents: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.LongTensor, torch.Tensor, torch.LongTensor]:
        latent_shape = latents.shape[:-1]
        flat_latents = latents.reshape(-1, latents.shape[-1])
        flat_latents = self.dil_model.denormalize_mean(flat_latents)
        decode_chunk_size = chunk_size or self.decode_chunk_size
        if decode_chunk_size > 0 and flat_latents.shape[0] > decode_chunk_size:
            logits_list = []
            for start in range(0, flat_latents.shape[0], decode_chunk_size):
                logits_chunk = self.dil_model.decode_from_latents(
                    flat_latents[start : start + decode_chunk_size]
                )
                logits_list.append(logits_chunk)
            logits = torch.cat(logits_list, dim=0)
        else:
            logits = self.dil_model.decode_from_latents(flat_latents)
        logits = logits.float()
        token_ids = logits.argmax(dim=-1)
        eos_mask = token_ids.eq(self.config.eos_token_id)
        positions = torch.arange(self.max_word_bytes, device=latents.device).unsqueeze(0)
        first_eos = torch.where(
            eos_mask.any(dim=-1),
            eos_mask.float().argmax(dim=-1),
            torch.full(token_ids.shape[:1], self.max_word_bytes, device=latents.device),
        ).long()
        masks = positions < first_eos.unsqueeze(-1)
        lengths = first_eos
        token_ids = token_ids.masked_fill(~masks, self.pad_token_id)
        return (
            token_ids.reshape(*latent_shape, self.max_word_bytes),
            masks.reshape(*latent_shape, self.max_word_bytes),
            lengths.reshape(*latent_shape),
        )

    @torch.no_grad()
    def roundtrip_semantic_cosine(
        self,
        normalized_latents: torch.Tensor,
        token_ids: torch.LongTensor,
        token_masks: torch.Tensor,
    ) -> torch.Tensor:
        unit_mask = token_masks.any(dim=-1)
        context_ids, context_masks, active = self.dil_context_inputs(
            token_ids,
            token_masks,
            unit_mask,
        )
        latent_states = self.dil_model.encode(
            input_ids=context_ids,
            word_masks=context_masks,
        )
        mean, log_std = torch.chunk(latent_states, 2, dim=-1)
        mean, _ = self.dil_model.normalize_distribution(mean.float(), log_std.float())
        flat_latents = normalized_latents.reshape(-1, normalized_latents.shape[-1])
        scores = torch.zeros(flat_latents.shape[0], dtype=mean.dtype, device=mean.device)
        scores[active] = F.cosine_similarity(mean, flat_latents[active].float(), dim=-1)
        return scores.reshape(normalized_latents.shape[:-1])

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        word_masks: torch.Tensor,
        unit_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 16,
        num_samples: int = 64,
        roundtrip_check: bool = False,
    ) -> NazGenerationOutput:
        self.eval()
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        if input_ids.dim() != 3 or word_masks.shape != input_ids.shape:
            raise ValueError("input_ids and word_masks must be shaped [batch, units, bytes]")

        unit_mask = unit_mask if unit_mask is not None else word_masks.any(dim=-1)
        if unit_mask.shape != input_ids.shape[:2]:
            raise ValueError("unit_mask must be shaped [batch, units]")
        if not bool(unit_mask.all().detach().cpu()):
            raise ValueError("Naz.generate expects packed prompts without unit padding")

        del num_samples
        current_input_embeds = self.semantic_embeddings(input_ids, word_masks, unit_mask)
        past_key_values = None
        generated_means = []
        generated_log_stds = []

        for _ in range(max_new_tokens):
            outputs = self.transformer(
                inputs_embeds=current_input_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            last_hidden = outputs.last_hidden_state[:, -1, :]
            next_mean, next_log_std = self.generative_head.sample_distribution(last_hidden)
            next_mean, next_log_std = self.dil_model.guard_normalized_distribution(
                next_mean,
                next_log_std,
            )
            generated_means.append(next_mean)
            generated_log_stds.append(next_log_std)
            current_input_embeds = self.student_core.embed_distribution(
                next_mean,
                next_log_std,
            ).unsqueeze(1)

        generated_mean = torch.stack(generated_means, dim=1)
        generated_log_std = torch.stack(generated_log_stds, dim=1)
        generated_raw_mean = self.dil_model.denormalize_mean(generated_mean)
        generated_ids, generated_masks, generated_lengths = self.decode_latent_tokens(generated_mean)
        roundtrip_cosine = (
            self.roundtrip_semantic_cosine(generated_mean, generated_ids, generated_masks)
            if roundtrip_check
            else None
        )
        sequences = torch.cat((input_ids, generated_ids), dim=1)
        sequence_word_masks = torch.cat((word_masks, generated_masks), dim=1)
        sequence_unit_mask = torch.cat(
            (
                unit_mask,
                torch.ones(
                    generated_lengths.shape,
                    dtype=torch.bool,
                    device=generated_lengths.device,
                ),
            ),
            dim=1,
        )

        return NazGenerationOutput(
            sequences=sequences,
            word_masks=sequence_word_masks,
            unit_mask=sequence_unit_mask,
            generated_lengths=generated_lengths,
            generated_mean=generated_mean,
            generated_log_std=generated_log_std,
            generated_raw_mean=generated_raw_mean,
            roundtrip_cosine=roundtrip_cosine,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        word_masks: torch.Tensor,
        target_input_ids: torch.LongTensor,
        target_word_masks: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> NazOutput:
        target_mean, target_log_std = self.target_distribution(
            target_input_ids,
            target_word_masks,
            unit_mask,
        )
        return self.forward_semantic(
            self.semantic_states(input_ids, word_masks, unit_mask),
            target_mean,
            target_log_std,
            unit_mask,
        )

