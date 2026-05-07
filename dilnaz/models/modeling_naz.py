from dataclasses import dataclass
from collections import Counter
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
    energy_loss: Optional[torch.FloatTensor] = None
    mean_loss: Optional[torch.FloatTensor] = None
    cosine_loss: Optional[torch.FloatTensor] = None
    writer_loss: Optional[torch.FloatTensor] = None
    stop_loss: Optional[torch.FloatTensor] = None
    latent_cos: Optional[torch.FloatTensor] = None
    candidate_cos: Optional[torch.FloatTensor] = None
    byte_acc: Optional[torch.FloatTensor] = None
    latent_predictions: Optional[torch.FloatTensor] = None
    predicted_mean: Optional[torch.FloatTensor] = None
    stop_prob: Optional[torch.FloatTensor] = None
    target_mean: Optional[torch.FloatTensor] = None
    target_log_std: Optional[torch.FloatTensor] = None


@dataclass
class NazGenerationOutput(ModelOutput):
    sequences: Optional[torch.LongTensor] = None
    word_masks: Optional[torch.Tensor] = None
    unit_mask: Optional[torch.Tensor] = None
    generated_lengths: Optional[torch.LongTensor] = None
    generated_latents: Optional[torch.FloatTensor] = None
    generated_raw_latents: Optional[torch.FloatTensor] = None
    roundtrip_cosine: Optional[torch.FloatTensor] = None


@dataclass
class NazGenerationStep(ModelOutput):
    token_ids: Optional[torch.LongTensor] = None
    word_masks: Optional[torch.Tensor] = None
    lengths: Optional[torch.LongTensor] = None
    latent: Optional[torch.FloatTensor] = None
    feedback_latent: Optional[torch.FloatTensor] = None
    roundtrip_score: Optional[torch.FloatTensor] = None
    likelihood_score: Optional[torch.FloatTensor] = None
    stop_probability: Optional[torch.FloatTensor] = None
    should_stop: Optional[torch.Tensor] = None


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
        self.noise_size = config.noise_size
        self.noise_embd = nn.Linear(config.noise_size, config.hidden_size)
        self.hidden_embd = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm_hidden = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm_noise = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.mlp_blocks = nn.ModuleList(
            [NazMLPBlock(config.hidden_size) for _ in range(config.num_mlp_layers)]
        )
        self.final_layer = NazFinalLayer(config.hidden_size, config.latent_size)

    def initialize_weights(self):
        nn.init.constant_(self.final_layer.linears[-1].weight, 0)
        nn.init.constant_(self.final_layer.linears[-1].bias, 0)

    def sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        noise = torch.rand(
            (*hidden_states.shape[:-1], self.noise_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        ) - 0.5
        condition = self.norm_hidden(self.hidden_embd(hidden_states))
        states = self.norm_noise(self.noise_embd(noise))

        for block in self.mlp_blocks:
            states = block(states, condition)

        return self.final_layer(states)


class NazContextualWriterLayer(nn.Module):
    def __init__(self, config: NazConfig):
        super().__init__()
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size, bias=True),
            nn.SiLU(),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=True),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))


class NazContextualWriter(nn.Module):
    def __init__(self, config: NazConfig):
        super().__init__()
        self.max_word_bytes = config.max_word_bytes
        self.latent_proj = nn.Linear(config.latent_size, config.hidden_size)
        self.hidden_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.fuse_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.expand_layer = nn.Linear(config.hidden_size, config.max_word_bytes * config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_word_bytes, config.hidden_size)
        self.layers = nn.ModuleList([NazContextualWriterLayer(config) for _ in range(config.num_writer_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

    def initialize_weights(self, std: float, pad_token_id: int):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
        if 0 <= pad_token_id < self.embed_tokens.num_embeddings:
            self.embed_tokens.weight.data[pad_token_id].zero_()

    def forward(self, latents: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        if latents.dim() == hidden_states.dim() + 1:
            hidden_states = hidden_states.unsqueeze(0).expand(*latents.shape[:-1], hidden_states.shape[-1])
        latent_shape = latents.shape[:-1]
        flat_latents = latents.reshape(-1, latents.shape[-1])
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        fused = self.fuse_norm(self.latent_proj(flat_latents) + self.hidden_proj(flat_hidden))
        byte_states = self.expand_layer(fused).reshape(flat_latents.shape[0], self.max_word_bytes, -1)
        positions = torch.arange(self.max_word_bytes, device=latents.device)
        byte_states = byte_states + self.position_embeddings(positions).unsqueeze(0)
        for layer in self.layers:
            byte_states = layer(byte_states)
        logits = F.linear(self.norm(byte_states), self.embed_tokens.weight)
        return logits.reshape(*latent_shape, self.max_word_bytes, -1)


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
        self.mean_head = NazFinalLayer(config.hidden_size, config.latent_size)
        self.generative_head = NazLatentGenerator(config)
        self.writer = NazContextualWriter(config)
        self.stop_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size, eps=1e-6),
            nn.Linear(config.hidden_size, 1, bias=True),
        )

    def initialize_weights(self, config: NazConfig):
        self.generative_head.initialize_weights()
        self.writer.initialize_weights(config.initializer_range, config.pad_token_id)

    def embed_semantic_states(self, semantic_states: torch.Tensor) -> torch.Tensor:
        return self.semantic_embed_proj(semantic_states)

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
        return outputs.last_hidden_state


class Naz(PreTrainedModel):
    config_class = NazConfig

    def __init__(self, config: NazConfig):
        super().__init__(config)
        if config.dil_path is None:
            raise ValueError("NazConfig.dil_path is required")
        if config.num_samples < 3:
            raise ValueError("num_samples must be >= 3: one mean anchor plus at least two energy samples")

        self.student_core = NazStudentCore(config)
        self.beta = config.beta
        self.num_samples = config.num_samples
        self.energy_target_samples = config.energy_target_samples
        self.max_word_bytes = config.max_word_bytes
        self.pad_token_id = config.pad_token_id
        self.decode_chunk_size = config.decode_chunk_size
        self.mean_loss_weight = config.mean_loss_weight
        self.cosine_loss_weight = config.cosine_loss_weight
        self.energy_loss_weight = config.energy_loss_weight
        self.writer_loss_weight = config.writer_loss_weight
        self.writer_target_warmup_steps = config.writer_target_warmup_steps
        self.writer_candidate_start_step = config.writer_candidate_start_step
        self.writer_candidate_probability = config.writer_candidate_probability
        self.stop_loss_weight = config.stop_loss_weight
        self.stop_threshold = config.stop_threshold
        self.repetition_cos_threshold = config.repetition_cos_threshold
        self.min_new_tokens = config.min_new_tokens

        self.student_core.initialize_weights(config)
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

    @property
    def mean_head(self):
        return self.student_core.mean_head

    @property
    def writer(self):
        return self.student_core.writer

    @property
    def stop_head(self):
        return self.student_core.stop_head

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
        if not bool(model.semantic_normalizer.fitted.detach().cpu()):
            raise RuntimeError("Naz requires a DIL checkpoint with fitted robust semantic normalizer")
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

    def set_compiled_writer_forward(self, compiled_forward=None):
        object.__setattr__(self, "_compiled_writer_forward", compiled_forward)

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

    target_distribution = latent_distribution

    def _student_hidden(
        self,
        semantic_states: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor:
        compiled_forward = getattr(self, "_student_core_forward", None)
        if compiled_forward is not None:
            return compiled_forward(semantic_states, unit_mask)
        return self.student_core(semantic_states, unit_mask)

    def guard_normalized_latents(self, latents: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(latents)
        latents, _ = self.dil_model.guard_normalized_distribution(latents, zeros)
        return latents

    def writer_labels(
        self,
        target_input_ids: torch.LongTensor,
        target_word_masks: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> torch.LongTensor:
        labels = torch.full_like(target_input_ids, -100)
        labels = torch.where(target_word_masks, target_input_ids, labels)
        lengths = target_word_masks.long().sum(dim=-1)
        eos_positions = lengths.clamp_max(self.max_word_bytes - 1)
        eos_valid = unit_mask & lengths.lt(self.max_word_bytes)
        batch_idx, unit_idx = torch.where(eos_valid)
        labels[batch_idx, unit_idx, eos_positions[eos_valid]] = self.config.eos_token_id
        return labels.masked_fill(~unit_mask.unsqueeze(-1), -100)

    def writer_logits(self, latents: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        compiled_forward = getattr(self, "_compiled_writer_forward", None)
        if compiled_forward is not None:
            return compiled_forward(latents, hidden_states)
        return self.writer(latents, hidden_states)

    def writer_training_latents(
        self,
        target_mean: torch.Tensor,
        predicted_mean: torch.Tensor,
        sample_predictions: torch.Tensor,
        training_step: Optional[int],
    ) -> torch.Tensor:
        if training_step is None or training_step < self.writer_target_warmup_steps:
            return target_mean.detach()

        writer_latents = predicted_mean.detach()
        if (
            self.writer_candidate_probability <= 0.0
            or training_step < self.writer_candidate_start_step
            or sample_predictions.shape[0] == 0
        ):
            return writer_latents

        sample_idx = torch.randint(
            sample_predictions.shape[0],
            (writer_latents.shape[0],),
            device=writer_latents.device,
        )
        row_idx = torch.arange(writer_latents.shape[0], device=writer_latents.device)
        candidate_latents = sample_predictions.detach()[sample_idx, row_idx]
        use_candidate = torch.rand(
            (writer_latents.shape[0], 1),
            device=writer_latents.device,
            dtype=writer_latents.dtype,
        ).lt(self.writer_candidate_probability)
        return torch.where(use_candidate, candidate_latents, writer_latents)

    def predict_semantic_mean(
        self,
        semantic_states: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self._student_hidden(semantic_states, unit_mask)
        return self.guard_normalized_latents(self.mean_head(hidden_states).float())

    def predict_mean(
        self,
        input_ids: torch.LongTensor,
        word_masks: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.predict_semantic_mean(
            self.semantic_states(input_ids, word_masks, unit_mask),
            unit_mask,
        )

    def candidate_latents_from_hidden(
        self,
        hidden_states: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")
        predicted_mean = self.guard_normalized_latents(self.mean_head(hidden_states).float())
        anchor = predicted_mean.unsqueeze(0)
        if num_samples == 1:
            return anchor
        sample_count = num_samples - 1
        repeated_hidden = hidden_states.unsqueeze(0).repeat(sample_count, *([1] * hidden_states.dim()))
        sampled_offsets = self.generative_head.sample(repeated_hidden).float()
        samples = self.guard_normalized_latents(predicted_mean.unsqueeze(0) + sampled_offsets)
        return torch.cat((anchor, samples), dim=0)

    def sample_semantic_latents(
        self,
        semantic_states: torch.Tensor,
        unit_mask: torch.Tensor,
        num_samples: Optional[int] = None,
    ) -> torch.Tensor:
        sample_count = num_samples or self.num_samples
        hidden_states = self._student_hidden(semantic_states, unit_mask)
        active_hidden = hidden_states[unit_mask]
        return self.candidate_latents_from_hidden(active_hidden, sample_count)

    def sample_latents(
        self,
        input_ids: torch.LongTensor,
        word_masks: torch.Tensor,
        unit_mask: torch.Tensor,
        num_samples: Optional[int] = None,
    ) -> torch.Tensor:
        return self.sample_semantic_latents(
            self.semantic_states(input_ids, word_masks, unit_mask),
            unit_mask,
            num_samples,
        )

    def forward_semantic(
        self,
        semantic_states: torch.Tensor,
        target_mean: torch.Tensor,
        target_log_std: torch.Tensor,
        unit_mask: torch.Tensor,
        target_input_ids: torch.LongTensor,
        target_word_masks: torch.Tensor,
        stop_targets: Optional[torch.Tensor] = None,
        training_step: Optional[int] = None,
    ) -> NazOutput:
        hidden_states = self._student_hidden(semantic_states, unit_mask)
        active_hidden = hidden_states[unit_mask]
        active_target_mean = target_mean if target_mean.dim() == 2 else target_mean[unit_mask]
        active_target_log_std = target_log_std if target_log_std.dim() == 2 else target_log_std[unit_mask]
        predicted_mean = self.guard_normalized_latents(self.mean_head(hidden_states).float())[unit_mask]
        anchor_prediction = predicted_mean.unsqueeze(0)
        sample_count = self.num_samples - 1
        repeated_hidden = active_hidden.unsqueeze(0).repeat(sample_count, 1, 1)
        sampled_offsets = self.generative_head.sample(repeated_hidden).float()
        sample_predictions = self.guard_normalized_latents(
            predicted_mean.unsqueeze(0) + sampled_offsets
        )
        energy = self.energy_score(
            sample_predictions,
            active_target_mean,
            active_target_log_std,
        ).mean()
        energy_loss = -energy
        mean_loss = F.mse_loss(predicted_mean, active_target_mean)
        cosine = F.cosine_similarity(
            predicted_mean,
            active_target_mean,
            dim=-1,
        )
        latent_cos = cosine.mean()
        expanded_sample_target_mean = active_target_mean.unsqueeze(0).expand_as(sample_predictions)
        candidate_cos = F.cosine_similarity(
            sample_predictions.detach(),
            expanded_sample_target_mean,
            dim=-1,
        ).mean()
        cosine_loss = (1.0 - cosine).mean()
        writer_loss = active_hidden.new_zeros(())
        byte_acc = active_hidden.new_zeros(())
        if self.writer_loss_weight > 0.0:
            labels = self.writer_labels(target_input_ids, target_word_masks, unit_mask)
            active_labels = labels[unit_mask].to(active_hidden.device)
            writer_latents = self.writer_training_latents(
                active_target_mean,
                predicted_mean,
                sample_predictions,
                training_step,
            )
            writer_logits = self.writer_logits(writer_latents, active_hidden.detach()).float()
            writer_loss = F.cross_entropy(
                writer_logits.reshape(-1, self.config.vocab_size),
                active_labels.reshape(-1),
                ignore_index=-100,
            )
            valid_labels = active_labels.ne(-100)
            byte_correct = writer_logits.argmax(dim=-1).eq(active_labels) & valid_labels
            byte_acc = byte_correct.sum().float() / valid_labels.sum().clamp_min(1).float()
        stop_logits = self.stop_head(hidden_states).squeeze(-1)
        if stop_targets is None:
            stop_targets = torch.zeros_like(unit_mask, dtype=stop_logits.dtype)
        active_stop_targets = stop_targets.to(stop_logits.device, dtype=stop_logits.dtype)[unit_mask]
        stop_loss = F.binary_cross_entropy_with_logits(stop_logits[unit_mask].float(), active_stop_targets.float())
        loss = (
            self.energy_loss_weight * energy_loss
            + self.mean_loss_weight * mean_loss
            + self.cosine_loss_weight * cosine_loss
            + self.writer_loss_weight * writer_loss
            + self.stop_loss_weight * stop_loss
        )
        return NazOutput(
            loss=loss,
            energy=energy,
            energy_loss=energy_loss,
            mean_loss=mean_loss,
            cosine_loss=cosine_loss,
            writer_loss=writer_loss,
            stop_loss=stop_loss,
            latent_cos=latent_cos,
            candidate_cos=candidate_cos,
            byte_acc=byte_acc,
            latent_predictions=torch.cat((anchor_prediction, sample_predictions), dim=0),
            predicted_mean=predicted_mean,
            stop_prob=torch.sigmoid(stop_logits[unit_mask].float()),
            target_mean=active_target_mean,
            target_log_std=active_target_log_std,
        )

    def decode_latent_tokens(
        self,
        latents: torch.Tensor,
        hidden_states: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.LongTensor, torch.Tensor, torch.LongTensor]:
        latent_shape = latents.shape[:-1]
        flat_latents = latents.reshape(-1, latents.shape[-1])
        if latents.dim() == hidden_states.dim() + 1:
            hidden_states = hidden_states.unsqueeze(0).expand(*latents.shape[:-1], hidden_states.shape[-1])
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        decode_chunk_size = chunk_size or self.decode_chunk_size
        if decode_chunk_size > 0 and flat_latents.shape[0] > decode_chunk_size:
            logits_list = []
            for start in range(0, flat_latents.shape[0], decode_chunk_size):
                logits_chunk = self.writer_logits(
                    flat_latents[start : start + decode_chunk_size],
                    flat_hidden[start : start + decode_chunk_size],
                )
                logits_list.append(logits_chunk)
            logits = torch.cat(logits_list, dim=0)
        else:
            logits = self.writer_logits(flat_latents, flat_hidden)
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

    def sample_next_latents(self, hidden_states: torch.Tensor, num_samples: int) -> torch.Tensor:
        return self.candidate_latents_from_hidden(hidden_states, num_samples)

    def feedback_distribution_for_candidates(
        self,
        sequences: torch.LongTensor,
        sequence_word_masks: torch.Tensor,
        sequence_unit_mask: torch.Tensor,
        candidate_ids: torch.LongTensor,
        candidate_masks: torch.Tensor,
    ) -> torch.Tensor:
        sample_count, batch_size, byte_width = candidate_ids.shape
        sequence_length = sequences.shape[1]
        repeated_sequences = sequences.unsqueeze(0).expand(sample_count, -1, -1, -1)
        repeated_word_masks = sequence_word_masks.unsqueeze(0).expand(sample_count, -1, -1, -1)
        repeated_unit_mask = sequence_unit_mask.unsqueeze(0).expand(sample_count, -1, -1)
        flat_sequences = repeated_sequences.reshape(sample_count * batch_size, sequence_length, byte_width)
        flat_word_masks = repeated_word_masks.reshape(sample_count * batch_size, sequence_length, byte_width)
        flat_unit_mask = repeated_unit_mask.reshape(sample_count * batch_size, sequence_length)
        flat_candidate_ids = candidate_ids.reshape(sample_count * batch_size, byte_width)
        flat_candidate_masks = candidate_masks.reshape(sample_count * batch_size, byte_width)
        candidate_sequences = torch.cat((flat_sequences, flat_candidate_ids.unsqueeze(1)), dim=1)
        candidate_word_masks = torch.cat((flat_word_masks, flat_candidate_masks.unsqueeze(1)), dim=1)
        candidate_unit_mask = torch.cat(
            (
                flat_unit_mask,
                torch.ones(
                    (sample_count * batch_size, 1),
                    dtype=torch.bool,
                    device=flat_unit_mask.device,
                ),
            ),
            dim=1,
        )
        feedback_unit_mask = torch.zeros_like(candidate_unit_mask)
        feedback_unit_mask[:, -1] = True
        feedback_mean, feedback_log_std = self.latent_distribution(
            candidate_sequences,
            candidate_word_masks,
            feedback_unit_mask,
        )
        return (
            feedback_mean.reshape(sample_count, batch_size, -1),
            feedback_log_std.reshape(sample_count, batch_size, -1),
        )

    def select_written_candidates(
        self,
        candidate_latents: torch.Tensor,
        candidate_ids: torch.LongTensor,
        candidate_masks: torch.Tensor,
        candidate_lengths: torch.LongTensor,
        feedback_means: torch.Tensor,
        feedback_log_stds: torch.Tensor,
        stop_probability: Optional[torch.Tensor] = None,
        should_stop: Optional[torch.Tensor] = None,
    ) -> NazGenerationStep:
        scores = F.cosine_similarity(candidate_latents, feedback_means, dim=-1)
        log_var = feedback_log_stds * 2.0
        likelihood_scores = -0.5 * (
            ((candidate_latents - feedback_means).pow(2) / log_var.exp().clamp_min(1e-6))
            + log_var
        ).mean(dim=-1)
        sample_count, batch_size, latent_size = candidate_latents.shape
        selected = []
        for batch_idx in range(batch_size):
            keys = [
                tuple(candidate_ids[sample_idx, batch_idx][candidate_masks[sample_idx, batch_idx]].tolist())
                for sample_idx in range(sample_count)
            ]
            counts = Counter(keys)
            best_idx = max(
                range(sample_count),
                key=lambda sample_idx: (
                    counts[keys[sample_idx]],
                    float(scores[sample_idx, batch_idx].detach().cpu()),
                    float(likelihood_scores[sample_idx, batch_idx].detach().cpu()),
                ),
            )
            selected.append(best_idx)
        selected_idx = torch.tensor(selected, dtype=torch.long, device=candidate_latents.device)
        latent_index = selected_idx.reshape(1, batch_size, 1).expand(1, batch_size, latent_size)
        byte_index = selected_idx.reshape(1, batch_size, 1).expand(1, batch_size, self.max_word_bytes)
        scalar_index = selected_idx.reshape(1, batch_size)
        return NazGenerationStep(
            token_ids=candidate_ids.gather(0, byte_index).squeeze(0),
            word_masks=candidate_masks.gather(0, byte_index).squeeze(0),
            lengths=candidate_lengths.gather(0, scalar_index).squeeze(0),
            latent=candidate_latents.gather(0, latent_index).squeeze(0),
            feedback_latent=feedback_means.gather(0, latent_index).squeeze(0),
            roundtrip_score=scores.gather(0, scalar_index).squeeze(0),
            likelihood_score=likelihood_scores.gather(0, scalar_index).squeeze(0),
            stop_probability=stop_probability,
            should_stop=should_stop,
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

    def generate(
        self,
        input_ids: torch.LongTensor,
        word_masks: torch.Tensor,
        unit_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 16,
        num_samples: int = 4,
        roundtrip_check: bool = False,
        min_new_tokens: Optional[int] = None,
        stop_threshold: Optional[float] = None,
        repetition_cos_threshold: Optional[float] = None,
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

        sequences = input_ids
        sequence_word_masks = word_masks
        sequence_unit_mask = unit_mask
        generated_latents = []
        generated_ids = []
        generated_masks = []
        generated_lengths = []

        for step in self.generate_stream(
            input_ids=input_ids,
            word_masks=word_masks,
            unit_mask=unit_mask,
            max_new_tokens=max_new_tokens,
            num_samples=num_samples,
            min_new_tokens=min_new_tokens,
            stop_threshold=stop_threshold,
            repetition_cos_threshold=repetition_cos_threshold,
        ):
            generated_latents.append(step.latent)
            generated_ids.append(step.token_ids)
            generated_masks.append(step.word_masks)
            generated_lengths.append(step.lengths)
            sequences = torch.cat((sequences, step.token_ids.unsqueeze(1)), dim=1)
            sequence_word_masks = torch.cat((sequence_word_masks, step.word_masks.unsqueeze(1)), dim=1)
            sequence_unit_mask = torch.cat(
                (
                    sequence_unit_mask,
                    torch.ones(
                        (sequence_unit_mask.shape[0], 1),
                        dtype=torch.bool,
                        device=sequence_unit_mask.device,
                    ),
                ),
                dim=1,
            )

        generated_latents = torch.stack(generated_latents, dim=1)
        generated_raw_latents = self.dil_model.denormalize_mean(generated_latents)
        generated_ids = torch.stack(generated_ids, dim=1)
        generated_masks = torch.stack(generated_masks, dim=1)
        generated_lengths = torch.stack(generated_lengths, dim=1)
        roundtrip_cosine = (
            self.roundtrip_semantic_cosine(generated_latents, generated_ids, generated_masks)
            if roundtrip_check
            else None
        )
        return NazGenerationOutput(
            sequences=sequences,
            word_masks=sequence_word_masks,
            unit_mask=sequence_unit_mask,
            generated_lengths=generated_lengths,
            generated_latents=generated_latents,
            generated_raw_latents=generated_raw_latents,
            roundtrip_cosine=roundtrip_cosine,
        )

    def generate_stream(
        self,
        input_ids: torch.LongTensor,
        word_masks: torch.Tensor,
        unit_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 16,
        num_samples: int = 4,
        min_new_tokens: Optional[int] = None,
        stop_threshold: Optional[float] = None,
        repetition_cos_threshold: Optional[float] = None,
    ):
        with torch.no_grad():
            self.eval()
            if max_new_tokens <= 0:
                raise ValueError("max_new_tokens must be > 0")
            if num_samples <= 0:
                raise ValueError("num_samples must be > 0")
            min_new_tokens = self.min_new_tokens if min_new_tokens is None else min_new_tokens
            stop_threshold = self.stop_threshold if stop_threshold is None else stop_threshold
            repetition_cos_threshold = (
                self.repetition_cos_threshold if repetition_cos_threshold is None else repetition_cos_threshold
            )
            if min_new_tokens < 0:
                raise ValueError("min_new_tokens must be >= 0")
            if input_ids.dim() != 3 or word_masks.shape != input_ids.shape:
                raise ValueError("input_ids and word_masks must be shaped [batch, units, bytes]")

            unit_mask = unit_mask if unit_mask is not None else word_masks.any(dim=-1)
            if unit_mask.shape != input_ids.shape[:2]:
                raise ValueError("unit_mask must be shaped [batch, units]")
            if not bool(unit_mask.all().detach().cpu()):
                raise ValueError("Naz.generate_stream expects packed prompts without unit padding")

            current_semantic_states = self.semantic_states(input_ids, word_masks, unit_mask)
            previous_feedback_latent = current_semantic_states[:, -1]
            previous_token_ids = input_ids[:, -1]
            previous_token_masks = word_masks[:, -1]
            current_input_embeds = self.student_core.embed_semantic_states(current_semantic_states)
            sequences = input_ids
            sequence_word_masks = word_masks
            sequence_unit_mask = unit_mask
            past_key_values = None

            for generated_idx in range(max_new_tokens):
                outputs = self.transformer(
                    inputs_embeds=current_input_embeds,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                last_hidden = outputs.last_hidden_state[:, -1, :]
                stop_probability = torch.sigmoid(self.stop_head(last_hidden).squeeze(-1).float())
                candidate_latents = self.sample_next_latents(last_hidden, num_samples)
                candidate_ids, candidate_masks, candidate_lengths = self.decode_latent_tokens(candidate_latents, last_hidden)
                feedback_means, feedback_log_stds = self.feedback_distribution_for_candidates(
                    sequences,
                    sequence_word_masks,
                    sequence_unit_mask,
                    candidate_ids,
                    candidate_masks,
                )
                selected = self.select_written_candidates(
                    candidate_latents,
                    candidate_ids,
                    candidate_masks,
                    candidate_lengths,
                    feedback_means,
                    feedback_log_stds,
                )
                repeated_latent = F.cosine_similarity(
                    previous_feedback_latent,
                    selected.feedback_latent,
                    dim=-1,
                ).ge(repetition_cos_threshold)
                repeated_surface = (
                    selected.word_masks.eq(previous_token_masks)
                    & selected.token_ids.eq(previous_token_ids)
                ).all(dim=-1)
                should_stop = (
                    (stop_probability.ge(stop_threshold) | repeated_latent | repeated_surface)
                    & torch.full_like(repeated_latent, generated_idx + 1 >= min_new_tokens)
                )
                selected.stop_probability = stop_probability
                selected.should_stop = should_stop

                yield selected

                sequences = torch.cat((sequences, selected.token_ids.unsqueeze(1)), dim=1)
                sequence_word_masks = torch.cat((sequence_word_masks, selected.word_masks.unsqueeze(1)), dim=1)
                sequence_unit_mask = torch.cat(
                    (
                        sequence_unit_mask,
                        torch.ones(
                            (sequence_unit_mask.shape[0], 1),
                            dtype=torch.bool,
                            device=sequence_unit_mask.device,
                        ),
                    ),
                    dim=1,
                )
                current_input_embeds = self.student_core.embed_semantic_states(selected.feedback_latent).unsqueeze(1)
                previous_feedback_latent = selected.feedback_latent
                previous_token_ids = selected.token_ids
                previous_token_masks = selected.word_masks
                if bool(should_stop.all().detach().cpu()):
                    break

    def forward(
        self,
        input_ids: torch.LongTensor,
        word_masks: torch.Tensor,
        target_input_ids: torch.LongTensor,
        target_word_masks: torch.Tensor,
        unit_mask: torch.Tensor,
        stop_targets: Optional[torch.Tensor] = None,
        training_step: Optional[int] = None,
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
            target_input_ids=target_input_ids,
            target_word_masks=target_word_masks,
            stop_targets=stop_targets,
            training_step=training_step,
        )

