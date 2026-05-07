from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel

from .configuration_dil import DilConfig


@dataclass
class DilOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    mean: Optional[torch.FloatTensor] = None
    log_std: Optional[torch.FloatTensor] = None
    kl_loss: Optional[torch.FloatTensor] = None
    distill_loss: Optional[torch.FloatTensor] = None
    layer_geometry_losses: Optional[torch.FloatTensor] = None
    mean_geometry_loss: Optional[torch.FloatTensor] = None
    variance_loss: Optional[torch.FloatTensor] = None


class DilRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class DilGatedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class DilLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = DilGatedMLP(config)
        self.layernorm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class SemanticDistributionNormalizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps = config.semantic_normalizer_eps
        self.z_clip = config.semantic_normalizer_z_clip
        self.quantile_min = config.semantic_normalizer_quantile_min
        self.quantile_max = config.semantic_normalizer_quantile_max
        self.log_std_min = config.normalized_log_std_min
        self.log_std_max = config.normalized_log_std_max
        self.register_buffer("center", torch.zeros(config.latent_size))
        self.register_buffer("scale", torch.ones(config.latent_size))
        self.register_buffer("log_scale", torch.zeros(config.latent_size))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def fit(self, mean: torch.Tensor):
        flat = mean.detach().float().reshape(-1, mean.shape[-1])
        if flat.numel() == 0:
            raise ValueError("semantic normalizer calibration received no latents")
        quantiles = torch.quantile(
            flat,
            torch.tensor([self.quantile_min, 0.5, self.quantile_max], device=flat.device),
            dim=0,
        )
        low, median, high = quantiles.unbind(dim=0)
        gaussian_iqr = 1.3489795003921634
        scale = ((high - low) / gaussian_iqr).clamp_min(self.eps)
        self.center.copy_(median.to(self.center.device, dtype=self.center.dtype))
        self.scale.copy_(scale.to(self.scale.device, dtype=self.scale.dtype))
        self.initialized.fill_(True)
        self.fitted.fill_(True)
        self.scale.clamp_(min=self.eps)
        self.log_scale.copy_(self.scale.log())

    def require_fitted(self):
        if not bool(self.fitted.detach().cpu()):
            raise RuntimeError("DIL semantic normalizer is not fitted; run DIL calibration before NAZ")

    def normalize_mean(self, mean: torch.Tensor) -> torch.Tensor:
        self.require_fitted()
        center = self.center.to(device=mean.device, dtype=mean.dtype)
        scale = self.scale.to(device=mean.device, dtype=mean.dtype)
        return (mean - center) / scale

    def denormalize_mean(self, mean: torch.Tensor) -> torch.Tensor:
        self.require_fitted()
        center = self.center.to(device=mean.device, dtype=mean.dtype)
        scale = self.scale.to(device=mean.device, dtype=mean.dtype)
        return mean * scale + center

    def normalize_distribution(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_scale = self.log_scale.to(device=log_std.device, dtype=log_std.dtype)
        return self.normalize_mean(mean), log_std - log_scale

    def denormalize_distribution(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_scale = self.log_scale.to(device=log_std.device, dtype=log_std.dtype)
        return self.denormalize_mean(mean), log_std + log_scale

    def guard_distribution(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            mean.clamp(min=-self.z_clip, max=self.z_clip),
            log_std.clamp(min=self.log_std_min, max=self.log_std_max),
        )


def dil_context_attention_heads(hidden_size: int) -> int:
    return next(heads for heads in (8, 4, 2, 1) if hidden_size % heads == 0)


class DilEncoderCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_word_bytes = config.max_word_bytes
        self.context_size = config.context_size
        self.target_index = config.target_index
        self.latent_size = config.latent_size
        self.context_attention_heads = dil_context_attention_heads(config.hidden_size)
        self.context_head_dim = config.hidden_size // self.context_attention_heads

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.encoder_layers = nn.ModuleList(
            [DilLayer(config) for _ in range(config.num_encoder_layers)]
        )
        self.num_stage_layers = config.num_encoder_layers // 2
        self.token_squeeze = nn.Linear(
            config.max_word_bytes * config.hidden_size,
            config.hidden_size,
        )
        self.context_offset_embeddings = nn.Embedding(config.context_size, config.hidden_size)
        self.context_norm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.target_norm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.context_q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.context_k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.context_v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.context_out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.context_gate = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.hidden_to_latent = nn.Linear(config.hidden_size, config.latent_size * 2)
        self.norm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        indices = torch.arange(config.context_size)
        self.register_buffer("context_indices", indices[indices != self.target_index], persistent=False)

    def pooled_target_vector(
        self,
        hidden_states: torch.Tensor,
        word_masks: torch.Tensor,
    ) -> torch.Tensor:
        target_states = hidden_states[:, self.target_index]
        target_masks = word_masks[:, self.target_index].unsqueeze(-1).to(target_states.dtype)
        denom = target_masks.sum(dim=1).clamp_min(1.0)
        return (target_states * target_masks).sum(dim=1) / denom

    def target_conditioned_by_context(
        self,
        token_states: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        target_state = token_states[:, self.target_index]
        context_states = token_states.index_select(1, self.context_indices)
        context_mask = token_mask.index_select(1, self.context_indices)
        if context_states.shape[1] == 0:
            return target_state

        batch_size = target_state.shape[0]
        query = self.context_q_proj(self.target_norm(target_state)).reshape(
            batch_size,
            self.context_attention_heads,
            self.context_head_dim,
        )
        keys = self.context_k_proj(self.context_norm(context_states)).reshape(
            batch_size,
            context_states.shape[1],
            self.context_attention_heads,
            self.context_head_dim,
        )
        values = self.context_v_proj(context_states).reshape(
            batch_size,
            context_states.shape[1],
            self.context_attention_heads,
            self.context_head_dim,
        )
        scores = torch.einsum("bhd,bchd->bhc", query, keys) / (self.context_head_dim ** 0.5)
        context_mask = context_mask.unsqueeze(1)
        scores = scores.masked_fill(~context_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores.float(), dim=-1).to(scores.dtype).masked_fill(~context_mask, 0.0)
        context_delta = torch.einsum("bhc,bchd->bhd", attention, values).reshape(batch_size, -1)
        context_delta = self.context_out_proj(context_delta)
        gate_input = torch.cat(
            [
                target_state,
                context_delta,
                target_state * context_delta,
                target_state - context_delta,
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.context_gate(gate_input))
        return target_state + gate * context_delta

    def forward(
        self,
        input_ids: torch.LongTensor,
        word_masks: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> torch.Tensor:
        if input_ids.dim() == 2:
            input_ids = input_ids.unsqueeze(1)
            word_masks = word_masks.unsqueeze(1)
        batch_size, context_size, byte_width = input_ids.shape
        if context_size != self.context_size:
            raise ValueError(
                f"input_ids context width {context_size} != context_size {self.context_size}"
            )
        if byte_width != self.max_word_bytes:
            raise ValueError(
                f"input_ids width {byte_width} != max_word_bytes {self.max_word_bytes}"
            )

        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states * word_masks.unsqueeze(-1).to(hidden_states.dtype)

        layer_vectors = []
        for layer_idx in range(self.num_stage_layers):
            hidden_states = self.encoder_layers[layer_idx](hidden_states)
            layer_vectors.append(self.pooled_target_vector(hidden_states, word_masks))

        hidden_states = hidden_states.reshape(batch_size, context_size, -1)
        token_states = self.token_squeeze(hidden_states)
        offsets = torch.arange(context_size, device=token_states.device)
        token_states = token_states + self.context_offset_embeddings(offsets).unsqueeze(0)
        token_mask = word_masks.any(dim=-1)
        hidden_states = self.target_conditioned_by_context(token_states, token_mask)

        for layer_idx in range(self.num_stage_layers):
            encoder_idx = self.num_stage_layers + layer_idx
            hidden_states = self.encoder_layers[encoder_idx](hidden_states)
            layer_vectors.append(hidden_states)

        hidden_states = self.norm(hidden_states)
        latent_states = self.hidden_to_latent(hidden_states)
        if output_hidden_states:
            return latent_states, tuple(layer_vectors)
        return latent_states


class Dil(PreTrainedModel):
    config_class = DilConfig

    def __init__(self, config):
        super().__init__(config)
        if config.checkpoint_format_version != 15:
            raise ValueError("Dil encoder-only checkpoints require checkpoint_format_version=15")
        if config.pad_token_id >= config.vocab_size:
            raise ValueError("pad_token_id must be inside the tokenizer vocabulary")

        self.encoder = DilEncoderCore(config)
        self.semantic_normalizer = SemanticDistributionNormalizer(config)
        self.dil_dropout = config.dil_dropout
        self.kl_clamp = config.kl_clamp
        self.kl_weight = config.kl_weight
        self.distillation_weight = config.distillation_weight
        self.layer_geometry_weight = config.layer_geometry_weight
        self.mean_geometry_weight = config.mean_geometry_weight
        self.variance_weight = config.variance_weight

        self.post_init()

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
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

    def fit_semantic_normalizer(self, mean: torch.Tensor):
        self.semantic_normalizer.fit(mean)

    def normalize_distribution(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.semantic_normalizer.normalize_distribution(mean, log_std)

    def denormalize_distribution(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.semantic_normalizer.denormalize_distribution(mean, log_std)

    def denormalize_mean(self, mean: torch.Tensor) -> torch.Tensor:
        return self.semantic_normalizer.denormalize_mean(mean)

    def guard_normalized_distribution(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.semantic_normalizer.guard_distribution(mean, log_std)

    def encode(
        self,
        input_ids: torch.LongTensor,
        word_masks: torch.Tensor,
        output_hidden_states: bool = False,
    ):
        compiled_forward = getattr(self, "_compiled_encoder_forward", None)
        if compiled_forward is not None:
            return compiled_forward(input_ids, word_masks, output_hidden_states)
        return self.encoder(
            input_ids=input_ids,
            word_masks=word_masks,
            output_hidden_states=output_hidden_states,
        )

    def set_compiled_forwards(self, encoder_forward=None):
        object.__setattr__(self, "_compiled_encoder_forward", encoder_forward)

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
        off_diagonal = ~torch.eye(
            model_sim.shape[0],
            dtype=torch.bool,
            device=model_sim.device,
        )
        return F.mse_loss(model_sim[off_diagonal], teacher_sim[off_diagonal])

    def variance_regularizer(
        self,
        model_vectors: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            model_vectors = model_vectors[mask]
        if model_vectors.shape[0] < 2:
            return model_vectors.new_zeros(())
        std = torch.sqrt(model_vectors.float().var(dim=0, unbiased=False) + 1e-4)
        return F.relu(1.0 - std).mean()

    def forward(
        self,
        input_ids: torch.LongTensor,
        word_masks: torch.Tensor,
        teacher_layers: Optional[torch.Tensor] = None,
        teacher_mask: Optional[torch.Tensor] = None,
    ) -> DilOutput:
        encoder_masks = word_masks
        if self.training and self.dil_dropout > 0:
            keep = torch.rand_like(word_masks.float()) >= self.dil_dropout
            encoder_masks = word_masks * keep.to(word_masks.dtype)

        latent_states, layer_vectors = self.encode(
            input_ids=input_ids,
            word_masks=encoder_masks,
            output_hidden_states=True,
        )
        mean, log_std = torch.chunk(latent_states, 2, dim=-1)
        std = torch.exp(log_std)

        kl_loss = 0.5 * (mean.pow(2) + std.pow(2) - 1 - log_std * 2)
        kl_loss = torch.clamp(kl_loss, min=self.kl_clamp)
        kl_loss = torch.mean(torch.sum(kl_loss, dim=-1))

        distill_loss = None
        layer_geometry_losses = None
        mean_geometry_loss = None
        variance_loss = None
        loss = kl_loss * self.kl_weight

        if teacher_layers is not None:
            teacher_layers = teacher_layers.to(mean.device, dtype=torch.float32)
            if teacher_mask is None:
                teacher_mask = torch.ones(
                    teacher_layers.shape[0],
                    dtype=torch.bool,
                    device=mean.device,
                )
            else:
                teacher_mask = teacher_mask.to(mean.device, dtype=torch.bool)
            mean = mean.float()
            layer_count = min(len(layer_vectors), teacher_layers.shape[1])
            losses = [
                self.geometry_loss(layer_vectors[idx].float(), teacher_layers[:, idx], teacher_mask)
                for idx in range(layer_count)
            ]
            layer_geometry_losses = torch.stack(losses) if losses else mean.new_zeros((0,))
            mean_geometry_loss = self.geometry_loss(mean, teacher_layers[:, layer_count - 1], teacher_mask)
            variance_terms = [self.variance_regularizer(mean, teacher_mask)]
            variance_terms.extend(
                self.variance_regularizer(layer_vectors[idx].float(), teacher_mask)
                for idx in range(layer_count)
            )
            variance_loss = torch.stack(variance_terms).mean()
            distill_loss = (
                layer_geometry_losses.mean() * self.layer_geometry_weight
                + mean_geometry_loss * self.mean_geometry_weight
                + variance_loss * self.variance_weight
            )
            loss = loss + distill_loss * self.distillation_weight

        return DilOutput(
            loss=loss,
            mean=mean,
            log_std=log_std,
            kl_loss=kl_loss,
            distill_loss=distill_loss,
            layer_geometry_losses=layer_geometry_losses,
            mean_geometry_loss=mean_geometry_loss,
            variance_loss=variance_loss,
        )

