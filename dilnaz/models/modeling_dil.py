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
    logits: Optional[torch.FloatTensor] = None
    length_logits: Optional[torch.FloatTensor] = None
    mean: Optional[torch.FloatTensor] = None
    log_std: Optional[torch.FloatTensor] = None
    ce_loss: Optional[torch.FloatTensor] = None
    length_loss: Optional[torch.FloatTensor] = None
    kl_loss: Optional[torch.FloatTensor] = None
    distill_loss: Optional[torch.FloatTensor] = None
    layer_geometry_losses: Optional[torch.FloatTensor] = None
    mean_geometry_loss: Optional[torch.FloatTensor] = None
    variance_loss: Optional[torch.FloatTensor] = None
    byte_acc: Optional[torch.FloatTensor] = None
    length_acc: Optional[torch.FloatTensor] = None


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


class DilEncoderCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_word_bytes = config.max_word_bytes
        self.context_size = config.context_size
        self.latent_size = config.latent_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.encoder_layers = nn.ModuleList(
            [DilLayer(config) for _ in range(config.num_encoder_layers)]
        )
        self.num_stage_layers = config.num_encoder_layers // 2
        self.token_squeeze = nn.Linear(
            config.max_word_bytes * config.hidden_size,
            config.hidden_size,
        )
        self.context_norm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.target_norm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.context_fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 4, config.hidden_size * 2),
            nn.SiLU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
        )
        self.hidden_to_latent = nn.Linear(config.hidden_size, config.latent_size * 2)
        self.norm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def pooled_target_vector(
        self,
        hidden_states: torch.Tensor,
        word_masks: torch.Tensor,
    ) -> torch.Tensor:
        target_states = hidden_states[:, -1]
        target_masks = word_masks[:, -1].unsqueeze(-1).to(target_states.dtype)
        denom = target_masks.sum(dim=1).clamp_min(1.0)
        return (target_states * target_masks).sum(dim=1) / denom

    def target_conditioned_by_context(
        self,
        token_states: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        target_state = token_states[:, -1]
        context_states = token_states[:, :-1]
        context_mask = token_mask[:, :-1]
        if context_states.shape[1] == 0:
            return target_state

        query = self.target_norm(target_state).unsqueeze(1)
        keys = self.context_norm(context_states)
        scores = (query * keys).sum(dim=-1) / (keys.shape[-1] ** 0.5)
        scores = scores.masked_fill(~context_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1).masked_fill(~context_mask, 0.0)
        context_state = torch.sum(context_states * attention.unsqueeze(-1), dim=1)
        fusion_input = torch.cat(
            [
                target_state,
                context_state,
                target_state * context_state,
                target_state - context_state,
            ],
            dim=-1,
        )
        return target_state + self.context_fusion(fusion_input)

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


class DilDecoderRenderer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_word_bytes = config.max_word_bytes
        self.num_stage_layers = config.num_decoder_layers // 2

        self.latent_to_hidden = nn.Linear(config.latent_size, config.hidden_size)
        self.decoder_layers = nn.ModuleList(
            [DilLayer(config) for _ in range(config.num_decoder_layers)]
        )
        self.expand_layer = nn.Linear(
            config.hidden_size,
            config.max_word_bytes * config.hidden_size,
        )
        self.norm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, latent_states: torch.Tensor) -> torch.Tensor:
        batch_size = latent_states.shape[0]
        hidden_states = self.latent_to_hidden(latent_states).unsqueeze(1)

        for layer_idx in range(self.num_stage_layers):
            hidden_states = self.decoder_layers[layer_idx](hidden_states)

        hidden_states = self.expand_layer(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, self.max_word_bytes, -1)

        for layer_idx in range(self.num_stage_layers):
            decoder_idx = self.num_stage_layers + layer_idx
            hidden_states = self.decoder_layers[decoder_idx](hidden_states)

        hidden_states = self.norm(hidden_states)
        return F.linear(hidden_states, self.lm_head_weight)


class Dil(PreTrainedModel):
    config_class = DilConfig
    _tied_weights_keys = ["decoder.lm_head_weight"]

    def __init__(self, config):
        super().__init__(config)
        if config.pad_token_id >= config.vocab_size:
            raise ValueError("pad_token_id must be inside the tokenizer vocabulary")

        self.encoder = DilEncoderCore(config)
        self.decoder = DilDecoderRenderer(config)
        self.decoder.lm_head_weight = self.encoder.embed_tokens.weight
        self.length_head = nn.Sequential(
            nn.LayerNorm(config.latent_size, eps=1e-6),
            nn.Linear(config.latent_size, config.max_word_bytes),
        )
        self.dil_dropout = config.dil_dropout
        self.kl_clamp = config.kl_clamp
        self.kl_weight = config.kl_weight
        self.ce_weight = config.ce_weight
        self.length_loss_weight = config.length_loss_weight
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
        self.decoder.lm_head_weight = value.weight

    def decode_from_latents(
        self,
        latent_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        compiled_forward = getattr(self, "_compiled_decode_forward", None)
        if compiled_forward is not None:
            return compiled_forward(latent_states)
        return self._decode_from_latents_impl(latent_states)

    def _decode_from_latents_impl(
        self,
        latent_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(latent_states), self.length_head(latent_states.float())

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

    def set_compiled_forwards(self, encoder_forward=None, decode_forward=None):
        object.__setattr__(self, "_compiled_encoder_forward", encoder_forward)
        object.__setattr__(self, "_compiled_decode_forward", decode_forward)

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
        labels: Optional[torch.LongTensor] = None,
        teacher_layers: Optional[torch.Tensor] = None,
        teacher_mask: Optional[torch.Tensor] = None,
        length_labels: Optional[torch.LongTensor] = None,
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
        latent_states = mean + torch.randn_like(mean) * std
        latent_states = F.dropout(latent_states, p=self.dil_dropout, training=self.training)

        kl_loss = 0.5 * (mean.pow(2) + std.pow(2) - 1 - log_std * 2)
        kl_loss = torch.clamp(kl_loss, min=self.kl_clamp)
        kl_loss = torch.mean(torch.sum(kl_loss, dim=-1))

        logits, length_logits = self.decode_from_latents(latent_states)
        logits = logits.float()
        length_logits = length_logits.float()
        ce_loss = None
        length_loss = None
        distill_loss = None
        layer_geometry_losses = None
        mean_geometry_loss = None
        variance_loss = None
        byte_acc = None
        length_acc = None
        loss = None

        if labels is not None:
            if length_labels is None:
                length_labels = labels.ne(-100).sum(dim=-1).clamp(
                    min=1,
                    max=self.config.max_word_bytes,
                ) - 1
            ce_loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1).to(logits.device),
                ignore_index=-100,
            )
            length_labels = length_labels.to(length_logits.device)
            length_loss = F.cross_entropy(length_logits, length_labels)
            valid = labels.ne(-100)
            correct = logits.argmax(dim=-1).eq(labels.to(logits.device)) & valid.to(logits.device)
            byte_acc = correct.sum().float() / valid.to(logits.device).sum().clamp_min(1).float()
            length_acc = length_logits.argmax(dim=-1).eq(length_labels).float().mean()

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

        if ce_loss is not None and length_loss is not None and distill_loss is not None:
            loss = (
                ce_loss * self.ce_weight
                + length_loss * self.length_loss_weight
                + kl_loss * self.kl_weight
                + distill_loss * self.distillation_weight
            )

        return DilOutput(
            loss=loss,
            logits=logits,
            length_logits=length_logits,
            mean=mean,
            log_std=log_std,
            ce_loss=ce_loss,
            length_loss=length_loss,
            kl_loss=kl_loss,
            distill_loss=distill_loss,
            layer_geometry_losses=layer_geometry_losses,
            mean_geometry_loss=mean_geometry_loss,
            variance_loss=variance_loss,
            byte_acc=byte_acc,
            length_acc=length_acc,
        )

