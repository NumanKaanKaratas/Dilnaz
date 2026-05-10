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
    semantic: Optional[torch.FloatTensor] = None
    distill_loss: Optional[torch.FloatTensor] = None
    writer_loss: Optional[torch.FloatTensor] = None
    layer_geometry_losses: Optional[torch.FloatTensor] = None
    mean_geometry_loss: Optional[torch.FloatTensor] = None
    variance_loss: Optional[torch.FloatTensor] = None
    byte_acc: Optional[torch.FloatTensor] = None
    token_exact: Optional[torch.FloatTensor] = None


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
        self.encoder_layers = nn.ModuleList([DilLayer(config) for _ in range(config.num_encoder_layers)])
        self.num_stage_layers = config.num_encoder_layers // 2
        self.token_squeeze = nn.Linear(config.max_word_bytes * config.hidden_size, config.hidden_size)
        self.context_offset_embeddings = nn.Embedding(config.context_size, config.hidden_size)
        self.context_norm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.target_norm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.context_q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.context_k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.context_v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.context_out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.context_gate = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.hidden_to_semantic = nn.Linear(config.hidden_size, config.latent_size)
        self.norm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        indices = torch.arange(config.context_size)
        self.register_buffer("context_indices", indices[indices != self.target_index], persistent=False)

    def pooled_target_vector(self, hidden_states: torch.Tensor, word_masks: torch.Tensor) -> torch.Tensor:
        target_states = hidden_states[:, self.target_index]
        target_masks = word_masks[:, self.target_index].unsqueeze(-1).to(target_states.dtype)
        denom = target_masks.sum(dim=1).clamp_min(1.0)
        return (target_states * target_masks).sum(dim=1) / denom

    def target_conditioned_by_context(self, token_states: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
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
        scores = torch.einsum("bhd,bchd->bhc", query, keys) / (self.context_head_dim**0.5)
        context_mask = context_mask.unsqueeze(1)
        scores = scores.masked_fill(~context_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores.float(), dim=-1).to(scores.dtype).masked_fill(~context_mask, 0.0)
        context_delta = torch.einsum("bhc,bchd->bhd", attention, values).reshape(batch_size, -1)
        context_delta = self.context_out_proj(context_delta)
        gate_input = torch.cat(
            [target_state, context_delta, target_state * context_delta, target_state - context_delta],
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
            raise ValueError(f"input_ids context width {context_size} != context_size {self.context_size}")
        if byte_width != self.max_word_bytes:
            raise ValueError(f"input_ids width {byte_width} != max_word_bytes {self.max_word_bytes}")

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
        semantic = self.hidden_to_semantic(hidden_states)
        if output_hidden_states:
            return semantic, tuple(layer_vectors)
        return semantic


class DilConditionalWriter(nn.Module):
    def __init__(self, config: DilConfig):
        super().__init__()
        if config.hidden_size % config.writer_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by writer_attention_heads")
        self.max_word_bytes = config.max_word_bytes
        self.pad_token_id = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.decoder_start_token_id = config.decoder_start_token_id
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_word_bytes, config.hidden_size)
        self.semantic_memory_proj = nn.Linear(config.latent_size, config.hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.writer_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.writer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.writer_num_layers)
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.writer_dropout)

    def decoder_inputs_from_labels(self, labels: torch.LongTensor) -> torch.LongTensor:
        decoder_input_ids = torch.full_like(labels, self.pad_token_id)
        decoder_input_ids[:, 0] = self.decoder_start_token_id
        if labels.shape[1] > 1:
            previous = labels[:, :-1].masked_fill(labels[:, :-1].eq(-100), self.pad_token_id)
            decoder_input_ids[:, 1:] = previous
        return decoder_input_ids

    def forward(self, semantic: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        if labels.dim() != 2:
            raise ValueError("labels must be shaped [batch, target_length]")
        if labels.shape[1] != self.max_word_bytes:
            raise ValueError(f"labels width {labels.shape[1]} != max_word_bytes {self.max_word_bytes}")
        decoder_input_ids = self.decoder_inputs_from_labels(labels)
        positions = torch.arange(labels.shape[1], device=labels.device).unsqueeze(0)
        hidden_states = self.token_embeddings(decoder_input_ids) + self.position_embeddings(positions)
        hidden_states = self.dropout(hidden_states)
        memory = self.semantic_memory_proj(semantic).unsqueeze(1)
        target_mask = torch.triu(
            torch.ones(labels.shape[1], labels.shape[1], dtype=torch.bool, device=labels.device),
            diagonal=1,
        )
        target_padding_mask = decoder_input_ids.eq(self.pad_token_id)
        hidden_states = self.decoder(
            tgt=hidden_states,
            memory=memory,
            tgt_mask=target_mask,
            tgt_key_padding_mask=target_padding_mask,
        )
        hidden_states = self.final_norm(hidden_states)
        return F.linear(hidden_states, self.token_embeddings.weight)

    @torch.no_grad()
    def generate(self, semantic: torch.Tensor) -> tuple[torch.LongTensor, torch.Tensor, torch.LongTensor]:
        batch_size = semantic.shape[0]
        generated = torch.full(
            (batch_size, self.max_word_bytes),
            self.pad_token_id,
            dtype=torch.long,
            device=semantic.device,
        )
        labels = torch.full_like(generated, -100)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=semantic.device)
        for idx in range(self.max_word_bytes):
            logits = self.forward(semantic, labels)
            next_ids = logits[:, idx].argmax(dim=-1)
            next_ids = torch.where(finished, torch.full_like(next_ids, self.pad_token_id), next_ids)
            generated[:, idx] = next_ids
            labels[:, idx] = next_ids
            finished = finished | next_ids.eq(self.eos_token_id)
        eos_hits = generated.eq(self.eos_token_id)
        has_eos = eos_hits.any(dim=-1)
        first_eos = eos_hits.float().argmax(dim=-1).long()
        lengths = torch.where(has_eos, first_eos, torch.full_like(first_eos, self.max_word_bytes))
        mask = torch.arange(self.max_word_bytes, device=semantic.device).unsqueeze(0) < lengths.unsqueeze(1)
        return generated.masked_fill(~mask, self.pad_token_id), mask, lengths


class Dil(PreTrainedModel):
    config_class = DilConfig

    def __init__(self, config):
        super().__init__(config)
        if config.checkpoint_format_version != 17:
            raise ValueError("Dil SONAR-style encoder-decoder checkpoints require checkpoint_format_version=17")
        if config.pad_token_id >= config.vocab_size:
            raise ValueError("pad_token_id must be inside the tokenizer vocabulary")
        if config.eos_token_id >= config.vocab_size:
            raise ValueError("eos_token_id must be inside the tokenizer vocabulary")
        if config.decoder_start_token_id >= config.vocab_size:
            raise ValueError("decoder_start_token_id must be inside the tokenizer vocabulary")

        self.encoder = DilEncoderCore(config)
        self.writer = DilConditionalWriter(config)
        self.dil_dropout = config.dil_dropout
        self.distillation_weight = config.distillation_weight
        self.layer_geometry_weight = config.layer_geometry_weight
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
            return compiled_forward(input_ids, word_masks, output_hidden_states)
        return self.encoder(input_ids=input_ids, word_masks=word_masks, output_hidden_states=output_hidden_states)

    def writer_logits(self, semantic: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        compiled_forward = getattr(self, "_compiled_writer_forward", None)
        if compiled_forward is not None:
            return compiled_forward(semantic, labels)
        return self.writer(semantic, labels)

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

    def writer_metrics(self, logits: torch.Tensor, labels: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = labels.ne(-100)
        predictions = logits.argmax(dim=-1)
        byte_acc = (predictions.eq(labels) & valid).sum().float() / valid.sum().clamp_min(1).float()
        row_valid = valid.any(dim=-1)
        exact = ((predictions.eq(labels) | ~valid).all(dim=-1) & row_valid).sum().float()
        token_exact = exact / row_valid.sum().clamp_min(1).float()
        return byte_acc, token_exact

    def forward(
        self,
        input_ids: torch.LongTensor,
        word_masks: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        teacher_layers: Optional[torch.Tensor] = None,
        teacher_mask: Optional[torch.Tensor] = None,
    ) -> DilOutput:
        encoder_masks = word_masks
        if self.training and self.dil_dropout > 0:
            keep = torch.rand_like(word_masks.float()) >= self.dil_dropout
            encoder_masks = word_masks * keep.to(word_masks.dtype)

        semantic, layer_vectors = self.encode(input_ids=input_ids, word_masks=encoder_masks, output_hidden_states=True)
        semantic = semantic.float()
        loss = semantic.new_zeros(())
        distill_loss = semantic.new_zeros(())
        layer_geometry_losses = semantic.new_zeros((0,))
        mean_geometry_loss = semantic.new_zeros(())
        variance_loss = semantic.new_zeros(())

        if teacher_layers is not None:
            teacher_layers = teacher_layers.to(semantic.device, dtype=torch.float32)
            if teacher_mask is None:
                teacher_mask = torch.ones(teacher_layers.shape[0], dtype=torch.bool, device=semantic.device)
            else:
                teacher_mask = teacher_mask.to(semantic.device, dtype=torch.bool)
            layer_count = min(len(layer_vectors), teacher_layers.shape[1])
            losses = [
                self.geometry_loss(layer_vectors[idx].float(), teacher_layers[:, idx], teacher_mask)
                for idx in range(layer_count)
            ]
            layer_geometry_losses = torch.stack(losses) if losses else semantic.new_zeros((0,))
            mean_geometry_loss = self.geometry_loss(semantic, teacher_layers[:, layer_count - 1], teacher_mask)
            variance_terms = [self.variance_regularizer(semantic, teacher_mask)]
            variance_terms.extend(self.variance_regularizer(layer_vectors[idx].float(), teacher_mask) for idx in range(layer_count))
            variance_loss = torch.stack(variance_terms).mean()
            distill_loss = (
                layer_geometry_losses.mean() * self.layer_geometry_weight
                + mean_geometry_loss * self.mean_geometry_weight
                + variance_loss * self.variance_weight
            )
            loss = loss + distill_loss * self.distillation_weight

        writer_loss = semantic.new_zeros(())
        byte_acc = semantic.new_zeros(())
        token_exact = semantic.new_zeros(())
        if labels is not None and self.writer_loss_weight > 0.0:
            writer_semantic = semantic.detach()
            logits = self.writer_logits(writer_semantic, labels.to(semantic.device)).float()
            writer_loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.to(logits.device).reshape(-1),
                ignore_index=-100,
            )
            byte_acc, token_exact = self.writer_metrics(logits, labels.to(logits.device))
            loss = loss + writer_loss * self.writer_loss_weight

        return DilOutput(
            loss=loss,
            semantic=semantic,
            distill_loss=distill_loss,
            writer_loss=writer_loss,
            layer_geometry_losses=layer_geometry_losses,
            mean_geometry_loss=mean_geometry_loss,
            variance_loss=variance_loss,
            byte_acc=byte_acc,
            token_exact=token_exact,
        )

    @torch.no_grad()
    def decode_semantic(self, semantic: torch.Tensor) -> tuple[torch.LongTensor, torch.Tensor, torch.LongTensor]:
        return self.writer.generate(semantic)
