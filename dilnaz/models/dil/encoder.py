import torch
from torch import nn

from ..common.norms import DilRMSNorm
from .layers import DilConvSwiGLUBlock, DilLayer


class DilByteConvStem(nn.Module):
    def __init__(self, config):
        super().__init__()
        intermediate_size = config.hidden_size * config.byte_conv_expansion
        self.layers = nn.ModuleList(
            [
                DilConvSwiGLUBlock(
                    config.hidden_size,
                    intermediate_size,
                    config.byte_conv_kernel_size,
                    config.rms_norm_eps,
                    config.mlp_bias,
                    config.dil_dropout,
                )
                for _ in range(config.byte_conv_layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor, word_masks: torch.Tensor) -> torch.Tensor:
        batch_size, context_size, byte_width, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size * context_size, byte_width, hidden_size)
        masks = word_masks.reshape(batch_size * context_size, byte_width)
        for layer in self.layers:
            hidden_states = layer(hidden_states, masks)
        return hidden_states.reshape(batch_size, context_size, byte_width, hidden_size)


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
        self.byte_stem = DilByteConvStem(config)
        self.encoder_layers = nn.ModuleList([DilLayer(config) for _ in range(config.num_encoder_layers)])
        self.num_stage_layers = config.num_encoder_layers // 2
        self.pool_norm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pool_score = nn.Linear(config.hidden_size, 1, bias=False)
        self.pool_value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
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

    def pool_token_states(self, hidden_states: torch.Tensor, word_masks: torch.Tensor) -> torch.Tensor:
        scores = self.pool_score(self.pool_norm(hidden_states)).squeeze(-1)
        scores = scores.masked_fill(~word_masks, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        attention = attention * word_masks.to(attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(attention.dtype).tiny)
        values = self.pool_value(hidden_states)
        return (values * attention.unsqueeze(-1)).sum(dim=2)

    def target_conditioned_by_context(self, token_states: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        target_state = token_states[:, self.target_index]
        context_states = token_states.index_select(1, self.context_indices).detach()
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
        hidden_states = self.byte_stem(hidden_states, word_masks)

        layer_vectors = [] if output_hidden_states else None
        for layer_idx in range(self.num_stage_layers):
            hidden_states = self.encoder_layers[layer_idx](hidden_states)
            if output_hidden_states:
                layer_vectors.append(self.pooled_target_vector(hidden_states, word_masks))

        token_states = self.pool_token_states(hidden_states, word_masks)
        offsets = torch.arange(context_size, device=token_states.device)
        token_states = token_states + self.context_offset_embeddings(offsets).unsqueeze(0)
        token_mask = word_masks.any(dim=-1)
        hidden_states = self.target_conditioned_by_context(token_states, token_mask)

        for layer_idx in range(self.num_stage_layers):
            encoder_idx = self.num_stage_layers + layer_idx
            hidden_states = self.encoder_layers[encoder_idx](hidden_states)
            if output_hidden_states:
                layer_vectors.append(hidden_states)

        hidden_states = self.norm(hidden_states)
        semantic = self.hidden_to_semantic(hidden_states)
        if output_hidden_states:
            return semantic, tuple(layer_vectors)
        return semantic
