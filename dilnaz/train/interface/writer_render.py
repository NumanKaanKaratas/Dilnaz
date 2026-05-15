from __future__ import annotations

import torch

from dilnaz.models.dil import Dil


def _decode_token_ids(tokenizer, token_ids: torch.Tensor, token_mask: torch.Tensor) -> str:
    ids = token_ids[token_mask].detach().cpu().tolist()
    return tokenizer.decode(ids)


def render_latents_with_unit_writer(
    model: Dil,
    tokenizer,
    latents: torch.Tensor,
    unit_mask: torch.Tensor | None = None,
    *,
    microbatch_size: int = 64,
) -> list[str]:
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be > 0")
    if latents.dim() == 2:
        latents = latents.unsqueeze(0)
    if latents.dim() != 3 or latents.shape[0] != 1:
        raise ValueError(f"latents must be shaped [units, latent] or [1, units, latent], got {latents.shape}")
    latents = latents[0]
    device = latents.device

    if unit_mask is None:
        unit_mask = torch.ones(latents.shape[0], dtype=torch.bool, device=device)
    else:
        unit_mask = unit_mask.to(device, dtype=torch.bool)
    if unit_mask.shape[0] != latents.shape[0]:
        raise ValueError("unit_mask must match latents unit dimension")

    valid_indices = unit_mask.nonzero(as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        return []

    output_tokens: list[str] = []
    for start in range(0, int(valid_indices.numel()), microbatch_size):
        indices = valid_indices[start : start + microbatch_size]
        semantic = latents.index_select(0, indices)
        batch_size = semantic.shape[0]
        generation = model.decode_semantic(semantic)
        for row_idx in range(batch_size):
            token = _decode_token_ids(
                tokenizer,
                generation.token_ids[row_idx],
                generation.token_mask[row_idx],
            )
            if token:
                output_tokens.append(token)
    return output_tokens
