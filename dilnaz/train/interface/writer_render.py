from __future__ import annotations

import torch

from dilnaz.models.dil import Dil, DilConfig


def _future_horizons(config: DilConfig) -> int:
    return min(config.writer_right_guard, config.writer_sliding_window_size - 1)


def _build_future_latents(
    config: DilConfig,
    semantic: torch.Tensor,
    window_mask: torch.Tensor,
) -> torch.Tensor | None:
    horizons = _future_horizons(config)
    if horizons <= 0:
        return None
    batch_size, window_size, latent_size = semantic.shape
    future = semantic.new_zeros((batch_size, window_size, horizons, latent_size))
    valid_count = 0
    for h in range(horizons):
        offset = h + 1
        if offset >= window_size:
            break
        future[:, :-offset, h] = semantic[:, offset:] * window_mask[:, offset:].unsqueeze(-1).to(semantic.dtype)
        valid_count += 1
    if valid_count == 0:
        return None
    valid = future.float().norm(dim=-1).gt(1e-6)
    if not valid.any():
        return None
    return future


def _decode_window(
    model: Dil,
    semantic: torch.Tensor,
    zone_ids: torch.Tensor,
    window_mask: torch.Tensor,
    future_latents: torch.Tensor | None,
    position_age: torch.Tensor,
):
    generation = model.decode_semantic_window(
        semantic,
        zone_ids=zone_ids,
        window_mask=window_mask,
        future_latents=future_latents,
        position_age=position_age,
    )
    return generation


def _decode_token_ids(tokenizer, token_ids: torch.Tensor, token_mask: torch.Tensor) -> str:
    ids = token_ids[token_mask].detach().cpu().tolist()
    return tokenizer.decode(ids)


def render_latents_with_sliding_writer(
    model: Dil,
    tokenizer,
    latents: torch.Tensor,
    unit_mask: torch.Tensor | None = None,
    *,
    stride: int | None = None,
) -> list[str]:
    config = model.config
    window_size = config.writer_sliding_window_size
    left_frozen = config.writer_left_frozen
    active_size = config.writer_active_size
    if stride is None:
        stride = config.writer_stride
    max_position_age = config.writer_max_position_age

    if latents.dim() == 2:
        latents = latents.unsqueeze(0)
    if latents.dim() != 3 or latents.shape[0] != 1:
        raise ValueError(f"latents must be shaped [units, latent] or [1, units, latent], got {latents.shape}")
    latents = latents[0]
    device = latents.device
    dtype = latents.dtype

    if unit_mask is None:
        unit_mask = torch.ones(latents.shape[0], dtype=torch.bool, device=device)
    else:
        unit_mask = unit_mask.to(device, dtype=torch.bool)
    if unit_mask.shape[0] != latents.shape[0]:
        raise ValueError("unit_mask must match latents unit dimension")

    unit_count = latents.shape[0]

    zone_template = torch.full((window_size,), 1, dtype=torch.long, device=device)
    zone_template[:left_frozen] = 0
    zone_template[left_frozen + active_size:] = 2

    output_tokens: list[str] = []
    window_semantic = torch.zeros((1, window_size, config.latent_size), dtype=dtype, device=device)
    window_mask_tensor = torch.zeros((1, window_size), dtype=torch.bool, device=device)
    window_zone = zone_template.unsqueeze(0)
    window_age = torch.zeros((1, window_size), dtype=torch.long, device=device)

    for active_start in range(0, unit_count, stride):
        window_start = active_start - left_frozen

        window_semantic.zero_()
        window_mask_tensor.zero_()
        window_age.zero_()

        for slot in range(window_size):
            unit_idx = window_start + slot
            if 0 <= unit_idx < unit_count and unit_mask[unit_idx]:
                window_semantic[0, slot] = latents[unit_idx]
                window_mask_tensor[0, slot] = True
                if slot < left_frozen:
                    window_age[0, slot] = max_position_age

        future_latents = _build_future_latents(config, window_semantic, window_mask_tensor)
        generation = _decode_window(model, window_semantic, window_zone, window_mask_tensor, future_latents, window_age)

        for slot in range(left_frozen, window_size):
            unit_idx = window_start + slot
            if unit_idx >= unit_count:
                break
            if slot >= left_frozen + active_size:
                if unit_idx < unit_count:
                    continue
                break
            if not unit_mask[unit_idx]:
                continue
            token = _decode_token_ids(tokenizer, generation.token_ids[0, slot], generation.token_mask[0, slot])
            if token:
                output_tokens.append(token)

        if window_start + window_size >= unit_count:
            break

    return output_tokens
