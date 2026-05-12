from __future__ import annotations

import torch

from .packing import writer_query_from_lengths
from .types import PackedSurface, PackedSurfaceState, PackedWriterTarget


STATE_EMPTY = 0
STATE_DRAFT = 1
STATE_KNOWN = 2


def _state_from_query(
    query: PackedSurface,
    *,
    ids: torch.Tensor,
    state_kind: torch.Tensor,
    frozen: torch.Tensor,
    mask: torch.Tensor,
) -> PackedSurfaceState:
    return PackedSurfaceState(
        ids=ids,
        state_kind=state_kind,
        frozen=frozen,
        mask=mask,
        unit_ids=query.unit_ids,
        pos_in_unit=query.pos_in_unit,
        unit_lengths=query.unit_lengths,
        unit_offsets=query.unit_offsets,
        unit_mask=query.unit_mask,
    )


def empty_surface_state(query: PackedSurface, empty_token_id: int) -> PackedSurfaceState:
    return _state_from_query(
        query,
        ids=torch.full_like(query.ids, empty_token_id),
        state_kind=torch.zeros_like(query.ids),
        frozen=torch.zeros_like(query.mask),
        mask=torch.zeros_like(query.mask),
    )


def known_surface_state(target: PackedWriterTarget, empty_token_id: int) -> PackedSurfaceState:
    ids = torch.where(target.label_mask, target.labels, torch.full_like(target.labels, empty_token_id))
    return _state_from_query(
        target.query,
        ids=ids,
        state_kind=torch.where(target.label_mask, torch.full_like(target.labels, STATE_KNOWN), torch.zeros_like(target.labels)),
        frozen=target.label_mask.clone(),
        mask=target.label_mask.clone(),
    )


def synthetic_state_from_targets(
    target: PackedWriterTarget,
    *,
    zone_ids: torch.LongTensor,
    window_mask: torch.BoolTensor,
    empty_token_id: int,
    vocab_size: int,
    mask_ratio: float,
    draft_max_ratio: float,
) -> PackedSurfaceState:
    query = target.query
    device = target.labels.device
    zone_per_pos = zone_ids.gather(1, query.unit_ids.clamp_max(zone_ids.shape[1] - 1))
    window_per_pos = window_mask.gather(1, query.unit_ids.clamp_max(window_mask.shape[1] - 1))
    valid = target.label_mask & window_per_pos
    left = zone_per_pos.eq(0) & valid
    target_scope = valid & ~left
    draft_ratio = min(draft_max_ratio, max(0.0, 1.0 - mask_ratio))
    draft = target_scope & torch.rand(target.labels.shape, device=device).lt(draft_ratio)
    random_tokens = torch.randint(vocab_size, target.labels.shape, device=device, dtype=torch.long)
    ids = torch.full_like(target.labels, empty_token_id)
    state_kind = torch.zeros_like(target.labels)
    frozen = torch.zeros_like(target.label_mask)
    ids[draft] = random_tokens[draft]
    ids[left] = target.labels[left]
    state_kind[draft] = STATE_DRAFT
    state_kind[left] = STATE_KNOWN
    frozen[left] = True
    return _state_from_query(
        query,
        ids=ids,
        state_kind=state_kind,
        frozen=frozen,
        mask=draft | left,
    )


def merge_frozen_state(base: PackedSurfaceState, draft_ids: torch.Tensor, draft_mask: torch.Tensor) -> PackedSurfaceState:
    ids = torch.where(base.frozen, base.ids, draft_ids)
    state_kind = torch.where(base.frozen, base.state_kind, torch.full_like(base.state_kind, STATE_DRAFT))
    state_kind = torch.where(draft_mask | base.frozen, state_kind, torch.zeros_like(state_kind))
    return PackedSurfaceState(
        ids=ids,
        state_kind=state_kind,
        frozen=base.frozen,
        mask=draft_mask | base.frozen,
        unit_ids=base.unit_ids,
        pos_in_unit=base.pos_in_unit,
        unit_lengths=base.unit_lengths,
        unit_offsets=base.unit_offsets,
        unit_mask=base.unit_mask,
    )
