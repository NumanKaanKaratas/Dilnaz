from __future__ import annotations

import torch

from .types import PackedSurface


def generated_unit_tensors(
    generated: torch.LongTensor,
    query: PackedSurface,
    *,
    stop_token_id: int,
    pad_token_id: int,
) -> tuple[torch.LongTensor, torch.BoolTensor, torch.LongTensor]:
    batch_size, unit_count = query.unit_lengths.shape
    max_length = int(query.unit_lengths.max().detach().cpu().item())
    ids = torch.full((batch_size, unit_count, max(max_length - 1, 1)), pad_token_id, dtype=torch.long, device=generated.device)
    masks = torch.zeros_like(ids, dtype=torch.bool)
    lengths = torch.zeros((batch_size, unit_count), dtype=torch.long, device=generated.device)
    for row_idx in range(batch_size):
        for unit_idx in range(unit_count):
            start = int(query.unit_offsets[row_idx, unit_idx].detach().cpu())
            width = int(query.unit_lengths[row_idx, unit_idx].detach().cpu())
            if width <= 0:
                continue
            values = generated[row_idx, start : start + width]
            stop_hits = values.eq(stop_token_id)
            if bool(stop_hits.any().detach().cpu()):
                length = int(stop_hits.float().argmax().detach().cpu())
            else:
                length = max(width - 1, 0)
            lengths[row_idx, unit_idx] = length
            if length:
                ids[row_idx, unit_idx, :length] = values[:length]
                masks[row_idx, unit_idx, :length] = True
    return ids, masks, lengths


def decode_packed_units(tokenizer, ids: torch.LongTensor, masks: torch.BoolTensor) -> list[list[str]]:
    rows: list[list[str]] = []
    for row_ids, row_mask in zip(ids.detach().cpu(), masks.detach().cpu()):
        row: list[str] = []
        for unit_ids, unit_mask in zip(row_ids, row_mask):
            row.append(tokenizer.decode(unit_ids[unit_mask].tolist()))
        rows.append(row)
    return rows
