from __future__ import annotations

from collections.abc import Sequence

import torch

from dilnaz.tokenization import TokenSegment

from .buckets import choose_bucket_size
from .types import PackedSurface, PackedWriterTarget


def segment_piece_ids(segment: TokenSegment | None) -> list[int]:
    if segment is None:
        return []
    return [piece.token_id for piece in segment.pieces]


def _validate_unit(ids: Sequence[int], max_pieces_per_unit: int) -> None:
    if len(ids) > max_pieces_per_unit:
        raise ValueError(f"surface unit has {len(ids)} pieces; max_surface_pieces_per_unit={max_pieces_per_unit}")


def pack_token_units(
    rows: Sequence[Sequence[Sequence[int]]],
    *,
    pad_token_id: int,
    bucket_sizes: Sequence[int],
    max_pieces_per_unit: int,
    device: torch.device | None = None,
) -> PackedSurface:
    batch_size = len(rows)
    if batch_size == 0:
        raise ValueError("cannot pack an empty batch")
    unit_count = max((len(row) for row in rows), default=0)
    if unit_count == 0:
        raise ValueError("cannot pack rows with no surface units")

    required_width = 0
    lengths = torch.zeros((batch_size, unit_count), dtype=torch.long)
    for row_idx, row in enumerate(rows):
        offset = 0
        for unit_idx, ids in enumerate(row):
            _validate_unit(ids, max_pieces_per_unit)
            lengths[row_idx, unit_idx] = len(ids)
            offset += len(ids)
        required_width = max(required_width, offset)
    width = choose_bucket_size(max(required_width, 1), tuple(bucket_sizes))

    ids_tensor = torch.full((batch_size, width), pad_token_id, dtype=torch.long)
    mask = torch.zeros((batch_size, width), dtype=torch.bool)
    unit_ids = torch.zeros((batch_size, width), dtype=torch.long)
    pos_in_unit = torch.zeros((batch_size, width), dtype=torch.long)
    offsets = torch.zeros((batch_size, unit_count + 1), dtype=torch.long)
    unit_mask = lengths.gt(0)

    for row_idx, row in enumerate(rows):
        cursor = 0
        for unit_idx, ids in enumerate(row):
            offsets[row_idx, unit_idx] = cursor
            width_i = len(ids)
            if width_i:
                end = cursor + width_i
                ids_tensor[row_idx, cursor:end] = torch.tensor(ids, dtype=torch.long)
                mask[row_idx, cursor:end] = True
                unit_ids[row_idx, cursor:end] = unit_idx
                pos_in_unit[row_idx, cursor:end] = torch.arange(width_i, dtype=torch.long)
                cursor = end
        offsets[row_idx, len(row) :] = cursor

    packed = PackedSurface(
        ids=ids_tensor,
        mask=mask,
        unit_ids=unit_ids,
        pos_in_unit=pos_in_unit,
        unit_lengths=lengths,
        unit_offsets=offsets,
        unit_mask=unit_mask,
    )
    return packed if device is None else packed.to(device)


def pack_segment_units(
    rows: Sequence[Sequence[TokenSegment | None]],
    *,
    pad_token_id: int,
    bucket_sizes: Sequence[int],
    max_pieces_per_unit: int,
    device: torch.device | None = None,
) -> PackedSurface:
    return pack_token_units(
        [[segment_piece_ids(segment) for segment in row] for row in rows],
        pad_token_id=pad_token_id,
        bucket_sizes=bucket_sizes,
        max_pieces_per_unit=max_pieces_per_unit,
        device=device,
    )


def writer_query_from_lengths(
    unit_lengths: torch.LongTensor,
    *,
    pad_token_id: int,
    surface_bucket_sizes: Sequence[int],
) -> PackedSurface:
    if unit_lengths.dim() != 2:
        raise ValueError("unit_lengths must be shaped [batch, units]")
    batch_size, unit_count = unit_lengths.shape
    required_width = int(unit_lengths.sum(dim=1).max().detach().cpu())
    width = choose_bucket_size(max(required_width, 1), tuple(surface_bucket_sizes))
    device = unit_lengths.device
    ids = torch.full((batch_size, width), pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros((batch_size, width), dtype=torch.bool, device=device)
    unit_ids = torch.zeros((batch_size, width), dtype=torch.long, device=device)
    pos_in_unit = torch.zeros((batch_size, width), dtype=torch.long, device=device)
    offsets = torch.zeros((batch_size, unit_count + 1), dtype=torch.long, device=device)
    for row_idx in range(batch_size):
        cursor = 0
        for unit_idx in range(unit_count):
            length = int(unit_lengths[row_idx, unit_idx].detach().cpu())
            offsets[row_idx, unit_idx] = cursor
            if length:
                end = cursor + length
                mask[row_idx, cursor:end] = True
                unit_ids[row_idx, cursor:end] = unit_idx
                pos_in_unit[row_idx, cursor:end] = torch.arange(length, dtype=torch.long, device=device)
                cursor = end
        offsets[row_idx, unit_count] = cursor
    return PackedSurface(
        ids=ids,
        mask=mask,
        unit_ids=unit_ids,
        pos_in_unit=pos_in_unit,
        unit_lengths=unit_lengths.long(),
        unit_offsets=offsets,
        unit_mask=unit_lengths.gt(0),
    )


def pack_writer_targets(
    rows: Sequence[Sequence[Sequence[int]]],
    *,
    pad_token_id: int,
    bos_token_id: int,
    stop_token_id: int,
    surface_bucket_sizes: Sequence[int],
    max_pieces_per_unit: int,
    device: torch.device | None = None,
) -> PackedWriterTarget:
    batch_size = len(rows)
    if batch_size == 0:
        raise ValueError("cannot pack an empty writer target batch")
    unit_count = max((len(row) for row in rows), default=0)
    true_lengths = torch.zeros((batch_size, unit_count), dtype=torch.long)
    for row_idx, row in enumerate(rows):
        for unit_idx, ids in enumerate(row):
            _validate_unit(ids, max_pieces_per_unit)
            if ids:
                true_lengths[row_idx, unit_idx] = len(ids) + 1
    query = writer_query_from_lengths(
        true_lengths,
        pad_token_id=pad_token_id,
        surface_bucket_sizes=surface_bucket_sizes,
    )
    labels = torch.full_like(query.ids, -100)
    label_mask = torch.zeros_like(query.mask)
    for row_idx, row in enumerate(rows):
        for unit_idx, ids in enumerate(row):
            start = int(query.unit_offsets[row_idx, unit_idx])
            if not ids:
                continue
            input_ids = [bos_token_id] + list(ids)
            ids_with_stop = list(ids) + [stop_token_id]
            end = start + len(ids_with_stop)
            query.ids[row_idx, start:end] = torch.tensor(input_ids, dtype=torch.long)
            labels[row_idx, start:end] = torch.tensor(ids_with_stop, dtype=torch.long)
            label_mask[row_idx, start:end] = True
    target = PackedWriterTarget(
        query=query,
        labels=labels,
        label_mask=label_mask,
        true_lengths=true_lengths,
    )
    return target if device is None else target.to(device)
