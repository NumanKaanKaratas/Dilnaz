from __future__ import annotations

from collections.abc import Sequence

import torch

from dilnaz.models.dil import DilConfig
from dilnaz.surface import PackedWriterTarget, pack_writer_targets


def build_writer_window_view(
    unit_rows: Sequence[Sequence[Sequence[int]]],
    config: DilConfig,
) -> dict:
    target_rows: list[list[list[int]]] = []
    source_rows: list[int] = []
    unit_indices = []
    window_mask_rows = []
    window_size = config.writer_sliding_window_size
    left_frozen = config.writer_left_frozen
    active_size = config.writer_active_size
    right_guard = config.writer_right_guard
    stride = config.writer_stride
    if left_frozen + active_size + right_guard != window_size:
        raise ValueError("writer window zones must sum to writer_sliding_window_size")

    for row_idx, units in enumerate(unit_rows):
        if not units:
            continue
        for active_start in range(0, len(units), stride):
            window_start = active_start - left_frozen
            target_row: list[list[int]] = []
            index_row: list[int] = []
            mask_row: list[bool] = []
            for window_idx in range(window_size):
                unit_idx = window_start + window_idx
                if 0 <= unit_idx < len(units):
                    target_row.append(list(units[unit_idx]))
                    index_row.append(unit_idx)
                    mask_row.append(True)
                else:
                    target_row.append([])
                    index_row.append(0)
                    mask_row.append(False)
            target_rows.append(target_row)
            source_rows.append(row_idx)
            unit_indices.append(index_row)
            window_mask_rows.append(mask_row)

    if not target_rows:
        raise ValueError("writer window view has no trainable windows")

    zone_template = torch.full((window_size,), 1, dtype=torch.long)
    zone_template[:left_frozen] = 0
    zone_template[left_frozen + active_size :] = 2
    window_count = len(target_rows)
    return {
        "writer_labels": pack_writer_targets(
            target_rows,
            pad_token_id=config.pad_token_id,
            stop_token_id=config.writer_stop_token_id,
            bos_token_id=config.writer_bos_token_id,
            empty_token_id=config.writer_empty_token_id,
            surface_bucket_sizes=config.surface_bucket_sizes,
            max_pieces_per_unit=config.max_surface_pieces_per_unit,
        ),
        "writer_source_rows": torch.tensor(source_rows, dtype=torch.long),
        "writer_unit_indices": torch.tensor(unit_indices, dtype=torch.long),
        "writer_zone_ids": zone_template.unsqueeze(0).expand(window_count, -1).clone(),
        "writer_window_mask": torch.tensor(window_mask_rows, dtype=torch.bool),
    }


def gather_writer_semantic(
    semantic: torch.Tensor,
    source_rows: torch.Tensor,
    unit_indices: torch.Tensor,
    window_mask: torch.Tensor,
) -> torch.Tensor:
    source_rows = source_rows.to(semantic.device, dtype=torch.long)
    unit_indices = unit_indices.to(semantic.device, dtype=torch.long)
    window_mask = window_mask.to(semantic.device, dtype=torch.bool)
    gathered = semantic[source_rows.unsqueeze(1), unit_indices]
    return gathered * window_mask.unsqueeze(-1).to(gathered.dtype)


def writer_window_counts(target: PackedWriterTarget) -> tuple[int, int]:
    return int(target.label_mask.sum().detach().cpu()), int(target.true_lengths.gt(0).sum().detach().cpu())
