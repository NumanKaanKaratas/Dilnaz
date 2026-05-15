from __future__ import annotations

from collections.abc import Sequence

import torch

from dilnaz.models.dil import DilConfig
from dilnaz.surface import PackedWriterTarget, pack_writer_targets


def build_writer_unit_view(
    unit_rows: Sequence[Sequence[Sequence[int]]],
    config: DilConfig,
) -> dict:
    target_rows: list[list[list[int]]] = []
    source_rows: list[int] = []
    unit_indices: list[int] = []

    for row_idx, units in enumerate(unit_rows):
        for unit_idx, unit in enumerate(units):
            if unit:
                target_rows.append([list(unit)])
                source_rows.append(row_idx)
                unit_indices.append(unit_idx)

    if not target_rows:
        raise ValueError("writer unit view has no trainable units")

    return {
        "writer_labels": pack_writer_targets(
            target_rows,
            pad_token_id=config.pad_token_id,
            bos_token_id=config.decoder_start_token_id,
            stop_token_id=config.eos_token_id,
            surface_bucket_sizes=config.surface_bucket_sizes,
            max_pieces_per_unit=config.max_surface_pieces_per_unit,
        ),
        "writer_source_rows": torch.tensor(source_rows, dtype=torch.long),
        "writer_unit_indices": torch.tensor(unit_indices, dtype=torch.long),
    }


def gather_writer_semantic(
    semantic: torch.Tensor,
    source_rows: torch.Tensor,
    unit_indices: torch.Tensor,
) -> torch.Tensor:
    return semantic[
        source_rows.to(semantic.device, dtype=torch.long),
        unit_indices.to(semantic.device, dtype=torch.long),
    ]


def writer_unit_counts(target: PackedWriterTarget) -> tuple[int, int]:
    return int(target.label_mask.sum().detach().cpu()), int(target.true_lengths.gt(0).sum().detach().cpu())
