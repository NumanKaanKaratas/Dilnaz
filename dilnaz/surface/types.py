from __future__ import annotations

from dataclasses import dataclass

import torch


def _move(value: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return value.to(*args, **kwargs)


@dataclass(frozen=True)
class PackedSurface:
    ids: torch.LongTensor
    mask: torch.BoolTensor
    unit_ids: torch.LongTensor
    pos_in_unit: torch.LongTensor
    unit_lengths: torch.LongTensor
    unit_offsets: torch.LongTensor
    unit_mask: torch.BoolTensor

    @property
    def batch_size(self) -> int:
        return int(self.ids.shape[0])

    @property
    def unit_count(self) -> int:
        return int(self.unit_lengths.shape[1])

    @property
    def surface_width(self) -> int:
        return int(self.ids.shape[1])

    def to(self, *args, **kwargs) -> "PackedSurface":
        return PackedSurface(
            ids=_move(self.ids, *args, **kwargs),
            mask=_move(self.mask, *args, **kwargs),
            unit_ids=_move(self.unit_ids, *args, **kwargs),
            pos_in_unit=_move(self.pos_in_unit, *args, **kwargs),
            unit_lengths=_move(self.unit_lengths, *args, **kwargs),
            unit_offsets=_move(self.unit_offsets, *args, **kwargs),
            unit_mask=_move(self.unit_mask, *args, **kwargs),
        )

    def detach(self) -> "PackedSurface":
        return PackedSurface(
            ids=self.ids.detach(),
            mask=self.mask.detach(),
            unit_ids=self.unit_ids.detach(),
            pos_in_unit=self.pos_in_unit.detach(),
            unit_lengths=self.unit_lengths.detach(),
            unit_offsets=self.unit_offsets.detach(),
            unit_mask=self.unit_mask.detach(),
        )

    def cpu(self) -> "PackedSurface":
        return self.to("cpu")


@dataclass(frozen=True)
class PackedSurfaceState:
    ids: torch.LongTensor
    state_kind: torch.LongTensor
    frozen: torch.BoolTensor
    mask: torch.BoolTensor
    unit_ids: torch.LongTensor
    pos_in_unit: torch.LongTensor
    unit_lengths: torch.LongTensor
    unit_offsets: torch.LongTensor
    unit_mask: torch.BoolTensor

    @property
    def batch_size(self) -> int:
        return int(self.ids.shape[0])

    @property
    def unit_count(self) -> int:
        return int(self.unit_lengths.shape[1])

    def to(self, *args, **kwargs) -> "PackedSurfaceState":
        return PackedSurfaceState(
            ids=_move(self.ids, *args, **kwargs),
            state_kind=_move(self.state_kind, *args, **kwargs),
            frozen=_move(self.frozen, *args, **kwargs),
            mask=_move(self.mask, *args, **kwargs),
            unit_ids=_move(self.unit_ids, *args, **kwargs),
            pos_in_unit=_move(self.pos_in_unit, *args, **kwargs),
            unit_lengths=_move(self.unit_lengths, *args, **kwargs),
            unit_offsets=_move(self.unit_offsets, *args, **kwargs),
            unit_mask=_move(self.unit_mask, *args, **kwargs),
        )

    def detach(self) -> "PackedSurfaceState":
        return PackedSurfaceState(
            ids=self.ids.detach(),
            state_kind=self.state_kind.detach(),
            frozen=self.frozen.detach(),
            mask=self.mask.detach(),
            unit_ids=self.unit_ids.detach(),
            pos_in_unit=self.pos_in_unit.detach(),
            unit_lengths=self.unit_lengths.detach(),
            unit_offsets=self.unit_offsets.detach(),
            unit_mask=self.unit_mask.detach(),
        )

    def cpu(self) -> "PackedSurfaceState":
        return self.to("cpu")


@dataclass(frozen=True)
class PackedWriterTarget:
    query: PackedSurface
    labels: torch.LongTensor
    label_mask: torch.BoolTensor
    true_lengths: torch.LongTensor

    def to(self, *args, **kwargs) -> "PackedWriterTarget":
        return PackedWriterTarget(
            query=self.query.to(*args, **kwargs),
            labels=_move(self.labels, *args, **kwargs),
            label_mask=_move(self.label_mask, *args, **kwargs),
            true_lengths=_move(self.true_lengths, *args, **kwargs),
        )

    def detach(self) -> "PackedWriterTarget":
        return PackedWriterTarget(
            query=self.query.detach(),
            labels=self.labels.detach(),
            label_mask=self.label_mask.detach(),
            true_lengths=self.true_lengths.detach(),
        )

    def cpu(self) -> "PackedWriterTarget":
        return self.to("cpu")
