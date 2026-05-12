from __future__ import annotations

import torch


def gather_unit_values(values: torch.Tensor, unit_ids: torch.LongTensor) -> torch.Tensor:
    expanded = unit_ids.clamp_min(0).unsqueeze(-1).expand(*unit_ids.shape, values.shape[-1])
    return values.gather(dim=1, index=expanded)


def scatter_sum_by_unit(values: torch.Tensor, unit_ids: torch.LongTensor, unit_count: int, mask: torch.Tensor) -> torch.Tensor:
    batch_size, _, hidden_size = values.shape
    index = unit_ids.clamp_min(0).unsqueeze(-1).expand(-1, -1, hidden_size)
    source = values * mask.unsqueeze(-1).to(values.dtype)
    output = values.new_zeros((batch_size, unit_count, hidden_size))
    output.scatter_add_(1, index, source)
    return output


def scatter_mean_by_unit(values: torch.Tensor, unit_ids: torch.LongTensor, unit_count: int, mask: torch.Tensor) -> torch.Tensor:
    summed = scatter_sum_by_unit(values, unit_ids, unit_count, mask)
    counts = values.new_zeros((values.shape[0], unit_count, 1))
    counts.scatter_add_(1, unit_ids.clamp_min(0).unsqueeze(-1), mask.unsqueeze(-1).to(values.dtype))
    return summed / counts.clamp_min(1.0)


def scatter_max_by_unit(values: torch.Tensor, unit_ids: torch.LongTensor, unit_count: int, mask: torch.Tensor) -> torch.Tensor:
    floor = torch.finfo(values.dtype).min
    source = values.masked_fill(~mask, floor)
    output = values.new_full((values.shape[0], unit_count), floor)
    output.scatter_reduce_(1, unit_ids.clamp_min(0), source, reduce="amax", include_self=True)
    return output


def scatter_softmax_by_unit(scores: torch.Tensor, unit_ids: torch.LongTensor, unit_count: int, mask: torch.Tensor) -> torch.Tensor:
    max_scores = scatter_max_by_unit(scores.float(), unit_ids, unit_count, mask)
    gathered_max = max_scores.gather(1, unit_ids.clamp_min(0))
    exp_scores = torch.exp((scores.float() - gathered_max).masked_fill(~mask, torch.finfo(torch.float32).min))
    sums = scores.new_zeros((scores.shape[0], unit_count), dtype=torch.float32)
    sums.scatter_add_(1, unit_ids.clamp_min(0), exp_scores)
    weights = exp_scores / sums.gather(1, unit_ids.clamp_min(0)).clamp_min(torch.finfo(torch.float32).tiny)
    return weights.to(scores.dtype).masked_fill(~mask, 0.0)
