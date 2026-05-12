from __future__ import annotations

import torch


def choose_bucket_size(required: int, buckets: tuple[int, ...] | list[int]) -> int:
    for bucket in buckets:
        if required <= int(bucket):
            return int(bucket)
    raise ValueError(f"surface length {required} exceeds largest bucket {max(buckets)}")


def bucketize_lengths(lengths: torch.Tensor, buckets: tuple[int, ...] | list[int]) -> torch.LongTensor:
    bucket_values = torch.tensor(tuple(int(bucket) for bucket in buckets), device=lengths.device, dtype=torch.long)
    fits = lengths.unsqueeze(-1) <= bucket_values.view(*([1] * lengths.dim()), -1)
    if not bool(fits.any(dim=-1).all().detach().cpu()):
        largest = int(bucket_values[-1].detach().cpu())
        raise ValueError(f"surface length exceeds largest writer output bucket {largest}")
    return fits.float().argmax(dim=-1).long()
