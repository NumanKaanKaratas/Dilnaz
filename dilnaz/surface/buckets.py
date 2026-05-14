from __future__ import annotations


def choose_bucket_size(required: int, buckets: tuple[int, ...] | list[int]) -> int:
    for bucket in buckets:
        if required <= int(bucket):
            return int(bucket)
    raise ValueError(f"surface length {required} exceeds largest bucket {max(buckets)}")
