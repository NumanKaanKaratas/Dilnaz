import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from models.configuration_dil import DilConfig
from tokenization import HybridTokenizer


WHITESPACE_PATTERN = re.compile(r"\s+", re.UNICODE)
TOKEN_CACHE_FORMAT_VERSION = 2


def last_whitespace_end(text: str) -> int:
    boundary = -1
    for match in WHITESPACE_PATTERN.finditer(text):
        boundary = match.end()
    return boundary


def stream_text_blocks(path: Path, read_chars: int):
    carry = ""
    emitted = False
    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(read_chars)
            if not chunk:
                break
            text = carry + chunk
            boundary = last_whitespace_end(text)
            if boundary < 0:
                carry = text
                continue
            block = text[:boundary]
            if block.strip():
                emitted = True
                yield block
            carry = text[boundary:]
    if carry.strip():
        emitted = True
        yield carry
    if not emitted:
        raise ValueError(f"{path} produced no text blocks")


def stream_token_pieces(path: Path, tokenizer: HybridTokenizer, max_word_bytes: int, read_chars: int):
    for block in stream_text_blocks(path, read_chars):
        for segment in tokenizer.encode_segments(block):
            if segment.kind == "space" or segment.piece_len > max_word_bytes:
                continue
            token_ids = [piece.token_id for piece in segment.pieces]
            if not token_ids:
                continue
            yield token_ids


def vocab_fingerprint(tokenizer: HybridTokenizer) -> str:
    payload = {
        "vocab_size": tokenizer.vocab_size,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "token_to_id": tokenizer.token_to_id,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def token_cache_key(
    path: Path,
    tokenizer: HybridTokenizer,
    max_word_bytes: int,
    read_chars: int,
) -> str:
    stat = path.stat()
    payload = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "max_word_bytes": max_word_bytes,
        "read_chars": read_chars,
        "vocab": vocab_fingerprint(tokenizer),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def token_cache_paths(cache_dir: Path, key: str) -> tuple[Path, Path, Path]:
    return (
        cache_dir / f"{key}.ids.npy",
        cache_dir / f"{key}.lengths.npy",
        cache_dir / f"{key}.json",
    )


def build_token_cache(
    path: Path,
    tokenizer: HybridTokenizer,
    max_word_bytes: int,
    pad_token_id: int,
    read_chars: int,
    cache_dir: Path,
) -> tuple[Path, Path, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = token_cache_key(path, tokenizer, max_word_bytes, read_chars)
    ids_path, lengths_path, meta_path = token_cache_paths(cache_dir, key)
    if ids_path.exists() and lengths_path.exists() and meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        if meta["format_version"] == TOKEN_CACHE_FORMAT_VERSION:
            return ids_path, lengths_path, int(meta["token_count"])

    token_count = sum(1 for _ in stream_token_pieces(path, tokenizer, max_word_bytes, read_chars))
    if token_count < 2:
        raise ValueError(f"{path} needs at least two tokens for sequence training")
    byte_ids = np.lib.format.open_memmap(ids_path, mode="w+", dtype=np.uint16, shape=(token_count, max_word_bytes))
    lengths = np.lib.format.open_memmap(lengths_path, mode="w+", dtype=np.uint8, shape=(token_count,))
    byte_ids[:] = pad_token_id
    for token_idx, ids in enumerate(stream_token_pieces(path, tokenizer, max_word_bytes, read_chars)):
        width = len(ids)
        byte_ids[token_idx, :width] = np.asarray(ids, dtype=np.uint16)
        lengths[token_idx] = width
    byte_ids.flush()
    lengths.flush()

    stat = path.stat()
    meta = {
        "format_version": TOKEN_CACHE_FORMAT_VERSION,
        "source_path": str(path.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "token_count": token_count,
        "max_word_bytes": max_word_bytes,
        "pad_token_id": pad_token_id,
        "read_chars": read_chars,
        "vocab_size": tokenizer.vocab_size,
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    return ids_path, lengths_path, token_count


class RandomWindowNazDataset(IterableDataset):
    def __init__(
        self,
        ids_path: Path,
        lengths_path: Path,
        token_count: int,
        config: DilConfig,
        sequence_length: int,
        batch_size: int,
        seed: int,
    ):
        super().__init__()
        self.ids_path = ids_path
        self.lengths_path = lengths_path
        self.token_count = token_count
        self.max_word_bytes = config.max_word_bytes
        self.pad_token_id = config.pad_token_id
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.seed = seed
        self._byte_ids = None
        self._lengths = None
        self.offsets = np.arange(sequence_length, dtype=np.int64).reshape(1, sequence_length)
        self.positions = torch.arange(self.max_word_bytes).reshape(1, 1, self.max_word_bytes)

    @property
    def byte_ids(self):
        if self._byte_ids is None:
            self._byte_ids = np.load(self.ids_path, mmap_mode="r")
        return self._byte_ids

    @property
    def lengths(self):
        if self._lengths is None:
            self._lengths = np.load(self.lengths_path, mmap_mode="r")
        return self._lengths

    def make_batch(self, starts: list[int]):
        starts_array = np.asarray(starts, dtype=np.int64).reshape(-1, 1)
        source_idx = starts_array + self.offsets
        unit_mask_np = source_idx < (self.token_count - 1)
        source_idx = np.minimum(source_idx, self.token_count - 1)
        target_idx = np.minimum(source_idx + 1, self.token_count - 1)

        input_ids = torch.from_numpy(np.asarray(self.byte_ids[source_idx], dtype=np.int64))
        target_input_ids = torch.from_numpy(np.asarray(self.byte_ids[target_idx], dtype=np.int64))
        source_lengths = torch.from_numpy(np.asarray(self.lengths[source_idx], dtype=np.int64))
        target_lengths = torch.from_numpy(np.asarray(self.lengths[target_idx], dtype=np.int64))
        unit_mask = torch.from_numpy(unit_mask_np)
        word_masks = (self.positions < source_lengths.unsqueeze(-1)) & unit_mask.unsqueeze(-1)
        target_word_masks = (self.positions < target_lengths.unsqueeze(-1)) & unit_mask.unsqueeze(-1)
        return {
            "input_ids": input_ids,
            "word_masks": word_masks,
            "target_input_ids": target_input_ids,
            "target_word_masks": target_word_masks,
            "unit_mask": unit_mask,
        }

    def sample_start(self, rng: random.Random) -> int:
        if self.token_count <= self.sequence_length:
            return 0
        return rng.randint(0, self.token_count - self.sequence_length - 1)

    def iter_batches(self, rng: random.Random):
        while True:
            starts = [self.sample_start(rng) for _ in range(self.batch_size)]
            yield self.make_batch(starts)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        rng = random.Random(self.seed + worker_id * 1_000_003)
        yield from self.iter_batches(rng)


def make_naz_loader(dataset, num_workers: int, pin_memory: bool, prefetch_factor: int):
    loader_kwargs = {"batch_size": None, "num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = True
    return DataLoader(dataset, **loader_kwargs)


class ResidentNazBatcher:
    def __init__(
        self,
        ids_path: Path,
        lengths_path: Path,
        token_count: int,
        config: DilConfig,
        sequence_length: int,
        batch_size: int,
        device: torch.device,
        seed: int,
    ):
        self.byte_ids = torch.from_numpy(np.asarray(np.load(ids_path, mmap_mode="r"), dtype=np.int64)).to(device)
        self.lengths = torch.from_numpy(np.asarray(np.load(lengths_path, mmap_mode="r"), dtype=np.int64)).to(device)
        self.token_count = token_count
        self.max_word_bytes = config.max_word_bytes
        self.pad_token_id = config.pad_token_id
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.device = device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)
        self.positions = torch.arange(self.max_word_bytes, device=device).reshape(1, 1, self.max_word_bytes)
        self.offsets = torch.arange(sequence_length, device=device).reshape(1, sequence_length)
        if token_count < 2:
            raise ValueError("resident Naz data needs at least two tokens")

    def __iter__(self):
        return self

    def __next__(self):
        max_start = max(self.token_count - self.sequence_length - 1, 0)
        starts = torch.randint(
            max_start + 1,
            (self.batch_size, 1),
            generator=self.generator,
            device=self.device,
        )
        return self.make_batch(starts)

    def make_batch(self, starts: torch.Tensor):
        source_idx = starts + self.offsets
        unit_mask = source_idx < (self.token_count - 1)
        source_idx = source_idx.clamp_max(self.token_count - 1)
        target_idx = (source_idx + 1).clamp_max(self.token_count - 1)
        batch_size = starts.shape[0]

        input_ids = self.byte_ids.index_select(0, source_idx.reshape(-1)).reshape(
            batch_size,
            self.sequence_length,
            self.max_word_bytes,
        )
        target_input_ids = self.byte_ids.index_select(0, target_idx.reshape(-1)).reshape(
            batch_size,
            self.sequence_length,
            self.max_word_bytes,
        )
        source_lengths = self.lengths.index_select(0, source_idx.reshape(-1)).reshape(
            batch_size,
            self.sequence_length,
            1,
        )
        target_lengths = self.lengths.index_select(0, target_idx.reshape(-1)).reshape(
            batch_size,
            self.sequence_length,
            1,
        )
        return {
            "input_ids": input_ids,
            "word_masks": (self.positions < source_lengths) & unit_mask.unsqueeze(-1),
            "target_input_ids": target_input_ids,
            "target_word_masks": (self.positions < target_lengths) & unit_mask.unsqueeze(-1),
            "unit_mask": unit_mask,
        }


class ResidentNazEvalLoader:
    def __init__(self, batcher: ResidentNazBatcher, batch_size: int):
        self.batcher = batcher
        self.batch_size = batch_size

    def __iter__(self):
        max_start = max(self.batcher.token_count - self.batcher.sequence_length - 1, 0)
        starts = torch.arange(max_start + 1, device=self.batcher.device)
        for offset in range(0, starts.numel(), self.batch_size):
            yield self.batcher.make_batch(starts[offset : offset + self.batch_size].reshape(-1, 1))


class ResidentNazSemanticBatcher:
    def __init__(
        self,
        semantic_states: torch.Tensor,
        target_mean: torch.Tensor,
        target_log_std: torch.Tensor,
        sequence_length: int,
        batch_size: int,
        seed: int,
    ):
        if semantic_states.shape[:-1] != target_mean.shape[:-1]:
            raise ValueError("semantic and target caches must have matching token dimensions")
        if target_mean.shape != target_log_std.shape:
            raise ValueError("target mean/log_std caches must have matching shape")
        self.semantic_states = semantic_states
        self.target_mean = target_mean
        self.target_log_std = target_log_std
        self.token_count = semantic_states.shape[0]
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.device = semantic_states.device
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)
        self.offsets = torch.arange(sequence_length, device=self.device).reshape(1, sequence_length)
        if self.token_count < 2:
            raise ValueError("resident semantic Naz data needs at least two tokens")

    def __iter__(self):
        return self

    def __next__(self):
        max_start = max(self.token_count - self.sequence_length - 1, 0)
        starts = torch.randint(
            max_start + 1,
            (self.batch_size, 1),
            generator=self.generator,
            device=self.device,
        )
        return self.make_batch(starts)

    def make_batch(self, starts: torch.Tensor):
        source_idx = starts + self.offsets
        unit_mask = source_idx < (self.token_count - 1)
        source_idx = source_idx.clamp_max(self.token_count - 1)
        target_idx = (source_idx + 1).clamp_max(self.token_count - 1)
        batch_size = starts.shape[0]
        return {
            "semantic_states": self.semantic_states.index_select(0, source_idx.reshape(-1)).reshape(
                batch_size,
                self.sequence_length,
                -1,
            ),
            "target_mean": self.target_mean.index_select(0, target_idx.reshape(-1)).reshape(
                batch_size,
                self.sequence_length,
                -1,
            ),
            "target_log_std": self.target_log_std.index_select(0, target_idx.reshape(-1)).reshape(
                batch_size,
                self.sequence_length,
                -1,
            ),
            "unit_mask": unit_mask,
        }


class ResidentNazSemanticEvalLoader:
    def __init__(self, batcher: ResidentNazSemanticBatcher, batch_size: int):
        self.batcher = batcher
        self.batch_size = batch_size

    def __iter__(self):
        max_start = max(self.batcher.token_count - self.batcher.sequence_length - 1, 0)
        starts = torch.arange(max_start + 1, device=self.batcher.device)
        for offset in range(0, starts.numel(), self.batch_size):
            yield self.batcher.make_batch(starts[offset : offset + self.batch_size].reshape(-1, 1))
