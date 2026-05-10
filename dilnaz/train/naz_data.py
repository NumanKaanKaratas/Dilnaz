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
TOKEN_CACHE_FORMAT_VERSION = 5


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
            next_carry = text[boundary:]
            trailing_space_count = len(block) - len(block.rstrip(" "))
            if trailing_space_count:
                next_carry = block[-trailing_space_count:] + next_carry
                block = block[:-trailing_space_count]
            if block.strip():
                emitted = True
                yield block
            carry = next_carry
    if carry.strip():
        emitted = True
        yield carry
    if not emitted:
        raise ValueError(f"{path} produced no text blocks")


def stream_text_lines(path: Path, read_chars: int):
    emitted = False
    carry = ""
    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(read_chars)
            if not chunk:
                break
            lines = (carry + chunk).splitlines(keepends=True)
            carry = ""
            if lines and not lines[-1].endswith(("\n", "\r")):
                carry = lines.pop()
            for line in lines:
                if not line.strip():
                    continue
                emitted = True
                yield line
    if carry.strip():
        emitted = True
        yield carry
    if not emitted:
        raise ValueError(f"{path} produced no text blocks")


def stream_prompt_answer_rows(path: Path, read_chars: int):
    for line_idx, line in enumerate(stream_text_lines(path, read_chars)):
        text = line.rstrip("\r\n")
        parts = text.split("\t")
        if len(parts) != 2:
            raise ValueError(f"{path}:{line_idx + 1} must be a tab-separated prompt/answer row")
        prompt, answer = parts
        if not prompt or not answer:
            raise ValueError(f"{path}:{line_idx + 1} prompt and answer must be non-empty")
        yield prompt, answer


def stream_token_pieces(path: Path, tokenizer: HybridTokenizer, max_word_bytes: int, read_chars: int):
    for line in stream_text_lines(path, read_chars):
        emitted = False
        for segment in tokenizer.encode_segments(line):
            if segment.piece_len > max_word_bytes:
                continue
            token_ids = [piece.token_id for piece in segment.pieces]
            if not token_ids:
                continue
            emitted = True
            yield token_ids
        if emitted:
            yield [tokenizer.eos_token_id]


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


class StreamingTextNazDataset(IterableDataset):
    def __init__(
        self,
        train_file: Path,
        tokenizer: HybridTokenizer,
        config: DilConfig,
        sequence_length: int,
        batch_size: int,
        read_chars: int,
        repeat: bool,
    ):
        super().__init__()
        self.train_file = train_file
        self.tokenizer = tokenizer
        self.max_word_bytes = config.max_word_bytes
        self.pad_token_id = config.pad_token_id
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.read_chars = read_chars
        self.repeat = repeat
        self.positions = torch.arange(self.max_word_bytes).reshape(1, 1, self.max_word_bytes)

    def make_batch(self, windows: list[list[list[int]]]) -> dict:
        batch_size = len(windows)
        source = [window[:-1] for window in windows]
        target = [window[1:] for window in windows]
        input_ids = torch.full(
            (batch_size, self.sequence_length, self.max_word_bytes),
            self.pad_token_id,
            dtype=torch.long,
        )
        target_input_ids = torch.full_like(input_ids, self.pad_token_id)
        source_lengths = torch.zeros((batch_size, self.sequence_length), dtype=torch.long)
        target_lengths = torch.zeros_like(source_lengths)
        for row_idx, (source_window, target_window) in enumerate(zip(source, target)):
            for token_idx, token_ids in enumerate(source_window):
                width = len(token_ids)
                input_ids[row_idx, token_idx, :width] = torch.tensor(token_ids, dtype=torch.long)
                source_lengths[row_idx, token_idx] = width
            for token_idx, token_ids in enumerate(target_window):
                width = len(token_ids)
                target_input_ids[row_idx, token_idx, :width] = torch.tensor(token_ids, dtype=torch.long)
                target_lengths[row_idx, token_idx] = width
        unit_mask = source_lengths.gt(0)
        word_masks = (self.positions < source_lengths.unsqueeze(-1)) & unit_mask.unsqueeze(-1)
        target_word_masks = (self.positions < target_lengths.unsqueeze(-1)) & unit_mask.unsqueeze(-1)
        return {
            "input_ids": input_ids,
            "word_masks": word_masks,
            "target_input_ids": target_input_ids,
            "target_word_masks": target_word_masks,
            "unit_mask": unit_mask,
        }

    def iter_windows_once(self, worker_id: int, worker_count: int):
        buffer: list[list[int]] = []
        window_idx = 0
        for token_ids in stream_token_pieces(
            self.train_file,
            self.tokenizer,
            self.max_word_bytes,
            self.read_chars,
        ):
            buffer.append(token_ids)
            if len(buffer) == self.sequence_length + 1:
                if window_idx % worker_count == worker_id:
                    yield list(buffer)
                window_idx += 1
                buffer.pop(0)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        while True:
            emitted = False
            batch: list[list[list[int]]] = []
            for window in self.iter_windows_once(worker_id, worker_count):
                emitted = True
                batch.append(window)
                if len(batch) == self.batch_size:
                    yield self.make_batch(batch)
                    batch = []
            if batch:
                yield self.make_batch(batch)
            if not emitted:
                raise ValueError(f"{self.train_file} produced no Naz streaming windows")
            if not self.repeat:
                return


class PromptAnswerNazDataset(IterableDataset):
    def __init__(
        self,
        train_file: Path,
        tokenizer: HybridTokenizer,
        config: DilConfig,
        sequence_length: int,
        batch_size: int,
        read_chars: int,
        repeat: bool,
    ):
        super().__init__()
        self.train_file = train_file
        self.tokenizer = tokenizer
        self.max_word_bytes = config.max_word_bytes
        self.pad_token_id = config.pad_token_id
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.read_chars = read_chars
        self.repeat = repeat
        self.positions = torch.arange(self.max_word_bytes).reshape(1, 1, self.max_word_bytes)

    def encode_row(self, prompt: str, answer: str) -> tuple[list[list[int]], list[bool]]:
        prompt_segments = [
            segment.token_ids
            for segment in self.tokenizer.encode_segments(prompt)
            if 0 < segment.piece_len <= self.max_word_bytes
        ]
        answer_segments = [
            segment.token_ids
            for segment in self.tokenizer.encode_segments(" " + answer, add_eos=True)
            if 0 < segment.piece_len <= self.max_word_bytes
        ]
        tokens = prompt_segments + answer_segments
        target_answer_mask = [False] * max(len(prompt_segments) - 1, 0) + [True] * len(answer_segments)
        source_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        if len(source_tokens) != len(target_answer_mask):
            raise ValueError("prompt-answer target mask construction mismatch")
        if not source_tokens:
            raise ValueError("prompt-answer row produced no trainable token transitions")
        return [source_tokens, target_tokens], target_answer_mask

    def make_batch(self, rows: list[tuple[list[list[int]], list[list[int]], list[bool]]]) -> dict:
        batch_size = len(rows)
        input_ids = torch.full(
            (batch_size, self.sequence_length, self.max_word_bytes),
            self.pad_token_id,
            dtype=torch.long,
        )
        target_input_ids = torch.full_like(input_ids, self.pad_token_id)
        source_lengths = torch.zeros((batch_size, self.sequence_length), dtype=torch.long)
        target_lengths = torch.zeros_like(source_lengths)
        unit_mask = torch.zeros((batch_size, self.sequence_length), dtype=torch.bool)
        loss_mask = torch.zeros_like(unit_mask)
        for row_idx, (source_tokens, target_tokens, target_answer_mask) in enumerate(rows):
            if len(source_tokens) > self.sequence_length:
                raise ValueError(
                    f"prompt-answer row has {len(source_tokens)} transitions, sequence_length={self.sequence_length}"
                )
            width = len(source_tokens)
            for token_idx in range(width):
                source_ids = source_tokens[token_idx]
                target_ids = target_tokens[token_idx]
                input_ids[row_idx, token_idx, : len(source_ids)] = torch.tensor(source_ids, dtype=torch.long)
                target_input_ids[row_idx, token_idx, : len(target_ids)] = torch.tensor(target_ids, dtype=torch.long)
                source_lengths[row_idx, token_idx] = len(source_ids)
                target_lengths[row_idx, token_idx] = len(target_ids)
                unit_mask[row_idx, token_idx] = True
                loss_mask[row_idx, token_idx] = target_answer_mask[token_idx]
        word_masks = (self.positions < source_lengths.unsqueeze(-1)) & unit_mask.unsqueeze(-1)
        target_word_masks = (self.positions < target_lengths.unsqueeze(-1)) & unit_mask.unsqueeze(-1)
        return {
            "input_ids": input_ids,
            "word_masks": word_masks,
            "target_input_ids": target_input_ids,
            "target_word_masks": target_word_masks,
            "unit_mask": unit_mask,
            "loss_mask": loss_mask,
        }

    def iter_rows_once(self, worker_id: int, worker_count: int):
        for row_idx, (prompt, answer) in enumerate(stream_prompt_answer_rows(self.train_file, self.read_chars)):
            if row_idx % worker_count != worker_id:
                continue
            source_target, target_answer_mask = self.encode_row(prompt, answer)
            source_tokens, target_tokens = source_target
            yield source_tokens, target_tokens, target_answer_mask

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        while True:
            emitted = False
            batch: list[tuple[list[list[int]], list[list[int]], list[bool]]] = []
            for row in self.iter_rows_once(worker_id, worker_count):
                emitted = True
                batch.append(row)
                if len(batch) == self.batch_size:
                    yield self.make_batch(batch)
                    batch = []
            if batch:
                yield self.make_batch(batch)
            if not emitted:
                raise ValueError(f"{self.train_file} produced no prompt-answer rows")
            if not self.repeat:
                return


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
        self.byte_ids = torch.from_numpy(np.array(np.load(ids_path, mmap_mode="r"), dtype=np.int64, copy=True)).to(device)
        self.lengths = torch.from_numpy(np.array(np.load(lengths_path, mmap_mode="r"), dtype=np.int64, copy=True)).to(device)
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
        byte_ids: torch.LongTensor,
        lengths: torch.LongTensor,
        sequence_length: int,
        batch_size: int,
        seed: int,
    ):
        if semantic_states.shape[:-1] != target_mean.shape[:-1]:
            raise ValueError("semantic and target caches must have matching token dimensions")
        if byte_ids.shape[0] != semantic_states.shape[0] or lengths.shape[0] != semantic_states.shape[0]:
            raise ValueError("surface cache must match semantic token count")
        self.semantic_states = semantic_states
        self.target_mean = target_mean
        self.byte_ids = byte_ids.long()
        self.lengths = lengths.long()
        self.token_count = semantic_states.shape[0]
        self.max_word_bytes = byte_ids.shape[1]
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.device = semantic_states.device
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)
        self.offsets = torch.arange(sequence_length, device=self.device).reshape(1, sequence_length)
        self.positions = torch.arange(self.max_word_bytes, device=self.device).reshape(1, 1, self.max_word_bytes)
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
