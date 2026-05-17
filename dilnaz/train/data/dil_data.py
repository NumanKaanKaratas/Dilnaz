from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info

from dilnaz.surface import pack_token_units, pack_writer_targets
from dilnaz.surface.types import PackedSurface
from dilnaz.tokenization import HybridTokenizer, TokenSegment
from dilnaz.train.configs.defaults import DIL_MODEL_DEFAULTS


NLLB_DEFAULT_NUM_GROUPS = 8
NLLB_DEFAULT_MAX_ENCODER_TOKENS = 1024
NLLB_TEACHER_CACHE_VERSION = 1
NLLB_TEXT_CHUNKING = "offset-token-budget-v1"
TEACHER_CENTERED_ADD_WEIGHT = 0.5


def build_nllb_layer_groups(num_layers: int, num_groups: int = 8) -> tuple[tuple[int, ...], ...]:
    num_layers = max(num_layers, num_groups)
    base_size = num_layers // num_groups
    extra = num_layers % num_groups
    groups = []
    start = 0
    for i in range(num_groups):
        size = base_size + (1 if i < extra else 0)
        groups.append(tuple(range(start, start + size)))
        start += size
    return tuple(groups)


NLLB_LAYER_GROUPS = ((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12))


def overlaps(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end


def teacher_distill_segment(segment, *_args, **_kwargs):
    del _args, _kwargs
    return segment is not None and segment.kind != "eos" and not segment.text.isspace()


def apply_teacher_centered_add_by_group(teacher, mask, group_ids):
    centered = teacher.clone()
    valid_groups = group_ids[mask].unique()
    for group_id in valid_groups:
        rows = mask & group_ids.eq(group_id)
        centered[rows] = teacher[rows] - teacher[rows].mean(dim=0, keepdim=True)
    return centered


def segment_piece_ids(segment: TokenSegment | None) -> list[int]:
    if segment is None:
        return []
    return list(segment.token_ids)


def trainable_segments(
    tokenizer: HybridTokenizer,
    text: str,
    max_surface_pieces_per_unit: int,
    add_eos: bool = True,
    min_pieces: int = 1,
) -> list[TokenSegment]:
    segments = [
        segment
        for segment in tokenizer.encode_segments(text)
        if segment.piece_len > 0 and segment.piece_len <= max_surface_pieces_per_unit
    ]
    if not segments:
        return []
    if add_eos:
        from dilnaz.tokenization.hybrid_tokenizer import TokenPiece
        eos_piece = TokenPiece(token_id=tokenizer.eos_token_id, text="", start=0, end=0, kind="eos")
        eos = TokenSegment(
            text="",
            start=segments[-1].end,
            end=segments[-1].end,
            kind="eos",
            pieces=(eos_piece,),
        )
        segments.append(eos)
    return segments


def stream_text_items(path: Path, read_chars: int) -> Iterator[tuple[int, str, bool]]:
    if path.suffix.casefold() == ".jsonl":
        emitted = False
        with path.open("r", encoding="utf-8") as handle:
            for line_idx, line in enumerate(handle):
                if not line.strip():
                    continue
                payload = json.loads(line)
                text = payload.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                emitted = True
                yield line_idx, text, True
        if not emitted:
            raise ValueError(f"{path} produced no JSONL text records")
        return

    with path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            text = line[:read_chars]
            if not text:
                continue
            yield line_idx, text, True


stream_teacher_text_items_with_eos = stream_text_items


def load_hybrid_tokenizer(vocab_path: Path) -> HybridTokenizer:
    return HybridTokenizer.from_file(str(vocab_path))


def make_dil_batch_loader(dataset, num_workers: int, pin_memory: bool, prefetch_factor: int):
    from torch.utils.data import DataLoader

    loader_kwargs = dict(
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=None,
        persistent_workers=bool(num_workers > 0),
    )
    return DataLoader(dataset, **loader_kwargs)


def context_windows(segments: list[TokenSegment], context_radius: int) -> list[list[TokenSegment | None]]:
    offsets = range(-context_radius, context_radius + 1)
    windows: list[list[TokenSegment | None]] = []
    for target in range(len(segments)):
        window = []
        for offset in offsets:
            idx = target + offset
            window.append(segments[idx] if 0 <= idx < len(segments) else None)
        windows.append(window)
    return windows


def align_spans_to_pieces(starts: list[int], ends: list[int], pieces: list) -> list[list[int]]:
    return [
        [
            idx
            for idx, piece in enumerate(pieces)
            if overlaps(start, end, int(piece[1]), int(piece[2]))
        ]
        for start, end in zip(starts, ends)
    ]


def apply_teacher_centered_add(
    teacher: torch.Tensor,
    teacher_mask: torch.Tensor,
    weight: float = TEACHER_CENTERED_ADD_WEIGHT,
) -> torch.Tensor:
    result = teacher.clone()
    valid_teacher = teacher[teacher_mask]
    if valid_teacher.numel() == 0:
        return result
    center = valid_teacher.mean(dim=0, keepdim=True)
    result[teacher_mask] = valid_teacher + (valid_teacher - center) * weight
    return result


@dataclass(frozen=True)
class NllbTextChunk:
    text: str
    start: int


@dataclass(frozen=True)
class NllbEncodedText:
    group_hidden: torch.Tensor
    pieces: tuple[tuple[str, int, int, int], ...]


NLLB_SENTENCE_BOUNDARY_CHARS = frozenset(".!?\u2026")
NLLB_CLAUSE_BOUNDARY_CHARS = frozenset(",;:")


def nllb_token_count(tokenizer, text: str) -> int:
    encoded = tokenizer(text)
    input_ids = encoded["input_ids"]
    if hasattr(input_ids, "shape"):
        return int(input_ids.shape[-1])
    if input_ids and isinstance(input_ids[0], (list, tuple)):
        return len(input_ids[0])
    return len(input_ids)


def nllb_content_budget(tokenizer, max_encoder_tokens: int) -> int:
    special_count = tokenizer.num_special_tokens_to_add(pair=False) if hasattr(tokenizer, "num_special_tokens_to_add") else 2
    budget = int(max_encoder_tokens) - int(special_count)
    if budget <= 0:
        raise ValueError("max_encoder_tokens must leave room for NLLB content tokens")
    return budget


def nllb_content_offsets(tokenizer, text: str) -> list[tuple[int, int]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return [
        (int(start), int(end))
        for start, end in encoded["offset_mapping"]
        if int(start) != int(end)
    ]


def nllb_split_boundary_rank(text: str, previous_end: int, next_start: int | None) -> int:
    gap = text[previous_end:next_start] if next_start is not None else text[previous_end:]
    left = text[:previous_end].rstrip()
    last = left[-1] if left else ""
    if "\n" in gap or last in NLLB_SENTENCE_BOUNDARY_CHARS:
        return 3
    if last in NLLB_CLAUSE_BOUNDARY_CHARS:
        return 2
    if gap and gap.strip() == "":
        return 1
    return 0


def choose_nllb_split_end(text: str, offsets: list[tuple[int, int]], start_idx: int, max_end_idx: int) -> int:
    if max_end_idx >= len(offsets):
        return len(offsets)
    min_end_idx = start_idx + max(1, (max_end_idx - start_idx) // 2)
    best_end_idx = max_end_idx
    best_rank = 0
    for end_idx in range(max_end_idx, min_end_idx - 1, -1):
        previous_end = offsets[end_idx - 1][1]
        next_start = offsets[end_idx][0]
        rank = nllb_split_boundary_rank(text, previous_end, next_start)
        if rank > best_rank:
            best_rank = rank
            best_end_idx = end_idx
            if rank == 3:
                break
    return best_end_idx


def split_text_for_nllb(text: str, tokenizer, max_encoder_tokens: int) -> tuple[NllbTextChunk, ...]:
    offsets = nllb_content_offsets(tokenizer, text)
    if not offsets:
        return (NllbTextChunk(text=text, start=0),)

    budget = nllb_content_budget(tokenizer, max_encoder_tokens)
    if len(offsets) <= budget and nllb_token_count(tokenizer, text) <= max_encoder_tokens:
        return (NllbTextChunk(text=text, start=0),)

    chunks: list[NllbTextChunk] = []
    start_idx = 0
    while start_idx < len(offsets):
        max_end_idx = min(start_idx + budget, len(offsets))
        end_idx = choose_nllb_split_end(text, offsets, start_idx, max_end_idx)
        while end_idx > start_idx:
            chunk_start = offsets[start_idx][0]
            chunk_end = offsets[end_idx - 1][1]
            chunk_text = text[chunk_start:chunk_end]
            if nllb_token_count(tokenizer, chunk_text) <= max_encoder_tokens:
                chunks.append(NllbTextChunk(text=chunk_text, start=chunk_start))
                start_idx = end_idx
                break
            end_idx -= 1
        else:
            raise ValueError("NLLB chunking failed to make progress")
    return tuple(chunks)


def nllb_piece_positions(tokenizer, input_ids: list[int], offsets, offset_shift: int = 0, index_shift: int = 0):
    pieces = tokenizer.convert_ids_to_tokens(input_ids)
    return tuple(
        (
            piece,
            int(offset[0]) + offset_shift,
            int(offset[1]) + offset_shift,
            token_idx + index_shift,
        )
        for token_idx, (piece, offset) in enumerate(zip(pieces, offsets))
        if int(offset[0]) != int(offset[1])
    )


class NllbTeacherTextCache:
    def __init__(
        self,
        cache_dir: Path,
        contract: dict,
        memory_items: int = 128,
        max_disk_bytes: int = 0,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.contract = dict(contract)
        self.memory_items = int(memory_items)
        self.max_disk_bytes = int(max_disk_bytes)
        self.memory_cache: dict[str, NllbEncodedText] = {}
        self.memory_order: list[str] = []
        self._remove_stale_tmp_files()
        self.disk_bytes = self._scan_disk_bytes()
        self._prune_disk_cache()

    def key(self, lang: str, text: str) -> str:
        payload = {
            "version": NLLB_TEACHER_CACHE_VERSION,
            "contract": self.contract,
            "lang": lang,
            "text": text,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def path_for(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pt"

    def get(self, lang: str, text: str) -> NllbEncodedText | None:
        key = self.key(lang, text)
        cached = self.memory_cache.get(key)
        if cached is not None:
            return cached
        path = self.path_for(key)
        if not path.exists():
            return None
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata", {})
        if metadata.get("cache_key") != key or metadata.get("contract") != self.contract:
            raise ValueError(f"NLLB teacher cache metadata mismatch: {path}")
        os.utime(path, None)
        encoded = NllbEncodedText(
            group_hidden=payload["group_hidden"].contiguous(),
            pieces=tuple(tuple(piece) for piece in payload["pieces"]),
        )
        self._remember(key, encoded)
        return encoded

    def put(self, lang: str, text: str, encoded: NllbEncodedText) -> None:
        key = self.key(lang, text)
        self._remember(key, encoded)
        path = self.path_for(key)
        tmp_path = path.with_suffix(".tmp")
        payload = {
            "metadata": {
                "version": NLLB_TEACHER_CACHE_VERSION,
                "cache_key": key,
                "contract": self.contract,
                "lang": lang,
            },
            "group_hidden": encoded.group_hidden.cpu().contiguous(),
            "pieces": tuple(encoded.pieces),
        }
        torch.save(payload, tmp_path)
        old_size = path.stat().st_size if path.exists() else 0
        os.replace(str(tmp_path), str(path))
        self.disk_bytes += path.stat().st_size - old_size
        self._prune_disk_cache()

    def _remember(self, key: str, encoded: NllbEncodedText) -> None:
        if self.memory_items <= 0:
            return
        if key in self.memory_cache:
            return
        self.memory_cache[key] = encoded
        self.memory_order.append(key)
        while len(self.memory_order) > self.memory_items:
            old_key = self.memory_order.pop(0)
            self.memory_cache.pop(old_key, None)

    def _scan_disk_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.cache_dir.glob("*.pt") if path.is_file())

    def _remove_stale_tmp_files(self) -> None:
        for path in self.cache_dir.glob("*.tmp"):
            if path.is_file():
                path.unlink()

    def _prune_disk_cache(self) -> None:
        if self.max_disk_bytes <= 0 or self.disk_bytes <= self.max_disk_bytes:
            return
        files = sorted(
            (path for path in self.cache_dir.glob("*.pt") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
        )
        for path in files:
            if self.disk_bytes <= self.max_disk_bytes:
                break
            key = path.stem
            size = path.stat().st_size
            path.unlink()
            self.disk_bytes -= size
            self.memory_cache.pop(key, None)
            if key in self.memory_order:
                self.memory_order.remove(key)


class NllbTeacher:
    def __init__(
        self,
        model_name: str,
        src_lang: str,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int = 64,
        cache_dir: Path | None = None,
        cache_memory_items: int = 128,
        cache_max_bytes: int = 0,
    ):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = src_lang
        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=dtype)
            .to(device)
            .eval()
        )
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model_name = model_name
        self.src_lang = src_lang
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_encoder_tokens = int(getattr(self.model.config, "max_position_embeddings", NLLB_DEFAULT_MAX_ENCODER_TOKENS))
        self.layer_groups = NLLB_LAYER_GROUPS
        self.cache = None
        if cache_dir is not None:
            self.cache = NllbTeacherTextCache(
                cache_dir,
                {
                    "model_name": model_name,
                    "layer_groups": tuple(tuple(int(layer) for layer in group) for group in self.layer_groups),
                    "max_encoder_tokens": self.max_encoder_tokens,
                    "text_chunking": NLLB_TEXT_CHUNKING,
                    "hidden_size": int(self.model.config.d_model),
                    "dtype": str(dtype),
                },
                memory_items=cache_memory_items,
                max_disk_bytes=cache_max_bytes,
            )
        self.last_stats: dict[str, float] = {}

    def teacher_layers(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        total_start = time.perf_counter()
        stats = {
            "cache_hits": 0.0,
            "cache_misses": 0.0,
            "unique_texts": 0.0,
            "tokenize_seconds": 0.0,
            "forward_seconds": 0.0,
            "pool_seconds": 0.0,
            "input_tokens": 0.0,
            "padded_tokens": 0.0,
        }
        texts = batch["teacher_texts"]
        text_indices = batch["teacher_text_indices"].detach().cpu().tolist()
        starts = batch["teacher_starts"].detach().cpu().tolist()
        ends = batch["teacher_ends"].detach().cpu().tolist()
        distill_mask = batch["teacher_distill_mask"].detach().cpu().tolist()
        sample_count = len(starts)
        num_groups = len(self.layer_groups)
        teacher = torch.zeros(
            (sample_count, num_groups, self.model.config.d_model),
            dtype=self.dtype,
            device=self.device,
        )
        teacher_mask = torch.zeros((sample_count,), dtype=torch.bool, device=self.device)

        rows_by_text: dict[str, list[int]] = {}
        for row_idx, text_idx in enumerate(text_indices):
            text = texts[text_idx]
            rows_by_text.setdefault(text, []).append(row_idx)
        unique_texts = list(rows_by_text)
        stats["unique_texts"] = float(len(unique_texts))
        encoded_texts = self.encoded_texts(unique_texts, stats)

        pool_start = time.perf_counter()
        for text, rows in rows_by_text.items():
            encoded = encoded_texts[text]
            if not rows or not encoded.pieces:
                continue
            alignments = align_spans_to_pieces(
                [starts[row_idx] for row_idx in rows],
                [ends[row_idx] for row_idx in rows],
                list(encoded.pieces),
            )
            group_hidden = encoded.group_hidden.to(self.device, non_blocking=True)
            for row_idx, positions in zip(rows, alignments):
                if not distill_mask[row_idx] or not positions:
                    continue
                teacher_mask[row_idx] = True
                hidden_positions = [encoded.pieces[position][3] for position in positions]
                pos = torch.tensor(hidden_positions, dtype=torch.long, device=self.device)
                teacher[row_idx] = group_hidden.index_select(1, pos).mean(dim=1)
        teacher = apply_teacher_centered_add(teacher, teacher_mask)
        stats["pool_seconds"] += time.perf_counter() - pool_start
        total_seconds = time.perf_counter() - total_start
        cache_total = stats["cache_hits"] + stats["cache_misses"]
        stats["total_seconds"] = total_seconds
        stats["cache_hit_rate"] = stats["cache_hits"] / max(cache_total, 1.0)
        stats["padding_ratio"] = (
            1.0 - (stats["input_tokens"] / stats["padded_tokens"])
            if stats["padded_tokens"] > 0
            else 0.0
        )
        stats["nllb_tokens_per_second"] = (
            stats["input_tokens"] / stats["forward_seconds"]
            if stats["forward_seconds"] > 0
            else 0.0
        )
        self.last_stats = stats
        return teacher, teacher_mask

    def encoded_texts(self, texts: list[str], stats: dict[str, float]) -> dict[str, NllbEncodedText]:
        result: dict[str, NllbEncodedText] = {}
        missing: list[str] = []
        for text in texts:
            cached = self.cache.get(self.src_lang, text) if self.cache is not None else None
            if cached is None:
                missing.append(text)
                stats["cache_misses"] += 1.0
            else:
                result[text] = cached
                stats["cache_hits"] += 1.0
        if missing:
            for text, encoded in self.encode_missing_texts(missing, stats).items():
                result[text] = encoded
                if self.cache is not None:
                    self.cache.put(self.src_lang, text, encoded)
        return result

    def encode_missing_texts(self, texts: list[str], stats: dict[str, float]) -> dict[str, NllbEncodedText]:
        encoded_texts: dict[str, NllbEncodedText] = {}
        records: list[tuple[str, int, NllbTextChunk]] = []
        chunk_count_by_text: dict[str, int] = {}
        for text in texts:
            chunks = split_text_for_nllb(text, self.tokenizer, self.max_encoder_tokens)
            chunk_count_by_text[text] = len(chunks)
            records.extend((text, chunk_idx, chunk) for chunk_idx, chunk in enumerate(chunks))

        grouped_chunks: dict[str, list[torch.Tensor | None]] = {
            text: [None] * chunk_count for text, chunk_count in chunk_count_by_text.items()
        }
        piece_chunks: dict[str, list[tuple[tuple[str, int, int, int], ...] | None]] = {
            text: [None] * chunk_count for text, chunk_count in chunk_count_by_text.items()
        }
        length_sorted = sorted(records, key=lambda record: len(record[2].text))
        for batch_start in range(0, len(length_sorted), self.batch_size):
            records_batch = length_sorted[batch_start : batch_start + self.batch_size]
            texts_batch = [record[2].text for record in records_batch]
            tokenize_start = time.perf_counter()
            encoded = self.tokenizer(
                texts_batch,
                return_tensors="pt",
                return_offsets_mapping=True,
                padding=True,
            )
            stats["tokenize_seconds"] += time.perf_counter() - tokenize_start
            if encoded["input_ids"].shape[1] > self.max_encoder_tokens:
                raise ValueError(
                    f"NLLB input has {encoded['input_ids'].shape[1]} tokens; "
                    f"max_encoder_tokens={self.max_encoder_tokens}"
                )
            offsets = encoded.pop("offset_mapping")
            input_ids_batch = encoded["input_ids"].tolist()
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                stats["input_tokens"] += float(attention_mask.sum().item())
                stats["padded_tokens"] += float(attention_mask.numel())
            else:
                stats["input_tokens"] += float(encoded["input_ids"].numel())
                stats["padded_tokens"] += float(encoded["input_ids"].numel())
            inputs = {key: value.to(self.device, non_blocking=True) for key, value in encoded.items()}
            forward_start = time.perf_counter()
            with torch.inference_mode():
                outputs = self.model.get_encoder()(**inputs, output_hidden_states=True, return_dict=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            stats["forward_seconds"] += time.perf_counter() - forward_start
            hidden_states = outputs.hidden_states
            grouped = torch.stack(
                [
                    torch.stack([hidden_states[layer] for layer in layers], dim=0).mean(dim=0)
                    for layers in self.layer_groups
                ],
                dim=1,
            ).to(dtype=self.dtype)
            for local_idx, (text, chunk_idx, chunk) in enumerate(records_batch):
                seq_len = (
                    int(attention_mask[local_idx].sum().item())
                    if attention_mask is not None
                    else int(encoded["input_ids"].shape[1])
                )
                grouped_chunks[text][chunk_idx] = grouped[local_idx, :, :seq_len].detach().cpu().contiguous()
                piece_chunks[text][chunk_idx] = nllb_piece_positions(
                    self.tokenizer,
                    input_ids_batch[local_idx],
                    offsets[local_idx].tolist(),
                    offset_shift=chunk.start,
                )
        for text in texts:
            text_group_chunks = grouped_chunks[text]
            text_piece_chunks = piece_chunks[text]
            group_hidden_parts: list[torch.Tensor] = []
            pieces: list[tuple[str, int, int, int]] = []
            index_shift = 0
            for group_hidden, piece_chunk in zip(text_group_chunks, text_piece_chunks):
                if group_hidden is None or piece_chunk is None:
                    raise ValueError("missing NLLB encoded chunk")
                pieces.extend(
                    (piece, start, end, encoder_index + index_shift)
                    for piece, start, end, encoder_index in piece_chunk
                )
                group_hidden_parts.append(group_hidden)
                index_shift += int(group_hidden.shape[1])
            encoded_texts[text] = NllbEncodedText(
                group_hidden=torch.cat(group_hidden_parts, dim=1).contiguous(),
                pieces=tuple(pieces),
            )
        return encoded_texts


class ContextDilBatchDataset(IterableDataset):
    def __init__(
        self,
        train_file: Path,
        config,
        tokenizer: HybridTokenizer,
        batch_size: int,
        read_chars: int,
        repeat: bool = True,
        max_samples: int = 0,
        teacher_tokenizer=None,
        teacher_max_tokens: int = 512,
    ):
        super().__init__()
        self.train_file = train_file
        self.config = config
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.read_chars = read_chars
        self.repeat = repeat
        self.max_samples = max_samples
        self.context_radius = getattr(config, "context_radius", 2)
        self.context_size = 2 * self.context_radius + 1
        self.max_pieces_per_unit = config.max_surface_pieces_per_unit
        self.surface_bucket_sizes = tuple(config.surface_bucket_sizes)
        self.pad_token_id = config.pad_token_id
        self.teacher_tokenizer = teacher_tokenizer
        self.teacher_max_tokens = teacher_max_tokens

    def make_batch(
        self,
        unit_rows: list[list[list[int]]],
        target_rows: list[list[list[int]]],
        texts: list[str],
        line_ids: list[int],
        text_indices: list[int],
        starts: list[int],
        ends: list[int],
        distill_mask: list[bool],
        source_row_starts: list[bool],
    ) -> dict:
        surface = pack_token_units(
            unit_rows,
            pad_token_id=self.pad_token_id,
            bucket_sizes=self.surface_bucket_sizes,
            max_pieces_per_unit=self.max_pieces_per_unit,
        )
        batch: dict = {
            "surface": surface,
            "writer_target": pack_writer_targets(
                target_rows,
                pad_token_id=self.pad_token_id,
                bos_token_id=self.config.decoder_start_token_id,
                stop_token_id=self.config.writer_stop_token_id,
                surface_bucket_sizes=self.surface_bucket_sizes,
                max_pieces_per_unit=self.max_pieces_per_unit,
            ),
            "teacher_texts": texts,
            "teacher_text_indices": torch.tensor(text_indices, dtype=torch.long),
            "teacher_starts": torch.tensor(starts, dtype=torch.long),
            "teacher_ends": torch.tensor(ends, dtype=torch.long),
            "teacher_distill_mask": torch.tensor(distill_mask, dtype=torch.bool),
            "source_line_ids": torch.tensor(line_ids, dtype=torch.long),
            "source_row_count": sum(1 for value in source_row_starts if value),
            "target_unit_count": len(target_rows),
        }
        return batch

    def iter_once(self, worker_id: int, worker_count: int):
        unit_rows: list[list[list[int]]] = []
        target_rows: list[list[list[int]]] = []
        texts: list[str] = []
        line_ids: list[int] = []
        text_indices: list[int] = []
        starts: list[int] = []
        ends: list[int] = []
        distill_mask: list[bool] = []
        source_row_starts: list[bool] = []
        produced = 0

        for item_idx, (source_line_id, text, add_eos) in enumerate(
            stream_text_items(self.train_file, self.read_chars)
        ):
            if item_idx % worker_count != worker_id:
                continue
            segments = trainable_segments(
                self.tokenizer,
                text,
                self.max_pieces_per_unit,
                add_eos=add_eos,
            )
            if not segments:
                continue
            windows = context_windows(segments, self.context_radius)
            for target_idx, window in enumerate(windows):
                segment = segments[target_idx]
                row = [segment_piece_ids(seg) for seg in window]
                text_idx = len(texts)
                texts.append(text)
                unit_rows.append(row)
                target_rows.append([segment_piece_ids(segment)])
                line_ids.append(source_line_id)
                text_indices.append(text_idx)
                starts.append(segment.start)
                ends.append(segment.end)
                distill_mask.append(teacher_distill_segment(segment))
                source_row_starts.append(target_idx == 0)
                produced += 1
                if len(unit_rows) >= self.batch_size:
                    yield self.make_batch(
                        unit_rows,
                        target_rows,
                        texts,
                        line_ids,
                        text_indices,
                        starts,
                        ends,
                        distill_mask,
                        source_row_starts,
                    )
                    unit_rows = []
                    target_rows = []
                    texts = []
                    line_ids = []
                    text_indices = []
                    starts = []
                    ends = []
                    distill_mask = []
                    source_row_starts = []
                if self.max_samples > 0 and produced >= self.max_samples:
                    if unit_rows:
                        yield self.make_batch(
                            unit_rows,
                            target_rows,
                            texts,
                            line_ids,
                            text_indices,
                            starts,
                            ends,
                            distill_mask,
                            source_row_starts,
                        )
                    return

        if unit_rows:
            yield self.make_batch(
                unit_rows,
                target_rows,
                texts,
                line_ids,
                text_indices,
                starts,
                ends,
                distill_mask,
                source_row_starts,
            )

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        while True:
            yielded = False
            for batch in self.iter_once(worker_id, worker_count):
                yielded = True
                yield batch
            if not yielded:
                raise ValueError(f"{self.train_file} produced no samples")
            if not self.repeat:
                return


class ResidentDilBatcher:
    def __init__(self, batches: list[dict], batch_size: int, device: torch.device, seed: int):
        self.batches = batches
        self.batch_size = batch_size
        self.device = device
        self.rng = random.Random(seed)

    @classmethod
    def from_dataset(cls, dataset, teacher, batch_size: int, device: torch.device, seed: int):
        from dilnaz.train.common.runtime import cuda_sync

        batches = []
        for batch in dataset.iter_once(worker_id=0, worker_count=1):
            teacher_layers, teacher_mask = teacher.teacher_layers(batch)
            batch["teacher_layers"] = teacher_layers
            batch["teacher_mask"] = teacher_mask
            batch["_teacher_stats"] = dict(getattr(teacher, "last_stats", {}))
            batch["_teacher_reuse_count"] = 1
            cuda_sync(device)
            batches.append(batch)
        if not batches:
            raise ValueError(f"{dataset.train_file} produced no resident batches")
        return cls(batches, batch_size, device, seed)

    def __iter__(self):
        return self

    def __next__(self):
        idx = self.rng.randrange(len(self.batches))
        return {
            key: value.to(self.device, non_blocking=True) if hasattr(value, "to") else value
            for key, value in self.batches[idx].items()
        }

    def __len__(self):
        return len(self.batches)


class ResidentDilEvalLoader:
    def __init__(self, batcher: ResidentDilBatcher):
        self.batches = batcher.batches
        self.device = batcher.device

    def __iter__(self):
        for idx in range(len(self.batches)):
            batch = self.batches[idx]
            yield {
                key: value.to(self.device, non_blocking=True) if hasattr(value, "to") else value
                for key, value in batch.items()
            }

    def __len__(self):
        return len(self.batches)
