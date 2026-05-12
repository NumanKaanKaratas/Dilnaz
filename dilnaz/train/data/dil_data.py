from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from dilnaz.models.dil import DilConfig
from dilnaz.surface import pack_context_segments, pack_writer_targets
from dilnaz.tokenization import HybridTokenizer, TokenSegment, default_vocab_path
from dilnaz.train.common.runtime import move_to_device


SENTENCE_CHUNK_PATTERN = re.compile(r".+?(?:[.!?]+(?:\s+|$)|\n+|$)", re.UNICODE | re.DOTALL)
NLLB_LAYER_GROUPS = ((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12))
TEACHER_CENTERED_ADD_WEIGHT = 0.5
NLLB_DEFAULT_MAX_ENCODER_TOKENS = 1024
DILNAZ_READY_FORMAT = "dilnaz-ready-teacher-v2"
DILNAZ_READY_REQUIRED_COLUMNS = [
    "surface_ids",
    "surface_offsets",
    "target_ids",
    "teacher_layers",
    "teacher_mask",
]


@dataclass(frozen=True)
class BatchSampleRef:
    text_idx: int
    token_idx: int


def stream_text_lines(path: Path, read_chars: int):
    for _, line in stream_text_line_items(path, read_chars):
        yield line


def is_jsonl_path(path: Path) -> bool:
    return path.suffix.casefold() == ".jsonl"


def jsonl_text_records(path: Path):
    emitted = False
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_idx + 1} is not valid JSONL") from exc
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            emitted = True
            yield line_idx, text
    if not emitted:
        raise ValueError(f"{path} produced no JSONL text records")


def stream_text_line_items(path: Path, read_chars: int):
    if is_jsonl_path(path):
        yield from jsonl_text_records(path)
        return

    emitted = False
    carry = ""
    line_idx = 0
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
                yield line_idx, line
                line_idx += 1
    if carry.strip():
        emitted = True
        yield line_idx, carry
    if not emitted:
        raise ValueError(f"{path} produced no text blocks")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parquet_metadata(path: Path) -> dict[str, str]:
    raw_metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    return {
        key.decode("utf-8") if isinstance(key, bytes) else str(key): (
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
        )
        for key, value in raw_metadata.items()
    }


def _metadata_int(metadata: dict[str, str], key: str) -> int:
    try:
        return int(metadata[key])
    except KeyError as exc:
        raise ValueError(f"Dilnaz-ready parquet metadata missing {key!r}") from exc
    except ValueError as exc:
        raise ValueError(f"Dilnaz-ready parquet metadata {key!r} must be int, got {metadata[key]!r}") from exc


def validate_dilnaz_ready_parquet(path: Path, config: DilConfig, vocab_path: Path) -> dict[str, str]:
    parquet = pq.ParquetFile(path)
    schema_names = set(parquet.schema_arrow.names)
    missing = [column for column in DILNAZ_READY_REQUIRED_COLUMNS if column not in schema_names]
    if missing:
        raise ValueError(f"Dilnaz-ready parquet is missing required columns: {missing}")
    if parquet.metadata is None or int(parquet.metadata.num_rows or 0) <= 0:
        raise ValueError(f"Dilnaz-ready parquet has no rows: {path}")

    metadata = parquet_metadata(path)
    if metadata.get("format") != DILNAZ_READY_FORMAT:
        raise ValueError(f"unsupported Dilnaz parquet format={metadata.get('format')!r}")
    expected = {
        "tokenizer_vocab_size": config.vocab_size,
        "pad_token_id": config.pad_token_id,
        "eos_token_id": config.eos_token_id,
        "max_surface_pieces_per_unit": config.max_surface_pieces_per_unit,
        "context_radius": config.context_radius,
        "context_size": config.context_size,
        "target_index": config.target_index,
        "teacher_dim": 1024,
        "teacher_layer_count": len(NLLB_LAYER_GROUPS),
    }
    for key, expected_value in expected.items():
        actual_value = _metadata_int(metadata, key)
        if actual_value != int(expected_value):
            raise ValueError(f"Dilnaz-ready parquet metadata {key}={actual_value}, expected {expected_value}")
    vocab_hash = metadata.get("tokenizer_vocab_sha256")
    expected_hash = file_sha256(vocab_path)
    if vocab_hash != expected_hash:
        raise ValueError(f"Dilnaz-ready parquet vocab hash mismatch: {vocab_hash} != {expected_hash}")
    if metadata.get("teacher_formula") != "centered_add_w050_grouped":
        raise ValueError(f"unsupported teacher_formula={metadata.get('teacher_formula')!r}")
    return metadata


def nllb_token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def split_oversized_span(text: str, tokenizer, max_tokens: int) -> list[str]:
    words = re.findall(r"\S+\s*", text, re.UNICODE)
    chunks: list[str] = []
    current = ""
    for word in words:
        candidate = current + word
        if current and nllb_token_count(tokenizer, candidate) > max_tokens:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def split_text_for_nllb(text: str, tokenizer, max_tokens: int) -> list[str]:
    if tokenizer is None or nllb_token_count(tokenizer, text) <= max_tokens:
        return [text]

    chunks: list[str] = []
    current = ""
    for match in SENTENCE_CHUNK_PATTERN.finditer(text):
        sentence = match.group(0)
        if not sentence:
            continue
        candidate = current + sentence
        if current and nllb_token_count(tokenizer, candidate) > max_tokens:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
        if nllb_token_count(tokenizer, current) > max_tokens:
            chunks.extend(split_oversized_span(current, tokenizer, max_tokens))
            current = ""
    if current:
        chunks.append(current)
    return chunks


def stream_teacher_texts(path: Path, read_chars: int, tokenizer=None, max_tokens: int = NLLB_DEFAULT_MAX_ENCODER_TOKENS):
    for _, line in stream_text_line_items(path, read_chars):
        yield from split_text_for_nllb(line, tokenizer, max_tokens)


def stream_teacher_text_items(
    path: Path,
    read_chars: int,
    tokenizer=None,
    max_tokens: int = NLLB_DEFAULT_MAX_ENCODER_TOKENS,
):
    for line_idx, line in stream_text_line_items(path, read_chars):
        for text in split_text_for_nllb(line, tokenizer, max_tokens):
            yield line_idx, text


def overlaps(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return max(left_start, right_start) < min(left_end, right_end)


def align_spans_to_pieces(
    starts: list[int],
    ends: list[int],
    pieces: list[tuple],
) -> list[list[int]]:
    alignments = []
    for start, end in zip(starts, ends):
        alignments.append(
            [
                idx
                for idx, piece in enumerate(pieces)
                if overlaps(start, end, int(piece[1]), int(piece[2]))
            ]
        )
    return alignments


def segment_piece_ids(segment: TokenSegment) -> list[int]:
    return [piece.token_id for piece in segment.pieces]


def context_offsets(context_radius: int) -> range:
    return range(-context_radius, context_radius + 1)


def trainable_segments(tokenizer: HybridTokenizer, text: str, max_surface_pieces_per_unit: int) -> list[TokenSegment]:
    return [
        segment
        for segment in tokenizer.encode_segments(text)
        if 0 < segment.piece_len <= max_surface_pieces_per_unit
    ]


def teacher_distill_segment(segment: TokenSegment) -> bool:
    return not segment.text.isspace()


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


def apply_teacher_centered_add_by_group(
    teacher: torch.Tensor,
    teacher_mask: torch.Tensor,
    group_ids: torch.Tensor,
    weight: float = TEACHER_CENTERED_ADD_WEIGHT,
) -> torch.Tensor:
    result = teacher.clone()
    for group_id in torch.unique(group_ids[teacher_mask]):
        group_mask = teacher_mask & (group_ids == group_id)
        valid_teacher = teacher[group_mask]
        center = valid_teacher.mean(dim=0, keepdim=True)
        result[group_mask] = valid_teacher + (valid_teacher - center) * weight
    return result


class HybridDilBatchDataset(IterableDataset):
    def __init__(
        self,
        train_file: Path,
        config: DilConfig,
        tokenizer: HybridTokenizer,
        batch_size: int,
        read_chars: int,
        repeat: bool = True,
        max_samples: int = 0,
        teacher_tokenizer=None,
        teacher_max_tokens: int = NLLB_DEFAULT_MAX_ENCODER_TOKENS,
    ):
        super().__init__()
        self.train_file = train_file
        self.context_radius = config.context_radius
        self.context_size = config.context_size
        self.max_surface_pieces_per_unit = config.max_surface_pieces_per_unit
        self.surface_bucket_sizes = tuple(config.surface_bucket_sizes)
        self.pad_token_id = config.pad_token_id
        self.writer_stop_token_id = config.writer_stop_token_id
        self.batch_size = batch_size
        self.read_chars = read_chars
        self.repeat = repeat
        self.max_samples = max_samples
        self.tokenizer = tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.teacher_max_tokens = teacher_max_tokens
        self._carry_texts: list[str] = []
        self._carry_line_ids: list[int] = []
        self._carry_segments_by_text: list[list[TokenSegment]] = []
        self._carry_refs: list[BatchSampleRef] = []
        self._produced = 0

    def make_batch(
        self,
        texts: list[str],
        line_ids: list[int],
        segments_by_text: list[list[TokenSegment]],
        refs: list[BatchSampleRef],
    ):
        size = len(refs)
        context_rows: list[list[TokenSegment | None]] = []
        target_rows: list[list[list[int]]] = []
        teacher_text_indices = np.zeros((size,), dtype=np.int64)
        teacher_starts = np.zeros((size,), dtype=np.int64)
        teacher_ends = np.zeros((size,), dtype=np.int64)
        teacher_distill_mask = np.zeros((size,), dtype=np.bool_)
        source_line_ids = np.zeros((size,), dtype=np.int64)

        for row_idx, ref in enumerate(refs):
            segments = segments_by_text[ref.text_idx]
            token_idx = ref.token_idx
            segment = segments[token_idx]
            source_line_ids[row_idx] = line_ids[ref.text_idx]
            context_row: list[TokenSegment | None] = []
            for context_idx, offset in enumerate(context_offsets(self.context_radius)):
                source_idx = token_idx + offset
                if 0 <= source_idx < len(segments):
                    context_row.append(segments[source_idx])
                else:
                    context_row.append(None)
            context_rows.append(context_row)

            target_rows.append([segment_piece_ids(segment)])
            teacher_text_indices[row_idx] = ref.text_idx
            teacher_starts[row_idx] = segment.start
            teacher_ends[row_idx] = segment.end
            teacher_distill_mask[row_idx] = teacher_distill_segment(segment)

        return {
            "surface": pack_context_segments(
                context_rows,
                pad_token_id=self.pad_token_id,
                bucket_sizes=self.surface_bucket_sizes,
                max_pieces_per_unit=self.max_surface_pieces_per_unit,
            ),
            "labels": pack_writer_targets(
                target_rows,
                pad_token_id=self.pad_token_id,
                stop_token_id=self.writer_stop_token_id,
                surface_bucket_sizes=self.surface_bucket_sizes,
                max_pieces_per_unit=self.max_surface_pieces_per_unit,
            ),
            "teacher_texts": texts,
            "teacher_text_indices": torch.from_numpy(teacher_text_indices),
            "teacher_starts": torch.from_numpy(teacher_starts),
            "teacher_ends": torch.from_numpy(teacher_ends),
            "teacher_distill_mask": torch.from_numpy(teacher_distill_mask),
            "source_line_ids": torch.from_numpy(source_line_ids),
        }

    def iter_once(self, worker_id: int, worker_count: int):
        texts = self._carry_texts
        line_ids = self._carry_line_ids
        segments_by_text = self._carry_segments_by_text
        refs = self._carry_refs
        self._carry_texts = []
        self._carry_line_ids = []
        self._carry_segments_by_text = []
        self._carry_refs = []

        for text_idx, (source_line_id, text) in enumerate(
            stream_teacher_text_items(
                self.train_file,
                self.read_chars,
                self.teacher_tokenizer,
                self.teacher_max_tokens,
            )
        ):
            if text_idx % worker_count != worker_id:
                continue
            segments = trainable_segments(self.tokenizer, text, self.max_surface_pieces_per_unit)
            if not segments:
                continue

            local_text_idx = len(texts)
            texts.append(text)
            line_ids.append(source_line_id)
            segments_by_text.append(segments)
            for token_idx in range(len(segments)):
                refs.append(BatchSampleRef(local_text_idx, token_idx))
                self._produced += 1
                if len(refs) == self.batch_size:
                    yield self.make_batch(texts, line_ids, segments_by_text, refs)
                    texts, line_ids, segments_by_text, refs = [], [], [], []
                    if token_idx + 1 < len(segments):
                        local_text_idx = 0
                        texts.append(text)
                        line_ids.append(source_line_id)
                        segments_by_text.append(segments)
                if self.max_samples > 0 and self._produced >= self.max_samples:
                    if refs:
                        yield self.make_batch(texts, line_ids, segments_by_text, refs)
                    return

        if refs and not self.repeat:
            yield self.make_batch(texts, line_ids, segments_by_text, refs)
        elif refs:
            self._carry_texts = texts
            self._carry_line_ids = line_ids
            self._carry_segments_by_text = segments_by_text
            self._carry_refs = refs

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        while True:
            yielded = False
            for batch in self.iter_once(worker_id, worker_count):
                yielded = True
                yield batch
            if not yielded and not self._carry_refs:
                raise ValueError(f"{self.train_file} produced no trainable tokenizer segments")
            if not self.repeat:
                return


class ResidentDilBatcher:
    def __init__(
        self,
        batches: list[dict],
        batch_size: int,
        device: torch.device,
        seed: int,
    ):
        if not batches:
            raise ValueError("resident Dil data has no batches")
        self.batches = [
            {
                key: move_to_device(value, device)
                for key, value in batch.items()
            }
            for batch in batches
        ]
        self.batch_size = batch_size
        self.device = device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    @classmethod
    def materialize_hybrid_batch(cls, dataset: HybridDilBatchDataset, batch: dict, teacher: "NllbTeacher") -> dict:
        teacher_layers, teacher_mask = teacher.teacher_layers(batch)
        batch["teacher_layers"] = teacher_layers.detach().cpu()
        batch["teacher_mask"] = teacher_mask.detach().cpu()
        return {
            key: value.detach().cpu() if hasattr(value, "detach") else value
            for key, value in batch.items()
        }

    @classmethod
    def from_dataset(
        cls,
        dataset: HybridDilBatchDataset,
        teacher: "NllbTeacher",
        batch_size: int,
        device: torch.device,
        seed: int,
    ) -> "ResidentDilBatcher":
        batches = []
        for batch in dataset.iter_once(worker_id=0, worker_count=1):
            batches.append(cls.materialize_hybrid_batch(dataset, batch, teacher))
        if dataset._carry_refs:
            batch = dataset.make_batch(
                dataset._carry_texts,
                dataset._carry_line_ids,
                dataset._carry_segments_by_text,
                dataset._carry_refs,
            )
            batches.append(cls.materialize_hybrid_batch(dataset, batch, teacher))
        return cls(batches, batch_size=batch_size, device=device, seed=seed)

    def __iter__(self):
        return self

    def __next__(self):
        batch_idx = torch.randint(
            len(self.batches),
            (1,),
            generator=self.generator,
            device=self.device,
        ).item()
        return self.batches[int(batch_idx)]


class ReadyParquetDilBatchDataset(IterableDataset):
    def __init__(
        self,
        parquet_path: Path,
        config: DilConfig,
        batch_size: int,
        repeat: bool = True,
        max_samples: int = 0,
    ):
        super().__init__()
        raise ValueError("Dilnaz-ready fixed surface parquet is not supported; regenerate packed surface streaming data")
        self.parquet_path = parquet_path
        self.context_size = config.context_size
        self.max_surface_pieces_per_unit = config.max_surface_pieces_per_unit
        self.teacher_layer_count = len(NLLB_LAYER_GROUPS)
        self.teacher_dim = 1024
        self.batch_size = batch_size
        self.repeat = repeat
        self.max_samples = max_samples
        self.columns = list(DILNAZ_READY_REQUIRED_COLUMNS)
        self._carry: list[dict] = []
        self._produced = 0

    def rows_from_record_batch(self, record_batch) -> list[dict]:
        columns = {
            name: record_batch.column(record_batch.schema.get_field_index(name)).to_pylist()
            for name in self.columns
        }
        return [
            {name: columns[name][row_idx] for name in self.columns}
            for row_idx in range(record_batch.num_rows)
        ]

    def make_batch(self, rows: list[dict]) -> dict:
        raise ValueError("packed surface parquet batching is not implemented; use streaming data")

    def iter_once(self, worker_id: int, worker_count: int):
        rows = self._carry
        self._carry = []
        parquet = pq.ParquetFile(self.parquet_path)
        for batch_idx, record_batch in enumerate(
            parquet.iter_batches(batch_size=self.batch_size, columns=self.columns)
        ):
            if batch_idx % worker_count != worker_id:
                continue
            for row in self.rows_from_record_batch(record_batch):
                rows.append(row)
                self._produced += 1
                if len(rows) == self.batch_size:
                    yield self.make_batch(rows)
                    rows = []
                if self.max_samples > 0 and self._produced >= self.max_samples:
                    if rows:
                        yield self.make_batch(rows)
                    return

        if rows and not self.repeat:
            yield self.make_batch(rows)
        elif rows:
            self._carry = rows

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        while True:
            yielded = False
            for batch in self.iter_once(worker_id, worker_count):
                yielded = True
                yield batch
            if not yielded and not self._carry:
                raise ValueError(f"{self.parquet_path} produced no Dilnaz-ready rows")
            if not self.repeat:
                return


class ResidentReadyParquetBatcher(ResidentDilBatcher):
    @classmethod
    def from_dataset(
        cls,
        dataset: ReadyParquetDilBatchDataset,
        batch_size: int,
        device: torch.device,
        seed: int,
    ) -> "ResidentReadyParquetBatcher":
        batches = [
            {key: value.detach().cpu() for key, value in batch.items()}
            for batch in dataset.iter_once(worker_id=0, worker_count=1)
        ]
        if dataset._carry:
            batches.append(
                {
                    key: value.detach().cpu()
                    for key, value in dataset.make_batch(dataset._carry).items()
                }
            )
        return cls(batches, batch_size=batch_size, device=device, seed=seed)


class ResidentDilEvalLoader:
    def __init__(self, batcher: ResidentDilBatcher):
        self.batches = batcher.batches

    def __iter__(self):
        yield from self.batches


def make_dil_batch_loader(dataset, num_workers: int, pin_memory: bool, prefetch_factor: int):
    loader_kwargs = {"batch_size": None, "num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = True
    return DataLoader(dataset, **loader_kwargs)


class NllbTeacher:
    def __init__(
        self,
        model_name: str,
        src_lang: str,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = src_lang
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=dtype).to(device)
        self.model.eval()
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_encoder_tokens = int(
            getattr(self.model.config, "max_position_embeddings", NLLB_DEFAULT_MAX_ENCODER_TOKENS)
        )

    def piece_positions(self, input_ids: list[int], offsets) -> list[tuple[str, int, int, int]]:
        pieces = self.tokenizer.convert_ids_to_tokens(input_ids)
        return [
            (piece, int(offset[0]), int(offset[1]), token_idx)
            for token_idx, (piece, offset) in enumerate(zip(pieces, offsets))
            if int(offset[0]) != int(offset[1])
        ]

    def align_positions(
        self,
        starts: list[int],
        ends: list[int],
        pieces: list[tuple[str, int, int, int]],
    ) -> list[list[int]]:
        return align_spans_to_pieces(starts, ends, pieces)

    def teacher_layers(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        texts = batch["teacher_texts"]
        text_indices = batch["teacher_text_indices"].detach().cpu().tolist()
        starts = batch["teacher_starts"].detach().cpu().tolist()
        ends = batch["teacher_ends"].detach().cpu().tolist()
        distill_mask = batch["teacher_distill_mask"].detach().cpu().tolist()
        sample_count = len(starts)
        teacher = torch.zeros(
            (sample_count, len(NLLB_LAYER_GROUPS), self.model.config.d_model),
            dtype=torch.float32,
            device=self.device,
        )
        teacher_mask = torch.zeros((sample_count,), dtype=torch.bool, device=self.device)

        sample_rows_by_text: list[list[int]] = [[] for _ in texts]
        for row_idx, text_idx in enumerate(text_indices):
            sample_rows_by_text[text_idx].append(row_idx)

        for batch_start in range(0, len(texts), self.batch_size):
            batch_texts = texts[batch_start : batch_start + self.batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            if encoded["input_ids"].shape[1] > self.max_encoder_tokens:
                raise ValueError(
                    f"NLLB input has {encoded['input_ids'].shape[1]} tokens; "
                    f"max_encoder_tokens={self.max_encoder_tokens}"
                )
            offsets_batch = encoded.pop("offset_mapping").tolist()
            input_ids_batch = encoded["input_ids"].tolist()
            inputs = {key: value.to(self.device) for key, value in encoded.items()}

            with torch.no_grad():
                outputs = self.model.get_encoder()(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )

            hidden_states = outputs.hidden_states
            for local_text_idx, (input_ids, offsets) in enumerate(zip(input_ids_batch, offsets_batch)):
                text_idx = batch_start + local_text_idx
                rows = sample_rows_by_text[text_idx]
                if not rows:
                    continue
                pieces = self.piece_positions(input_ids, offsets)
                alignments = self.align_positions(
                    [starts[row_idx] for row_idx in rows],
                    [ends[row_idx] for row_idx in rows],
                    pieces,
                )
                for row_idx, positions in zip(rows, alignments):
                    if not distill_mask[row_idx] or not positions:
                        continue
                    teacher_mask[row_idx] = True
                    hidden_positions = [pieces[position][3] for position in positions]
                    pos = torch.tensor(hidden_positions, dtype=torch.long, device=self.device)
                    for group_idx, layers in enumerate(NLLB_LAYER_GROUPS):
                        layer_vectors = [
                            hidden_states[layer][local_text_idx, pos].float().mean(dim=0)
                            for layer in layers
                        ]
                        teacher[row_idx, group_idx] = torch.stack(layer_vectors).mean(dim=0)

        group_ids = batch["teacher_text_indices"].to(self.device, dtype=torch.long)
        teacher = apply_teacher_centered_add_by_group(teacher, teacher_mask, group_ids)
        return teacher, teacher_mask


def load_hybrid_tokenizer(vocab_path: Path | None = None) -> HybridTokenizer:
    return HybridTokenizer.from_file(vocab_path or default_vocab_path())
