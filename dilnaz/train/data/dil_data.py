from __future__ import annotations

import json
import random
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


class NllbTeacher:
    def __init__(
        self,
        model_name: str,
        src_lang: str,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int = 64,
    ):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = src_lang
        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=torch.float32)
            .to(device)
            .eval()
        )
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_encoder_tokens = int(getattr(self.model.config, "max_position_embeddings", NLLB_DEFAULT_MAX_ENCODER_TOKENS))
        self.layer_groups = NLLB_LAYER_GROUPS

    def teacher_layers(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
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

        for batch_start in range(0, len(unique_texts), self.batch_size):
            texts_batch = unique_texts[batch_start : batch_start + self.batch_size]
            encoded = self.tokenizer(
                texts_batch,
                return_tensors="pt",
                return_offsets_mapping=True,
                padding=True,
            )
            if encoded["input_ids"].shape[1] > self.max_encoder_tokens:
                raise ValueError(
                    f"NLLB input has {encoded['input_ids'].shape[1]} tokens; "
                    f"max_encoder_tokens={self.max_encoder_tokens}"
                )
            offsets = encoded.pop("offset_mapping")
            inputs = {key: value.to(self.device) for key, value in encoded.items()}
            pieces = [
                [
                    (
                        self.tokenizer.convert_ids_to_tokens(int(inputs["input_ids"][row, pos])),
                        int(offsets[row, pos, 0]),
                        int(offsets[row, pos, 1]),
                        int(pos),
                    )
                    for pos in range(int(inputs["input_ids"].shape[1]))
                    if int(offsets[row, pos, 0]) != int(offsets[row, pos, 1])
                ]
                for row in range(len(texts_batch))
            ]
            with torch.no_grad():
                outputs = self.model.get_encoder()(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states
            for uni_idx in range(len(texts_batch)):
                text = texts_batch[uni_idx]
                rows = rows_by_text[text]
                if not rows or not pieces[uni_idx]:
                    continue
                alignments = align_spans_to_pieces(
                    [starts[row_idx] for row_idx in rows],
                    [ends[row_idx] for row_idx in rows],
                    pieces[uni_idx],
                )
                for row_idx, positions in zip(rows, alignments):
                    if not distill_mask[row_idx] or not positions:
                        continue
                    teacher_mask[row_idx] = True
                    hidden_positions = [pieces[uni_idx][position][3] for position in positions]
                    pos = torch.tensor(hidden_positions, dtype=torch.long, device=self.device)
                    for group_idx, layers in enumerate(self.layer_groups):
                        layer_vectors = [
                            hidden_states[layer][uni_idx, pos].float().mean(dim=0)
                            for layer in layers
                        ]
                        teacher[row_idx, group_idx] = torch.stack(layer_vectors).mean(dim=0)
        teacher = apply_teacher_centered_add(teacher, teacher_mask)
        return teacher, teacher_mask


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
    ) -> dict:
        surface = pack_token_units(
            unit_rows,
            pad_token_id=self.pad_token_id,
            bucket_sizes=self.surface_bucket_sizes,
            max_pieces_per_unit=self.max_pieces_per_unit,
        )
        batch: dict = {
            "surface": surface,
            "labels": pack_writer_targets(
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
                produced += 1
                if len(unit_rows) >= self.batch_size:
                    yield self.make_batch(unit_rows, target_rows, texts, line_ids, text_indices, starts, ends, distill_mask)
                    unit_rows = []
                    target_rows = []
                    texts = []
                    line_ids = []
                    text_indices = []
                    starts = []
                    ends = []
                    distill_mask = []
                if self.max_samples > 0 and produced >= self.max_samples:
                    if unit_rows:
                        yield self.make_batch(unit_rows, target_rows, texts, line_ids, text_indices, starts, ends, distill_mask)
                    return

        if unit_rows:
            yield self.make_batch(unit_rows, target_rows, texts, line_ids, text_indices, starts, ends, distill_mask)

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
        teacher_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for batch in dataset.iter_once(worker_id=0, worker_count=1):
            text_list = batch.pop("texts", None)
            if text_list is None:
                teacher_layers, teacher_mask = teacher.teacher_layers(batch)
            else:
                missing_texts = list(dict.fromkeys(text for text in text_list if text not in teacher_cache))
                if missing_texts:
                    missing_layers, missing_mask = teacher.teacher_layers(batch, texts=missing_texts)
                    for row_idx, text in enumerate(missing_texts):
                        teacher_cache[text] = (missing_layers[row_idx], missing_mask[row_idx])
                teacher_layers = torch.stack([teacher_cache[text][0] for text in text_list], dim=0)
                teacher_mask = torch.stack([teacher_cache[text][1] for text in text_list], dim=0)
            batch["teacher_layers"] = teacher_layers
            batch["teacher_mask"] = teacher_mask
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
