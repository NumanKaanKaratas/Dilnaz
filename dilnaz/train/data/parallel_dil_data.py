from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from dilnaz.train.data.dil_data import (
    NLLB_DEFAULT_MAX_ENCODER_TOKENS,
    NLLB_LAYER_GROUPS,
    align_spans_to_pieces,
    apply_teacher_centered_add_by_group,
    context_offsets,
    overlaps,
    segment_piece_ids,
    teacher_distill_segment,
    trainable_segments,
)
from dilnaz.models.dil import DilConfig
from dilnaz.tokenization import HybridTokenizer, TokenSegment


ALIGN_THRESHOLD = 1e-4
MAX_TARGET_DISTANCE = 3
MAX_TARGETS_PER_SOURCE = 3
MAX_PHRASE_TARGET_SPAN = 8
TARGET_MULTI_OWNER_RATIO = 0.995
TARGET_MULTI_OWNER_EPS = 1e-6
SOURCE_SIDE = 0
TARGET_SIDE = 1
DEFAULT_PARALLEL_NLLB_MODEL = "facebook/nllb-200-3.3B"
DEFAULT_SOURCE_LANG = "tur_Latn"
DEFAULT_TARGET_LANG = "eng_Latn"
TARGET_PHRASE_PREFIXES = frozenset(
    {
        "a",
        "an",
        "the",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "from",
        "by",
        "with",
        "without",
        "into",
        "onto",
        "over",
        "under",
        "until",
        "before",
        "after",
        "during",
        "through",
        "between",
        "among",
        "about",
        "as",
        "than",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
    }
)


@dataclass(frozen=True)
class ParallelTextPair:
    index: int
    source_text: str
    target_text: str


@dataclass(frozen=True)
class ParallelPairSegments:
    pair: ParallelTextPair
    source_segments: list[TokenSegment]
    target_segments: list[TokenSegment]


@dataclass(frozen=True)
class ParallelBatchRow:
    item: ParallelPairSegments
    side_id: int
    token_idx: int


@dataclass(frozen=True)
class EncoderPiece:
    text: str
    start: int
    end: int
    encoder_index: int


@dataclass(frozen=True)
class EncodedText:
    hidden_states: tuple[torch.Tensor, ...]
    align_hidden: torch.Tensor
    pieces: list[EncoderPiece]


@dataclass(frozen=True)
class RowAlignment:
    source_row: int
    target_row: int
    score: float


@dataclass(frozen=True)
class ParallelAlignmentGroup:
    source_rows: tuple[int, ...]
    target_rows: tuple[int, ...]
    score: float


def parse_parallel_line(line: str) -> tuple[str, str] | None:
    line = line.rstrip("\n")
    parts = line.split("\t")
    if len(parts) >= 4 and parts[0] == "eng" and parts[1] == "tur":
        return "\t".join(parts[3:]), parts[2]
    if len(parts) == 3 and parts[0] == "eng" and parts[1] == "tur":
        payload = parts[2]
    elif line.startswith("eng tur "):
        payload = line[len("eng tur ") :]
    else:
        return None
    split = re.split(r"\s{2,}", payload, maxsplit=1)
    if len(split) != 2:
        return None
    return split[1], split[0]


def iter_parallel_pairs(path: Path) -> Iterator[ParallelTextPair]:
    emitted = False
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            parsed = parse_parallel_line(line)
            if parsed is None:
                continue
            source_text, target_text = parsed
            emitted = True
            yield ParallelTextPair(line_idx, source_text=source_text, target_text=target_text)
    if not emitted:
        raise ValueError(f"{path} produced no eng/tur parallel rows")


def contiguous_runs(indices: list[int], positions: dict[int, int]) -> list[tuple[int, ...]]:
    runs = []
    for row_idx in sorted(indices, key=lambda idx: positions[idx]):
        if not runs or positions[row_idx] != positions[runs[-1][-1]] + 1:
            runs.append([row_idx])
        else:
            runs[-1].append(row_idx)
    return [tuple(run) for run in runs]


def rows_by_pair_side(batch: dict) -> dict[tuple[int, int], list[int]]:
    pair_indices = batch["row_pair_indices"].detach().cpu().tolist()
    side_ids = batch["row_side_ids"].detach().cpu().tolist()
    token_indices = batch["row_token_indices"].detach().cpu().tolist()
    rows: dict[tuple[int, int], list[int]] = {}
    for row_idx, (pair_idx, side_id) in enumerate(zip(pair_indices, side_ids)):
        rows.setdefault((pair_idx, side_id), []).append(row_idx)
    for key, values in rows.items():
        values.sort(key=lambda row_idx: token_indices[row_idx])
    return rows


def row_positions(batch: dict) -> dict[int, int]:
    return {
        row_idx: token_idx
        for row_idx, token_idx in enumerate(batch["row_token_indices"].detach().cpu().tolist())
    }


def write_segment(
    target: np.ndarray,
    mask: np.ndarray,
    row_idx: int,
    context_idx: int,
    segment: TokenSegment,
    max_word_bytes: int,
):
    piece_ids = np.asarray(segment_piece_ids(segment), dtype=np.int64)
    width = piece_ids.shape[0]
    if width <= 0 or width > max_word_bytes:
        return
    target[row_idx, context_idx, :width] = piece_ids
    mask[row_idx, context_idx, :width] = True


class ParallelDilBatchDataset(IterableDataset):
    def __init__(
        self,
        train_file: Path,
        config: DilConfig,
        tokenizer: HybridTokenizer,
        batch_size: int,
        repeat: bool = True,
        max_samples: int = 0,
    ):
        super().__init__()
        self.train_file = train_file
        self.context_radius = config.context_radius
        self.context_size = config.context_size
        self.max_word_bytes = config.max_word_bytes
        self.writer_max_positions = config.writer_max_positions
        self.pad_token_id = config.pad_token_id
        self.writer_stop_token_id = config.writer_stop_token_id
        self.batch_size = batch_size
        self.repeat = repeat
        self.max_samples = max_samples
        self.tokenizer = tokenizer
        self._carry_rows: list[ParallelBatchRow] = []
        self._produced_pairs = 0

    def pair_rows(self, item: ParallelPairSegments) -> Iterator[ParallelBatchRow]:
        source_count = len(item.source_segments)
        target_count = len(item.target_segments)
        for token_idx in range(max(source_count, target_count)):
            if token_idx < source_count:
                yield ParallelBatchRow(item, SOURCE_SIDE, token_idx)
            if token_idx < target_count:
                yield ParallelBatchRow(item, TARGET_SIDE, token_idx)

    def make_batch(self, rows: list[ParallelBatchRow]) -> dict:
        size = len(rows)
        input_ids = np.full(
            (size, self.context_size, self.max_word_bytes),
            self.pad_token_id,
            dtype=np.int64,
        )
        word_masks = np.zeros((size, self.context_size, self.max_word_bytes), dtype=np.bool_)
        labels = np.full((size, self.writer_max_positions), -100, dtype=np.int64)
        teacher_text_indices = np.zeros((size,), dtype=np.int64)
        teacher_starts = np.zeros((size,), dtype=np.int64)
        teacher_ends = np.zeros((size,), dtype=np.int64)
        teacher_distill_mask = np.zeros((size,), dtype=np.bool_)
        row_pair_indices = np.zeros((size,), dtype=np.int64)
        row_side_ids = np.zeros((size,), dtype=np.int64)
        row_token_indices = np.zeros((size,), dtype=np.int64)
        source_line_ids = np.zeros((size,), dtype=np.int64)
        teacher_texts: list[str] = []
        row_texts: list[str] = []
        pair_slots: dict[int, int] = {}
        pair_items: list[ParallelPairSegments] = []

        for row in rows:
            item_key = id(row.item)
            if item_key in pair_slots:
                continue
            pair_slots[item_key] = len(pair_items)
            pair_items.append(row.item)

        pair_text_indices = np.zeros((len(pair_items), 2), dtype=np.int64)
        for pair_idx, item in enumerate(pair_items):
            source_text_idx = len(teacher_texts)
            teacher_texts.append(item.pair.source_text)
            target_text_idx = len(teacher_texts)
            teacher_texts.append(item.pair.target_text)
            pair_text_indices[pair_idx] = (source_text_idx, target_text_idx)

        for row_idx, row in enumerate(rows):
            pair_idx = pair_slots[id(row.item)]
            segments = row.item.source_segments if row.side_id == SOURCE_SIDE else row.item.target_segments
            text_idx = pair_text_indices[pair_idx, row.side_id]
            segment = segments[row.token_idx]
            for context_idx, offset in enumerate(context_offsets(self.context_radius)):
                source_idx = row.token_idx + offset
                if 0 <= source_idx < len(segments):
                    write_segment(
                        input_ids,
                        word_masks,
                        row_idx,
                        context_idx,
                        segments[source_idx],
                        self.max_word_bytes,
                    )

            piece_ids = np.asarray(segment_piece_ids(segment), dtype=np.int64)
            labels[row_idx, : piece_ids.shape[0]] = piece_ids
            labels[row_idx, piece_ids.shape[0]] = self.writer_stop_token_id
            teacher_text_indices[row_idx] = text_idx
            teacher_starts[row_idx] = segment.start
            teacher_ends[row_idx] = segment.end
            teacher_distill_mask[row_idx] = teacher_distill_segment(segment)
            row_pair_indices[row_idx] = pair_idx
            row_side_ids[row_idx] = row.side_id
            row_token_indices[row_idx] = row.token_idx
            source_line_ids[row_idx] = row.item.pair.index
            row_texts.append(segment.text)

        teacher_text_side_ids = np.asarray(
            [side_id for _ in pair_items for side_id in (SOURCE_SIDE, TARGET_SIDE)],
            dtype=np.int64,
        )
        return {
            "input_ids": torch.from_numpy(input_ids),
            "word_masks": torch.from_numpy(word_masks),
            "labels": torch.from_numpy(labels),
            "teacher_texts": teacher_texts,
            "teacher_text_side_ids": torch.from_numpy(teacher_text_side_ids),
            "teacher_text_indices": torch.from_numpy(teacher_text_indices),
            "teacher_starts": torch.from_numpy(teacher_starts),
            "teacher_ends": torch.from_numpy(teacher_ends),
            "teacher_distill_mask": torch.from_numpy(teacher_distill_mask),
            "pair_text_indices": torch.from_numpy(pair_text_indices),
            "row_pair_indices": torch.from_numpy(row_pair_indices),
            "row_side_ids": torch.from_numpy(row_side_ids),
            "row_token_indices": torch.from_numpy(row_token_indices),
            "row_texts": row_texts,
            "source_line_ids": torch.from_numpy(source_line_ids),
        }

    def iter_once(self, worker_id: int, worker_count: int):
        rows = self._carry_rows
        self._carry_rows = []

        for pair in iter_parallel_pairs(self.train_file):
            if pair.index % worker_count != worker_id:
                continue
            source_segments = trainable_segments(self.tokenizer, pair.source_text, self.max_word_bytes)
            target_segments = trainable_segments(self.tokenizer, pair.target_text, self.max_word_bytes)
            if not source_segments or not target_segments:
                continue

            pair_item = ParallelPairSegments(pair, source_segments, target_segments)
            for row in self.pair_rows(pair_item):
                rows.append(row)
                if len(rows) == self.batch_size:
                    yield self.make_batch(rows)
                    rows = []
            self._produced_pairs += 1
            if self.max_samples > 0 and self._produced_pairs >= self.max_samples:
                if rows:
                    yield self.make_batch(rows)
                return

        if rows and not self.repeat:
            yield self.make_batch(rows)
        elif rows:
            self._carry_rows = rows

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        while True:
            yielded = False
            for batch in self.iter_once(worker_id, worker_count):
                yielded = True
                yield batch
            if not yielded and not self._carry_rows:
                raise ValueError(f"{self.train_file} produced no trainable parallel segments")
            if not self.repeat:
                return


def piece_positions(tokenizer, input_ids: list[int], offsets: list[list[int]]) -> list[EncoderPiece]:
    pieces = tokenizer.convert_ids_to_tokens(input_ids)
    return [
        EncoderPiece(piece, int(offset[0]), int(offset[1]), token_idx)
        for token_idx, (piece, offset) in enumerate(zip(pieces, offsets))
        if int(offset[0]) != int(offset[1])
    ]


def piece_to_rows(row_indices: list[int], batch: dict, pieces: list[EncoderPiece]) -> list[list[int]]:
    max_piece_index = max((piece.encoder_index for piece in pieces), default=-1)
    mapping: list[list[int]] = [[] for _ in range(max_piece_index + 1)]
    starts = batch["teacher_starts"].detach().cpu().tolist()
    ends = batch["teacher_ends"].detach().cpu().tolist()
    distill_mask = batch["teacher_distill_mask"].detach().cpu().tolist()
    for row_idx in row_indices:
        if not distill_mask[row_idx]:
            continue
        for piece in pieces:
            if overlaps(starts[row_idx], ends[row_idx], piece.start, piece.end):
                mapping[piece.encoder_index].append(row_idx)
    return mapping


def target_claims(groups: list[ParallelAlignmentGroup]) -> dict[int, set[int]]:
    claims: dict[int, set[int]] = {}
    for group_idx, group in enumerate(groups):
        for row_idx in group.target_rows:
            claims.setdefault(row_idx, set()).add(group_idx)
    return claims


def target_span(groups: list[ParallelAlignmentGroup], positions: dict[int, int]) -> tuple[int, int]:
    indices = [positions[row_idx] for group in groups for row_idx in group.target_rows]
    return min(indices), max(indices)


def source_adjacent(left: ParallelAlignmentGroup, right: ParallelAlignmentGroup, positions: dict[int, int]) -> bool:
    return positions[left.source_rows[-1]] + 1 == positions[right.source_rows[0]]


def target_span_is_private(
    pair_target_rows: list[int],
    start: int,
    end: int,
    group_indices: set[int],
    claims: dict[int, set[int]],
    positions: dict[int, int],
) -> bool:
    return all(
        not claims.get(row_idx, set()) - group_indices
        for row_idx in pair_target_rows
        if start <= positions[row_idx] <= end
    )


def should_merge_phrase_groups(
    current: list[tuple[int, ParallelAlignmentGroup]],
    candidate: tuple[int, ParallelAlignmentGroup],
    pair_target_rows: list[int],
    claims: dict[int, set[int]],
    positions: dict[int, int],
) -> bool:
    previous = current[-1][1]
    candidate_idx, candidate_group = candidate
    if not source_adjacent(previous, candidate_group, positions):
        return False

    group_indices = {group_idx for group_idx, _ in current} | {candidate_idx}
    start, end = target_span([group for _, group in current] + [candidate_group], positions)
    span_width = sum(1 for row_idx in pair_target_rows if start <= positions[row_idx] <= end)
    if span_width > MAX_PHRASE_TARGET_SPAN:
        return False
    if not target_span_is_private(pair_target_rows, start, end, group_indices, claims, positions):
        return False

    previous_start, _ = target_span([previous], positions)
    candidate_start, _ = target_span([candidate_group], positions)
    inverted_order = candidate_start < previous_start
    unclaimed_gap = any(
        not claims.get(row_idx)
        for row_idx in pair_target_rows
        if start <= positions[row_idx] <= end
    )
    return inverted_order or unclaimed_gap


def phrase_prefixed_target_rows(
    target_rows: tuple[int, ...],
    pair_target_rows: list[int],
    batch: dict,
    claims: dict[int, set[int]],
    positions: dict[int, int],
) -> tuple[int, ...]:
    start = min(positions[row_idx] for row_idx in target_rows)
    end = max(positions[row_idx] for row_idx in target_rows)
    if start == end:
        return target_rows
    row_by_position = {positions[row_idx]: row_idx for row_idx in pair_target_rows}
    content_positions = sorted(row_by_position)
    start_offset = content_positions.index(start)
    row_texts = batch["row_texts"]
    while start_offset > 0:
        prefix_position = content_positions[start_offset - 1]
        prefix_row = row_by_position[prefix_position]
        prefix = row_texts[prefix_row].lower()
        if claims.get(prefix_row) or prefix not in TARGET_PHRASE_PREFIXES:
            break
        start = prefix_position
        start_offset -= 1
    return tuple(row_by_position[position] for position in content_positions if start <= position <= end)


def build_alignment_group(
    source_rows: tuple[int, ...],
    target_rows: tuple[int, ...],
    score: float,
    batch: dict,
) -> ParallelAlignmentGroup | None:
    distill_mask = batch["teacher_distill_mask"].detach().cpu().tolist()
    source = tuple(row_idx for row_idx in source_rows if distill_mask[row_idx])
    target = tuple(row_idx for row_idx in target_rows if distill_mask[row_idx])
    if not source or not target:
        return None
    return ParallelAlignmentGroup(source, target, score)


def expand_phrase_groups(
    base_groups: list[ParallelAlignmentGroup],
    pair_target_rows: list[int],
    batch: dict,
    positions: dict[int, int],
) -> list[ParallelAlignmentGroup]:
    claims = target_claims(base_groups)
    ordered_groups = sorted(
        enumerate(base_groups),
        key=lambda item: (positions[item[1].source_rows[0]], item[0]),
    )
    merged_runs: list[list[tuple[int, ParallelAlignmentGroup]]] = []
    current_run: list[tuple[int, ParallelAlignmentGroup]] = []
    for item in ordered_groups:
        if current_run and should_merge_phrase_groups(current_run, item, pair_target_rows, claims, positions):
            current_run.append(item)
        else:
            if current_run:
                merged_runs.append(current_run)
            current_run = [item]
    if current_run:
        merged_runs.append(current_run)

    groups = []
    for run in merged_runs:
        source_rows = tuple(row_idx for _, group in run for row_idx in group.source_rows)
        raw_target_rows = tuple(
            sorted({row_idx for _, group in run for row_idx in group.target_rows}, key=lambda idx: positions[idx])
        )
        target_rows = phrase_prefixed_target_rows(raw_target_rows, pair_target_rows, batch, claims, positions)
        groups.append(
            ParallelAlignmentGroup(
                source_rows=source_rows,
                target_rows=target_rows,
                score=max(group.score for _, group in run),
            )
        )
    return groups


def group_row_alignments(
    alignments: list[RowAlignment],
    pair_target_rows: list[int],
    batch: dict,
    positions: dict[int, int],
) -> list[ParallelAlignmentGroup]:
    if not alignments:
        return []

    source_primary: dict[int, RowAlignment] = {}
    target_best_score: dict[int, float] = {}
    for alignment in alignments:
        if alignment.source_row not in source_primary or alignment.score > source_primary[alignment.source_row].score:
            source_primary[alignment.source_row] = alignment
        target_best_score[alignment.target_row] = max(
            alignment.score,
            target_best_score.get(alignment.target_row, alignment.score),
        )

    target_owners: dict[int, set[int]] = {}
    for alignment in alignments:
        best_score = target_best_score[alignment.target_row]
        if alignment.score + TARGET_MULTI_OWNER_EPS >= best_score * TARGET_MULTI_OWNER_RATIO:
            target_owners.setdefault(alignment.target_row, set()).add(alignment.source_row)

    kept_by_source: dict[int, list[RowAlignment]] = {}
    for alignment in alignments:
        if alignment.source_row not in target_owners[alignment.target_row]:
            continue
        primary = source_primary[alignment.source_row]
        if abs(positions[alignment.target_row] - positions[primary.target_row]) > MAX_TARGET_DISTANCE:
            continue
        kept_by_source.setdefault(alignment.source_row, []).append(alignment)

    for source_row, items in list(kept_by_source.items()):
        kept_by_source[source_row] = sorted(items, key=lambda item: (-item.score, positions[item.target_row]))[
            :MAX_TARGETS_PER_SOURCE
        ]

    sources_by_primary_target: dict[int, list[int]] = {}
    for source_row, primary in source_primary.items():
        if source_row in kept_by_source:
            sources_by_primary_target.setdefault(primary.target_row, []).append(source_row)

    base_groups = []
    for _, source_rows_ in sorted(
        sources_by_primary_target.items(),
        key=lambda item: (min(positions[row_idx] for row_idx in item[1]), positions[item[0]]),
    ):
        for source_rows in contiguous_runs(source_rows_, positions):
            selected_alignments = [
                alignment
                for source_row in source_rows
                for alignment in kept_by_source[source_row]
            ]
            target_rows = tuple(
                sorted({alignment.target_row for alignment in selected_alignments}, key=lambda idx: positions[idx])
            )
            group = build_alignment_group(
                source_rows,
                target_rows,
                max(alignment.score for alignment in selected_alignments),
                batch,
            )
            if group is not None:
                base_groups.append(group)
    return expand_phrase_groups(base_groups, pair_target_rows, batch, positions)


def apply_one_to_one_shared_teacher(
    teacher: torch.Tensor,
    teacher_mask: torch.Tensor,
    groups: list[ParallelAlignmentGroup],
) -> torch.Tensor:
    result = teacher.clone()
    for group in groups:
        if len(group.source_rows) != 1 or len(group.target_rows) != 1:
            continue
        source_row = group.source_rows[0]
        target_row = group.target_rows[0]
        if not bool(teacher_mask[source_row]) or not bool(teacher_mask[target_row]):
            continue
        shared = (teacher[source_row] + teacher[target_row]) * 0.5
        result[source_row] = shared
        result[target_row] = shared
    return result


def alignment_groups_to_tensors(groups: list[ParallelAlignmentGroup], device: torch.device) -> dict[str, torch.Tensor]:
    group_count = len(groups)
    max_source = max((len(group.source_rows) for group in groups), default=1)
    max_target = max((len(group.target_rows) for group in groups), default=1)
    source_rows = torch.full((group_count, max_source), -1, dtype=torch.long, device=device)
    target_rows = torch.full((group_count, max_target), -1, dtype=torch.long, device=device)
    source_mask = torch.zeros((group_count, max_source), dtype=torch.bool, device=device)
    target_mask = torch.zeros((group_count, max_target), dtype=torch.bool, device=device)
    scores = torch.zeros((group_count,), dtype=torch.float32, device=device)
    for group_idx, group in enumerate(groups):
        source_rows[group_idx, : len(group.source_rows)] = torch.tensor(group.source_rows, dtype=torch.long, device=device)
        target_rows[group_idx, : len(group.target_rows)] = torch.tensor(group.target_rows, dtype=torch.long, device=device)
        source_mask[group_idx, : len(group.source_rows)] = True
        target_mask[group_idx, : len(group.target_rows)] = True
        scores[group_idx] = float(group.score)
    return {
        "parallel_source_rows": source_rows,
        "parallel_source_mask": source_mask,
        "parallel_target_rows": target_rows,
        "parallel_target_mask": target_mask,
        "parallel_alignment_scores": scores,
    }


def grouped_mean(vectors: torch.Tensor, row_indices: torch.Tensor, row_mask: torch.Tensor) -> torch.Tensor:
    safe_indices = row_indices.clamp_min(0)
    gathered = vectors.index_select(0, safe_indices.reshape(-1)).reshape(*safe_indices.shape, vectors.shape[-1])
    weights = row_mask.unsqueeze(-1).to(gathered.dtype)
    return (gathered * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def parallel_alignment_loss(mean: torch.Tensor, batch: dict) -> torch.Tensor:
    source_rows = batch["parallel_source_rows"].to(mean.device)
    target_rows = batch["parallel_target_rows"].to(mean.device)
    if source_rows.shape[0] == 0:
        return mean.new_zeros(())
    source_vectors = grouped_mean(mean, source_rows, batch["parallel_source_mask"].to(mean.device))
    target_vectors = grouped_mean(mean, target_rows, batch["parallel_target_mask"].to(mean.device))
    return (1.0 - F.cosine_similarity(source_vectors.float(), target_vectors.float(), dim=-1)).mean()


def parallel_total_loss(outputs, batch: dict, parallel_alignment_weight: float) -> tuple[torch.Tensor, torch.Tensor]:
    if outputs.loss is None:
        raise ValueError("DIL outputs.loss is required for parallel training")
    alignment_loss = parallel_alignment_loss(outputs.semantic, batch)
    return outputs.loss + alignment_loss * parallel_alignment_weight, alignment_loss


class ParallelNllbTeacher:
    def __init__(
        self,
        model_name: str,
        source_lang: str,
        target_lang: str,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        align_layer: int = -1,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=dtype).to(device)
        self.model.eval()
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.align_layer = align_layer
        self.max_encoder_tokens = int(
            getattr(self.model.config, "max_position_embeddings", NLLB_DEFAULT_MAX_ENCODER_TOKENS)
        )

    def set_lang(self, lang: str):
        if hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = lang

    def encode_texts(self, text_indices: list[int], texts: list[str], lang: str) -> dict[int, EncodedText]:
        encoded_texts = {}
        self.set_lang(lang)
        for batch_start in range(0, len(text_indices), self.batch_size):
            chunk_indices = text_indices[batch_start : batch_start + self.batch_size]
            batch_texts = [texts[text_idx] for text_idx in chunk_indices]
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
            hidden_states = tuple(state.float() for state in outputs.hidden_states)
            for local_idx, text_idx in enumerate(chunk_indices):
                text_hidden_states = tuple(layer[local_idx] for layer in hidden_states)
                encoded_texts[text_idx] = EncodedText(
                    hidden_states=text_hidden_states,
                    align_hidden=hidden_states[self.align_layer][local_idx],
                    pieces=piece_positions(self.tokenizer, input_ids_batch[local_idx], offsets_batch[local_idx]),
                )
        return encoded_texts

    def encode_batch_texts(self, batch: dict) -> dict[int, EncodedText]:
        texts = batch["teacher_texts"]
        side_ids = batch["teacher_text_side_ids"].detach().cpu().tolist()
        source_indices = [idx for idx, side_id in enumerate(side_ids) if side_id == SOURCE_SIDE]
        target_indices = [idx for idx, side_id in enumerate(side_ids) if side_id == TARGET_SIDE]
        encoded_texts = {}
        encoded_texts.update(self.encode_texts(source_indices, texts, self.source_lang))
        encoded_texts.update(self.encode_texts(target_indices, texts, self.target_lang))
        return encoded_texts

    def raw_teacher_layers(self, batch: dict, encoded_texts: dict[int, EncodedText]) -> tuple[torch.Tensor, torch.Tensor]:
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
        rows_by_text: list[list[int]] = [[] for _ in batch["teacher_texts"]]
        for row_idx, text_idx in enumerate(text_indices):
            rows_by_text[text_idx].append(row_idx)

        for text_idx, rows in enumerate(rows_by_text):
            if not rows:
                continue
            encoded = encoded_texts[text_idx]
            pieces = encoded.pieces
            alignments = align_spans_to_pieces(
                [starts[row_idx] for row_idx in rows],
                [ends[row_idx] for row_idx in rows],
                [(piece.text, piece.start, piece.end, piece.encoder_index) for piece in pieces],
            )
            for row_idx, positions in zip(rows, alignments):
                if not distill_mask[row_idx] or not positions:
                    continue
                teacher_mask[row_idx] = True
                hidden_positions = [pieces[position].encoder_index for position in positions]
                pos = torch.tensor(hidden_positions, dtype=torch.long, device=self.device)
                for group_idx, layers in enumerate(NLLB_LAYER_GROUPS):
                    layer_vectors = [
                        encoded.hidden_states[layer].index_select(0, pos).mean(dim=0)
                        for layer in layers
                    ]
                    teacher[row_idx, group_idx] = torch.stack(layer_vectors).mean(dim=0)

        group_ids = batch["teacher_text_indices"].to(self.device, dtype=torch.long)
        return apply_teacher_centered_add_by_group(teacher, teacher_mask, group_ids), teacher_mask

    def alignment_candidates(
        self,
        source_rows: list[int],
        target_rows: list[int],
        source_encoded: EncodedText,
        target_encoded: EncodedText,
        batch: dict,
    ) -> list[RowAlignment]:
        if not source_encoded.pieces or not target_encoded.pieces:
            return []

        source_positions = torch.tensor(
            [piece.encoder_index for piece in source_encoded.pieces],
            dtype=torch.long,
            device=self.device,
        )
        target_positions = torch.tensor(
            [piece.encoder_index for piece in target_encoded.pieces],
            dtype=torch.long,
            device=self.device,
        )
        source_vectors = source_encoded.align_hidden.index_select(0, source_positions)
        target_vectors = target_encoded.align_hidden.index_select(0, target_positions)
        dot_product = torch.matmul(source_vectors, target_vectors.T)
        softmax_fwd = torch.softmax(dot_product, dim=-1)
        softmax_bwd = torch.softmax(dot_product, dim=-2)
        align_matrix = (softmax_fwd > ALIGN_THRESHOLD) & (softmax_bwd > ALIGN_THRESHOLD)
        intersection_score = torch.sqrt(softmax_fwd * softmax_bwd)
        source_piece_to_rows = piece_to_rows(source_rows, batch, source_encoded.pieces)
        target_piece_to_rows = piece_to_rows(target_rows, batch, target_encoded.pieces)
        best_scores: dict[tuple[int, int], float] = {}
        for source_piece_idx, target_piece_idx in torch.nonzero(align_matrix, as_tuple=False).tolist():
            source_encoder_idx = source_encoded.pieces[source_piece_idx].encoder_index
            target_encoder_idx = target_encoded.pieces[target_piece_idx].encoder_index
            score = float(intersection_score[source_piece_idx, target_piece_idx].item())
            for source_row in source_piece_to_rows[source_encoder_idx]:
                for target_row in target_piece_to_rows[target_encoder_idx]:
                    key = (source_row, target_row)
                    best_scores[key] = max(score, best_scores.get(key, score))
        return [
            RowAlignment(source_row, target_row, score)
            for (source_row, target_row), score in sorted(best_scores.items())
        ]

    def alignment_groups(self, batch: dict, encoded_texts: dict[int, EncodedText]) -> list[ParallelAlignmentGroup]:
        pair_text_indices = batch["pair_text_indices"].detach().cpu().tolist()
        rows = rows_by_pair_side(batch)
        positions = row_positions(batch)
        distill_mask = batch["teacher_distill_mask"].detach().cpu().tolist()
        groups = []
        for pair_idx, (source_text_idx, target_text_idx) in enumerate(pair_text_indices):
            source_rows = rows.get((pair_idx, SOURCE_SIDE), [])
            target_rows = rows.get((pair_idx, TARGET_SIDE), [])
            target_content_rows = [row_idx for row_idx in target_rows if distill_mask[row_idx]]
            alignments = self.alignment_candidates(
                source_rows,
                target_rows,
                encoded_texts[source_text_idx],
                encoded_texts[target_text_idx],
                batch,
            )
            groups.extend(group_row_alignments(alignments, target_content_rows, batch, positions))
        return groups

    def materialize(self, batch: dict) -> dict:
        encoded_texts = self.encode_batch_texts(batch)
        teacher_layers, teacher_mask = self.raw_teacher_layers(batch, encoded_texts)
        groups = self.alignment_groups(batch, encoded_texts)
        batch["teacher_layers"] = apply_one_to_one_shared_teacher(teacher_layers, teacher_mask, groups)
        batch["teacher_mask"] = teacher_mask
        batch.update(alignment_groups_to_tensors(groups, self.device))
        return batch
