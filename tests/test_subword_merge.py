#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import string
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
from torch import Tensor

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(r"D:\Projects\Ai\DataCreator")
sys.path.insert(0, str(PROJECT_ROOT))

from converter_parts.embedding_runtime import (
    NLLB_13B_SPEC,
    NLLBTeacherWordEncoder,
    TeacherModelSpec,
    _collect_token_units,
    _resolve_device,
)

TUR = "tur_Latn"
ENG = "eng_Latn"
DEFAULT_GOLD_PATH = Path(__file__).resolve().with_name("subword_merge_gold_template.jsonl")
NLLB_33B_SPEC = TeacherModelSpec(
    label="NLLB 3.3B",
    hf_model_id="facebook/nllb-200-3.3B",
    formula="raw_nllb_subword_hidden_state",
)
NLLB_600M_SPEC = TeacherModelSpec(
    label="NLLB distilled 600M",
    hf_model_id="facebook/nllb-200-distilled-600M",
    formula="raw_nllb_subword_hidden_state",
)
PoolFn = Callable[[Tensor], Tensor]
def normalize_surface(text: str) -> str:
    stripped = text.strip().strip(string.punctuation + "“”‘’«»…")
    return " ".join(stripped.casefold().split())


def append_encoded_word(
    *,
    words: list["EncodedWord"],
    sentence: SentenceSpec,
    span: tuple[int, int],
    piece_group: list[int],
    hidden_states: Tensor,
) -> None:
    start, end = span
    surface = sentence.text[start:end].strip()
    if not surface:
        return
    words.append(
        EncodedWord(
            surface=surface,
            normalized_surface=normalize_surface(surface),
            sentence=sentence.text,
            language=sentence.language,
            span=(start, end),
            word_index=len(words),
            piece_indices=tuple(piece_group),
            layer_piece_vectors=torch.stack([hidden_states[:, index, :] for index in piece_group], dim=1),
        )
    )


def pool_mean(pieces: Tensor) -> Tensor:
    return pieces.mean(dim=0)


def pool_last(pieces: Tensor) -> Tensor:
    return pieces[-1]


def pool_first(pieces: Tensor) -> Tensor:
    return pieces[0]


def pool_max(pieces: Tensor) -> Tensor:
    return pieces.max(dim=0).values


def pool_w70(pieces: Tensor) -> Tensor:
    if pieces.shape[0] == 1:
        return pieces[0]
    weights = torch.zeros(pieces.shape[0], dtype=pieces.dtype, device=pieces.device)
    weights[0] = 0.7
    weights[1:] = 0.3 / (pieces.shape[0] - 1)
    return (pieces * weights.unsqueeze(1)).sum(dim=0)


POOLS: dict[str, PoolFn] = {
    "mean": pool_mean,
    "last": pool_last,
    "first": pool_first,
    "max": pool_max,
    "w70": pool_w70,
}
REPRESENTATIONS: dict[str, float] = {
    "word_only": 0.0,
    "ctx_avg_w025": 0.25,
    "ctx_avg_w050": 0.50,
    "ctx_avg_w100": 1.00,
    "target_plus_attended_context": -1.0,
    "target_minus_sentence_mean": -2.0,
    "target_minus_local_context_mean": -3.0,
    "target_concat_residual": -4.0,
    "concat_word_and_sentence_residual": -5.0,
    "centered_add_plus_local_projection_deflation": -6.0,
    "isolated_anchor_plus_sentence_residual": -7.0,
    "isolated_anchor_plus_local_residual": -8.0,
    "adaptive_anchor_local_residual": -9.0,
    "target_centered_add_w050": -10.0,
}

CENTERED_ADD_WEIGHT = 0.25
LOCAL_PROJECTION_DEFLATION_WEIGHT = 0.15
ANCHOR_SENTENCE_RESIDUAL_WEIGHT = 0.20
ANCHOR_LOCAL_RESIDUAL_WEIGHT = 0.30
ADAPTIVE_ANCHOR_SENTENCE_WEIGHT = 0.10
ADAPTIVE_ANCHOR_LOCAL_MAX_WEIGHT = 0.70
LOCAL_CONTEXT_DISTANCE_WEIGHTS = {
    1: 1.0,
    2: 0.5,
}


@dataclass(frozen=True, slots=True)
class SentenceSpec:
    text: str
    language: str


@dataclass(frozen=True, slots=True)
class TargetRef:
    sentence: str
    language: str
    surface: str
    occurrence: int = 0


@dataclass(frozen=True, slots=True)
class PairExpectation:
    label: str
    category: str
    relation: str
    left: TargetRef
    right: TargetRef
    note: str = ""


@dataclass(frozen=True, slots=True)
class EncodedWord:
    surface: str
    normalized_surface: str
    sentence: str
    language: str
    span: tuple[int, int]
    word_index: int
    piece_indices: tuple[int, ...]
    layer_piece_vectors: Tensor


@dataclass(frozen=True, slots=True)
class PairScore:
    label: str
    category: str
    relation: str
    representation: str
    similarity: float


@dataclass(frozen=True, slots=True)
class CategorySummary:
    category: str
    count: int
    pair_accuracy: float
    ranking_accuracy: float | None
    positive_mean: float | None
    negative_mean: float | None
    related_mean: float | None
    margin: float | None


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    pair_accuracy: float
    ranking_accuracy: float
    positive_mean: float
    negative_mean: float
    margin: float
    pair_scores: tuple[PairScore, ...]
    category_summaries: tuple[CategorySummary, ...]


class BenchmarkEncoder:
    label: str
    model_id: str
    device: torch.device

    def load(self) -> None:
        raise NotImplementedError

    def encode_sentence(self, sentence: SentenceSpec) -> tuple[list[str], list[EncodedWord], Tensor]:
        raise NotImplementedError


class NLLBBenchmarkEncoder(BenchmarkEncoder):
    def __init__(self, device: torch.device, model_spec: TeacherModelSpec) -> None:
        self.label = model_spec.label
        self.model_id = model_spec.hf_model_id
        self.device = device
        self.model_spec = model_spec
        self._encoders: dict[str, NLLBTeacherWordEncoder] = {}

    def load(self) -> None:
        for language in (TUR, ENG):
            encoder = NLLBTeacherWordEncoder(self.model_spec, source_language=language, device=self.device)
            encoder.load()
            self._encoders[language] = encoder

    def encode_sentence(self, sentence: SentenceSpec) -> tuple[list[str], list[EncodedWord], Tensor]:
        encoder = self._encoders[sentence.language]
        encoded = encoder.tokenizer(
            [sentence.text],
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True,
            padding=True,
        )
        model_inputs = {key: value.to(encoder.device) for key, value in encoded.items() if key != "offset_mapping"}

        with torch.inference_mode():
            model_output = encoder.model(**model_inputs, return_dict=True, output_hidden_states=True)
            hidden_states = torch.stack(model_output.hidden_states, dim=0)[:, 0, :, :].float().cpu()

        tokens = encoder.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        offsets = encoded["offset_mapping"][0].tolist()
        attention_mask = encoded["attention_mask"][0].tolist()
        words = collect_words(
            sentence=sentence,
            tokens=tokens,
            offsets=offsets,
            attention_mask=attention_mask,
            hidden_states=hidden_states,
        )
        return tokens, words, hidden_states


def pair_matches_expectation(similarity: float, relation: str) -> bool:
    if relation == "positive":
        return similarity >= 0.60
    if relation == "negative":
        return similarity <= 0.45
    if relation == "related":
        return 0.35 <= similarity <= 0.90
    raise ValueError(f"Unknown relation: {relation}")


def cosine_similarity(a: Tensor, b: Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def collect_words(
    *,
    sentence: SentenceSpec,
    tokens: list[str],
    offsets: list[list[int]],
    attention_mask: list[int],
    hidden_states: Tensor,
) -> list[EncodedWord]:
    words: list[EncodedWord] = []
    token_units = _collect_token_units(
        sentence_text=sentence.text,
        tokens=tokens,
        offsets=offsets,
        attention_mask=attention_mask,
    )
    for token_unit in token_units:
        append_encoded_word(
            words=words,
            sentence=sentence,
            span=token_unit.span,
            piece_group=list(token_unit.piece_indices),
            hidden_states=hidden_states,
        )
    return words


def merge_word_vector(word: EncodedWord, layer_indices: tuple[int, ...], pool_fn: PoolFn) -> Tensor:
    selected_layers = word.layer_piece_vectors[list(layer_indices)]
    merged_layers = selected_layers.mean(dim=0)
    return pool_fn(merged_layers)


def build_context_vector(
    *,
    words_for_sentence: list[EncodedWord],
    target_word: EncodedWord,
    layer_indices: tuple[int, ...],
    pool_fn: PoolFn,
    context_weight: float,
) -> Tensor:
    target_vector = merge_word_vector(target_word, layer_indices, pool_fn)
    if context_weight <= 0:
        return target_vector

    neighbor_vectors: list[Tensor] = []
    for offset in (-1, 1):
        neighbor_index = target_word.word_index + offset
        if 0 <= neighbor_index < len(words_for_sentence):
            neighbor_vectors.append(merge_word_vector(words_for_sentence[neighbor_index], layer_indices, pool_fn))

    if not neighbor_vectors:
        return target_vector
    context_vector = torch.stack(neighbor_vectors).mean(dim=0)
    return target_vector + (context_weight * context_vector)


def build_attended_context_vector(
    *,
    words_for_sentence: list[EncodedWord],
    target_word: EncodedWord,
    layer_indices: tuple[int, ...],
    pool_fn: PoolFn,
) -> Tensor:
    target_vector = merge_word_vector(target_word, layer_indices, pool_fn)
    neighbor_vectors: list[Tensor] = []
    for index, word in enumerate(words_for_sentence):
        if index == target_word.word_index:
            continue
        distance = abs(index - target_word.word_index)
        if distance > 2:
            continue
        neighbor_vectors.append(merge_word_vector(word, layer_indices, pool_fn))

    if not neighbor_vectors:
        return target_vector

    neighbor_stack = torch.stack(neighbor_vectors, dim=0)
    scores = torch.matmul(neighbor_stack, target_vector)
    weights = torch.softmax(scores, dim=0)
    context_vector = (neighbor_stack * weights.unsqueeze(1)).sum(dim=0)
    return target_vector + (0.5 * context_vector)


def build_sentence_residual_vector(
    *,
    words_for_sentence: list[EncodedWord],
    target_word: EncodedWord,
    layer_indices: tuple[int, ...],
    pool_fn: PoolFn,
) -> Tensor:
    target_vector = merge_word_vector(target_word, layer_indices, pool_fn)
    sentence_vectors = [merge_word_vector(word, layer_indices, pool_fn) for word in words_for_sentence]
    if len(sentence_vectors) <= 1:
        return target_vector
    sentence_mean = torch.stack(sentence_vectors, dim=0).mean(dim=0)
    return target_vector - sentence_mean


def build_centered_add_vector(
    *,
    words_for_sentence: list[EncodedWord],
    target_word: EncodedWord,
    layer_indices: tuple[int, ...],
    pool_fn: PoolFn,
    weight: float,
) -> Tensor:
    target_vector = merge_word_vector(target_word, layer_indices, pool_fn)
    sentence_mean = build_sentence_content_mean_vector(
        words_for_sentence=words_for_sentence,
        target_word=target_word,
        layer_indices=layer_indices,
        pool_fn=pool_fn,
    )
    if sentence_mean is None:
        return target_vector
    return target_vector + ((target_vector - sentence_mean) * weight)


def build_local_context_mean_vector(
    *,
    words_for_sentence: list[EncodedWord],
    target_word: EncodedWord,
    layer_indices: tuple[int, ...],
    pool_fn: PoolFn,
) -> Tensor | None:
    neighbor_vectors: list[Tensor] = []
    for offset in (-2, -1, 1, 2):
        neighbor_index = target_word.word_index + offset
        if 0 <= neighbor_index < len(words_for_sentence):
            neighbor_vectors.append(merge_word_vector(words_for_sentence[neighbor_index], layer_indices, pool_fn))
    if not neighbor_vectors:
        return None
    return torch.stack(neighbor_vectors, dim=0).mean(dim=0)


def is_semantic_word(word: EncodedWord) -> bool:
    return bool(word.normalized_surface and any(char.isalnum() for char in word.normalized_surface))


def build_sentence_content_mean_vector(
    *,
    words_for_sentence: list[EncodedWord],
    target_word: EncodedWord,
    layer_indices: tuple[int, ...],
    pool_fn: PoolFn,
) -> Tensor | None:
    content_vectors = [
        merge_word_vector(word, layer_indices, pool_fn)
        for word in words_for_sentence
        if is_semantic_word(word)
    ]
    if not content_vectors:
        return None
    return torch.stack(content_vectors, dim=0).mean(dim=0)


def build_weighted_local_context_mean_vector(
    *,
    words_for_sentence: list[EncodedWord],
    target_word: EncodedWord,
    layer_indices: tuple[int, ...],
    pool_fn: PoolFn,
) -> Tensor | None:
    weighted_vectors: list[Tensor] = []
    weights: list[float] = []
    for offset, distance_weight in LOCAL_CONTEXT_DISTANCE_WEIGHTS.items():
        for signed_offset in (-offset, offset):
            neighbor_index = target_word.word_index + signed_offset
            if not (0 <= neighbor_index < len(words_for_sentence)):
                continue
            neighbor_word = words_for_sentence[neighbor_index]
            if not is_semantic_word(neighbor_word):
                continue
            if neighbor_word.normalized_surface == target_word.normalized_surface:
                continue
            weighted_vectors.append(merge_word_vector(neighbor_word, layer_indices, pool_fn))
            weights.append(distance_weight)
    if not weighted_vectors:
        return None
    weight_tensor = torch.tensor(weights, dtype=weighted_vectors[0].dtype, device=weighted_vectors[0].device)
    stacked = torch.stack(weighted_vectors, dim=0)
    return (stacked * weight_tensor.unsqueeze(1)).sum(dim=0) / weight_tensor.sum()


def build_local_residual_vector(
    *,
    words_for_sentence: list[EncodedWord],
    target_word: EncodedWord,
    layer_indices: tuple[int, ...],
    pool_fn: PoolFn,
) -> Tensor:
    target_vector = merge_word_vector(target_word, layer_indices, pool_fn)
    context_mean = build_local_context_mean_vector(
        words_for_sentence=words_for_sentence,
        target_word=target_word,
        layer_indices=layer_indices,
        pool_fn=pool_fn,
    )
    if context_mean is None:
        return target_vector
    return target_vector - context_mean


def build_concat_residual_vector(
    *,
    words_for_sentence: list[EncodedWord],
    target_word: EncodedWord,
    layer_indices: tuple[int, ...],
    pool_fn: PoolFn,
) -> Tensor:
    target_vector = merge_word_vector(target_word, layer_indices, pool_fn)
    context_mean = build_local_context_mean_vector(
        words_for_sentence=words_for_sentence,
        target_word=target_word,
        layer_indices=layer_indices,
        pool_fn=pool_fn,
    )
    if context_mean is None:
        return torch.cat([target_vector, target_vector], dim=0)
    residual_vector = target_vector - context_mean
    return torch.cat([target_vector, residual_vector], dim=0)


def build_concat_sentence_residual_vector(
    *,
    words_for_sentence: list[EncodedWord],
    target_word: EncodedWord,
    layer_indices: tuple[int, ...],
    pool_fn: PoolFn,
) -> Tensor:
    target_vector = merge_word_vector(target_word, layer_indices, pool_fn)
    sentence_residual = build_sentence_residual_vector(
        words_for_sentence=words_for_sentence,
        target_word=target_word,
        layer_indices=layer_indices,
        pool_fn=pool_fn,
    )
    return torch.cat([target_vector, sentence_residual], dim=0)


def build_centered_add_local_projection_deflation_vector(
    *,
    words_for_sentence: list[EncodedWord],
    target_word: EncodedWord,
    layer_indices: tuple[int, ...],
    pool_fn: PoolFn,
) -> Tensor:
    target_vector = merge_word_vector(target_word, layer_indices, pool_fn)
    sentence_mean = build_sentence_content_mean_vector(
        words_for_sentence=words_for_sentence,
        target_word=target_word,
        layer_indices=layer_indices,
        pool_fn=pool_fn,
    )
    if sentence_mean is None:
        base_vector = target_vector
    else:
        base_vector = target_vector + (target_vector - sentence_mean) * CENTERED_ADD_WEIGHT

    local_mean = build_weighted_local_context_mean_vector(
        words_for_sentence=words_for_sentence,
        target_word=target_word,
        layer_indices=layer_indices,
        pool_fn=pool_fn,
    )
    if local_mean is None:
        return base_vector
    local_norm = torch.linalg.vector_norm(local_mean)
    if local_norm <= 1e-8:
        return base_vector
    local_direction = local_mean / local_norm
    projected_component = torch.dot(base_vector, local_direction) * local_direction
    return base_vector - (LOCAL_PROJECTION_DEFLATION_WEIGHT * projected_component)


def resolve_target(words_by_sentence: dict[tuple[str, str], list[EncodedWord]], ref: TargetRef) -> EncodedWord:
    words = words_by_sentence[(ref.sentence, ref.language)]
    normalized_target = normalize_surface(ref.surface)
    matches = [word for word in words if word.normalized_surface == normalized_target]
    if not matches:
        raise ValueError(f"Target not found: {ref.surface!r} in {ref.language} sentence {ref.sentence!r}")
    if ref.occurrence >= len(matches):
        raise ValueError(
            f"Occurrence {ref.occurrence} out of range for {ref.surface!r} in sentence {ref.sentence!r}; found {len(matches)} matches"
        )
    return matches[ref.occurrence]


def build_anchor_cache(
    encoder: BenchmarkEncoder,
    expectations: tuple[PairExpectation, ...],
) -> dict[tuple[str, str], EncodedWord]:
    cache: dict[tuple[str, str], EncodedWord] = {}
    seen: set[tuple[str, str]] = set()
    for expectation in expectations:
        for ref in (expectation.left, expectation.right):
            key = (normalize_surface(ref.surface), ref.language)
            if key in seen:
                continue
            seen.add(key)
            _, words, _ = encoder.encode_sentence(SentenceSpec(text=ref.surface, language=ref.language))
            matches = [word for word in words if word.normalized_surface == key[0]]
            if not matches:
                raise ValueError(f"Anchor token not found for surface={ref.surface!r} language={ref.language}")
            cache[key] = matches[0]
    return cache


def resolve_anchor_word(anchor_cache: dict[tuple[str, str], EncodedWord], target_word: EncodedWord) -> EncodedWord:
    key = (target_word.normalized_surface, target_word.language)
    try:
        return anchor_cache[key]
    except KeyError as exc:
        raise KeyError(f"Anchor word not cached for {target_word.surface!r} ({target_word.language})") from exc


def normalize_vector(vector: Tensor) -> Tensor:
    norm = torch.linalg.vector_norm(vector)
    if norm <= 1e-8:
        return vector
    return vector / norm


def build_anchor_plus_sentence_residual_vector(
    *,
    anchor_cache: dict[tuple[str, str], EncodedWord],
    words_for_sentence: list[EncodedWord],
    target_word: EncodedWord,
    layer_indices: tuple[int, ...],
    pool_fn: PoolFn,
) -> Tensor:
    anchor_vector = merge_word_vector(resolve_anchor_word(anchor_cache, target_word), layer_indices, pool_fn)
    sentence_residual = build_sentence_residual_vector(
        words_for_sentence=words_for_sentence,
        target_word=target_word,
        layer_indices=layer_indices,
        pool_fn=pool_fn,
    )
    return normalize_vector(anchor_vector + (ANCHOR_SENTENCE_RESIDUAL_WEIGHT * sentence_residual))


def build_anchor_plus_local_residual_vector(
    *,
    anchor_cache: dict[tuple[str, str], EncodedWord],
    words_for_sentence: list[EncodedWord],
    target_word: EncodedWord,
    layer_indices: tuple[int, ...],
    pool_fn: PoolFn,
) -> Tensor:
    anchor_vector = merge_word_vector(resolve_anchor_word(anchor_cache, target_word), layer_indices, pool_fn)
    local_residual = build_local_residual_vector(
        words_for_sentence=words_for_sentence,
        target_word=target_word,
        layer_indices=layer_indices,
        pool_fn=pool_fn,
    )
    return normalize_vector(anchor_vector + (ANCHOR_LOCAL_RESIDUAL_WEIGHT * local_residual))


def build_adaptive_anchor_local_residual_vector(
    *,
    anchor_cache: dict[tuple[str, str], EncodedWord],
    words_for_sentence: list[EncodedWord],
    target_word: EncodedWord,
    layer_indices: tuple[int, ...],
    pool_fn: PoolFn,
) -> Tensor:
    anchor_vector = merge_word_vector(resolve_anchor_word(anchor_cache, target_word), layer_indices, pool_fn)
    target_vector = merge_word_vector(target_word, layer_indices, pool_fn)
    sentence_residual = build_sentence_residual_vector(
        words_for_sentence=words_for_sentence,
        target_word=target_word,
        layer_indices=layer_indices,
        pool_fn=pool_fn,
    )
    local_residual = build_local_residual_vector(
        words_for_sentence=words_for_sentence,
        target_word=target_word,
        layer_indices=layer_indices,
        pool_fn=pool_fn,
    )
    anchor_alignment = max(-1.0, min(1.0, cosine_similarity(anchor_vector, target_vector)))
    context_shift_gate = max(0.0, 1.0 - anchor_alignment)
    return normalize_vector(
        anchor_vector
        + (ADAPTIVE_ANCHOR_SENTENCE_WEIGHT * sentence_residual)
        + (context_shift_gate * ADAPTIVE_ANCHOR_LOCAL_MAX_WEIGHT * local_residual)
    )


def format_layers(layer_indices: tuple[int, ...]) -> str:
    if len(layer_indices) == 1:
        return f"L{layer_indices[0]:02d}"
    return "+".join(f"L{index:02d}" for index in layer_indices)


def build_band_sets(num_layers: int) -> tuple[tuple[int, ...], ...]:
    last_four_start = max(0, num_layers - 4)
    middle_start = max(0, (num_layers // 2) - 2)
    middle_end = min(num_layers, middle_start + 4)
    upper_middle_start = max(0, num_layers - 8)
    upper_middle_end = min(num_layers, upper_middle_start + 4)

    candidates = (
        tuple(range(0, min(4, num_layers))),
        tuple(range(middle_start, middle_end)),
        tuple(range(upper_middle_start, upper_middle_end)),
        tuple(range(last_four_start, num_layers)),
    )
    unique_candidates: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)
    return tuple(unique_candidates)


def rank_summaries(items: list[tuple[Any, EvaluationSummary]]) -> list[tuple[Any, EvaluationSummary]]:
    return sorted(
        items,
        key=lambda item: (item[1].ranking_accuracy, item[1].pair_accuracy, item[1].margin),
        reverse=True,
    )


def build_category_summaries(pair_scores: list[PairScore]) -> tuple[CategorySummary, ...]:
    grouped: dict[str, list[PairScore]] = defaultdict(list)
    for item in pair_scores:
        grouped[item.category].append(item)

    summaries: list[CategorySummary] = []
    for category in sorted(grouped):
        items = grouped[category]
        positives = [item.similarity for item in items if item.relation == "positive"]
        negatives = [item.similarity for item in items if item.relation == "negative"]
        relateds = [item.similarity for item in items if item.relation == "related"]
        pair_accuracy = sum(pair_matches_expectation(item.similarity, item.relation) for item in items) / len(items)

        ranking_accuracy: float | None = None
        margin: float | None = None
        if positives and negatives:
            ranking_total = len(positives) * len(negatives)
            ranking_hits = sum(1 for pos in positives for neg in negatives if pos > neg)
            ranking_accuracy = ranking_hits / ranking_total if ranking_total else None
            margin = (sum(positives) / len(positives)) - (sum(negatives) / len(negatives))

        summaries.append(
            CategorySummary(
                category=category,
                count=len(items),
                pair_accuracy=pair_accuracy,
                ranking_accuracy=ranking_accuracy,
                positive_mean=(sum(positives) / len(positives)) if positives else None,
                negative_mean=(sum(negatives) / len(negatives)) if negatives else None,
                related_mean=(sum(relateds) / len(relateds)) if relateds else None,
                margin=margin,
            )
        )
    return tuple(summaries)


def evaluate_configuration(
    *,
    words_by_sentence: dict[tuple[str, str], list[EncodedWord]],
    anchor_cache: dict[tuple[str, str], EncodedWord] | None,
    expectations: tuple[PairExpectation, ...],
    layer_indices: tuple[int, ...],
    pool_name: str,
    representation_name: str,
) -> EvaluationSummary:
    pool_fn = POOLS[pool_name]
    context_weight = REPRESENTATIONS[representation_name]
    pair_scores: list[PairScore] = []

    for expectation in expectations:
        left = resolve_target(words_by_sentence, expectation.left)
        right = resolve_target(words_by_sentence, expectation.right)
        left_sentence_words = words_by_sentence[(expectation.left.sentence, expectation.left.language)]
        right_sentence_words = words_by_sentence[(expectation.right.sentence, expectation.right.language)]
        if representation_name == "target_plus_attended_context":
            left_vector = build_attended_context_vector(
                words_for_sentence=left_sentence_words,
                target_word=left,
                layer_indices=layer_indices,
                pool_fn=pool_fn,
            )
            right_vector = build_attended_context_vector(
                words_for_sentence=right_sentence_words,
                target_word=right,
                layer_indices=layer_indices,
                pool_fn=pool_fn,
            )
        elif representation_name == "target_minus_sentence_mean":
            left_vector = build_sentence_residual_vector(
                words_for_sentence=left_sentence_words,
                target_word=left,
                layer_indices=layer_indices,
                pool_fn=pool_fn,
            )
            right_vector = build_sentence_residual_vector(
                words_for_sentence=right_sentence_words,
                target_word=right,
                layer_indices=layer_indices,
                pool_fn=pool_fn,
            )
        elif representation_name == "target_minus_local_context_mean":
            left_vector = build_local_residual_vector(
                words_for_sentence=left_sentence_words,
                target_word=left,
                layer_indices=layer_indices,
                pool_fn=pool_fn,
            )
            right_vector = build_local_residual_vector(
                words_for_sentence=right_sentence_words,
                target_word=right,
                layer_indices=layer_indices,
                pool_fn=pool_fn,
            )
        elif representation_name == "target_concat_residual":
            left_vector = build_concat_residual_vector(
                words_for_sentence=left_sentence_words,
                target_word=left,
                layer_indices=layer_indices,
                pool_fn=pool_fn,
            )
            right_vector = build_concat_residual_vector(
                words_for_sentence=right_sentence_words,
                target_word=right,
                layer_indices=layer_indices,
                pool_fn=pool_fn,
            )
        elif representation_name == "concat_word_and_sentence_residual":
            left_vector = build_concat_sentence_residual_vector(
                words_for_sentence=left_sentence_words,
                target_word=left,
                layer_indices=layer_indices,
                pool_fn=pool_fn,
            )
            right_vector = build_concat_sentence_residual_vector(
                words_for_sentence=right_sentence_words,
                target_word=right,
                layer_indices=layer_indices,
                pool_fn=pool_fn,
            )
        elif representation_name == "centered_add_plus_local_projection_deflation":
            left_vector = build_centered_add_local_projection_deflation_vector(
                words_for_sentence=left_sentence_words,
                target_word=left,
                layer_indices=layer_indices,
                pool_fn=pool_fn,
            )
            right_vector = build_centered_add_local_projection_deflation_vector(
                words_for_sentence=right_sentence_words,
                target_word=right,
                layer_indices=layer_indices,
                pool_fn=pool_fn,
            )
        elif representation_name == "target_centered_add_w050":
            left_vector = build_centered_add_vector(
                words_for_sentence=left_sentence_words,
                target_word=left,
                layer_indices=layer_indices,
                pool_fn=pool_fn,
                weight=0.50,
            )
            right_vector = build_centered_add_vector(
                words_for_sentence=right_sentence_words,
                target_word=right,
                layer_indices=layer_indices,
                pool_fn=pool_fn,
                weight=0.50,
            )
        elif representation_name == "isolated_anchor_plus_sentence_residual":
            if anchor_cache is None:
                raise RuntimeError("isolated anchor representation requires anchor cache")
            left_vector = build_anchor_plus_sentence_residual_vector(
                anchor_cache=anchor_cache,
                words_for_sentence=left_sentence_words,
                target_word=left,
                layer_indices=layer_indices,
                pool_fn=pool_fn,
            )
            right_vector = build_anchor_plus_sentence_residual_vector(
                anchor_cache=anchor_cache,
                words_for_sentence=right_sentence_words,
                target_word=right,
                layer_indices=layer_indices,
                pool_fn=pool_fn,
            )
        elif representation_name == "isolated_anchor_plus_local_residual":
            if anchor_cache is None:
                raise RuntimeError("isolated anchor representation requires anchor cache")
            left_vector = build_anchor_plus_local_residual_vector(
                anchor_cache=anchor_cache,
                words_for_sentence=left_sentence_words,
                target_word=left,
                layer_indices=layer_indices,
                pool_fn=pool_fn,
            )
            right_vector = build_anchor_plus_local_residual_vector(
                anchor_cache=anchor_cache,
                words_for_sentence=right_sentence_words,
                target_word=right,
                layer_indices=layer_indices,
                pool_fn=pool_fn,
            )
        elif representation_name == "adaptive_anchor_local_residual":
            if anchor_cache is None:
                raise RuntimeError("adaptive anchor representation requires anchor cache")
            left_vector = build_adaptive_anchor_local_residual_vector(
                anchor_cache=anchor_cache,
                words_for_sentence=left_sentence_words,
                target_word=left,
                layer_indices=layer_indices,
                pool_fn=pool_fn,
            )
            right_vector = build_adaptive_anchor_local_residual_vector(
                anchor_cache=anchor_cache,
                words_for_sentence=right_sentence_words,
                target_word=right,
                layer_indices=layer_indices,
                pool_fn=pool_fn,
            )
        else:
            left_vector = build_context_vector(
                words_for_sentence=left_sentence_words,
                target_word=left,
                layer_indices=layer_indices,
                pool_fn=pool_fn,
                context_weight=context_weight,
            )
            right_vector = build_context_vector(
                words_for_sentence=right_sentence_words,
                target_word=right,
                layer_indices=layer_indices,
                pool_fn=pool_fn,
                context_weight=context_weight,
            )
        pair_scores.append(
            PairScore(
                label=expectation.label,
                category=expectation.category,
                relation=expectation.relation,
                representation=representation_name,
                similarity=cosine_similarity(left_vector, right_vector),
            )
        )

    positives = [item.similarity for item in pair_scores if item.relation == "positive"]
    negatives = [item.similarity for item in pair_scores if item.relation == "negative"]
    pair_accuracy = sum(pair_matches_expectation(item.similarity, item.relation) for item in pair_scores) / len(pair_scores)

    ranking_total = len(positives) * len(negatives)
    ranking_hits = sum(1 for pos in positives for neg in negatives if pos > neg)
    positive_mean = sum(positives) / len(positives)
    negative_mean = sum(negatives) / len(negatives)

    return EvaluationSummary(
        pair_accuracy=pair_accuracy,
        ranking_accuracy=ranking_hits / ranking_total if ranking_total else 0.0,
        positive_mean=positive_mean,
        negative_mean=negative_mean,
        margin=positive_mean - negative_mean,
        pair_scores=tuple(pair_scores),
        category_summaries=build_category_summaries(pair_scores),
    )


def load_gold_cases(path: Path) -> tuple[tuple[SentenceSpec, ...], tuple[PairExpectation, ...]]:
    if not path.exists():
        raise FileNotFoundError(f"Gold test file not found: {path}")

    expectations: list[PairExpectation] = []
    sentence_map: dict[tuple[str, str], SentenceSpec] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            payload = json.loads(line)
            left = TargetRef(
                sentence=payload["left_sentence"],
                language=payload["left_language"],
                surface=payload["left_surface"],
                occurrence=int(payload.get("left_occurrence", 0)),
            )
            right = TargetRef(
                sentence=payload["right_sentence"],
                language=payload["right_language"],
                surface=payload["right_surface"],
                occurrence=int(payload.get("right_occurrence", 0)),
            )
            expectation = PairExpectation(
                label=payload["label"],
                category=payload["category"],
                relation=payload["relation"],
                left=left,
                right=right,
                note=payload.get("note", ""),
            )
            expectations.append(expectation)
            sentence_map[(left.sentence, left.language)] = SentenceSpec(left.sentence, left.language)
            sentence_map[(right.sentence, right.language)] = SentenceSpec(right.sentence, right.language)

    if not expectations:
        raise ValueError(f"No usable gold cases found in {path}")
    return tuple(sentence_map.values()), tuple(expectations)


def print_sentence_debug(tokens: list[str], words: list[EncodedWord], sentence: SentenceSpec) -> None:
    print(f"  [{sentence.language}] {sentence.text}")
    print(f"    Parcalar: {tokens}")
    print(f"    Kelimeler: {[f'{word.surface}({len(word.piece_indices)}p,{word.span[0]}:{word.span[1]})' for word in words]}")


def print_category_table(summary: EvaluationSummary) -> None:
    print("\n" + "=" * 120)
    print("BOLUM 4: KATEGORI OZETI")
    print("=" * 120)
    print(
        f"\n{'Kategori':20s} {'Adet':>5s} {'PairAcc':>8s} {'RankAcc':>8s} {'PosMean':>8s} "
        f"{'NegMean':>8s} {'RelMean':>8s} {'Margin':>8s}"
    )
    print("-" * 88)
    for item in summary.category_summaries:
        rank_acc = f"{item.ranking_accuracy:.3f}" if item.ranking_accuracy is not None else "-"
        pos_mean = f"{item.positive_mean:.4f}" if item.positive_mean is not None else "-"
        neg_mean = f"{item.negative_mean:.4f}" if item.negative_mean is not None else "-"
        rel_mean = f"{item.related_mean:.4f}" if item.related_mean is not None else "-"
        margin = f"{item.margin:.4f}" if item.margin is not None else "-"
        print(
            f"{item.category:20s} {item.count:>5d} {item.pair_accuracy:>8.3f} {rank_acc:>8s} "
            f"{pos_mean:>8s} {neg_mean:>8s} {rel_mean:>8s} {margin:>8s}"
        )


def print_pair_breakdown(title: str, summary: EvaluationSummary) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    print(f"\n{'Pair':36s} {'Kategori':20s} {'Temsil':16s} {'Relation':>9s} {'Similarity':>10s} {'OK':>3s}")
    print("-" * 106)
    for item in summary.pair_scores:
        ok = "+" if pair_matches_expectation(item.similarity, item.relation) else "-"
        print(
            f"{item.label:36s} {item.category:20s} {item.representation:16s} "
            f"{item.relation:>9s} {item.similarity:>10.4f} {ok:>3s}"
        )


def run_model_benchmark(
    encoder: BenchmarkEncoder,
    sentences: tuple[SentenceSpec, ...],
    expectations: tuple[PairExpectation, ...],
) -> None:
    print("\n" + "#" * 120)
    print(f"MODEL: {encoder.label} | {encoder.model_id}")
    print("#" * 120)
    print("[1] Model yukleniyor...")
    encoder.load()

    print(f"[2] {len(sentences)} cumle encode ediliyor...")
    words_by_sentence: dict[tuple[str, str], list[EncodedWord]] = {}
    num_layers: int | None = None
    for sentence in sentences:
        tokens, words, hidden_states = encoder.encode_sentence(sentence)
        words_by_sentence[(sentence.text, sentence.language)] = words
        num_layers = hidden_states.shape[0]
        print_sentence_debug(tokens, words, sentence)

    if num_layers is None:
        raise RuntimeError(f"{encoder.label} hic cumle encode etmedi")

    anchor_cache = build_anchor_cache(encoder, expectations)
    print(f"\n[3] Toplam katman sayisi: {num_layers} (L00 embedding, L{num_layers - 1:02d} son katman)")

    single_results = [
        ((layer_index, pool_name, representation_name), evaluate_configuration(
            words_by_sentence=words_by_sentence,
            anchor_cache=anchor_cache,
            expectations=expectations,
            layer_indices=(layer_index,),
            pool_name=pool_name,
            representation_name=representation_name,
        ))
        for representation_name in REPRESENTATIONS
        for pool_name in POOLS
        for layer_index in range(num_layers)
    ]
    band_results = [
        ((layer_indices, pool_name, representation_name), evaluate_configuration(
            words_by_sentence=words_by_sentence,
            anchor_cache=anchor_cache,
            expectations=expectations,
            layer_indices=layer_indices,
            pool_name=pool_name,
            representation_name=representation_name,
        ))
        for representation_name in REPRESENTATIONS
        for layer_indices in build_band_sets(num_layers)
        for pool_name in POOLS
    ]
    combo_candidates = tuple(range(max(0, num_layers - 12), num_layers))
    combo_results = [
        (((layer_a, layer_b), pool_name, representation_name), evaluate_configuration(
            words_by_sentence=words_by_sentence,
            anchor_cache=anchor_cache,
            expectations=expectations,
            layer_indices=(layer_a, layer_b),
            pool_name=pool_name,
            representation_name=representation_name,
        ))
        for representation_name in REPRESENTATIONS
        for layer_a, layer_b in combinations(combo_candidates, 2)
        for pool_name in POOLS
    ]

    ranked_single = rank_summaries(single_results)
    ranked_band = rank_summaries(band_results)
    ranked_combo = rank_summaries(combo_results)

    print("\n" + "=" * 120)
    print("BOLUM 1: TEK KATMAN x POOLING")
    print("=" * 120)
    print(
        f"\n{'Katman':>8s}  {'Pool':>6s}  {'Temsil':16s}  {'PairAcc':>8s}  {'RankAcc':>8s}  "
        f"{'PosMean':>8s}  {'NegMean':>8s}  {'Margin':>8s}"
    )
    print("-" * 102)
    for (layer_index, pool_name, representation_name), summary in ranked_single[:30]:
        print(
            f"{format_layers((layer_index,)):>8s}  {pool_name:>6s}  {representation_name:16s}  {summary.pair_accuracy:>8.3f}  "
            f"{summary.ranking_accuracy:>8.3f}  {summary.positive_mean:>8.4f}  {summary.negative_mean:>8.4f}  {summary.margin:>8.4f}"
        )

    print("\n" + "=" * 120)
    print("BOLUM 2: KATMAN BANDLARI")
    print("=" * 120)
    print(
        f"\n{'Katmanlar':>18s}  {'Pool':>6s}  {'Temsil':16s}  {'PairAcc':>8s}  {'RankAcc':>8s}  "
        f"{'PosMean':>8s}  {'NegMean':>8s}  {'Margin':>8s}"
    )
    print("-" * 114)
    for (layer_indices, pool_name, representation_name), summary in ranked_band[:30]:
        print(
            f"{format_layers(layer_indices):>18s}  {pool_name:>6s}  {representation_name:16s}  {summary.pair_accuracy:>8.3f}  "
            f"{summary.ranking_accuracy:>8.3f}  {summary.positive_mean:>8.4f}  {summary.negative_mean:>8.4f}  {summary.margin:>8.4f}"
        )

    print("\n" + "=" * 120)
    print("BOLUM 3: IKILI KATMAN KOMBINASYONLARI")
    print("=" * 120)
    print(
        f"\n{'Katmanlar':>12s}  {'Pool':>6s}  {'Temsil':16s}  {'PairAcc':>8s}  {'RankAcc':>8s}  "
        f"{'PosMean':>8s}  {'NegMean':>8s}  {'Margin':>8s}"
    )
    print("-" * 108)
    for (layer_indices, pool_name, representation_name), summary in ranked_combo[:30]:
        print(
            f"{format_layers(layer_indices):>12s}  {pool_name:>6s}  {representation_name:16s}  {summary.pair_accuracy:>8.3f}  "
            f"{summary.ranking_accuracy:>8.3f}  {summary.positive_mean:>8.4f}  {summary.negative_mean:>8.4f}  {summary.margin:>8.4f}"
        )

    best_single = ranked_single[0]
    best_band = ranked_band[0]
    best_combo = ranked_combo[0]

    print("\n" + "=" * 120)
    print("BOLUM 5: FINAL OZET")
    print("=" * 120)
    print(
        f"\nEn iyi tek katman: {format_layers((best_single[0][0],))}+{best_single[0][1]}+{best_single[0][2]} | "
        f"pair_acc={best_single[1].pair_accuracy:.3f} rank_acc={best_single[1].ranking_accuracy:.3f} margin={best_single[1].margin:.4f}"
    )
    print(
        f"En iyi katman bandi: {format_layers(best_band[0][0])}+{best_band[0][1]}+{best_band[0][2]} | "
        f"pair_acc={best_band[1].pair_accuracy:.3f} rank_acc={best_band[1].ranking_accuracy:.3f} margin={best_band[1].margin:.4f}"
    )
    print(
        f"En iyi ikili kombinasyon: {format_layers(best_combo[0][0])}+{best_combo[0][1]}+{best_combo[0][2]} | "
        f"pair_acc={best_combo[1].pair_accuracy:.3f} rank_acc={best_combo[1].ranking_accuracy:.3f} margin={best_combo[1].margin:.4f}"
    )

    print_category_table(best_single[1])
    print_pair_breakdown("BOLUM 6: EN IYI TEK KATMAN DETAYI", best_single[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NLLB subword merge benchmark")
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help="JSONL gold case dosyasi. Varsayilan: tests/subword_merge_gold_template.jsonl",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=("nllb13b", "nllb33b"),
        choices=("nllb600m", "nllb13b", "nllb33b"),
        help="Calistirilacak modeller",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    sentences, expectations = load_gold_cases(args.gold_path)

    print("=" * 120)
    print("SUBWORD MERGE BENCHMARK")
    print("=" * 120)
    print("Amac: kelimeyi olusturan subword parcalarini hangi layer ve pooling ile birlestirmenin")
    print("semantik ayrim icin daha iyi oldugunu kategori bazli gold set ile olcmek.")
    print(f"\nGold dosyasi: {args.gold_path}")
    print(f"Cumle sayisi: {len(sentences)} | Pair sayisi: {len(expectations)}")

    categories = sorted({item.category for item in expectations})
    print(f"Kategoriler: {', '.join(categories)}")

    device = _resolve_device()
    print(f"Cihaz: {device.type}")

    selected_encoders: list[BenchmarkEncoder] = []
    if "nllb600m" in args.models:
        selected_encoders.append(NLLBBenchmarkEncoder(device, NLLB_600M_SPEC))
    if "nllb13b" in args.models:
        selected_encoders.append(NLLBBenchmarkEncoder(device, NLLB_13B_SPEC))
    if "nllb33b" in args.models:
        selected_encoders.append(NLLBBenchmarkEncoder(device, NLLB_33B_SPEC))

    for encoder in selected_encoders:
        run_model_benchmark(encoder, sentences, expectations)


if __name__ == "__main__":
    run()
