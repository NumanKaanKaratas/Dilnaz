from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import IterableDataset, get_worker_info

from dilnaz.train.common.runtime import (
    COMPILE_MODE_CHOICES,
    DeviceBatchPrefetcher,
    compile_forward,
    effective_compile_mode,
    validate_compile_environment,
)
from dilnaz.train.data.dil_data import load_hybrid_tokenizer, make_dil_batch_loader, segment_piece_ids, trainable_segments
from dilnaz.surface import pack_token_units, pack_writer_targets
from dilnaz.train.configs.defaults import DIL_MODEL_DEFAULTS, DIL_TRAIN_DEFAULTS
from dilnaz.models.dil import DilConfig
from dilnaz.models.dil import Dil
from dilnaz.tokenization import HybridTokenizer, TokenSegment, default_vocab_path
from dilnaz.train.dil.train import is_dataloader_worker_exit, restore_checkpoint, save_checkpoint
from dilnaz.train.dil.writer_windows import build_writer_window_view, gather_writer_semantic
from dilnaz.train.common.trainer_core import BaseTrainer, StepResult, make_scheduler


@dataclass(frozen=True)
class JsonlParallelPair:
    line_id: int
    tr: str
    en: str


@dataclass(frozen=True)
class TokenizedParallelPair:
    line_id: int
    tr_segments: list[TokenSegment]
    en_segments: list[TokenSegment]


@dataclass
class TeacherlessDilOutput:
    loss: torch.Tensor
    sentence_loss: torch.Tensor
    writer_loss: torch.Tensor
    writer_token_loss: torch.Tensor
    variance_loss: torch.Tensor
    token_set_loss: torch.Tensor
    token_balance_loss: torch.Tensor
    covariance_loss: torch.Tensor
    byte_acc: torch.Tensor
    token_exact: torch.Tensor
    stop_acc: torch.Tensor
    token_set_weight: float
    token_balance_weight: float
    covariance_weight: float


def parse_parallel_jsonl_line(line: str, line_id: int) -> JsonlParallelPair | None:
    payload = json.loads(line)
    tr = str(payload["tr"]).strip()
    en = str(payload["en"]).strip()
    if not tr or not en:
        return None
    return JsonlParallelPair(line_id=line_id, tr=tr, en=en)


def iter_parallel_jsonl(path: Path, worker_id: int, worker_count: int) -> Iterator[JsonlParallelPair]:
    with path.open("r", encoding="utf-8") as handle:
        for line_id, line in enumerate(handle):
            if line_id % worker_count != worker_id:
                continue
            pair = parse_parallel_jsonl_line(line, line_id)
            if pair is not None:
                yield pair


class TeacherlessParallelJsonlDataset(IterableDataset):
    def __init__(
        self,
        train_file: Path,
        config: DilConfig,
        tokenizer: HybridTokenizer,
        batch_size: int,
        max_segments: int,
        min_segments: int,
        min_length_ratio: float,
        max_length_ratio: float,
        shuffle_buffer_size: int,
        seed: int,
        repeat: bool,
        max_samples: int = 0,
    ):
        super().__init__()
        self.train_file = train_file
        self.config = config
        self.max_surface_pieces_per_unit = config.max_surface_pieces_per_unit
        self.surface_bucket_sizes = tuple(config.surface_bucket_sizes)
        self.pad_token_id = config.pad_token_id
        self.writer_stop_token_id = config.writer_stop_token_id
        self.writer_bos_token_id = config.writer_bos_token_id
        self.writer_empty_token_id = config.writer_empty_token_id
        self.batch_size = batch_size
        self.max_segments = max_segments
        self.min_segments = min_segments
        self.min_length_ratio = min_length_ratio
        self.max_length_ratio = max_length_ratio
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.repeat = repeat
        self.max_samples = max_samples
        self.tokenizer = tokenizer

    def tokenized_pair(self, pair: JsonlParallelPair) -> TokenizedParallelPair | None:
        tr_segments = trainable_segments(
            self.tokenizer,
            pair.tr,
            self.max_surface_pieces_per_unit,
            add_eos=True,
        )
        en_segments = trainable_segments(
            self.tokenizer,
            pair.en,
            self.max_surface_pieces_per_unit,
            add_eos=True,
        )
        if len(tr_segments) < self.min_segments or len(en_segments) < self.min_segments:
            return None
        ratio = len(tr_segments) / max(len(en_segments), 1)
        if ratio < self.min_length_ratio or ratio > self.max_length_ratio:
            return None
        return TokenizedParallelPair(
            line_id=pair.line_id,
            tr_segments=tr_segments[: self.max_segments],
            en_segments=en_segments[: self.max_segments],
        )

    def side_packed(self, pairs: list[TokenizedParallelPair], prefix: str):
        size = len(pairs)
        surface_rows: list[list[list[int]]] = []
        target_rows: list[list[list[int]]] = []
        unit_mask = torch.zeros((size, self.max_segments), dtype=torch.bool)
        segment_counts = torch.zeros((size,), dtype=torch.long)
        pair_segments = [pair.tr_segments if prefix == "tr" else pair.en_segments for pair in pairs]
        for row_idx, segments in enumerate(pair_segments):
            count = min(len(segments), self.max_segments)
            unit_mask[row_idx, :count] = True
            segment_counts[row_idx] = count
            surface_row: list[list[int]] = []
            target_row: list[list[int]] = []
            for segment_idx in range(self.max_segments):
                if segment_idx >= count:
                    surface_row.append([])
                    target_row.append([])
                    continue
                pieces = segment_piece_ids(segments[segment_idx])
                surface_row.append(pieces)
                target_row.append(pieces)
            surface_rows.append(surface_row)
            target_rows.append(target_row)
        writer_view = build_writer_window_view(target_rows, self.config)
        return {
            f"{prefix}_surface": pack_token_units(
                surface_rows,
                pad_token_id=self.pad_token_id,
                bucket_sizes=self.surface_bucket_sizes,
                max_pieces_per_unit=self.max_surface_pieces_per_unit,
            ),
            f"{prefix}_labels": pack_writer_targets(
                target_rows,
                pad_token_id=self.pad_token_id,
                stop_token_id=self.writer_stop_token_id,
                bos_token_id=self.writer_bos_token_id,
                empty_token_id=self.writer_empty_token_id,
                surface_bucket_sizes=self.surface_bucket_sizes,
                max_pieces_per_unit=self.max_surface_pieces_per_unit,
            ),
            f"{prefix}_unit_mask": unit_mask,
            f"{prefix}_segment_counts": segment_counts,
            f"{prefix}_writer_labels": writer_view["writer_labels"],
            f"{prefix}_writer_source_rows": writer_view["writer_source_rows"],
            f"{prefix}_writer_unit_indices": writer_view["writer_unit_indices"],
            f"{prefix}_writer_zone_ids": writer_view["writer_zone_ids"],
            f"{prefix}_writer_window_mask": writer_view["writer_window_mask"],
        }

    def fill_side(
        self,
        batch: dict[str, torch.Tensor],
        prefix: str,
        row_idx: int,
        segments: list[TokenSegment],
    ) -> None:
        count = min(len(segments), self.max_segments)
        batch[f"{prefix}_unit_mask"][row_idx, :count] = True
        batch[f"{prefix}_segment_counts"][row_idx] = count

    def make_batch(self, pairs: list[TokenizedParallelPair]) -> dict:
        size = len(pairs)
        batch = {
            **self.side_packed(pairs, "tr"),
            **self.side_packed(pairs, "en"),
            "source_line_ids": torch.zeros((size,), dtype=torch.long),
        }
        for row_idx, pair in enumerate(pairs):
            batch["source_line_ids"][row_idx] = int(pair.line_id)
        return batch

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        rng = random.Random(self.seed + worker_id)
        buffer: list[TokenizedParallelPair] = []
        batch: list[TokenizedParallelPair] = []
        produced = 0

        while True:
            accepted_this_pass = 0
            for pair in iter_parallel_jsonl(self.train_file, worker_id, worker_count):
                tokenized = self.tokenized_pair(pair)
                if tokenized is None:
                    continue
                buffer.append(tokenized)
                accepted_this_pass += 1
                while len(buffer) >= self.shuffle_buffer_size:
                    batch.append(buffer.pop(rng.randrange(len(buffer))))
                    if len(batch) == self.batch_size or (
                        self.max_samples > 0 and produced + len(batch) >= self.max_samples
                    ):
                        produced += len(batch)
                        yield self.make_batch(batch)
                        batch = []
                        if self.max_samples > 0 and produced >= self.max_samples:
                            return

            while buffer:
                batch.append(buffer.pop(rng.randrange(len(buffer))))
                if len(batch) == self.batch_size or (
                    self.max_samples > 0 and produced + len(batch) >= self.max_samples
                ):
                    produced += len(batch)
                    yield self.make_batch(batch)
                    batch = []
                    if self.max_samples > 0 and produced >= self.max_samples:
                        return
            if batch:
                produced += len(batch)
                yield self.make_batch(batch)
                batch = []
                if self.max_samples > 0 and produced >= self.max_samples:
                    return

            if accepted_this_pass == 0 and produced == 0:
                raise ValueError(f"{self.train_file} produced no trainable TR-EN JSONL pairs")
            if not self.repeat:
                return


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def sentence_contrastive_loss(
    tr_sentence: torch.Tensor,
    en_sentence: torch.Tensor,
    temperature: float,
    margin: float,
) -> torch.Tensor:
    tr_unit = F.normalize(tr_sentence.float(), dim=-1, eps=1e-6)
    en_unit = F.normalize(en_sentence.float(), dim=-1, eps=1e-6)
    similarity = tr_unit @ en_unit.T
    labels = torch.arange(similarity.shape[0], device=similarity.device)
    margin_matrix = torch.eye(similarity.shape[0], device=similarity.device, dtype=similarity.dtype) * margin
    logits = (similarity - margin_matrix) / temperature
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) * 0.5


def variance_regularizer(sentence_latents: torch.Tensor, target_std: float) -> torch.Tensor:
    if sentence_latents.shape[0] < 2:
        return sentence_latents.new_zeros(())
    std = torch.sqrt(sentence_latents.float().var(dim=0, unbiased=False) + 1e-4)
    return F.relu(target_std - std).mean()


def token_balance_loss(latents: torch.Tensor, unit_mask: torch.Tensor, target_std: float) -> torch.Tensor:
    active_rows = unit_mask.sum(dim=1).ge(2)
    if not bool(active_rows.any()):
        return latents.new_zeros(())
    values = latents[active_rows].float()
    mask = unit_mask[active_rows]
    weights = mask.unsqueeze(-1).to(values.dtype)
    counts = weights.sum(dim=1).clamp_min(1.0)
    mean = (values * weights).sum(dim=1) / counts
    variance = ((values - mean.unsqueeze(1)) * weights).pow(2).sum(dim=1) / counts
    std = torch.sqrt(variance + 1e-4)
    return F.relu(target_std - std).mean()


def covariance_regularizer(sentence_latents: torch.Tensor) -> torch.Tensor:
    if sentence_latents.shape[0] < 2:
        return sentence_latents.new_zeros(())
    centered = sentence_latents.float() - sentence_latents.float().mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    covariance = covariance - torch.diag_embed(torch.diagonal(covariance))
    return covariance.pow(2).sum() / sentence_latents.shape[-1]


def valid_label_mask(labels: torch.Tensor, writer_stop_token_id: int, pad_token_id: int, eos_token_id: int) -> torch.Tensor:
    return (
        labels.ge(0)
        & labels.ne(writer_stop_token_id)
        & labels.ne(pad_token_id)
        & labels.ne(eos_token_id)
    )


def batch_token_set_targets(
    labels,
    writer_stop_token_id: int,
    pad_token_id: int,
    eos_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = labels.labels if hasattr(labels, "labels") else labels
    flat = labels.reshape(labels.shape[0], -1)
    valid = valid_label_mask(flat, writer_stop_token_id, pad_token_id, eos_token_id)
    vocab_ids = torch.unique(flat[valid])
    targets = torch.zeros((flat.shape[0], vocab_ids.numel()), dtype=torch.float32, device=labels.device)
    if vocab_ids.numel() == 0:
        return vocab_ids, targets
    nonnegative = flat.ge(0)
    lookup_size = int(flat[nonnegative].max().detach().cpu().item()) + 1
    lookup = torch.full((lookup_size,), -1, dtype=torch.long, device=labels.device)
    lookup[vocab_ids] = torch.arange(vocab_ids.numel(), device=labels.device)
    columns = lookup[flat.clamp_min(0)]
    rows = torch.arange(flat.shape[0], device=labels.device).unsqueeze(1).expand_as(flat)
    active = valid & columns.ge(0)
    targets[rows[active], columns[active]] = 1.0
    return vocab_ids, targets


def balanced_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, positive_weight: float) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    positive = targets.bool()
    negative = ~positive
    positive_loss = losses[positive].mean() if bool(positive.any()) else logits.new_zeros(())
    negative_loss = losses[negative].mean() if bool(negative.any()) else logits.new_zeros(())
    return positive_loss * positive_weight + negative_loss


def token_set_bow_loss(
    source_sentence: torch.Tensor,
    target_labels: torch.Tensor,
    token_embedding: torch.Tensor,
    writer_stop_token_id: int,
    pad_token_id: int,
    eos_token_id: int,
    temperature: float,
    positive_weight: float,
) -> torch.Tensor:
    vocab_ids, targets = batch_token_set_targets(
        target_labels,
        writer_stop_token_id,
        pad_token_id,
        eos_token_id,
    )
    if vocab_ids.numel() == 0:
        return source_sentence.new_zeros(())
    source = F.normalize(source_sentence.float(), dim=-1, eps=1e-6)
    target_embedding = F.normalize(token_embedding.index_select(0, vocab_ids).float(), dim=-1, eps=1e-6)
    logits = source @ target_embedding.T / temperature
    return balanced_bce_with_logits(logits, targets, positive_weight)


def ramped_weight(step: int, start_step: int, ramp_steps: int, target_weight: float) -> float:
    if target_weight <= 0.0 or step < start_step:
        return 0.0
    if ramp_steps <= 0:
        return target_weight
    return target_weight * min(1.0, float(step - start_step + 1) / float(ramp_steps))


def runtime_training_state(args) -> dict:
    return {
        "trainer": "teacherless_parallel_dil",
        "train_file": str(args.train_file),
        "eval_file": str(args.eval_file) if args.eval_file is not None else "",
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_segments": args.max_segments,
        "min_segments": args.min_segments,
        "shuffle_buffer_size": args.shuffle_buffer_size,
        "min_length_ratio": args.min_length_ratio,
        "max_length_ratio": args.max_length_ratio,
        "sentence_loss_weight": args.sentence_loss_weight,
        "temperature": args.temperature,
        "margin": args.margin,
        "teacherless_variance_weight": args.teacherless_variance_weight,
        "variance_target_std": args.variance_target_std,
        "token_set_loss_weight": args.token_set_loss_weight,
        "token_set_start_step": args.token_set_start_step,
        "token_set_ramp_steps": args.token_set_ramp_steps,
        "token_set_temperature": args.token_set_temperature,
        "token_set_positive_weight": args.token_set_positive_weight,
        "token_balance_weight": args.token_balance_weight,
        "token_balance_start_step": args.token_balance_start_step,
        "token_balance_ramp_steps": args.token_balance_ramp_steps,
        "token_balance_target_std": args.token_balance_target_std,
        "covariance_weight": args.covariance_weight,
        "covariance_start_step": args.covariance_start_step,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "warmup_steps": args.warmup_steps,
        "max_grad_norm": args.max_grad_norm,
        "log_every": args.log_every,
        "checkpoint_every": args.checkpoint_every,
        "eval_every": args.eval_every,
        "max_eval_batches": args.max_eval_batches,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "seed": args.seed,
        "max_samples": args.max_samples,
    }


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--tokenizer-vocab", type=Path, default=default_vocab_path())
    parser.add_argument("--compile-mode", choices=COMPILE_MODE_CHOICES, default=None)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--max-segments", type=int, default=32)
    parser.add_argument("--min-segments", type=int, default=3)
    parser.add_argument("--min-length-ratio", type=float, default=0.5)
    parser.add_argument("--max-length-ratio", type=float, default=2.0)
    parser.add_argument("--shuffle-buffer-size", type=int, default=8192)
    parser.add_argument("--prefetch-factor", type=int, default=DIL_TRAIN_DEFAULTS["prefetch_factor"])
    parser.add_argument("--no-cuda-prefetch", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=DIL_TRAIN_DEFAULTS["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=DIL_TRAIN_DEFAULTS["weight_decay"])
    parser.add_argument("--adam-beta1", type=float, default=DIL_TRAIN_DEFAULTS["adam_beta1"])
    parser.add_argument("--adam-beta2", type=float, default=DIL_TRAIN_DEFAULTS["adam_beta2"])
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--max-grad-norm", type=float, default=DIL_TRAIN_DEFAULTS["max_grad_norm"])
    parser.add_argument("--log-every", type=int, default=DIL_TRAIN_DEFAULTS["log_every"])
    parser.add_argument("--checkpoint-every", type=int, default=DIL_TRAIN_DEFAULTS["checkpoint_every"])
    parser.add_argument("--eval-every", type=int, default=DIL_TRAIN_DEFAULTS["eval_every"])
    parser.add_argument("--max-eval-batches", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=DIL_TRAIN_DEFAULTS["num_workers"])
    parser.add_argument("--seed", type=int, default=DIL_TRAIN_DEFAULTS["seed"])
    parser.add_argument("--max-samples", type=int, default=DIL_TRAIN_DEFAULTS["max_samples"])
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=DIL_MODEL_DEFAULTS["hidden_size"])
    parser.add_argument("--intermediate-size", type=int, default=DIL_MODEL_DEFAULTS["intermediate_size"])
    parser.add_argument("--latent-size", type=int, default=DIL_MODEL_DEFAULTS["latent_size"])
    parser.add_argument("--max-surface-pieces-per-unit", type=int, default=DIL_MODEL_DEFAULTS["max_surface_pieces_per_unit"])
    parser.add_argument("--byte-conv-layers", type=int, default=DIL_MODEL_DEFAULTS["byte_conv_layers"])
    parser.add_argument("--byte-conv-kernel-size", type=int, default=DIL_MODEL_DEFAULTS["byte_conv_kernel_size"])
    parser.add_argument("--byte-conv-expansion", type=int, default=DIL_MODEL_DEFAULTS["byte_conv_expansion"])
    parser.add_argument("--dil-dropout", type=float, default=DIL_MODEL_DEFAULTS["dil_dropout"])
    parser.add_argument("--writer-loss-weight", type=float, default=DIL_MODEL_DEFAULTS["writer_loss_weight"])
    parser.add_argument("--writer-num-layers", type=int, default=DIL_MODEL_DEFAULTS["writer_num_layers"])
    parser.add_argument("--writer-conv-kernel-size", type=int, default=DIL_MODEL_DEFAULTS["writer_conv_kernel_size"])
    parser.add_argument("--writer-conv-expansion", type=int, default=DIL_MODEL_DEFAULTS["writer_conv_expansion"])
    parser.add_argument("--writer-dropout", type=float, default=DIL_MODEL_DEFAULTS["writer_dropout"])
    parser.add_argument("--sentence-loss-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--teacherless-variance-weight", type=float, default=0.5)
    parser.add_argument("--variance-target-std", type=float, default=1.0)
    parser.add_argument("--token-set-loss-weight", type=float, default=0.5)
    parser.add_argument("--token-set-start-step", type=int, default=2000)
    parser.add_argument("--token-set-ramp-steps", type=int, default=1000)
    parser.add_argument("--token-set-temperature", type=float, default=0.07)
    parser.add_argument("--token-set-positive-weight", type=float, default=1.0)
    parser.add_argument("--token-balance-weight", type=float, default=0.2)
    parser.add_argument("--token-balance-start-step", type=int, default=2000)
    parser.add_argument("--token-balance-ramp-steps", type=int, default=1000)
    parser.add_argument("--token-balance-target-std", type=float, default=0.5)
    parser.add_argument("--covariance-weight", type=float, default=0.05)
    parser.add_argument("--covariance-start-step", type=int, default=15000)
    return parser.parse_args(argv)


def validate_args(args) -> None:
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("--batch-size and --eval-batch-size must be > 0")
    if args.max_segments <= 0 or args.min_segments <= 0:
        raise ValueError("--max-segments and --min-segments must be > 0")
    if args.min_segments > args.max_segments:
        raise ValueError("--min-segments must be <= --max-segments")
    if args.min_length_ratio <= 0.0 or args.max_length_ratio < args.min_length_ratio:
        raise ValueError("length ratio must satisfy 0 < min <= max")
    if args.shuffle_buffer_size < args.batch_size:
        raise ValueError("--shuffle-buffer-size must be >= --batch-size")
    if args.byte_conv_layers < 0:
        raise ValueError("--byte-conv-layers must be >= 0")
    if args.byte_conv_kernel_size <= 0 or args.byte_conv_kernel_size % 2 == 0:
        raise ValueError("--byte-conv-kernel-size must be a positive odd integer")
    if args.byte_conv_expansion <= 0:
        raise ValueError("--byte-conv-expansion must be > 0")
    if args.writer_loss_weight < 0.0:
        raise ValueError("--writer-loss-weight must be >= 0")
    if args.writer_num_layers < 0:
        raise ValueError("--writer-num-layers must be >= 0")
    if args.writer_conv_kernel_size <= 0 or args.writer_conv_kernel_size % 2 == 0:
        raise ValueError("--writer-conv-kernel-size must be a positive odd integer")
    if args.writer_conv_expansion <= 0:
        raise ValueError("--writer-conv-expansion must be > 0")
    if not 0.0 <= args.writer_dropout < 1.0:
        raise ValueError("--writer-dropout must be inside [0, 1)")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be > 0")
    if args.eval_every < 0 or args.checkpoint_every < 0:
        raise ValueError("--eval-every and --checkpoint-every must be >= 0")
    if args.eval_every > 0 and args.eval_file is None:
        raise ValueError("--eval-file is required when --eval-every > 0")
    if args.max_eval_batches <= 0:
        raise ValueError("--max-eval-batches must be > 0")
    if args.temperature <= 0.0 or args.token_set_temperature <= 0.0:
        raise ValueError("contrastive and token-set temperatures must be > 0")
    if min(
        args.sentence_loss_weight,
        args.teacherless_variance_weight,
        args.token_set_loss_weight,
        args.token_balance_weight,
        args.covariance_weight,
    ) < 0.0:
        raise ValueError("loss weights must be >= 0")
    if args.token_set_loss_weight > 0.0 and args.latent_size != args.hidden_size and args.resume is None:
        raise ValueError("token-set BoW uses tied embed_tokens, so --latent-size must equal --hidden-size")


def build_config(args, tokenizer: HybridTokenizer) -> DilConfig:
    if args.resume is not None:
        return DilConfig.from_pretrained(args.resume.parent)
    return DilConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        latent_size=args.latent_size,
        max_surface_pieces_per_unit=args.max_surface_pieces_per_unit,
        byte_conv_layers=args.byte_conv_layers,
        byte_conv_kernel_size=args.byte_conv_kernel_size,
        byte_conv_expansion=args.byte_conv_expansion,
        dil_dropout=args.dil_dropout,
        distillation_weight=0.0,
        mean_geometry_weight=0.0,
        variance_weight=0.0,
        writer_loss_weight=args.writer_loss_weight,
        writer_num_layers=args.writer_num_layers,
        writer_conv_kernel_size=args.writer_conv_kernel_size,
        writer_conv_expansion=args.writer_conv_expansion,
        writer_dropout=args.writer_dropout,
        writer_noise_warmup_steps=DIL_MODEL_DEFAULTS["writer_noise_warmup_steps"],
        writer_noise_clean_ratio=DIL_MODEL_DEFAULTS["writer_noise_clean_ratio"],
        writer_noise_easy_ratio=DIL_MODEL_DEFAULTS["writer_noise_easy_ratio"],
        writer_noise_mid_ratio=DIL_MODEL_DEFAULTS["writer_noise_mid_ratio"],
        writer_noise_hard_ratio=DIL_MODEL_DEFAULTS["writer_noise_hard_ratio"],
        writer_noise_easy_min_cos=DIL_MODEL_DEFAULTS["writer_noise_easy_min_cos"],
        writer_noise_easy_max_cos=DIL_MODEL_DEFAULTS["writer_noise_easy_max_cos"],
        writer_noise_mid_min_cos=DIL_MODEL_DEFAULTS["writer_noise_mid_min_cos"],
        writer_noise_mid_max_cos=DIL_MODEL_DEFAULTS["writer_noise_mid_max_cos"],
        writer_noise_hard_min_cos=DIL_MODEL_DEFAULTS["writer_noise_hard_min_cos"],
        writer_noise_hard_max_cos=DIL_MODEL_DEFAULTS["writer_noise_hard_max_cos"],
        tokenizer_vocab_file=args.tokenizer_vocab.name,
    )


def empty_metric_sums() -> dict:
    return {
        "loss": 0.0,
        "sent": 0.0,
        "writer": 0.0,
        "writer_tok": 0.0,
        "var": 0.0,
        "token_set": 0.0,
        "token_set_w": 0.0,
        "token_balance": 0.0,
        "token_balance_w": 0.0,
        "cov": 0.0,
        "cov_w": 0.0,
        "byte_acc": 0.0,
        "token_exact": 0.0,
        "stop_acc": 0.0,
        "token_set_weight": 0.0,
        "token_balance_weight": 0.0,
        "covariance_weight": 0.0,
        "batches": 0,
        "source_line_ids": set(),
    }


def accumulate_output_metrics(total: dict, outputs: TeacherlessDilOutput, batch: dict) -> None:
    total["loss"] += float(outputs.loss.detach().cpu())
    total["sent"] += float(outputs.sentence_loss.detach().cpu())
    total["writer"] += float(outputs.writer_loss.detach().cpu())
    total["writer_tok"] += float(outputs.writer_token_loss.detach().cpu())
    total["var"] += float(outputs.variance_loss.detach().cpu())
    total["token_set"] += float(outputs.token_set_loss.detach().cpu())
    total["token_set_w"] += float((outputs.token_set_loss * outputs.token_set_weight).detach().cpu())
    total["token_balance"] += float(outputs.token_balance_loss.detach().cpu())
    total["token_balance_w"] += float((outputs.token_balance_loss * outputs.token_balance_weight).detach().cpu())
    total["cov"] += float(outputs.covariance_loss.detach().cpu())
    total["cov_w"] += float((outputs.covariance_loss * outputs.covariance_weight).detach().cpu())
    total["byte_acc"] += float(outputs.byte_acc.detach().cpu())
    total["token_exact"] += float(outputs.token_exact.detach().cpu())
    total["stop_acc"] += float(outputs.stop_acc.detach().cpu())
    total["token_set_weight"] += outputs.token_set_weight
    total["token_balance_weight"] += outputs.token_balance_weight
    total["covariance_weight"] += outputs.covariance_weight
    total["batches"] += 1
    total["source_line_ids"].update(int(line_id) for line_id in batch["source_line_ids"].detach().cpu().tolist())


def reduce_metric_sums(total: dict) -> dict[str, float]:
    batches = max(total["batches"], 1)
    metrics = {
        key: value / batches
        for key, value in total.items()
        if key not in {"batches", "source_line_ids"}
    }
    if total["source_line_ids"]:
        metrics["source_lines_seen"] = len(total["source_line_ids"])
    return metrics


def format_log(step: int, metrics: dict[str, float]) -> str:
    fields = [
        f"step={step}",
        f"loss={metrics['loss']:.4f}",
        f"sent={metrics['sent']:.4f}",
        f"writer={metrics['writer']:.4f}",
        f"writer_tok={metrics['writer_tok']:.4f}",
        f"var={metrics['var']:.4f}",
        f"token_set={metrics['token_set']:.4f}",
        f"token_set_w={metrics['token_set_w']:.4f}",
        f"token_balance={metrics['token_balance']:.4f}",
        f"token_balance_w={metrics['token_balance_w']:.4f}",
        f"cov={metrics['cov']:.4f}",
        f"cov_w={metrics['cov_w']:.4f}",
        f"byte_acc={metrics['byte_acc']:.4f}",
        f"token_exact={metrics['token_exact']:.4f}",
        f"stop_acc={metrics['stop_acc']:.4f}",
        f"bow_weight={metrics['token_set_weight']:.3f}",
        f"balance_weight={metrics['token_balance_weight']:.3f}",
        f"lr={metrics['lr']:.2e}",
        f"data_s={metrics['data_seconds']:.4f}",
        f"compute_s={metrics['compute_seconds']:.4f}",
        f"t/s={metrics['tokens_per_second']:.1f}",
        f"w/s={metrics['windows_per_second']:.1f}",
        f"step/s={metrics['steps_per_second']:.2f}",
    ]
    if "source_lines_seen" in metrics:
        fields.append(f"total/row={int(metrics['source_lines_seen'])}")
    for key in sorted(k for k in metrics if k.startswith("eval_")):
        fields.append(f"{key}={metrics[key]:.4f}")
    return " ".join(fields)


class TeacherlessParallelDilTrainer(BaseTrainer):
    def __init__(self, args):
        validate_args(args)
        super().__init__(args)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.tokenizer_vocab_path = self.resolve_tokenizer_vocab_path(args)
        self.tokenizer = load_hybrid_tokenizer(self.tokenizer_vocab_path)
        self.config = build_config(args, self.tokenizer)
        if args.token_set_loss_weight > 0.0 and self.config.latent_size != self.config.hidden_size:
            raise ValueError("token-set BoW uses tied embed_tokens, so config.latent_size must equal config.hidden_size")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
        self.compile_mode = effective_compile_mode(args.compile_mode, self.device)
        validate_compile_environment(self.compile_mode)
        self.autocast_enabled = bool(args.bf16 and self.device.type == "cuda")
        self.cuda_prefetch = bool(self.device.type == "cuda" and not args.no_cuda_prefetch)
        self.model = Dil(self.config).to(self.device)
        self.model.train()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.weight_decay,
        )
        self.scheduler = make_scheduler(self.optimizer, args.learning_rate, args.warmup_steps)
        if args.resume is not None:
            self.start_step, self.last_metrics = restore_checkpoint(
                args.resume,
                self.model,
                self.optimizer,
                self.scheduler,
                self.device,
            )
        self.model.set_compiled_forwards(
            encoder_forward=compile_forward(self.model.encoder.forward, self.compile_mode, "DilEncoderCore"),
            writer_forward=compile_forward(self.model.writer.forward, self.compile_mode, "DilConditionalWriter"),
        )
        self.train_loader = None
        self.eval_loader = None
        self.prepare_data_sources()

    def resolve_tokenizer_vocab_path(self, args) -> Path:
        if args.resume is None:
            return args.tokenizer_vocab
        resume_config = DilConfig.from_pretrained(args.resume.parent)
        return args.resume.parent / resume_config.tokenizer_vocab_file

    def make_dataset(self, path: Path, batch_size: int, repeat: bool) -> TeacherlessParallelJsonlDataset:
        return TeacherlessParallelJsonlDataset(
            path,
            self.config,
            self.tokenizer,
            batch_size=batch_size,
            max_segments=self.args.max_segments,
            min_segments=self.args.min_segments,
            min_length_ratio=self.args.min_length_ratio,
            max_length_ratio=self.args.max_length_ratio,
            shuffle_buffer_size=self.args.shuffle_buffer_size,
            seed=self.args.seed,
            repeat=repeat,
            max_samples=self.args.max_samples if repeat else 0,
        )

    def prepare_data_sources(self) -> None:
        train_dataset = self.make_dataset(self.args.train_file, self.args.batch_size, repeat=True)
        self.train_loader = make_dil_batch_loader(
            train_dataset,
            num_workers=self.args.num_workers,
            pin_memory=self.device.type == "cuda",
            prefetch_factor=self.args.prefetch_factor,
        )
        if self.args.eval_every > 0:
            eval_dataset = self.make_dataset(self.args.eval_file, self.args.eval_batch_size, repeat=False)
            self.eval_loader = make_dil_batch_loader(
                eval_dataset,
                num_workers=self.args.num_workers,
                pin_memory=self.device.type == "cuda",
                prefetch_factor=self.args.prefetch_factor,
            )

    def build_train_iterator(self):
        return DeviceBatchPrefetcher(self.train_loader, self.device, self.cuda_prefetch)

    def build_eval_iterator(self):
        if self.eval_loader is None:
            return None
        return DeviceBatchPrefetcher(self.eval_loader, self.device, self.cuda_prefetch)

    def has_eval(self) -> bool:
        return self.eval_loader is not None

    def empty_metric_sums(self) -> dict:
        return empty_metric_sums()

    def accumulate_metrics(self, total: dict, result: StepResult) -> None:
        accumulate_output_metrics(total, result.outputs, result.batch)

    def reduce_metrics(self, total: dict) -> dict[str, float]:
        return reduce_metric_sums(total)

    def encode_side(self, batch: dict, prefix: str) -> torch.Tensor:
        surface = batch[f"{prefix}_surface"]
        unit_mask = batch[f"{prefix}_unit_mask"]
        latents = self.model.encode(surface).float()
        if latents.shape[:2] != unit_mask.shape:
            raise ValueError("encoded sequence latents must match unit_mask shape")
        return latents * unit_mask.unsqueeze(-1).to(latents.dtype)

    def writer_side_loss(
        self,
        latents: torch.Tensor,
        batch: dict,
        prefix: str,
        step: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        window_mask = batch[f"{prefix}_writer_window_mask"].to(latents.device, dtype=torch.bool)
        writer_semantic = gather_writer_semantic(
            latents.detach(),
            batch[f"{prefix}_writer_source_rows"],
            batch[f"{prefix}_writer_unit_indices"],
            window_mask,
        )
        return self.model.writer_transition_loss_and_metrics(
            writer_semantic,
            batch[f"{prefix}_writer_labels"],
            batch[f"{prefix}_writer_zone_ids"],
            window_mask,
            training_step=step,
        )

    def forward_batch(self, batch: dict, step: int | None) -> TeacherlessDilOutput:
        tr_latents = self.encode_side(batch, "tr")
        en_latents = self.encode_side(batch, "en")
        tr_unit_mask = batch["tr_unit_mask"]
        en_unit_mask = batch["en_unit_mask"]
        tr_sentence = masked_mean(tr_latents, tr_unit_mask)
        en_sentence = masked_mean(en_latents, en_unit_mask)

        sentence_loss = sentence_contrastive_loss(
            tr_sentence,
            en_sentence,
            temperature=self.args.temperature,
            margin=self.args.margin,
        )
        tr_writer = self.writer_side_loss(tr_latents, batch, "tr", step)
        en_writer = self.writer_side_loss(en_latents, batch, "en", step)
        writer_loss = (tr_writer[0] + en_writer[0]) * 0.5
        writer_token_loss = (tr_writer[1] + en_writer[1]) * 0.5
        byte_acc = (tr_writer[2] + en_writer[2]) * 0.5
        token_exact = (tr_writer[3] + en_writer[3]) * 0.5
        stop_acc = (tr_writer[4] + en_writer[4]) * 0.5

        variance_loss = variance_regularizer(
            torch.cat([tr_sentence, en_sentence], dim=0),
            self.args.variance_target_std,
        )
        token_set_weight = ramped_weight(
            0 if step is None else step,
            self.args.token_set_start_step,
            self.args.token_set_ramp_steps,
            self.args.token_set_loss_weight,
        )
        token_balance_weight = ramped_weight(
            0 if step is None else step,
            self.args.token_balance_start_step,
            self.args.token_balance_ramp_steps,
            self.args.token_balance_weight,
        )
        covariance_weight = ramped_weight(
            0 if step is None else step,
            self.args.covariance_start_step,
            0,
            self.args.covariance_weight,
        )

        token_embedding = self.model.encoder.embed_tokens.weight
        token_set_loss = (
            token_set_bow_loss(
                tr_sentence,
                batch["en_labels"],
                token_embedding,
                self.config.writer_stop_token_id,
                self.config.pad_token_id,
                self.config.eos_token_id,
                self.args.token_set_temperature,
                self.args.token_set_positive_weight,
            )
            + token_set_bow_loss(
                en_sentence,
                batch["tr_labels"],
                token_embedding,
                self.config.writer_stop_token_id,
                self.config.pad_token_id,
                self.config.eos_token_id,
                self.args.token_set_temperature,
                self.args.token_set_positive_weight,
            )
        ) * 0.5
        balance_loss = (
            token_balance_loss(tr_latents, tr_unit_mask, self.args.token_balance_target_std)
            + token_balance_loss(en_latents, en_unit_mask, self.args.token_balance_target_std)
        ) * 0.5
        covariance_loss = covariance_regularizer(torch.cat([tr_sentence, en_sentence], dim=0))

        loss = (
            sentence_loss * self.args.sentence_loss_weight
            + writer_loss * self.config.writer_loss_weight
            + variance_loss * self.args.teacherless_variance_weight
            + token_set_loss * token_set_weight
            + balance_loss * token_balance_weight
            + covariance_loss * covariance_weight
        )
        return TeacherlessDilOutput(
            loss=loss,
            sentence_loss=sentence_loss,
            writer_loss=writer_loss,
            writer_token_loss=writer_token_loss,
            variance_loss=variance_loss,
            token_set_loss=token_set_loss,
            token_balance_loss=balance_loss,
            covariance_loss=covariance_loss,
            byte_acc=byte_acc,
            token_exact=token_exact,
            stop_acc=stop_acc,
            token_set_weight=token_set_weight,
            token_balance_weight=token_balance_weight,
            covariance_weight=covariance_weight,
        )

    def train_step(self, batch: dict, step: int) -> StepResult:
        outputs = self.forward_batch(batch, step)
        token_count = int(
            batch["tr_labels"].label_mask.sum().detach().cpu()
            + batch["en_labels"].label_mask.sum().detach().cpu()
        )
        window_count = int(batch["tr_unit_mask"].sum().detach().cpu() + batch["en_unit_mask"].sum().detach().cpu())
        return StepResult(outputs.loss, outputs, token_count=token_count, window_count=window_count, batch=batch)

    def eval_step(self, batch: dict) -> StepResult:
        outputs = self.forward_batch(batch, self.completed_step)
        token_count = int(
            batch["tr_labels"].label_mask.sum().detach().cpu()
            + batch["en_labels"].label_mask.sum().detach().cpu()
        )
        window_count = int(batch["tr_unit_mask"].sum().detach().cpu() + batch["en_unit_mask"].sum().detach().cpu())
        return StepResult(outputs.loss, outputs, token_count=token_count, window_count=window_count, batch=batch)

    def save_checkpoint(self, checkpoint_name: str, step: int, metrics: dict[str, float]):
        return save_checkpoint(
            self.args.output_dir,
            self.model,
            self.optimizer,
            self.scheduler,
            self.config,
            self.tokenizer_vocab_path,
            step,
            metrics,
            self.compile_mode,
            runtime_training_state(self.args),
            checkpoint_name=checkpoint_name,
        )

    def is_recoverable_runtime_error(self, error: RuntimeError) -> bool:
        return is_dataloader_worker_exit(error)

    def format_log(self, step: int, metrics: dict[str, float]) -> str:
        return format_log(step, metrics)

    def run(self) -> None:
        print(
            f"device={self.device.type} bf16={int(self.autocast_enabled)} compile_mode={self.compile_mode} "
            f"resume_step={self.start_step} teacher_source=none trainer=teacherless_parallel "
            f"vocab_size={self.config.vocab_size} hidden_size={self.config.hidden_size} "
            f"latent_size={self.config.latent_size} enc_layers={self.config.encoder_context_layers} "
            f"writer_layers={self.config.writer_num_layers} batch_pairs={self.args.batch_size} "
            f"max_segments={self.args.max_segments}",
            flush=True,
        )
        super().run()


def make_trainer(args) -> TeacherlessParallelDilTrainer:
    return TeacherlessParallelDilTrainer(args)


def main(argv: list[str] | None = None) -> None:
    trainer = make_trainer(parse_args(argv))
    trainer.run()


if __name__ == "__main__":
    main()
