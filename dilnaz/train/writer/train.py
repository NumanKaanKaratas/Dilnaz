from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import IterableDataset

from dilnaz.train.common.runtime import (
    COMPILE_MODE_CHOICES,
    DeviceBatchPrefetcher,
    autocast_context,
    compile_forward,
    cudagraph_step_begin,
    cuda_sync,
    effective_compile_mode,
    load_checkpoint,
    restore_rng_state,
    rng_state,
    validate_compile_environment,
)
from dilnaz.train.data.dil_data import (
    ResidentDilBatcher,
    ResidentDilEvalLoader,
    load_hybrid_tokenizer,
    make_dil_batch_loader,
    segment_piece_ids,
    stream_teacher_text_items_with_eos,
    trainable_segments,
)
from dilnaz.surface import pack_token_units
from dilnaz.train.configs.defaults import DIL_TRAIN_DEFAULTS
from dilnaz.models.dil import DilConfig
from dilnaz.models.dil import Dil
from dilnaz.train.common.trainer_core import make_adamw_param_groups, make_scheduler
from dilnaz.train.dil.writer_units import build_writer_unit_view, gather_writer_semantic, writer_unit_counts


CHECKPOINT_FORMAT_VERSION = 30
WRITER_OBJECTIVE = "unit_surface_writer_v3"
WRITER_METRIC_KEYS = (
    "loss",
    "token_loss",
    "teacher_byte_acc",
    "teacher_token_exact",
    "teacher_stop_acc",
)
FREE_METRIC_KEYS = (
    "free_byte_acc",
    "free_token_exact",
    "short_le4_exact",
    "short_le8_exact",
    "short_le16_exact",
    "first_token_exact",
    "empty_output_rate",
)


class HybridDilUnitWriterDataset(IterableDataset):
    def __init__(
        self,
        train_file: Path,
        config: DilConfig,
        tokenizer,
        batch_size: int,
        read_chars: int,
        repeat: bool = True,
        max_samples: int = 0,
        context_aug_max_units: int = 16,
        context_aug_stride: int = 8,
    ):
        super().__init__()
        self.train_file = train_file
        self.config = config
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.read_chars = read_chars
        self.repeat = repeat
        self.max_samples = max_samples
        self.context_aug_max_units = context_aug_max_units
        self.context_aug_stride = context_aug_stride
        self._carry_line_ids: list[int] = []
        self._carry_unit_rows: list[list[list[int]]] = []
        self._carry_unit_count = 0
        self._produced = 0

    @staticmethod
    def _span_key(row: list[list[int]]) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(unit) for unit in row)

    def context_augmented_rows(self, row: list[list[int]]) -> list[list[list[int]]]:
        row_len = len(row)
        if self.context_aug_max_units <= 0 or row_len <= 1:
            return []

        max_span_size = min(self.context_aug_max_units, row_len)
        rows: list[list[list[int]]] = []
        seen = {self._span_key(row)}

        def add_span(start: int, end: int) -> None:
            if end <= start or (start == 0 and end == row_len):
                return
            span = [list(unit) for unit in row[start:end]]
            key = self._span_key(span)
            if key not in seen:
                seen.add(key)
                rows.append(span)

        span_size = 1
        while span_size < max_span_size:
            add_span(0, span_size)
            add_span(row_len - span_size, row_len)
            span_size *= 2
        add_span(0, max_span_size)
        add_span(row_len - max_span_size, row_len)

        if max_span_size < row_len:
            start = 0
            while start < row_len:
                end = min(start + max_span_size, row_len)
                if end - start < max_span_size:
                    start = row_len - max_span_size
                    end = row_len
                add_span(start, end)
                if end == row_len:
                    break
                start += self.context_aug_stride
        return rows

    def append_training_row(
        self,
        row: list[list[int]],
        source_line_id: int,
        unit_rows: list[list[list[int]]],
        line_ids: list[int],
        unit_count: int,
    ) -> tuple[list[list[list[int]]], list[int], int, dict | None]:
        unit_rows.append(row)
        line_ids.append(source_line_id)
        unit_count += len(row)
        self._produced += len(row)
        if unit_count >= self.batch_size:
            batch = self.make_batch(unit_rows, line_ids)
            return [], [], 0, batch
        return unit_rows, line_ids, unit_count, None

    def make_batch(self, unit_rows: list[list[list[int]]], line_ids: list[int]):
        writer_view = build_writer_unit_view(unit_rows, self.config)
        return {
            "surface": pack_token_units(
                unit_rows,
                pad_token_id=self.config.pad_token_id,
                bucket_sizes=self.config.surface_bucket_sizes,
                max_pieces_per_unit=self.config.max_surface_pieces_per_unit,
            ),
            "writer_labels": writer_view["writer_labels"],
            "writer_source_rows": writer_view["writer_source_rows"],
            "writer_unit_indices": writer_view["writer_unit_indices"],
            "source_line_ids": torch.tensor(line_ids, dtype=torch.long),
        }

    def carry_batch(self):
        if not self._carry_unit_rows:
            return None
        batch = self.make_batch(self._carry_unit_rows, self._carry_line_ids)
        self._carry_line_ids = []
        self._carry_unit_rows = []
        self._carry_unit_count = 0
        return batch

    def iter_once(self, worker_id: int, worker_count: int):
        line_ids = self._carry_line_ids
        unit_rows = self._carry_unit_rows
        unit_count = self._carry_unit_count
        self._carry_line_ids = []
        self._carry_unit_rows = []
        self._carry_unit_count = 0

        for text_idx, (source_line_id, text, add_eos) in enumerate(
            stream_teacher_text_items_with_eos(self.train_file, self.read_chars)
        ):
            if text_idx % worker_count != worker_id:
                continue
            segments = trainable_segments(
                self.tokenizer,
                text,
                self.config.max_surface_pieces_per_unit,
                add_eos=add_eos,
            )
            if not segments:
                continue
            row = [segment_piece_ids(segment) for segment in segments]
            for candidate in (row, *self.context_augmented_rows(row)):
                unit_rows, line_ids, unit_count, batch = self.append_training_row(
                    candidate,
                    source_line_id,
                    unit_rows,
                    line_ids,
                    unit_count,
                )
                if batch is not None:
                    yield batch
                if self.max_samples > 0 and self._produced >= self.max_samples:
                    if unit_rows:
                        yield self.make_batch(unit_rows, line_ids)
                    return

        if unit_rows and not self.repeat:
            yield self.make_batch(unit_rows, line_ids)
        elif unit_rows:
            self._carry_line_ids = line_ids
            self._carry_unit_rows = unit_rows
            self._carry_unit_count = unit_count

    def __iter__(self):
        from torch.utils.data import get_worker_info

        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        while True:
            yielded = False
            for batch in self.iter_once(worker_id, worker_count):
                yielded = True
                yield batch
            if not yielded and not self._carry_unit_rows:
                raise ValueError(f"{self.train_file} produced no unit writer samples")
            if not self.repeat:
                return


def freeze_for_writer_only(model: Dil):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.writer.parameters():
        param.requires_grad = True
    model.shared_token_embeddings.weight.requires_grad = False
    model.encoder.eval()
    model.writer.train()


def free_writer_metrics(
    model: Dil,
    semantic: torch.Tensor,
    target,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        generation = model.decode_semantic(semantic)

    labels = target.labels.to(semantic.device)
    label_mask = target.label_mask.to(semantic.device)
    true_lengths = target.true_lengths.to(semantic.device)
    piece_lengths = (true_lengths[:, 0] - 1).clamp_min(0)
    width = min(generation.token_ids.shape[-1], labels.shape[-1])
    target_ids = labels[:, :width]
    target_mask = label_mask[:, :width] & target_ids.ne(model.config.eos_token_id)
    pred_ids = generation.token_ids[:, :width]
    pred_mask = generation.token_mask[:, :width]

    byte_matches = pred_mask & target_mask & pred_ids.eq(target_ids)
    free_byte_acc = byte_matches.sum().float() / target_mask.sum().clamp_min(1).float()
    mask_exact = pred_mask.eq(target_mask).all(dim=1)
    id_exact = ((~target_mask) | pred_ids.eq(target_ids)).all(dim=1)
    token_exact_rows = mask_exact & id_exact & true_lengths[:, 0].gt(0)
    valid_rows = true_lengths[:, 0].gt(0)
    free_token_exact = token_exact_rows.sum().float() / valid_rows.sum().clamp_min(1).float()
    first_token_exact = (
        pred_mask[:, 0] & target_mask[:, 0] & pred_ids[:, 0].eq(target_ids[:, 0])
    ).sum().float() / valid_rows.sum().clamp_min(1).float()
    empty_output_rate = generation.lengths.eq(0).float().mean()

    metrics = {
        "free_byte_acc": free_byte_acc,
        "free_token_exact": free_token_exact,
        "first_token_exact": first_token_exact,
        "empty_output_rate": empty_output_rate,
    }
    for threshold in (4, 8, 16):
        short = valid_rows & piece_lengths.le(threshold)
        metrics[f"short_le{threshold}_exact"] = token_exact_rows[short].sum().float() / short.sum().clamp_min(1).float()
    return metrics


def writer_only_metrics(
    model: Dil,
    batch: dict,
    compute_free_metrics: bool = False,
) -> dict[str, torch.Tensor]:
    surface = batch["surface"]
    target = batch["writer_labels"].to(surface.ids.device)
    with torch.no_grad():
        full_semantic = model.encode(surface).float()
    semantic = gather_writer_semantic(
        full_semantic,
        batch["writer_source_rows"],
        batch["writer_unit_indices"],
    )
    metrics = model.writer_loss_and_metrics(
        semantic.detach(),
        target,
        return_metrics=True,
    )
    output = {
        "loss": metrics["loss"],
        "token_loss": metrics["token_loss"],
        "teacher_byte_acc": metrics["byte_acc"],
        "teacher_token_exact": metrics["token_exact"],
        "teacher_stop_acc": metrics["stop_acc"],
    }
    if compute_free_metrics:
        output.update(free_writer_metrics(model, semantic.detach(), target))
    return output


def writer_only_forward(model: Dil, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    metrics = writer_only_metrics(model, batch)
    return metrics["loss"], metrics["teacher_byte_acc"], metrics["teacher_token_exact"], metrics["teacher_stop_acc"]


def materialize_writer_batches(dataset, device: torch.device, batch_size: int, seed: int):
    keep_keys = {
        "surface",
        "writer_labels",
        "writer_source_rows",
        "writer_unit_indices",
        "source_line_ids",
    }
    batches = [
        {
            key: value.detach().cpu() if hasattr(value, "detach") else value
            for key, value in batch.items()
            if key in keep_keys
        }
        for batch in dataset.iter_once(worker_id=0, worker_count=1)
    ]
    carry_batch = dataset.carry_batch() if hasattr(dataset, "carry_batch") else None
    if carry_batch is not None:
        batches.append(
            {
                key: value.detach().cpu() if hasattr(value, "detach") else value
                for key, value in carry_batch.items()
                if key in keep_keys
            }
        )
    return ResidentDilBatcher(batches, batch_size=batch_size, device=device, seed=seed)


def save_checkpoint(
    output_dir: Path,
    model: Dil,
    optimizer,
    scheduler,
    config: DilConfig,
    tokenizer_vocab_path: Path,
    step: int,
    metrics: dict,
    compile_mode: str,
    checkpoint_name: str = "",
):
    checkpoint_dir = output_dir / checkpoint_name if checkpoint_name else output_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(checkpoint_dir)
    dst_vocab = checkpoint_dir / config.tokenizer_vocab_file
    if tokenizer_vocab_path.resolve() != dst_vocab.resolve():
        shutil.copyfile(tokenizer_vocab_path, dst_vocab)
    training_state = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "objective": WRITER_OBJECTIVE,
        "step": step,
        "metrics": metrics,
        "compile_mode": compile_mode,
        "vocab_size": config.vocab_size,
        "pad_token_id": config.pad_token_id,
        "eos_token_id": config.eos_token_id,
        "max_surface_pieces_per_unit": config.max_surface_pieces_per_unit,
        "max_sequence_units": config.max_sequence_units,
        "encoder_context_layers": config.encoder_context_layers,
        "latent_size": config.latent_size,
        "writer_gradient_checkpointing": config.writer_gradient_checkpointing,
    }
    import os as _os

    tmp_path = checkpoint_dir / "checkpoint.pt.tmp"
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": model.state_dict(),
            "writer_optimizer_state_dict": optimizer.state_dict(),
            "writer_scheduler_state_dict": scheduler.state_dict(),
            "training_state": training_state,
            "rng_state": rng_state(),
        },
        tmp_path,
    )
    _os.replace(str(tmp_path), str(checkpoint_dir / "checkpoint.pt"))
    with (checkpoint_dir / "training_state.json").open("w", encoding="utf-8") as handle:
        json.dump(training_state, handle, indent=2)
    return checkpoint_dir


def load_model_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[Dil, DilConfig, dict]:
    config = DilConfig.from_pretrained(checkpoint_path.parent)
    model = Dil(config).to(device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"unsupported Dil checkpoint format_version={checkpoint.get('format_version')}")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config, checkpoint


@torch.no_grad()
def evaluate(
    model,
    eval_loader,
    device,
    compile_mode: str,
    autocast_enabled: bool,
    cuda_prefetch: bool,
    max_batches: int,
):
    model.eval()
    model.encoder.eval()
    total = {key: 0.0 for key in (*WRITER_METRIC_KEYS, *FREE_METRIC_KEYS)}
    total["batches"] = 0
    for batch_idx, batch in enumerate(DeviceBatchPrefetcher(eval_loader, device, cuda_prefetch), start=1):
        cudagraph_step_begin(device, compile_mode)
        with autocast_context(autocast_enabled):
            metrics = writer_only_metrics(
                model,
                batch,
                compute_free_metrics=True,
            )
        for key in total:
            if key == "batches":
                continue
            total[key] += float(metrics[key].detach().cpu())
        total["batches"] += 1
        if batch_idx >= max_batches:
            break
    model.train()
    model.encoder.eval()
    batches = max(total.pop("batches"), 1)
    return {f"eval_{key}": value / batches for key, value in total.items()}


def format_log(step: int, metrics: dict) -> str:
    fields = [
        f"step={step}",
        f"loss={metrics['loss']:.4f}",
        f"tok={metrics['token_loss']:.4f}",
        f"teacher_byte_acc={metrics['teacher_byte_acc']:.4f}",
        f"teacher_token_exact={metrics['teacher_token_exact']:.4f}",
        f"teacher_stop_acc={metrics['teacher_stop_acc']:.4f}",
        f"lr={metrics['lr']:.2e}",
        f"data_s={metrics['data_seconds']:.4f}",
        f"compute_s={metrics['compute_seconds']:.4f}",
        f"t/s={metrics['tokens_per_second']:.1f}",
        f"u/s={metrics['units_per_second']:.1f}",
        f"step/s={metrics['steps_per_second']:.2f}",
    ]
    if "source_lines_seen" in metrics:
        fields.append(f"total/row={int(metrics['source_lines_seen'])}")
    for key in sorted(k for k in metrics if k.startswith("eval_")):
        fields.append(f"{key}={metrics[key]:.4f}")
    return " ".join(fields)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compile-mode", choices=COMPILE_MODE_CHOICES, default=None)
    parser.add_argument("--data-mode", choices=("streaming", "resident"), default=DIL_TRAIN_DEFAULTS["data_mode"])
    parser.add_argument("--max-steps", type=int, default=DIL_TRAIN_DEFAULTS["max_steps"])
    parser.add_argument("--batch-size", type=int, default=DIL_TRAIN_DEFAULTS["batch_size"])
    parser.add_argument("--eval-batch-size", type=int, default=DIL_TRAIN_DEFAULTS["eval_batch_size"])
    parser.add_argument("--text-read-chars", type=int, default=DIL_TRAIN_DEFAULTS["text_read_chars"])
    parser.add_argument("--prefetch-factor", type=int, default=DIL_TRAIN_DEFAULTS["prefetch_factor"])
    parser.add_argument("--no-cuda-prefetch", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=DIL_TRAIN_DEFAULTS["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=DIL_TRAIN_DEFAULTS["weight_decay"])
    parser.add_argument("--adam-beta1", type=float, default=DIL_TRAIN_DEFAULTS["adam_beta1"])
    parser.add_argument("--adam-beta2", type=float, default=DIL_TRAIN_DEFAULTS["adam_beta2"])
    parser.add_argument("--warmup-steps", type=int, default=DIL_TRAIN_DEFAULTS["warmup_steps"])
    parser.add_argument("--gradient-accumulation-steps", type=int, default=DIL_TRAIN_DEFAULTS["gradient_accumulation_steps"])
    parser.add_argument("--max-grad-norm", type=float, default=DIL_TRAIN_DEFAULTS["max_grad_norm"])
    parser.add_argument("--log-every", type=int, default=DIL_TRAIN_DEFAULTS["log_every"])
    parser.add_argument("--checkpoint-every", type=int, default=DIL_TRAIN_DEFAULTS["checkpoint_every"])
    parser.add_argument("--eval-every", type=int, default=DIL_TRAIN_DEFAULTS["eval_every"])
    parser.add_argument("--max-eval-batches", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=DIL_TRAIN_DEFAULTS["num_workers"])
    parser.add_argument("--seed", type=int, default=DIL_TRAIN_DEFAULTS["seed"])
    parser.add_argument("--max-samples", type=int, default=DIL_TRAIN_DEFAULTS["max_samples"])
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--writer-gradient-checkpointing", action="store_true")
    parser.add_argument("--context-aug-max-units", type=int, default=16)
    parser.add_argument("--context-aug-stride", type=int, default=8)
    return parser.parse_args()


def validate_args(args):
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("--batch-size and --eval-batch-size must be > 0")
    if args.text_read_chars <= 0:
        raise ValueError("--text-read-chars must be > 0")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be > 0")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be > 0")
    if args.eval_every < 0 or args.checkpoint_every < 0:
        raise ValueError("--eval-every and --checkpoint-every must be >= 0")
    if args.eval_every > 0 and args.eval_file is None:
        raise ValueError("--eval-file is required when --eval-every > 0")
    if args.max_eval_batches <= 0:
        raise ValueError("--max-eval-batches must be > 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.data_mode == "resident" and args.max_samples > 0:
        raise ValueError("--max-samples is not supported with --data-mode resident")
    if args.context_aug_max_units < 0:
        raise ValueError("--context-aug-max-units must be >= 0")
    if args.context_aug_max_units > 0 and args.context_aug_stride <= 0:
        raise ValueError("--context-aug-stride must be > 0 when context augmentation is enabled")


def sync_writer_runtime_config(model: Dil, config: DilConfig) -> None:
    model.writer.gradient_checkpointing = config.writer_gradient_checkpointing


def main():
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    compile_mode = effective_compile_mode(args.compile_mode, device)
    validate_compile_environment(compile_mode)
    autocast_enabled = bool(args.bf16 and device.type == "cuda")
    cuda_prefetch = bool(device.type == "cuda" and not args.no_cuda_prefetch)

    model, config, checkpoint = load_model_checkpoint(args.checkpoint, device)
    config.writer_gradient_checkpointing = bool(args.writer_gradient_checkpointing)
    sync_writer_runtime_config(model, config)
    tokenizer_vocab_path = args.checkpoint.parent / config.tokenizer_vocab_file
    tokenizer = load_hybrid_tokenizer(tokenizer_vocab_path)
    writer_resume = checkpoint.get("training_state", {}).get("objective") == WRITER_OBJECTIVE
    if not writer_resume:
        fresh_model = Dil(config).to(device)
        model.writer.load_state_dict(fresh_model.writer.state_dict())
        del fresh_model
    freeze_for_writer_only(model)
    model.set_compiled_forwards(
        encoder_forward=compile_forward(model.encoder.forward, compile_mode, "DilEncoderCore"),
        writer_forward=compile_forward(model.writer.forward, compile_mode, "DilConditionalWriter"),
        transition_forward=compile_forward(model.writer.transition, compile_mode, "DilConditionalWriterTransition"),
    )
    writer_named_parameters = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad and name.startswith("writer.")
    ]
    optimizer = AdamW(
        make_adamw_param_groups(writer_named_parameters, args.weight_decay),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
    )
    scheduler = make_scheduler(optimizer, args.learning_rate, args.warmup_steps, args.max_steps)
    if writer_resume:
        optimizer.load_state_dict(checkpoint["writer_optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["writer_scheduler_state_dict"])
        restore_rng_state(checkpoint["rng_state"])

    train_dataset = HybridDilUnitWriterDataset(
        args.train_file,
        config,
        tokenizer,
        batch_size=args.batch_size,
        read_chars=args.text_read_chars,
        repeat=True,
        max_samples=args.max_samples,
        context_aug_max_units=args.context_aug_max_units,
        context_aug_stride=args.context_aug_stride,
    )
    eval_dataset = None
    if args.eval_every > 0:
        eval_dataset = HybridDilUnitWriterDataset(
            args.eval_file,
            config,
            tokenizer,
            batch_size=args.eval_batch_size,
            read_chars=args.text_read_chars,
            repeat=False,
            context_aug_max_units=args.context_aug_max_units,
            context_aug_stride=args.context_aug_stride,
        )

    if args.data_mode == "resident":
        print("resident_writer_data_prepare_start=1", flush=True)
        train_iter = materialize_writer_batches(train_dataset, device, args.batch_size, args.seed)
        print(f"resident_writer_data_prepare_done=1 batches={len(train_iter.batches)}", flush=True)
        eval_loader = None
        if eval_dataset is not None:
            print("resident_writer_eval_prepare_start=1", flush=True)
            eval_loader = ResidentDilEvalLoader(
                materialize_writer_batches(eval_dataset, device, args.eval_batch_size, args.seed + 1)
            )
            print(f"resident_writer_eval_prepare_done=1 batches={len(eval_loader.batches)}", flush=True)
    else:
        train_loader = make_dil_batch_loader(
            train_dataset,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            prefetch_factor=args.prefetch_factor,
        )
        train_iter = DeviceBatchPrefetcher(train_loader, device, cuda_prefetch)
        eval_loader = None
        if eval_dataset is not None:
            eval_loader = make_dil_batch_loader(
                eval_dataset,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                prefetch_factor=args.prefetch_factor,
            )

    print(
        f"device={device.type} bf16={int(autocast_enabled)} compile_mode={compile_mode} "
        f"data_mode={args.data_mode} objective={WRITER_OBJECTIVE} "
        f"vocab_size={config.vocab_size} latent_size={config.latent_size} hidden_size={config.hidden_size} "
        f"unit_local=1 shared_embedding=1 context_aug={args.context_aug_max_units}/{args.context_aug_stride} "
        f"writer_resume={int(writer_resume)}",
        flush=True,
    )

    log_start = time.perf_counter()
    log_tokens = 0
    log_units = 0
    log_steps = 0
    log_micro_steps = 0
    data_seconds = 0.0
    compute_seconds = 0.0
    source_lines_seen: set[int] = set()
    metric_sums = {key: 0.0 for key in WRITER_METRIC_KEYS}
    last_metrics = {}
    completed_step = checkpoint.get("training_state", {}).get("step", 0) if writer_resume else 0

    def save_current(checkpoint_name: str = ""):
        return save_checkpoint(
            args.output_dir,
            model,
            optimizer,
            scheduler,
            config,
            tokenizer_vocab_path,
            completed_step,
            last_metrics,
            compile_mode,
            checkpoint_name=checkpoint_name,
        )

    try:
        for step in range(completed_step + 1, args.max_steps + 1):
            compute_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            for _ in range(args.gradient_accumulation_steps):
                data_start = time.perf_counter()
                batch = next(train_iter)
                data_seconds += time.perf_counter() - data_start

                cudagraph_step_begin(device, compile_mode)
                with autocast_context(autocast_enabled):
                    metrics = writer_only_metrics(
                        model,
                        batch,
                    )
                    loss = metrics["loss"]
                (loss / args.gradient_accumulation_steps).backward()
                real_tokens, real_units = writer_unit_counts(batch["writer_labels"])
                log_tokens += real_tokens
                log_units += real_units
                log_micro_steps += 1
                if "source_line_ids" in batch:
                    source_lines_seen.update(int(line_id) for line_id in batch["source_line_ids"].detach().cpu().tolist())
                for key in WRITER_METRIC_KEYS:
                    metric_sums[key] += float(metrics[key].detach().cpu())
            torch.nn.utils.clip_grad_norm_(model.writer.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            cuda_sync(device)
            compute_seconds += time.perf_counter() - compute_start
            completed_step = step

            log_steps += 1

            should_log = step % args.log_every == 0 or step == 1 or step == args.max_steps
            should_eval = eval_loader is not None and args.eval_every > 0 and step % args.eval_every == 0
            if should_log or should_eval:
                elapsed = max(time.perf_counter() - log_start, 1e-9)
                averaged = {key: value / max(log_micro_steps, 1) for key, value in metric_sums.items()}
                averaged["lr"] = scheduler.get_last_lr()[0]
                averaged["data_seconds"] = data_seconds / max(log_steps, 1)
                averaged["compute_seconds"] = compute_seconds / max(log_steps, 1)
                averaged["tokens_per_second"] = log_tokens / elapsed
                averaged["units_per_second"] = log_units / elapsed
                averaged["steps_per_second"] = log_steps / elapsed
                if source_lines_seen:
                    averaged["source_lines_seen"] = len(source_lines_seen)
                if should_eval:
                    averaged.update(
                        evaluate(
                            model,
                            eval_loader,
                            device,
                            compile_mode,
                            autocast_enabled,
                            cuda_prefetch,
                            args.max_eval_batches,
                        )
                    )
                print(format_log(step, averaged), flush=True)
                last_metrics = averaged
                log_start = time.perf_counter()
                log_tokens = 0
                log_units = 0
                log_steps = 0
                log_micro_steps = 0
                data_seconds = 0.0
                compute_seconds = 0.0
                for key in metric_sums:
                    metric_sums[key] = 0.0

            if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                save_current(checkpoint_name=f"checkpoint-{step}")
    except KeyboardInterrupt:
        interrupted_dir = save_current()
        print(f"interrupted_saved={interrupted_dir}", flush=True)
        return

    final_dir = save_current()
    print(f"saved={final_dir}", flush=True)


if __name__ == "__main__":
    main()
