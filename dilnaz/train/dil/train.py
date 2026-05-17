import argparse
import json
import queue
import random
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from dilnaz.train.common.runtime import (
    COMPILE_MODE_CHOICES,
    DeviceBatchPrefetcher,
    compile_forward,
    cuda_sync,
    effective_compile_mode,
    load_checkpoint,
    restore_rng_state,
    rng_state,
    validate_compile_environment,
)
from dilnaz.train.common.objectives import DIL_OBJECTIVE, WRITER_OBJECTIVE
from dilnaz.train.data.dil_data import (
    ContextDilBatchDataset,
    NllbTeacher,
    ResidentDilBatcher,
    ResidentDilEvalLoader,
    load_hybrid_tokenizer,
    make_dil_batch_loader,
)
from dilnaz.train.configs.defaults import DIL_MODEL_DEFAULTS, DIL_TRAIN_DEFAULTS
from dilnaz.models.dil import DilConfig
from dilnaz.models.dil import Dil
from dilnaz.tokenization import default_vocab_path
from dilnaz.train.common.trainer_core import BaseTrainer, StepResult, make_scheduler
CHECKPOINT_FORMAT_VERSION = 33
DATALOADER_WORKER_EXIT = "DataLoader worker"


class AsyncTeacherBatchSource:
    def __init__(self, train_iter, teacher, device: torch.device, max_batch_reuse: int):
        self.train_iter = train_iter
        self.teacher = teacher
        self.device = device
        self.max_batch_reuse = max_batch_reuse
        self.ready: queue.Queue[dict | BaseException] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self.produce, daemon=True)
        self.worker.start()

    def produce(self):
        try:
            while not self.stop_event.is_set():
                batch = next(self.train_iter)
                teacher_layers, teacher_mask = self.teacher.teacher_layers(batch)
                batch["teacher_layers"] = teacher_layers
                batch["teacher_mask"] = teacher_mask
                batch["_teacher_stats"] = dict(getattr(self.teacher, "last_stats", {}))
                cuda_sync(self.device)
                self.ready.put(batch)
        except BaseException as error:
            self.ready.put(error)

    def unwrap(self, item: dict | BaseException) -> dict:
        if isinstance(item, BaseException):
            raise item
        return item

    def first(self) -> tuple[dict, int, float]:
        start = time.perf_counter()
        return self.unwrap(self.ready.get()), 1, time.perf_counter() - start

    def next_after_step(self, current: dict, seen_count: int) -> tuple[dict, int, float]:
        try:
            return self.unwrap(self.ready.get_nowait()), 1, 0.0
        except queue.Empty:
            if seen_count < self.max_batch_reuse:
                return current, seen_count + 1, 0.0
            start = time.perf_counter()
            return self.unwrap(self.ready.get()), 1, time.perf_counter() - start

    def close(self):
        self.stop_event.set()


def json_training_state(config, step: int, metrics: dict, compile_mode: str):
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "objective": DIL_OBJECTIVE,
        "step": step,
        "metrics": metrics,
        "compile_mode": compile_mode,
        "vocab_size": config.vocab_size,
        "pad_token_id": config.pad_token_id,
        "eos_token_id": config.eos_token_id,
        "max_surface_pieces_per_unit": config.max_surface_pieces_per_unit,
        "context_radius": config.context_radius,
        "latent_size": config.latent_size,
        "semantic_latent_size": config.semantic_latent_size,
        "surface_latent_size": config.surface_latent_size,
        "distillation_weight": config.distillation_weight,
    }


def runtime_training_state(args) -> dict:
    return {
        "data_mode": args.data_mode,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "nllb_batch_size": args.nllb_batch_size,
        "teacher_cache_dir": str(args.teacher_cache_dir) if args.teacher_cache_dir is not None else None,
        "no_teacher_cache": args.no_teacher_cache,
        "max_batch_reuse": args.max_batch_reuse,
        "text_read_chars": args.text_read_chars,
        "prefetch_factor": args.prefetch_factor,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "warmup_steps": args.warmup_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_grad_norm": args.max_grad_norm,
        "log_every": args.log_every,
        "checkpoint_every": args.checkpoint_every,
        "eval_every": args.eval_every,
        "max_eval_batches": args.max_eval_batches,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "max_samples": args.max_samples,
    }


def save_checkpoint(
    output_dir: Path,
    model,
    optimizer,
    scheduler,
    config,
    tokenizer_vocab_path: Path,
    step: int,
    metrics: dict,
    compile_mode: str,
    runtime: dict | None = None,
    checkpoint_name: str = "",
):
    checkpoint_dir = output_dir / checkpoint_name if checkpoint_name else output_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(checkpoint_dir)
    dst_vocab = checkpoint_dir / config.tokenizer_vocab_file
    if tokenizer_vocab_path.resolve() != dst_vocab.resolve():
        shutil.copyfile(tokenizer_vocab_path, dst_vocab)
    state = json_training_state(config, step, metrics, compile_mode)
    if runtime is not None:
        state["runtime"] = runtime
    import os as _os

    tmp_path = checkpoint_dir / "checkpoint.pt.tmp"
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "training_state": state,
            "rng_state": rng_state(),
        },
        tmp_path,
    )
    _os.replace(str(tmp_path), str(checkpoint_dir / "checkpoint.pt"))
    with (checkpoint_dir / "training_state.json").open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    return checkpoint_dir


def restore_checkpoint(path: Path, model, optimizer, scheduler, device: torch.device) -> tuple[int, dict]:
    checkpoint = load_checkpoint(path, device)
    if checkpoint["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format_version={checkpoint.get('format_version')}")
    model.load_state_dict(checkpoint["model_state_dict"])
    training_state = checkpoint["training_state"]
    objective = training_state.get("objective", DIL_OBJECTIVE)

    has_dil_optimizer = "optimizer_state_dict" in checkpoint and "scheduler_state_dict" in checkpoint
    if has_dil_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        restore_rng_state(checkpoint["rng_state"])
        if objective == WRITER_OBJECTIVE:
            return int(training_state.get("source_dil_step", 0)), dict(training_state.get("source_dil_metrics", {}))
        return int(training_state["step"]), dict(training_state["metrics"])

    if objective != WRITER_OBJECTIVE:
        raise ValueError("DIL resume checkpoint is missing optimizer_state_dict and scheduler_state_dict")
    if "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"])
    return 0, {}


def is_dataloader_worker_exit(error: RuntimeError) -> bool:
    message = str(error)
    return DATALOADER_WORKER_EXIT in message and "exited unexpectedly" in message


def model_inputs(batch: dict) -> dict:
    inputs = {
        "surface": batch["surface"],
        "teacher_layers": batch["teacher_layers"],
        "teacher_mask": batch["teacher_mask"],
    }
    if "writer_target" in batch:
        inputs["writer_target"] = batch["writer_target"]
    return inputs


def prepare_writer_for_surface_training(model: Dil) -> None:
    model.writer.train()
    for param in model.writer.parameters():
        param.requires_grad_(True)


class AsyncTeacherIterator:
    def __init__(self, batch_source: AsyncTeacherBatchSource):
        self.batch_source = batch_source
        self.current_batch = None
        self.current_batch_seen = 0
        self.last_data_seconds = 0.0
        self.last_transfer_seconds = 0.0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_batch is None:
            self.current_batch, self.current_batch_seen, self.last_data_seconds = self.batch_source.first()
        else:
            self.current_batch, self.current_batch_seen, self.last_data_seconds = self.batch_source.next_after_step(
                self.current_batch,
                self.current_batch_seen,
            )
        self.current_batch["_teacher_reuse_count"] = self.current_batch_seen
        self.last_transfer_seconds = 0.0
        return self.current_batch


def empty_metric_sums() -> dict:
    return {
        "loss": 0.0,
        "distill": 0.0,
        "sem_loss": 0.0,
        "surface_loss": 0.0,
        "surface_norm": 0.0,
        "geom_mean": 0.0,
        "var": 0.0,
        "writer_token_exact": 0.0,
        "writer_stop_acc": 0.0,
        "teacher_total_seconds": 0.0,
        "teacher_tokenize_seconds": 0.0,
        "teacher_forward_seconds": 0.0,
        "teacher_pool_seconds": 0.0,
        "teacher_cache_hit_rate": 0.0,
        "teacher_padding_ratio": 0.0,
        "teacher_nllb_tokens_per_second": 0.0,
        "teacher_unique_texts": 0.0,
        "teacher_cache_hits": 0.0,
        "teacher_cache_misses": 0.0,
        "teacher_batches": 0,
        "batches": 0,
        "source_line_ids": set(),
    }


def accumulate_output_metrics(total: dict, outputs, batch: dict) -> None:
    total["loss"] += float(outputs.loss.detach().cpu())
    total["distill"] += float(outputs.distill_loss.detach().cpu())
    total["sem_loss"] += float(outputs.semantic_loss.detach().cpu())
    total["surface_loss"] += float(outputs.surface_loss.detach().cpu())
    total["surface_norm"] += float(outputs.surface_norm.detach().cpu())
    total["geom_mean"] += float(outputs.mean_geometry_loss.detach().cpu())
    total["var"] += float(outputs.variance_loss.detach().cpu())
    total["writer_token_exact"] += float(outputs.token_exact.detach().cpu())
    total["writer_stop_acc"] += float(outputs.stop_acc.detach().cpu())
    total["batches"] += 1
    stats = batch.get("_teacher_stats")
    if stats and int(batch.get("_teacher_reuse_count", 1)) == 1:
        total["teacher_total_seconds"] += float(stats.get("total_seconds", 0.0))
        total["teacher_tokenize_seconds"] += float(stats.get("tokenize_seconds", 0.0))
        total["teacher_forward_seconds"] += float(stats.get("forward_seconds", 0.0))
        total["teacher_pool_seconds"] += float(stats.get("pool_seconds", 0.0))
        total["teacher_cache_hit_rate"] += float(stats.get("cache_hit_rate", 0.0))
        total["teacher_padding_ratio"] += float(stats.get("padding_ratio", 0.0))
        total["teacher_nllb_tokens_per_second"] += float(stats.get("nllb_tokens_per_second", 0.0))
        total["teacher_unique_texts"] += float(stats.get("unique_texts", 0.0))
        total["teacher_cache_hits"] += float(stats.get("cache_hits", 0.0))
        total["teacher_cache_misses"] += float(stats.get("cache_misses", 0.0))
        total["teacher_batches"] += 1
    if "source_line_ids" in batch:
        total["source_line_ids"].update(int(line_id) for line_id in batch["source_line_ids"].detach().cpu().tolist())


def reduce_metric_sums(total: dict) -> dict[str, float]:
    batches = max(total["batches"], 1)
    metrics = {
        key: value / batches
        for key, value in total.items()
        if key not in {"batches", "source_line_ids", "teacher_batches"}
    }
    teacher_batches = total.get("teacher_batches", 0)
    if teacher_batches > 0:
        for key in (
            "teacher_total_seconds",
            "teacher_tokenize_seconds",
            "teacher_forward_seconds",
            "teacher_pool_seconds",
            "teacher_cache_hit_rate",
            "teacher_padding_ratio",
            "teacher_nllb_tokens_per_second",
            "teacher_unique_texts",
            "teacher_cache_hits",
            "teacher_cache_misses",
        ):
            metrics[key] = total[key] / teacher_batches
    if total["source_line_ids"]:
        metrics["source_lines_seen"] = len(total["source_line_ids"])
    return metrics


def format_log(step, metrics):
    fields = [
        f"step={step}",
        f"loss={metrics['loss']:.4f}",
        f"distill={metrics['distill']:.4f}",
        f"sem_loss={metrics['sem_loss']:.4f}",
        f"surface_loss={metrics['surface_loss']:.4f}",
        f"surface_norm={metrics['surface_norm']:.4f}",
        f"geom_mean={metrics['geom_mean']:.4f}",
        f"var={metrics['var']:.4f}",
        f"writer_token_exact={metrics['writer_token_exact']:.4f}",
        f"writer_stop_acc={metrics['writer_stop_acc']:.4f}",
        f"lr={metrics['lr']:.2e}",
        f"data_s={metrics['data_seconds']:.4f}",
        f"compute_s={metrics['compute_seconds']:.4f}",
        f"t/s={metrics['tokens_per_second']:.1f}",
        f"w/s={metrics['windows_per_second']:.1f}",
        f"step/s={metrics['steps_per_second']:.2f}",
    ]
    if "source_lines_seen" in metrics:
        fields.append(f"total/row={int(metrics['source_lines_seen'])}")
    if "teacher_total_seconds" in metrics:
        fields.extend(
            [
                f"teacher_s={metrics['teacher_total_seconds']:.4f}",
                f"nllb_fwd_s={metrics['teacher_forward_seconds']:.4f}",
                f"nllb_tok_s={metrics['teacher_tokenize_seconds']:.4f}",
                f"teacher_cache={metrics['teacher_cache_hit_rate']:.2f}",
                f"nllb_pad={metrics['teacher_padding_ratio']:.2f}",
                f"nllb_tok/s={metrics['teacher_nllb_tokens_per_second']:.1f}",
            ]
        )
    for key in sorted(k for k in metrics if k.startswith("eval_")):
        fields.append(f"{key}={metrics[key]:.4f}")
    return " ".join(fields)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--tokenizer-vocab", type=Path, default=default_vocab_path())
    parser.add_argument("--compile-mode", choices=COMPILE_MODE_CHOICES, default=None)
    parser.add_argument(
        "--data-mode",
        choices=("streaming", "resident"),
        default=DIL_TRAIN_DEFAULTS["data_mode"],
    )
    parser.add_argument("--max-steps", type=int, default=DIL_TRAIN_DEFAULTS["max_steps"])
    parser.add_argument("--batch-size", type=int, default=DIL_TRAIN_DEFAULTS["batch_size"])
    parser.add_argument("--eval-batch-size", type=int, default=DIL_TRAIN_DEFAULTS["eval_batch_size"])
    parser.add_argument("--nllb-batch-size", type=int, default=DIL_TRAIN_DEFAULTS["nllb_batch_size"])
    parser.add_argument("--teacher-cache-dir", type=Path, default=None)
    parser.add_argument("--no-teacher-cache", action="store_true")
    parser.add_argument("--max-batch-reuse", type=int, default=DIL_TRAIN_DEFAULTS["max_batch_reuse"])
    parser.add_argument("--text-read-chars", type=int, default=DIL_TRAIN_DEFAULTS["text_read_chars"])
    parser.add_argument("--prefetch-factor", type=int, default=DIL_TRAIN_DEFAULTS["prefetch_factor"])
    parser.add_argument("--no-cuda-prefetch", action="store_true")
    parser.add_argument("--sync-timing", action="store_true")
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
    parser.add_argument("--hidden-size", type=int, default=DIL_MODEL_DEFAULTS["hidden_size"])
    parser.add_argument("--intermediate-size", type=int, default=DIL_MODEL_DEFAULTS["intermediate_size"])
    parser.add_argument("--latent-size", type=int, default=DIL_MODEL_DEFAULTS["latent_size"])
    parser.add_argument("--semantic-latent-size", type=int, default=DIL_MODEL_DEFAULTS["semantic_latent_size"])
    parser.add_argument("--surface-latent-size", type=int, default=DIL_MODEL_DEFAULTS["surface_latent_size"])
    parser.add_argument("--max-surface-pieces-per-unit", type=int, default=DIL_MODEL_DEFAULTS["max_surface_pieces_per_unit"])
    parser.add_argument("--byte-conv-layers", type=int, default=DIL_MODEL_DEFAULTS["byte_conv_layers"])
    parser.add_argument("--byte-conv-kernel-size", type=int, default=DIL_MODEL_DEFAULTS["byte_conv_kernel_size"])
    parser.add_argument("--byte-conv-expansion", type=int, default=DIL_MODEL_DEFAULTS["byte_conv_expansion"])
    parser.add_argument("--dil-dropout", type=float, default=DIL_MODEL_DEFAULTS["dil_dropout"])
    parser.add_argument("--distillation-weight", type=float, default=DIL_MODEL_DEFAULTS["distillation_weight"])
    parser.add_argument("--mean-geometry-weight", type=float, default=DIL_MODEL_DEFAULTS["mean_geometry_weight"])
    parser.add_argument("--variance-weight", type=float, default=DIL_MODEL_DEFAULTS["variance_weight"])
    parser.add_argument("--writer-num-layers", type=int, default=DIL_MODEL_DEFAULTS["writer_num_layers"])
    parser.add_argument("--writer-conv-kernel-size", type=int, default=DIL_MODEL_DEFAULTS["writer_conv_kernel_size"])
    parser.add_argument("--writer-conv-expansion", type=int, default=DIL_MODEL_DEFAULTS["writer_conv_expansion"])
    parser.add_argument("--writer-dropout", type=float, default=DIL_MODEL_DEFAULTS["writer_dropout"])
    parser.add_argument("--num-encoder-layers", type=int, default=DIL_MODEL_DEFAULTS.get("num_encoder_layers", 6))
    parser.add_argument("--nllb-model-name", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--nllb-src-lang", default="tur_Latn")
    return parser.parse_args(argv)


def validate_args(args):
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("--batch-size and --eval-batch-size must be > 0")
    if args.nllb_batch_size <= 0:
        raise ValueError("--nllb-batch-size must be > 0")
    if args.no_teacher_cache and args.teacher_cache_dir is not None:
        raise ValueError("--no-teacher-cache cannot be used with --teacher-cache-dir")
    if args.max_batch_reuse <= 0:
        raise ValueError("--max-batch-reuse must be > 0")
    if args.text_read_chars <= 0:
        raise ValueError("--text-read-chars must be > 0")
    if args.byte_conv_layers < 0:
        raise ValueError("--byte-conv-layers must be >= 0")
    if args.semantic_latent_size <= 0 or args.surface_latent_size <= 0:
        raise ValueError("--semantic-latent-size and --surface-latent-size must be > 0")
    if args.latent_size != args.semantic_latent_size + args.surface_latent_size:
        raise ValueError("--latent-size must equal semantic + surface latent sizes")
    if args.byte_conv_kernel_size <= 0 or args.byte_conv_kernel_size % 2 == 0:
        raise ValueError("--byte-conv-kernel-size must be a positive odd integer")
    if args.byte_conv_expansion <= 0:
        raise ValueError("--byte-conv-expansion must be > 0")
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
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be > 0")
    if args.eval_every < 0 or args.checkpoint_every < 0:
        raise ValueError("--eval-every and --checkpoint-every must be >= 0")
    if args.eval_every > 0 and args.eval_file is None:
        raise ValueError("--eval-file is required when --eval-every > 0")
    if args.max_eval_batches <= 0:
        raise ValueError("--max-eval-batches must be > 0")
    if args.data_mode == "resident" and args.max_samples > 0:
        raise ValueError("--max-samples is not supported with --data-mode resident")

def build_config(args, tokenizer):
    if args.resume is not None:
        return DilConfig.from_pretrained(args.resume.parent)
    return DilConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_encoder_layers=args.num_encoder_layers,
        latent_size=args.latent_size,
        semantic_latent_size=args.semantic_latent_size,
        surface_latent_size=args.surface_latent_size,
        max_surface_pieces_per_unit=args.max_surface_pieces_per_unit,
        byte_conv_layers=args.byte_conv_layers,
        byte_conv_kernel_size=args.byte_conv_kernel_size,
        byte_conv_expansion=args.byte_conv_expansion,
        dil_dropout=args.dil_dropout,
        distillation_weight=args.distillation_weight,
        mean_geometry_weight=args.mean_geometry_weight,
        variance_weight=args.variance_weight,
        writer_num_layers=args.writer_num_layers,
        writer_conv_kernel_size=args.writer_conv_kernel_size,
        writer_conv_expansion=args.writer_conv_expansion,
        writer_dropout=args.writer_dropout,
        tokenizer_vocab_file=args.tokenizer_vocab.name,
        nllb_model_name=args.nllb_model_name,
        nllb_src_lang=args.nllb_src_lang,
    )


class DilBaseTrainer(BaseTrainer):
    def __init__(self, args):
        validate_args(args)
        super().__init__(args)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.tokenizer_vocab_path = self.resolve_tokenizer_vocab_path(args)
        self.tokenizer = load_hybrid_tokenizer(self.tokenizer_vocab_path)
        self.config = build_config(args, self.tokenizer)
        self.validate_data_contracts()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
        self.compile_mode = effective_compile_mode(args.compile_mode, self.device)
        validate_compile_environment(self.compile_mode)
        self.autocast_enabled = bool(args.bf16 and self.device.type == "cuda")
        self.teacher_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.cuda_prefetch = bool(self.device.type == "cuda" and not args.no_cuda_prefetch)
        self.model = Dil(self.config).to(self.device)
        self.model.train()
        prepare_writer_for_surface_training(self.model)
        self.optimizer = AdamW(
            self.optimizer_param_groups(args.weight_decay),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
        )
        self.scheduler = make_scheduler(self.optimizer, args.learning_rate, args.warmup_steps, args.max_steps)
        self.teacher = self.build_teacher()
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
        )
        self.train_iterator = None
        self.eval_loader = None
        self.batch_source = None
        self.prepare_data_sources()

    def resolve_tokenizer_vocab_path(self, args) -> Path:
        if args.resume is None:
            return args.tokenizer_vocab
        resume_config = DilConfig.from_pretrained(args.resume.parent)
        return args.resume.parent / resume_config.tokenizer_vocab_file

    def validate_data_contracts(self) -> None:
        if self.args.train_file.suffix.casefold() == ".parquet":
            raise ValueError("fixed surface parquet caches are not part of the packed surface training contract")
        if self.args.eval_file is not None and self.args.eval_file.suffix.casefold() == ".parquet":
            raise ValueError("fixed surface parquet caches are not part of the packed surface training contract")

    def build_teacher(self):
        cache_dir = None if self.args.no_teacher_cache else self.args.teacher_cache_dir
        if cache_dir is None and not self.args.no_teacher_cache:
            cache_dir = self.args.output_dir / "nllb_teacher_cache"
        return NllbTeacher(
            self.config.nllb_model_name,
            self.config.nllb_src_lang,
            self.device,
            self.teacher_dtype,
            batch_size=self.args.nllb_batch_size,
            cache_dir=cache_dir,
        )

    def make_train_dataset(self):
        return ContextDilBatchDataset(
            self.args.train_file,
            self.config,
            self.tokenizer,
            batch_size=self.args.batch_size,
            read_chars=self.args.text_read_chars,
            repeat=True,
            max_samples=self.args.max_samples,
            teacher_tokenizer=self.teacher.tokenizer,
            teacher_max_tokens=self.teacher.max_encoder_tokens,
        )

    def make_eval_dataset(self):
        if self.args.eval_every <= 0:
            return None
        return ContextDilBatchDataset(
            self.args.eval_file,
            self.config,
            self.tokenizer,
            batch_size=self.args.eval_batch_size,
            read_chars=self.args.text_read_chars,
            repeat=False,
            teacher_tokenizer=self.teacher.tokenizer,
            teacher_max_tokens=self.teacher.max_encoder_tokens,
        )

    def prepare_data_sources(self) -> None:
        train_dataset = self.make_train_dataset()
        eval_dataset = self.make_eval_dataset()
        if self.args.data_mode == "resident":
            self.prepare_resident_sources(train_dataset, eval_dataset)
        else:
            self.prepare_streaming_sources(train_dataset, eval_dataset)

    def prepare_resident_sources(self, train_dataset, eval_dataset) -> None:
        print("resident_data_prepare_start=1", flush=True)
        self.train_iterator = ResidentDilBatcher.from_dataset(
            train_dataset,
            self.teacher,
            self.args.batch_size,
            self.device,
            self.args.seed + self.start_step,
        )
        print(f"resident_data_prepare_done=1 batches={len(self.train_iterator.batches)}", flush=True)
        if eval_dataset is None:
            return
        print("resident_eval_prepare_start=1", flush=True)
        eval_batcher = ResidentDilBatcher.from_dataset(
            eval_dataset,
            self.teacher,
            self.args.eval_batch_size,
            self.device,
            self.args.seed + 1,
        )
        self.eval_loader = ResidentDilEvalLoader(eval_batcher)
        print(f"resident_eval_prepare_done=1 batches={len(self.eval_loader.batches)}", flush=True)

    def prepare_streaming_sources(self, train_dataset, eval_dataset) -> None:
        train_loader = make_dil_batch_loader(
            train_dataset,
            num_workers=self.args.num_workers,
            pin_memory=self.device.type == "cuda",
            prefetch_factor=self.args.prefetch_factor,
        )
        train_prefetcher = DeviceBatchPrefetcher(train_loader, self.device, self.cuda_prefetch)
        self.batch_source = AsyncTeacherBatchSource(
            train_prefetcher,
            self.teacher,
            self.device,
            self.args.max_batch_reuse,
        )
        self.train_iterator = AsyncTeacherIterator(self.batch_source)
        if eval_dataset is not None:
            self.eval_loader = make_dil_batch_loader(
                eval_dataset,
                num_workers=self.args.num_workers,
                pin_memory=self.device.type == "cuda",
                prefetch_factor=self.args.prefetch_factor,
            )

    def build_train_iterator(self):
        return self.train_iterator

    def build_eval_iterator(self):
        if self.eval_loader is None:
            return None
        if self.args.data_mode == "resident":
            return iter(self.eval_loader)
        return DeviceBatchPrefetcher(self.eval_loader, self.device, self.cuda_prefetch)

    def has_eval(self) -> bool:
        return self.eval_loader is not None

    def empty_metric_sums(self) -> dict:
        return empty_metric_sums()

    def accumulate_metrics(self, total: dict, result: StepResult) -> None:
        accumulate_output_metrics(total, result.outputs, result.batch)

    def reduce_metrics(self, total: dict) -> dict[str, float]:
        return reduce_metric_sums(total)

    def materialize_eval_teacher(self, batch: dict) -> None:
        if "teacher_layers" in batch:
            return
        if self.teacher is None:
            raise ValueError("eval batch has no teacher_layers and no NLLB teacher is available")
        teacher_layers, teacher_mask = self.teacher.teacher_layers(batch)
        batch["teacher_layers"] = teacher_layers
        batch["teacher_mask"] = teacher_mask
        batch["_teacher_stats"] = dict(getattr(self.teacher, "last_stats", {}))
        batch["_teacher_reuse_count"] = 1

    def forward_batch(self, batch: dict, training_step: int | None):
        model_batch = model_inputs(batch)
        if training_step is not None:
            model_batch["training_step"] = training_step
        return self.model(**model_batch)

    def train_step(self, batch: dict, step: int) -> StepResult:
        outputs = self.forward_batch(batch, step)
        token_count = int(batch["surface"].mask.sum().detach().cpu())
        window_count = int(batch["surface"].unit_mask.sum().detach().cpu())
        return StepResult(
            loss=outputs.loss,
            outputs=outputs,
            token_count=token_count,
            window_count=window_count,
            batch=batch,
        )

    def eval_step(self, batch: dict) -> StepResult:
        self.materialize_eval_teacher(batch)
        outputs = self.forward_batch(batch, None)
        token_count = int(batch["surface"].mask.sum().detach().cpu())
        window_count = int(batch["surface"].unit_mask.sum().detach().cpu())
        return StepResult(
            loss=outputs.loss,
            outputs=outputs,
            token_count=token_count,
            window_count=window_count,
            batch=batch,
        )

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

    def close(self) -> None:
        if self.batch_source is not None:
            self.batch_source.close()

    def format_log(self, step: int, metrics: dict[str, float]) -> str:
        return format_log(step, metrics)

    def run(self) -> None:
        print(
            f"device={self.device.type} bf16={int(self.autocast_enabled)} compile_mode={self.compile_mode} "
            f"data_mode={self.args.data_mode} resume_step={self.start_step} teacher_source=online_nllb "
            f"teacher_dtype={str(self.teacher_dtype).replace('torch.', '')} nllb_batch={self.args.nllb_batch_size} "
            f"vocab_size={self.config.vocab_size} latent_size={self.config.latent_size} "
            f"semantic_latent_size={self.config.semantic_latent_size} surface_latent_size={self.config.surface_latent_size} "
            f"hidden_size={self.config.hidden_size}",
            flush=True,
        )
        super().run()


class DilPretrainTrainer(DilBaseTrainer):
    pass


def make_trainer(args) -> DilBaseTrainer:
    return DilPretrainTrainer(args)


def main(argv: list[str] | None = None):
    trainer = make_trainer(parse_args(argv))
    trainer.run()


if __name__ == "__main__":
    main()
