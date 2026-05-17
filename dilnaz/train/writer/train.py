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
    context_windows,
    load_hybrid_tokenizer,
    make_dil_batch_loader,
    segment_piece_ids,
    stream_text_items,
    trainable_segments,
)
from dilnaz.surface import pack_writer_targets
from dilnaz.surface.types import PackedWriterTarget
from dilnaz.train.configs.defaults import DIL_TRAIN_DEFAULTS
from dilnaz.models.dil import DilConfig
from dilnaz.models.dil import Dil
from dilnaz.train.common.trainer_core import make_adamw_param_groups, make_scheduler


CHECKPOINT_FORMAT_VERSION = 32
WRITER_OBJECTIVE = "factorized_writer_encoder_prior_v1"
WRITER_METRIC_KEYS = (
    "loss",
    "token_loss",
    "byte_acc",
    "token_exact",
    "stop_acc",
    "tokens_processed",
)


class WriterContextDataset(IterableDataset):
    def __init__(
        self,
        train_file: Path,
        config: DilConfig,
        tokenizer,
        batch_size: int,
        read_chars: int,
        repeat: bool = True,
        max_samples: int = 0,
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
        self.max_pieces_per_unit = config.max_surface_pieces_per_unit
        self.surface_bucket_sizes = tuple(config.surface_bucket_sizes)
        self.pad_token_id = config.pad_token_id
        self.stop_token_id = config.writer_stop_token_id
        self.produced = 0

    def make_batch(
        self,
        context_rows: list[list[list[int]]],
        target_rows: list[list[list[int]]],
        line_ids: list[int],
    ) -> dict:
        from dilnaz.surface import pack_token_units

        surface = pack_token_units(
            context_rows,
            pad_token_id=self.pad_token_id,
            bucket_sizes=self.surface_bucket_sizes,
            max_pieces_per_unit=self.max_pieces_per_unit,
        )
        target = pack_writer_targets(
            target_rows,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.config.decoder_start_token_id,
            stop_token_id=self.stop_token_id,
            surface_bucket_sizes=self.surface_bucket_sizes,
            max_pieces_per_unit=self.max_pieces_per_unit,
        )
        return {
            "surface": surface,
            "writer_target": target,
            "source_line_ids": torch.tensor(line_ids, dtype=torch.long),
        }

    def iter_once(self, worker_id: int, worker_count: int):
        context_rows: list[list[list[int]]] = []
        target_rows: list[list[list[int]]] = []
        line_ids: list[int] = []

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
            for window in context_windows(segments, self.context_radius):
                target_segment = window[self.context_radius]
                context_rows.append([segment_piece_ids(segment) for segment in window])
                target_rows.append([segment_piece_ids(target_segment)])
                line_ids.append(source_line_id)
                self.produced += 1
                if len(context_rows) >= self.batch_size:
                    yield self.make_batch(context_rows, target_rows, line_ids)
                    context_rows = []
                    target_rows = []
                    line_ids = []
                if self.max_samples > 0 and self.produced >= self.max_samples:
                    if context_rows:
                        yield self.make_batch(context_rows, target_rows, line_ids)
                    return

        if context_rows:
            yield self.make_batch(context_rows, target_rows, line_ids)

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
            if not yielded:
                raise ValueError(f"{self.train_file} produced no writer samples")
            if not self.repeat:
                return


def freeze_for_writer_only(model: Dil):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.writer.parameters():
        param.requires_grad = True
    model.encoder.eval()
    model.writer.train()


def writer_only_metrics(
    model: Dil,
    batch: dict,
    compute_free_metrics: bool = False,
) -> dict[str, torch.Tensor]:
    surface = batch["surface"]
    target = batch["writer_target"]
    with torch.no_grad():
        full_semantic = model.encode(surface).float()
    target = target.to(full_semantic.device)
    metrics = model.writer_loss_and_metrics(
        full_semantic.detach(),
        target,
        return_metrics=True,
    )
    output = {
        "loss": metrics["loss"],
        "token_loss": metrics["token_loss"],
        "byte_acc": metrics["byte_acc"],
        "token_exact": metrics["token_exact"],
        "stop_acc": metrics["stop_acc"],
        "tokens_processed": metrics["loss"].new_tensor(float(target.label_mask.sum().detach())),
    }
    return output


def materialize_writer_batches(dataset, device: torch.device, batch_size: int, seed: int):
    keep_keys = {"surface", "writer_target", "source_line_ids"}
    batches = [
        {
            key: value.detach().cpu() if hasattr(value, "detach") else value
            for key, value in batch.items()
            if key in keep_keys
        }
        for batch in dataset.iter_once(worker_id=0, worker_count=1)
    ]
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
        "writer_stop_token_id": config.writer_stop_token_id,
        "max_surface_pieces_per_unit": config.max_surface_pieces_per_unit,
        "latent_size": config.latent_size,
        "semantic_latent_size": config.semantic_latent_size,
        "surface_latent_size": config.surface_latent_size,
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
    total = {key: 0.0 for key in WRITER_METRIC_KEYS}
    total["batches"] = 0
    for batch_idx, batch in enumerate(DeviceBatchPrefetcher(eval_loader, device, cuda_prefetch), start=1):
        cudagraph_step_begin(device, compile_mode)
        with autocast_context(autocast_enabled):
            metrics = writer_only_metrics(model, batch)
        for key in WRITER_METRIC_KEYS:
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
        f"byte_acc={metrics['byte_acc']:.4f}",
        f"token_exact={metrics['token_exact']:.4f}",
        f"stop_acc={metrics['stop_acc']:.4f}",
        f"lr={metrics['lr']:.2e}",
        f"data_s={metrics['data_seconds']:.4f}",
        f"compute_s={metrics['compute_seconds']:.4f}",
        f"t/s={metrics['tokens_per_second']:.1f}",
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

    train_dataset = WriterContextDataset(
        args.train_file,
        config,
        tokenizer,
        batch_size=args.batch_size,
        read_chars=args.text_read_chars,
        repeat=True,
        max_samples=args.max_samples,
    )
    eval_dataset = None
    if args.eval_every > 0:
        eval_dataset = WriterContextDataset(
            args.eval_file,
            config,
            tokenizer,
            batch_size=args.eval_batch_size,
            read_chars=args.text_read_chars,
            repeat=False,
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
        f"writer_resume={int(writer_resume)}",
        flush=True,
    )

    log_start = time.perf_counter()
    log_tokens = 0
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
                    metrics = writer_only_metrics(model, batch)
                    loss = metrics["loss"]
                (loss / args.gradient_accumulation_steps).backward()
                log_tokens += int(metrics.get("tokens_processed", 0))
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

            should_log = step % args.log_every == 0 or step == 1 or step == args.max_steps
            should_eval = eval_loader is not None and args.eval_every > 0 and step % args.eval_every == 0
            if should_log or should_eval:
                elapsed = max(time.perf_counter() - log_start, 1e-9)
                averaged = {key: value / max(log_micro_steps, 1) for key, value in metric_sums.items()}
                averaged["lr"] = scheduler.get_last_lr()[0]
                averaged["data_seconds"] = data_seconds / max(log_micro_steps, 1)
                averaged["compute_seconds"] = compute_seconds / max(log_micro_steps, 1)
                averaged["tokens_per_second"] = log_tokens / elapsed
                averaged["steps_per_second"] = log_micro_steps / elapsed
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
                log_micro_steps = 0
                data_seconds = 0.0
                compute_seconds = 0.0
                for key in metric_sums:
                    metric_sums[key] = 0.0

            if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                save_current(checkpoint_name=f"checkpoint-{step}")
    except KeyboardInterrupt:
        print(f"interrupted_saved={save_current()}", flush=True)
        return

    print(f"saved={save_current()}", flush=True)


if __name__ == "__main__":
    main()
