import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from byte_trainer_utils import (  # noqa: E402
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
from dil_data import HybridDilBatchDataset, ResidentDilBatcher, ResidentDilEvalLoader, load_hybrid_tokenizer, make_dil_batch_loader  # noqa: E402
from dilnaz_config import DIL_TRAIN_DEFAULTS  # noqa: E402
from models.configuration_dil import DilConfig  # noqa: E402
from models.modeling_dil import Dil  # noqa: E402


CHECKPOINT_FORMAT_VERSION = 17
WRITER_OBJECTIVE = "plain_text_writer_only"


def make_scheduler(optimizer, learning_rate: float, warmup_steps: int):
    def lr_lambda(step):
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(step + 1) / float(warmup_steps))

    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def freeze_for_writer_only(model: Dil):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.writer.parameters():
        param.requires_grad = True
    model.encoder.eval()
    model.writer.train()


def writer_only_forward(model: Dil, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = batch["labels"].to(batch["input_ids"].device)
    with torch.no_grad():
        semantic = model.encode(batch["input_ids"], batch["word_masks"]).float()
    logits = model.writer_logits(semantic.detach(), labels).float()
    loss = F.cross_entropy(
        logits.reshape(-1, model.config.vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
    )
    byte_acc, token_exact = model.writer_metrics(logits, labels)
    return loss, byte_acc, token_exact


def materialize_writer_batches(dataset: HybridDilBatchDataset, device: torch.device, batch_size: int, seed: int):
    batches = [
        {
            key: value.detach().cpu()
            for key, value in batch.items()
            if key in ("input_ids", "word_masks", "labels", "source_line_ids")
        }
        for batch in dataset.iter_once(worker_id=0, worker_count=1)
    ]
    if dataset._carry_refs:
        batch = dataset.make_batch(
            dataset._carry_texts,
            dataset._carry_line_ids,
            dataset._carry_segments_by_text,
            dataset._carry_refs,
        )
        batches.append(
            {
                key: value.detach().cpu()
                for key, value in batch.items()
                if key in ("input_ids", "word_masks", "labels", "source_line_ids")
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
        "max_word_bytes": config.max_word_bytes,
        "context_radius": config.context_radius,
        "target_index": config.target_index,
        "latent_size": config.latent_size,
    }
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": model.state_dict(),
            "writer_optimizer_state_dict": optimizer.state_dict(),
            "writer_scheduler_state_dict": scheduler.state_dict(),
            "training_state": training_state,
            "rng_state": rng_state(),
        },
        checkpoint_dir / "checkpoint.pt",
    )
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
def evaluate(model, eval_loader, device, compile_mode: str, autocast_enabled: bool, cuda_prefetch: bool, max_batches: int):
    model.eval()
    model.encoder.eval()
    total = {"loss": 0.0, "byte_acc": 0.0, "token_exact": 0.0, "batches": 0}
    for batch_idx, batch in enumerate(DeviceBatchPrefetcher(eval_loader, device, cuda_prefetch), start=1):
        cudagraph_step_begin(device, compile_mode)
        with autocast_context(autocast_enabled):
            loss, byte_acc, token_exact = writer_only_forward(model, batch)
        total["loss"] += float(loss.detach().cpu())
        total["byte_acc"] += float(byte_acc.detach().cpu())
        total["token_exact"] += float(token_exact.detach().cpu())
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
        f"byte_acc={metrics['byte_acc']:.4f}",
        f"token_exact={metrics['token_exact']:.4f}",
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
    freeze_for_writer_only(model)
    model.set_compiled_forwards(
        encoder_forward=compile_forward(model.encoder.forward, compile_mode, "DilEncoderCore"),
        writer_forward=compile_forward(model.writer.forward, compile_mode, "DilConditionalWriter"),
    )
    optimizer = AdamW(
        model.writer.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(optimizer, args.learning_rate, args.warmup_steps)
    if checkpoint.get("training_state", {}).get("objective") == WRITER_OBJECTIVE:
        optimizer.load_state_dict(checkpoint["writer_optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["writer_scheduler_state_dict"])
        restore_rng_state(checkpoint["rng_state"])

    train_dataset = HybridDilBatchDataset(
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
        eval_dataset = HybridDilBatchDataset(
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
        f"vocab_size={config.vocab_size} latent_size={config.latent_size} hidden_size={config.hidden_size}",
        flush=True,
    )

    log_start = time.perf_counter()
    log_tokens = 0
    log_windows = 0
    log_steps = 0
    data_seconds = 0.0
    compute_seconds = 0.0
    source_lines_seen: set[int] = set()
    metric_sums = {"loss": 0.0, "byte_acc": 0.0, "token_exact": 0.0}
    last_metrics = {}
    completed_step = 0

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
        for step in range(1, args.max_steps + 1):
            data_start = time.perf_counter()
            batch = next(train_iter)
            data_seconds += time.perf_counter() - data_start

            compute_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            cudagraph_step_begin(device, compile_mode)
            with autocast_context(autocast_enabled):
                loss, byte_acc, token_exact = writer_only_forward(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.writer.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            cuda_sync(device)
            compute_seconds += time.perf_counter() - compute_start
            completed_step = step

            real_tokens = int(batch["labels"].ne(-100).sum().detach().cpu())
            log_tokens += real_tokens
            log_windows += int(batch["labels"].shape[0])
            log_steps += 1
            if "source_line_ids" in batch:
                source_lines_seen.update(int(line_id) for line_id in batch["source_line_ids"].detach().cpu().tolist())
            metric_sums["loss"] += float(loss.detach().cpu())
            metric_sums["byte_acc"] += float(byte_acc.detach().cpu())
            metric_sums["token_exact"] += float(token_exact.detach().cpu())

            should_log = step % args.log_every == 0 or step == 1 or step == args.max_steps
            should_eval = eval_loader is not None and args.eval_every > 0 and step % args.eval_every == 0
            if should_log or should_eval:
                elapsed = max(time.perf_counter() - log_start, 1e-9)
                averaged = {key: value / max(log_steps, 1) for key, value in metric_sums.items()}
                averaged["lr"] = scheduler.get_last_lr()[0]
                averaged["data_seconds"] = data_seconds / max(log_steps, 1)
                averaged["compute_seconds"] = compute_seconds / max(log_steps, 1)
                averaged["tokens_per_second"] = log_tokens / elapsed
                averaged["windows_per_second"] = log_windows / elapsed
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
                log_windows = 0
                log_steps = 0
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
