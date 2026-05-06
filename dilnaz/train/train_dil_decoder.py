import argparse
import json
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from byte_trainer_utils import (
    COMPILE_MODE_CHOICES,
    DeviceBatchPrefetcher,
    autocast_context,
    compile_forward,
    cuda_sync,
    effective_compile_mode,
    rng_state,
    validate_compile_environment,
)
from dil_data import (
    HybridDilBatchDataset,
    ResidentDilBatcher,
    ResidentDilEvalLoader,
    load_hybrid_tokenizer,
    make_dil_batch_loader,
)
from models.configuration_dil import DilConfig
from models.modeling_dil import Dil


CHECKPOINT_FORMAT_VERSION = 12
DATALOADER_WORKER_EXIT = "DataLoader worker"


@dataclass
class WriterOnlyOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    ce_loss: torch.Tensor
    byte_acc: torch.Tensor


def make_scheduler(optimizer, learning_rate: float, warmup_steps: int):
    def lr_lambda(step):
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(step + 1) / float(warmup_steps))

    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def configure_writer_only_training(model: Dil):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.decoder.parameters():
        param.requires_grad = True

    model.encoder.eval()
    model.decoder.train()


def writer_trainable_parameters(model: Dil):
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise ValueError("writer-only training has no trainable parameters")
    return params


def writer_train_mode(model: Dil):
    model.encoder.eval()
    model.decoder.train()


def writer_eval_mode(model: Dil):
    model.encoder.eval()
    model.decoder.eval()


def writer_only_forward(model: Dil, batch: dict) -> WriterOnlyOutput:
    with torch.no_grad():
        latent_states = model.encode(
            input_ids=batch["input_ids"],
            word_masks=batch["word_masks"],
            output_hidden_states=False,
        )
        mean, _ = torch.chunk(latent_states, 2, dim=-1)
        latent_states = mean.detach()

    decoder_input_ids = model.decoder_inputs_from_labels(batch["labels"].to(latent_states.device))
    logits = model.decode_from_latents(latent_states, decoder_input_ids)
    logits = logits.float()
    labels = batch["labels"].to(logits.device)
    ce_loss = F.cross_entropy(
        logits.reshape(-1, model.config.vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
    )
    loss = ce_loss

    valid = labels.ne(-100)
    correct = logits.argmax(dim=-1).eq(labels) & valid
    byte_acc = correct.sum().float() / valid.sum().clamp_min(1).float()
    return WriterOnlyOutput(
        loss=loss,
        logits=logits,
        ce_loss=ce_loss,
        byte_acc=byte_acc,
    )


def json_training_state(config, step: int, metrics: dict, compile_mode: str):
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "writer_only": True,
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
        "semantic_normalizer_z_clip": config.semantic_normalizer_z_clip,
        "trainable": "autoregressive_writer",
    }


def save_checkpoint(
    output_dir: Path,
    model: Dil,
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
    state = json_training_state(config, step, metrics, compile_mode)
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "training_state": state,
        "rng_state": rng_state(),
    }
    tmp_path = checkpoint_dir / "checkpoint.pt.tmp"
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(checkpoint_dir / "checkpoint.pt")
    with (checkpoint_dir / "training_state.json").open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    return checkpoint_dir


def load_model(checkpoint_dir: Path, device: torch.device):
    config = DilConfig.from_pretrained(checkpoint_dir)
    model = Dil(config).to(device)
    checkpoint = torch.load(
        checkpoint_dir / "checkpoint.pt",
        map_location=device,
        weights_only=False,
    )
    if checkpoint["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format_version={checkpoint.get('format_version')}")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config


def is_dataloader_worker_exit(error: RuntimeError) -> bool:
    message = str(error)
    return DATALOADER_WORKER_EXIT in message and "exited unexpectedly" in message


def materialize_writer_batches(dataset: HybridDilBatchDataset, device: torch.device, batch_size: int, seed: int):
    batches = [
        {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in batch.items()
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
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
        )
    return ResidentDilBatcher(batches, batch_size=batch_size, device=device, seed=seed)


@torch.no_grad()
def evaluate(model, eval_loader, autocast_enabled: bool, cuda_prefetch: bool, device: torch.device, max_batches: int):
    writer_eval_mode(model)
    total = {
        "loss": 0.0,
        "ce": 0.0,
        "byte_acc": 0.0,
        "batches": 0,
    }
    for batch_idx, batch in enumerate(DeviceBatchPrefetcher(eval_loader, device, cuda_prefetch), start=1):
        with autocast_context(autocast_enabled):
            outputs = writer_only_forward(model, batch)
        total["loss"] += float(outputs.loss.detach().cpu())
        total["ce"] += float(outputs.ce_loss.detach().cpu())
        total["byte_acc"] += float(outputs.byte_acc.detach().cpu())
        total["batches"] += 1
        if batch_idx >= max_batches:
            break

    writer_train_mode(model)
    batches = max(total.pop("batches"), 1)
    return {f"eval_{key}": value / batches for key, value in total.items()}


def format_log(step: int, metrics: dict) -> str:
    fields = [
        f"step={step}",
        f"loss={metrics['loss']:.4f}",
        f"ce={metrics['ce']:.4f}",
        f"byte_acc={metrics['byte_acc']:.4f}",
        f"lr={metrics['lr']:.2e}",
        f"data_s={metrics['data_seconds']:.4f}",
        f"compute_s={metrics['compute_seconds']:.4f}",
        f"t/s={metrics['tokens_per_second']:.1f}",
        f"w/s={metrics['windows_per_second']:.1f}",
        f"step/s={metrics['steps_per_second']:.2f}",
    ]
    for key in sorted(k for k in metrics if k.startswith("eval_")):
        fields.append(f"{key}={metrics[key]:.4f}")
    return " ".join(fields)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, default=None)
    parser.add_argument("--data-mode", choices=("streaming", "resident"), default="streaming")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--text-read-chars", type=int, default=1_000_000)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-cuda-prefetch", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile-mode", choices=COMPILE_MODE_CHOICES, default=None)
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("--batch-size and --eval-batch-size must be > 0")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be > 0")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be >= 0")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0")
    if args.max_grad_norm <= 0.0:
        raise ValueError("--max-grad-norm must be > 0")
    if args.log_every <= 0:
        raise ValueError("--log-every must be > 0")
    if args.eval_every < 0 or args.checkpoint_every < 0:
        raise ValueError("--eval-every and --checkpoint-every must be >= 0")
    if args.eval_every > 0 and args.eval_file is None:
        raise ValueError("--eval-file is required when --eval-every > 0")
    if args.max_eval_batches <= 0:
        raise ValueError("--max-eval-batches must be > 0")
    if args.text_read_chars <= 0:
        raise ValueError("--text-read-chars must be > 0")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be > 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")


def make_dataset(path: Path, config: DilConfig, tokenizer, batch_size: int, read_chars: int, repeat: bool):
    return HybridDilBatchDataset(
        path,
        config,
        tokenizer,
        batch_size=batch_size,
        read_chars=read_chars,
        repeat=repeat,
        max_samples=0,
        teacher_tokenizer=None,
        teacher_max_tokens=0,
    )


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

    model, config = load_model(args.checkpoint_dir, device)
    tokenizer_vocab_path = args.checkpoint_dir / config.tokenizer_vocab_file
    tokenizer = load_hybrid_tokenizer(tokenizer_vocab_path)
    configure_writer_only_training(model)
    model.set_compiled_forwards(
        encoder_forward=compile_forward(model.encoder.forward, compile_mode, "DilEncoderCore"),
        decode_forward=compile_forward(model._decode_from_latents_impl, compile_mode, "DilDecoderRenderer"),
    )
    optimizer = AdamW(
        writer_trainable_parameters(model),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(optimizer, args.learning_rate, args.warmup_steps)

    train_dataset = make_dataset(
        args.train_file,
        config,
        tokenizer,
        batch_size=args.batch_size,
        read_chars=args.text_read_chars,
        repeat=True,
    )
    eval_dataset = None
    if args.eval_file is not None:
        eval_dataset = make_dataset(
            args.eval_file,
            config,
            tokenizer,
            batch_size=args.eval_batch_size,
            read_chars=args.text_read_chars,
            repeat=False,
        )

    if args.data_mode == "resident":
        print("resident_data_prepare_start=1", flush=True)
        train_iter = materialize_writer_batches(
            train_dataset,
            device,
            args.batch_size,
            args.seed,
        )
        print(f"resident_data_prepare_done=1 batches={len(train_iter.batches)}", flush=True)
        eval_loader = None
        if eval_dataset is not None:
            print("resident_eval_prepare_start=1", flush=True)
            eval_loader = ResidentDilEvalLoader(
                materialize_writer_batches(
                    eval_dataset,
                    device,
                    args.eval_batch_size,
                    args.seed + 1,
                )
            )
            print(f"resident_eval_prepare_done=1 batches={len(eval_loader.batches)}", flush=True)
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
        f"data_mode={args.data_mode} writer_only=1 vocab_size={config.vocab_size} "
        f"latent_size={config.latent_size} hidden_size={config.hidden_size}",
        flush=True,
    )

    log_start = time.perf_counter()
    log_tokens = 0
    log_windows = 0
    log_steps = 0
    data_seconds = 0.0
    compute_seconds = 0.0
    metric_sums = {
        "loss": 0.0,
        "ce": 0.0,
        "byte_acc": 0.0,
    }
    last_metrics = {}
    completed_step = 0

    def save_interrupted():
        interrupted_dir = save_checkpoint(
            args.output_dir,
            model,
            config,
            tokenizer_vocab_path,
            completed_step,
            last_metrics,
            compile_mode,
        )
        print(f"interrupted_saved={interrupted_dir}", flush=True)

    try:
        writer_train_mode(model)
        for step in range(1, args.max_steps + 1):
            data_start = time.perf_counter()
            batch = next(train_iter)
            if args.data_mode == "resident":
                data_seconds += time.perf_counter() - data_start
            else:
                data_seconds += train_iter.last_data_seconds + train_iter.last_transfer_seconds

            cuda_sync(device)
            compute_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(autocast_enabled):
                outputs = writer_only_forward(model, batch)

            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(writer_trainable_parameters(model), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            cuda_sync(device)
            compute_seconds += time.perf_counter() - compute_start
            completed_step = step

            real_tokens = int(batch["labels"].ne(-100).sum().detach().cpu())
            log_tokens += real_tokens
            log_windows += int(batch["labels"].shape[0])
            log_steps += 1
            metric_sums["loss"] += float(outputs.loss.detach().cpu())
            metric_sums["ce"] += float(outputs.ce_loss.detach().cpu())
            metric_sums["byte_acc"] += float(outputs.byte_acc.detach().cpu())

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
                if should_eval:
                    averaged.update(
                        evaluate(
                            model,
                            eval_loader,
                            autocast_enabled,
                            cuda_prefetch,
                            device,
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
                save_checkpoint(
                    args.output_dir,
                    model,
                    config,
                    tokenizer_vocab_path,
                    step,
                    last_metrics,
                    compile_mode,
                    checkpoint_name=f"checkpoint-{step}",
                )
    except KeyboardInterrupt:
        save_interrupted()
        return
    except RuntimeError as error:
        if not is_dataloader_worker_exit(error):
            raise
        save_interrupted()
        return

    final_dir = save_checkpoint(
        args.output_dir,
        model,
        config,
        tokenizer_vocab_path,
        args.max_steps,
        last_metrics,
        compile_mode,
    )
    print(f"saved={final_dir}", flush=True)


if __name__ == "__main__":
    main()
