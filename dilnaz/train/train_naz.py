import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
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
    load_checkpoint,
    restore_rng_state,
    rng_state,
    validate_compile_environment,
)
from dilnaz_config import NAZ_MODEL_DEFAULTS, NAZ_TRAIN_DEFAULTS
from naz_data import (
    RandomWindowNazDataset,
    ResidentNazBatcher,
    ResidentNazSemanticBatcher,
    ResidentNazSemanticEvalLoader,
    build_token_cache,
    make_naz_loader,
)
from models.configuration_dil import DilConfig
from models.configuration_naz import NazConfig
from models.modeling_naz import Naz
from tokenization import HybridTokenizer


CHECKPOINT_FORMAT_VERSION = 18
DATALOADER_WORKER_EXIT = "DataLoader worker"


def make_scheduler(optimizer, learning_rate: float, warmup_steps: int):
    def lr_lambda(step):
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(step + 1) / float(warmup_steps))

    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def dil_checksum(model: Naz) -> str:
    digest = hashlib.sha256()
    for tensor in model.dil_model.state_dict().values():
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def json_training_state(config, step: int, metrics: dict, dil_checksum: str, compile_mode: str):
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "step": step,
        "metrics": metrics,
        "dil_checksum": dil_checksum,
        "compile_mode": compile_mode,
        "max_word_bytes": config.max_word_bytes,
        "latent_size": config.latent_size,
        "semantic_space": "dil_normalized_latent",
        "feedback": "dil_reencoded_contextual_mean",
        "byte_vocab_size": config.byte_vocab_size,
        "vocab_size": config.vocab_size,
        "pad_token_id": config.pad_token_id,
    }


def save_checkpoint(
    output_dir: Path,
    model,
    optimizer,
    scheduler,
    config,
    step: int,
    metrics: dict,
    dil_checksum: str,
    compile_mode: str,
    checkpoint_name: str = "",
):
    checkpoint_dir = output_dir / checkpoint_name if checkpoint_name else output_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(checkpoint_dir)
    state = json_training_state(config, step, metrics, dil_checksum, compile_mode)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": model.trainable_state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "training_state": state,
            "rng_state": rng_state(),
        },
        checkpoint_dir / "checkpoint.pt",
    )
    with (checkpoint_dir / "training_state.json").open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    return checkpoint_dir


def restore_checkpoint(path: Path, model, optimizer, scheduler, device: torch.device) -> tuple[int, dict]:
    checkpoint = load_checkpoint(path, device)
    if checkpoint["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format_version={checkpoint.get('format_version')}")
    model.load_trainable_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    restore_rng_state(checkpoint["rng_state"])
    training_state = checkpoint["training_state"]
    return int(training_state["step"]), dict(training_state["metrics"])


def is_dataloader_worker_exit(error: RuntimeError) -> bool:
    message = str(error)
    return DATALOADER_WORKER_EXIT in message and "exited unexpectedly" in message


@torch.no_grad()
def evaluate(model, eval_loader, device, autocast_enabled: bool, max_batches: int, cuda_prefetch: bool):
    model.eval()
    total = {
        "loss": 0.0,
        "energy": 0.0,
        "energy_loss": 0.0,
        "mean_loss": 0.0,
        "cosine_loss": 0.0,
        "writer_loss": 0.0,
        "latent_cos": 0.0,
        "candidate_cos": 0.0,
        "byte_acc": 0.0,
        "batches": 0,
    }
    for batch_idx, batch in enumerate(DeviceBatchPrefetcher(eval_loader, device, cuda_prefetch), start=1):
        with autocast_context(autocast_enabled):
            outputs = run_naz_batch(model, batch)
        total["loss"] += float(outputs.loss.detach().cpu())
        total["energy"] += float(outputs.energy.detach().cpu())
        total["energy_loss"] += float(outputs.energy_loss.detach().cpu())
        total["mean_loss"] += float(outputs.mean_loss.detach().cpu())
        total["cosine_loss"] += float(outputs.cosine_loss.detach().cpu())
        total["writer_loss"] += float(outputs.writer_loss.detach().cpu())
        total["latent_cos"] += float(outputs.latent_cos.detach().cpu())
        total["candidate_cos"] += float(outputs.candidate_cos.detach().cpu())
        total["byte_acc"] += float(outputs.byte_acc.detach().cpu())
        total["batches"] += 1
        if batch_idx >= max_batches:
            break

    model.train()
    batches = max(total.pop("batches"), 1)
    return {f"eval_{key}": value / batches for key, value in total.items()}


def format_log(step: int, metrics: dict) -> str:
    fields = [
        f"step={step}",
        f"loss={metrics['loss']:.4f}",
        f"energy={metrics['energy']:.4f}",
        f"energy_loss={metrics['energy_loss']:.4f}",
        f"mean_loss={metrics['mean_loss']:.4f}",
        f"cosine_loss={metrics['cosine_loss']:.4f}",
        f"writer_loss={metrics['writer_loss']:.4f}",
        f"latent_cos={metrics['latent_cos']:.4f}",
        f"candidate_cos={metrics['candidate_cos']:.4f}",
        f"byte_acc={metrics['byte_acc']:.4f}",
        f"lr={metrics['lr']:.2e}",
        f"data_s={metrics['data_seconds']:.4f}",
        f"transfer_s={metrics['transfer_seconds']:.4f}",
        f"compute_s={metrics['compute_seconds']:.4f}",
        f"tokens/s={metrics['tokens_per_second']:.1f}",
        f"step/s={metrics['steps_per_second']:.2f}",
    ]
    for key in sorted(k for k in metrics if k.startswith("eval_")):
        fields.append(f"{key}={metrics[key]:.4f}")
    return " ".join(fields)


@torch.no_grad()
def build_resident_semantic_cache(
    model: Naz,
    surface_batcher: ResidentNazBatcher,
    chunk_tokens: int,
    autocast_enabled: bool,
):
    model.eval()
    byte_ids = surface_batcher.byte_ids
    lengths = surface_batcher.lengths
    token_count = surface_batcher.token_count
    positions = surface_batcher.positions.reshape(1, 1, surface_batcher.max_word_bytes)
    context_radius = model.dil_config.context_radius
    means = []
    log_stds = []
    for start in range(0, token_count, chunk_tokens):
        end = min(start + chunk_tokens, token_count)
        context_start = max(0, start - context_radius)
        context_end = end
        ids = byte_ids[context_start:context_end].unsqueeze(0)
        token_lengths = lengths[context_start:context_end].reshape(1, -1, 1)
        masks = positions < token_lengths
        unit_mask = torch.ones(ids.shape[:2], dtype=torch.bool, device=ids.device)
        with autocast_context(autocast_enabled):
            mean, log_std = model.latent_distribution(ids, masks, unit_mask)
        local_start = start - context_start
        local_end = local_start + end - start
        mean = mean.reshape(1, context_end - context_start, -1)[:, local_start:local_end]
        log_std = log_std.reshape(1, context_end - context_start, -1)[:, local_start:local_end]
        means.append(mean.squeeze(0).float())
        log_stds.append(log_std.squeeze(0).float())
    mean_cache = torch.cat(means, dim=0)
    log_std_cache = torch.cat(log_stds, dim=0)
    semantic_states = mean_cache
    model.train()
    return semantic_states, mean_cache, log_std_cache, surface_batcher.byte_ids, surface_batcher.lengths


def run_naz_batch(model, batch, training_step: int | None = None):
    if "semantic_states" in batch:
        return model.forward_semantic(
            semantic_states=batch["semantic_states"],
            target_mean=batch["target_mean"],
            target_log_std=batch["target_log_std"],
            unit_mask=batch["unit_mask"],
            target_input_ids=batch["target_input_ids"],
            target_word_masks=batch["target_word_masks"],
            training_step=training_step,
        )
    return model(**batch, training_step=training_step)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, default=None)
    parser.add_argument("--dil-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--compile-mode", choices=COMPILE_MODE_CHOICES, default=None)
    parser.add_argument(
        "--data-mode",
        choices=("streaming", "resident"),
        default=NAZ_TRAIN_DEFAULTS["data_mode"],
    )
    parser.add_argument("--max-steps", type=int, default=NAZ_TRAIN_DEFAULTS["max_steps"])
    parser.add_argument("--batch-size", type=int, default=NAZ_TRAIN_DEFAULTS["batch_size"])
    parser.add_argument("--eval-batch-size", type=int, default=NAZ_TRAIN_DEFAULTS["eval_batch_size"])
    parser.add_argument("--sequence-length", type=int, default=NAZ_TRAIN_DEFAULTS["sequence_length"])
    parser.add_argument("--learning-rate", type=float, default=NAZ_TRAIN_DEFAULTS["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=NAZ_TRAIN_DEFAULTS["weight_decay"])
    parser.add_argument("--adam-beta1", type=float, default=NAZ_TRAIN_DEFAULTS["adam_beta1"])
    parser.add_argument("--adam-beta2", type=float, default=NAZ_TRAIN_DEFAULTS["adam_beta2"])
    parser.add_argument("--warmup-steps", type=int, default=NAZ_TRAIN_DEFAULTS["warmup_steps"])
    parser.add_argument("--max-grad-norm", type=float, default=NAZ_TRAIN_DEFAULTS["max_grad_norm"])
    parser.add_argument("--log-every", type=int, default=NAZ_TRAIN_DEFAULTS["log_every"])
    parser.add_argument("--checkpoint-every", type=int, default=NAZ_TRAIN_DEFAULTS["checkpoint_every"])
    parser.add_argument("--eval-every", type=int, default=NAZ_TRAIN_DEFAULTS["eval_every"])
    parser.add_argument("--max-eval-batches", type=int, default=NAZ_TRAIN_DEFAULTS["max_eval_batches"])
    parser.add_argument("--text-read-chars", type=int, default=NAZ_TRAIN_DEFAULTS["text_read_chars"])
    parser.add_argument("--token-cache-dir", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=NAZ_TRAIN_DEFAULTS["num_workers"])
    parser.add_argument("--prefetch-factor", type=int, default=NAZ_TRAIN_DEFAULTS["prefetch_factor"])
    parser.add_argument("--no-cuda-prefetch", action="store_true")
    parser.add_argument("--seed", type=int, default=NAZ_TRAIN_DEFAULTS["seed"])
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=NAZ_MODEL_DEFAULTS["hidden_size"])
    parser.add_argument("--intermediate-size", type=int, default=NAZ_MODEL_DEFAULTS["intermediate_size"])
    parser.add_argument("--num-hidden-layers", type=int, default=NAZ_MODEL_DEFAULTS["num_hidden_layers"])
    parser.add_argument("--num-attention-heads", type=int, default=NAZ_MODEL_DEFAULTS["num_attention_heads"])
    parser.add_argument("--num-key-value-heads", type=int, default=NAZ_MODEL_DEFAULTS["num_key_value_heads"])
    parser.add_argument("--head-dim", type=int, default=NAZ_MODEL_DEFAULTS["head_dim"])
    parser.add_argument("--full-attention-interval", type=int, default=NAZ_MODEL_DEFAULTS["full_attention_interval"])
    parser.add_argument("--linear-key-head-dim", type=int, default=NAZ_MODEL_DEFAULTS["linear_key_head_dim"])
    parser.add_argument("--linear-value-head-dim", type=int, default=NAZ_MODEL_DEFAULTS["linear_value_head_dim"])
    parser.add_argument("--linear-num-key-heads", type=int, default=NAZ_MODEL_DEFAULTS["linear_num_key_heads"])
    parser.add_argument("--linear-num-value-heads", type=int, default=NAZ_MODEL_DEFAULTS["linear_num_value_heads"])
    parser.add_argument("--linear-conv-kernel-size", type=int, default=NAZ_MODEL_DEFAULTS["linear_conv_kernel_size"])
    parser.add_argument("--partial-rotary-factor", type=float, default=NAZ_MODEL_DEFAULTS["partial_rotary_factor"])
    parser.add_argument("--rope-theta", type=float, default=NAZ_MODEL_DEFAULTS["rope_theta"])
    parser.add_argument("--num-mlp-layers", type=int, default=NAZ_MODEL_DEFAULTS["num_mlp_layers"])
    parser.add_argument("--num-samples", type=int, default=NAZ_MODEL_DEFAULTS["num_samples"])
    parser.add_argument("--energy-target-samples", type=int, default=NAZ_MODEL_DEFAULTS["energy_target_samples"])
    parser.add_argument("--noise-size", type=int, default=NAZ_MODEL_DEFAULTS["noise_size"])
    parser.add_argument("--decode-chunk-size", type=int, default=NAZ_MODEL_DEFAULTS["decode_chunk_size"])
    parser.add_argument("--num-writer-layers", type=int, default=NAZ_MODEL_DEFAULTS["num_writer_layers"])
    parser.add_argument("--beta", type=float, default=NAZ_MODEL_DEFAULTS["beta"])
    parser.add_argument("--mean-loss-weight", type=float, default=NAZ_MODEL_DEFAULTS["mean_loss_weight"])
    parser.add_argument("--cosine-loss-weight", type=float, default=NAZ_MODEL_DEFAULTS["cosine_loss_weight"])
    parser.add_argument("--energy-loss-weight", type=float, default=NAZ_MODEL_DEFAULTS["energy_loss_weight"])
    parser.add_argument("--writer-loss-weight", type=float, default=NAZ_MODEL_DEFAULTS["writer_loss_weight"])
    parser.add_argument("--writer-target-warmup-steps", type=int, default=NAZ_MODEL_DEFAULTS["writer_target_warmup_steps"])
    parser.add_argument("--writer-candidate-start-step", type=int, default=NAZ_MODEL_DEFAULTS["writer_candidate_start_step"])
    parser.add_argument("--writer-candidate-probability", type=float, default=NAZ_MODEL_DEFAULTS["writer_candidate_probability"])
    return parser.parse_args()


def validate_args(args):
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("--batch-size and --eval-batch-size must be > 0")
    if args.sequence_length <= 0:
        raise ValueError("--sequence-length must be > 0")
    if args.text_read_chars <= 0:
        raise ValueError("--text-read-chars must be > 0")
    if args.max_eval_batches <= 0:
        raise ValueError("--max-eval-batches must be > 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be > 0")
    if args.eval_every < 0 or args.checkpoint_every < 0:
        raise ValueError("--eval-every and --checkpoint-every must be >= 0")
    if args.eval_every > 0 and args.eval_file is None:
        raise ValueError("--eval-file is required when --eval-every > 0")
    if args.resume is None and args.dil_checkpoint_dir is None:
        raise ValueError("--dil-checkpoint-dir is required when --resume is not set")
    if args.noise_size <= 0:
        raise ValueError("--noise-size must be > 0")
    if args.num_samples < 3:
        raise ValueError("--num-samples must be >= 3")
    if args.energy_target_samples <= 0:
        raise ValueError("--energy-target-samples must be > 0")
    if args.beta <= 0.0:
        raise ValueError("--beta must be > 0")
    if args.decode_chunk_size <= 0:
        raise ValueError("--decode-chunk-size must be > 0")
    if args.num_writer_layers <= 0:
        raise ValueError("--num-writer-layers must be > 0")
    if (
        args.mean_loss_weight < 0.0
        or args.cosine_loss_weight < 0.0
        or args.energy_loss_weight < 0.0
        or args.writer_loss_weight < 0.0
    ):
        raise ValueError("--mean-loss-weight, --cosine-loss-weight, --energy-loss-weight and --writer-loss-weight must be >= 0")
    if args.writer_target_warmup_steps < 0 or args.writer_candidate_start_step < 0:
        raise ValueError("--writer-target-warmup-steps and --writer-candidate-start-step must be >= 0")
    if not 0.0 <= args.writer_candidate_probability <= 1.0:
        raise ValueError("--writer-candidate-probability must be inside [0, 1]")
    if args.full_attention_interval <= 0:
        raise ValueError("--full-attention-interval must be > 0")
    if args.num_attention_heads <= 0 or args.num_key_value_heads <= 0:
        raise ValueError("--num-attention-heads and --num-key-value-heads must be > 0")
    if args.num_attention_heads % args.num_key_value_heads != 0:
        raise ValueError("--num-attention-heads must be divisible by --num-key-value-heads")
    if args.hidden_size != args.num_attention_heads * args.head_dim:
        raise ValueError("--hidden-size must equal --num-attention-heads * --head-dim")
    if args.linear_key_head_dim <= 0 or args.linear_value_head_dim <= 0:
        raise ValueError("--linear-key-head-dim and --linear-value-head-dim must be > 0")
    if args.linear_num_key_heads <= 0 or args.linear_num_value_heads <= 0:
        raise ValueError("--linear-num-key-heads and --linear-num-value-heads must be > 0")
    if args.linear_conv_kernel_size <= 0:
        raise ValueError("--linear-conv-kernel-size must be > 0")
    if args.partial_rotary_factor <= 0.0 or args.partial_rotary_factor > 1.0:
        raise ValueError("--partial-rotary-factor must be in (0, 1]")
    if args.rope_theta <= 0.0:
        raise ValueError("--rope-theta must be > 0")


def build_config(args, dil_config: DilConfig):
    if args.resume is not None:
        return NazConfig.from_pretrained(args.resume.parent)
    return NazConfig(
        dil_path=str(args.dil_checkpoint_dir),
        byte_vocab_size=dil_config.byte_vocab_size,
        vocab_size=dil_config.vocab_size,
        pad_token_id=dil_config.pad_token_id,
        eos_token_id=dil_config.eos_token_id,
        max_word_bytes=dil_config.max_word_bytes,
        latent_size=dil_config.latent_size,
        num_mlp_layers=args.num_mlp_layers,
        num_samples=args.num_samples,
        energy_target_samples=args.energy_target_samples,
        beta=args.beta,
        noise_size=args.noise_size,
        decode_chunk_size=args.decode_chunk_size,
        num_writer_layers=args.num_writer_layers,
        mean_loss_weight=args.mean_loss_weight,
        cosine_loss_weight=args.cosine_loss_weight,
        energy_loss_weight=args.energy_loss_weight,
        writer_loss_weight=args.writer_loss_weight,
        writer_target_warmup_steps=args.writer_target_warmup_steps,
        writer_candidate_start_step=args.writer_candidate_start_step,
        writer_candidate_probability=args.writer_candidate_probability,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        head_dim=args.head_dim,
        full_attention_interval=args.full_attention_interval,
        linear_key_head_dim=args.linear_key_head_dim,
        linear_value_head_dim=args.linear_value_head_dim,
        linear_num_key_heads=args.linear_num_key_heads,
        linear_num_value_heads=args.linear_num_value_heads,
        linear_conv_kernel_size=args.linear_conv_kernel_size,
        partial_rotary_factor=args.partial_rotary_factor,
        rope_theta=args.rope_theta,
    )


def main():
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.resume is not None:
        resume_config = NazConfig.from_pretrained(args.resume.parent)
        dil_checkpoint_dir = Path(resume_config.dil_path)
    else:
        dil_checkpoint_dir = args.dil_checkpoint_dir
    dil_config = DilConfig.from_pretrained(dil_checkpoint_dir)
    config = build_config(args, dil_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    compile_mode = effective_compile_mode(args.compile_mode, device)
    validate_compile_environment(compile_mode)
    autocast_enabled = bool(args.bf16 and device.type == "cuda")
    cuda_prefetch = bool(device.type == "cuda" and not args.no_cuda_prefetch)
    tokenizer = HybridTokenizer.from_file(dil_checkpoint_dir / dil_config.tokenizer_vocab_file)

    token_cache_dir = args.token_cache_dir or args.output_dir / "naz_token_cache"
    train_ids_path, train_lengths_path, train_token_count = build_token_cache(
        args.train_file,
        tokenizer,
        dil_config.max_word_bytes,
        dil_config.pad_token_id,
        args.text_read_chars,
        token_cache_dir,
    )
    eval_cache = None
    if args.eval_every > 0:
        eval_cache = build_token_cache(
            args.eval_file,
            tokenizer,
            dil_config.max_word_bytes,
            dil_config.pad_token_id,
            args.text_read_chars,
            token_cache_dir,
        )

    base_model = Naz(config).to(device)
    base_model.train()
    initial_dil_checksum = dil_checksum(base_model)
    base_model.dil_model.set_compiled_forwards(
        encoder_forward=compile_forward(base_model.dil_model.encoder.forward, compile_mode, "DilEncoderCore"),
    )
    base_model.set_compiled_student_forward(
        compile_forward(base_model.student_core.forward, compile_mode, "NazStudentCore")
    )
    if config.writer_loss_weight > 0.0:
        base_model.set_compiled_writer_forward(
            compile_forward(base_model.writer.forward, compile_mode, "NazContextualWriter")
        )
    model = base_model
    optimizer = AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(optimizer, args.learning_rate, args.warmup_steps)

    start_step = 0
    last_metrics = {}
    if args.resume is not None:
        start_step, last_metrics = restore_checkpoint(args.resume, model, optimizer, scheduler, device)
        resume_dil_checksum = dil_checksum(model)
        expected_checksum = load_checkpoint(args.resume, device)["training_state"]["dil_checksum"]
        if resume_dil_checksum != expected_checksum:
            raise RuntimeError("resumed Dil checksum does not match checkpoint")
        initial_dil_checksum = resume_dil_checksum

    print(
        f"device={device.type} bf16={int(autocast_enabled)} compile_mode={compile_mode} "
        f"data_mode={args.data_mode} resume_step={start_step} train_tokens={train_token_count}",
        flush=True,
    )

    if args.data_mode == "resident":
        train_surface = ResidentNazBatcher(
            train_ids_path,
            train_lengths_path,
            train_token_count,
            dil_config,
            args.sequence_length,
            args.batch_size,
            device,
            args.seed + start_step,
        )
        semantic_cache = build_resident_semantic_cache(
            model,
            train_surface,
            chunk_tokens=4096,
            autocast_enabled=autocast_enabled,
        )
        train_iter = ResidentNazSemanticBatcher(
            *semantic_cache,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            seed=args.seed + start_step,
        )
        eval_loader = None
        if eval_cache is not None:
            eval_ids_path, eval_lengths_path, eval_token_count = eval_cache
            eval_surface = ResidentNazBatcher(
                eval_ids_path,
                eval_lengths_path,
                eval_token_count,
                dil_config,
                args.sequence_length,
                args.eval_batch_size,
                device,
                args.seed + 1,
            )
            eval_loader = ResidentNazSemanticEvalLoader(
                ResidentNazSemanticBatcher(
                    *build_resident_semantic_cache(
                        model,
                        eval_surface,
                        chunk_tokens=4096,
                        autocast_enabled=autocast_enabled,
                    ),
                    sequence_length=args.sequence_length,
                    batch_size=args.eval_batch_size,
                    seed=args.seed + 1,
                ),
                batch_size=args.eval_batch_size,
            )
    else:
        train_loader = make_naz_loader(
            RandomWindowNazDataset(
                train_ids_path,
                train_lengths_path,
                train_token_count,
                dil_config,
                args.sequence_length,
                args.batch_size,
                args.seed + start_step,
            ),
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            prefetch_factor=args.prefetch_factor,
        )
        eval_loader = None
        if eval_cache is not None:
            eval_ids_path, eval_lengths_path, eval_token_count = eval_cache
            eval_loader = make_naz_loader(
                RandomWindowNazDataset(
                    eval_ids_path,
                    eval_lengths_path,
                    eval_token_count,
                    dil_config,
                    args.sequence_length,
                    args.eval_batch_size,
                    args.seed + 1,
                ),
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                prefetch_factor=args.prefetch_factor,
            )
        train_iter = DeviceBatchPrefetcher(train_loader, device, cuda_prefetch)
    log_start = time.perf_counter()
    log_tokens = 0
    log_steps = 0
    data_seconds = 0.0
    transfer_seconds = 0.0
    compute_seconds = 0.0
    metric_sums = {
        "loss": 0.0,
        "energy": 0.0,
        "energy_loss": 0.0,
        "mean_loss": 0.0,
        "cosine_loss": 0.0,
        "writer_loss": 0.0,
        "latent_cos": 0.0,
        "candidate_cos": 0.0,
        "byte_acc": 0.0,
    }
    completed_step = start_step

    def save_interrupted():
        interrupted_dil_checksum = dil_checksum(model)
        if interrupted_dil_checksum != initial_dil_checksum:
            raise RuntimeError("frozen Dil checksum changed during training")
        interrupted_dir = save_checkpoint(
            args.output_dir,
            model,
            optimizer,
            scheduler,
            config,
            completed_step,
            last_metrics,
            interrupted_dil_checksum,
            compile_mode,
        )
        print(f"interrupted_saved={interrupted_dir}", flush=True)

    try:
        for step in range(start_step + 1, args.max_steps + 1):
            data_start = time.perf_counter()
            batch = next(train_iter)
            if args.data_mode == "resident":
                data_seconds += time.perf_counter() - data_start
                transfer_seconds += 0.0
            else:
                data_seconds += train_iter.last_data_seconds
                transfer_seconds += train_iter.last_transfer_seconds

            cuda_sync(device)
            compute_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(autocast_enabled):
                outputs = run_naz_batch(model, batch, training_step=step)

            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (param for param in model.parameters() if param.requires_grad),
                args.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            cuda_sync(device)
            compute_seconds += time.perf_counter() - compute_start
            completed_step = step

            real_tokens = int(batch["unit_mask"].sum().detach().cpu())
            log_tokens += real_tokens
            log_steps += 1
            metric_sums["loss"] += float(outputs.loss.detach().cpu())
            metric_sums["energy"] += float(outputs.energy.detach().cpu())
            metric_sums["energy_loss"] += float(outputs.energy_loss.detach().cpu())
            metric_sums["mean_loss"] += float(outputs.mean_loss.detach().cpu())
            metric_sums["cosine_loss"] += float(outputs.cosine_loss.detach().cpu())
            metric_sums["writer_loss"] += float(outputs.writer_loss.detach().cpu())
            metric_sums["latent_cos"] += float(outputs.latent_cos.detach().cpu())
            metric_sums["candidate_cos"] += float(outputs.candidate_cos.detach().cpu())
            metric_sums["byte_acc"] += float(outputs.byte_acc.detach().cpu())

            should_log = step % args.log_every == 0 or step == start_step + 1 or step == args.max_steps
            should_eval = eval_loader is not None and step % args.eval_every == 0
            if should_log or should_eval:
                elapsed = max(time.perf_counter() - log_start, 1e-9)
                averaged = {key: value / max(log_steps, 1) for key, value in metric_sums.items()}
                averaged["lr"] = scheduler.get_last_lr()[0]
                averaged["data_seconds"] = data_seconds / max(log_steps, 1)
                averaged["transfer_seconds"] = transfer_seconds / max(log_steps, 1)
                averaged["compute_seconds"] = compute_seconds / max(log_steps, 1)
                averaged["tokens_per_second"] = log_tokens / elapsed
                averaged["steps_per_second"] = log_steps / elapsed
                if should_eval:
                    averaged.update(
                        evaluate(
                            model,
                            eval_loader,
                            device,
                            autocast_enabled,
                            args.max_eval_batches,
                            cuda_prefetch,
                        )
                    )
                print(format_log(step, averaged), flush=True)
                last_metrics = averaged
                log_start = time.perf_counter()
                log_tokens = 0
                log_steps = 0
                data_seconds = 0.0
                transfer_seconds = 0.0
                compute_seconds = 0.0
                for key in metric_sums:
                    metric_sums[key] = 0.0

            if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                save_checkpoint(
                    args.output_dir,
                    model,
                    optimizer,
                    scheduler,
                    config,
                    step,
                    last_metrics,
                    initial_dil_checksum,
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

    final_dil_checksum = dil_checksum(model)
    if final_dil_checksum != initial_dil_checksum:
        raise RuntimeError("frozen Dil checksum changed during training")

    final_dir = save_checkpoint(
        args.output_dir,
        model,
        optimizer,
        scheduler,
        config,
        args.max_steps,
        last_metrics,
        final_dil_checksum,
        compile_mode,
    )
    print(f"saved={final_dir}", flush=True)


if __name__ == "__main__":
    main()


