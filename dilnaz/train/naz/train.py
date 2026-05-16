from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from dilnaz.train.common.runtime import (
    COMPILE_MODE_CHOICES,
    DeviceBatchPrefetcher,
    autocast_context,
    compile_forward,
    effective_compile_mode,
    load_checkpoint,
    restore_rng_state,
    rng_state,
    validate_compile_environment,
)
from dilnaz.train.configs.defaults import NAZ_FINETUNE_DEFAULTS, NAZ_MODEL_DEFAULTS, NAZ_TRAIN_DEFAULTS
from dilnaz.models.dil import DilConfig
from dilnaz.models.naz import NazConfig
from dilnaz.models.naz import Naz
from dilnaz.train.data.naz_data import (
    MemmapNazSemanticBatcher,
    MemmapNazSemanticEvalLoader,
    ResidentNazBatcher,
    ResidentNazSemanticBatcher,
    ResidentNazSemanticEvalLoader,
    StreamingTextNazDataset,
    build_token_cache,
    make_naz_loader,
)
from dilnaz.tokenization import HybridTokenizer
from dilnaz.train.common.trainer_core import BaseTrainer, StepResult, make_scheduler


CHECKPOINT_FORMAT_VERSION = 28
OBJECTIVE = "semantic_dynamics_moe_mtp_v1"
DATALOADER_WORKER_EXIT = "DataLoader worker"
SEMANTIC_CACHE_FORMAT_VERSION = 2
STAGES = ("pretrain", "finetune")
RUNTIME_STATE_FIELDS = (
    "data_mode",
    "sequence_length",
    "batch_size",
    "eval_batch_size",
    "learning_rate",
    "weight_decay",
    "adam_beta1",
    "adam_beta2",
    "warmup_steps",
    "gradient_accumulation_steps",
    "max_grad_norm",
    "log_every",
    "checkpoint_every",
    "eval_every",
    "max_eval_batches",
    "text_read_chars",
    "semantic_cache_chunk_tokens",
    "num_workers",
    "prefetch_factor",
    "seed",
)
RESUME_LOCKED_RUNTIME_FIELDS = ("data_mode", "sequence_length")
MODEL_OVERRIDE_OPTIONS = frozenset(
    {
        "--hidden-size",
        "--intermediate-size",
        "--num-hidden-layers",
        "--num-attention-heads",
        "--num-key-value-heads",
        "--head-dim",
        "--full-attention-interval",
        "--linear-key-head-dim",
        "--linear-value-head-dim",
        "--linear-num-key-heads",
        "--linear-num-value-heads",
        "--linear-conv-kernel-size",
        "--partial-rotary-factor",
        "--rope-theta",
        "--reconstruction-loss-weight",
        "--num-semantic-candidates",
        "--mtp-horizons",
        "--mtp-loss-weights",
        "--mixture-sigma",
        "--usage-balance-weight",
        "--router-responsibility-weight",
        "--moe-num-experts",
        "--moe-top-k",
        "--moe-layers",
        "--moe-balance-weight",
        "--naz-input-jitter-prob",
        "--naz-input-jitter-min-cos",
        "--naz-input-jitter-max-cos",
    }
)


def provided_options(argv: list[str] | None) -> set[str]:
    tokens = sys.argv[1:] if argv is None else argv
    return {token.split("=", 1)[0] for token in tokens if token.startswith("--")}


def dil_checksum(model: Naz) -> str:
    digest = hashlib.sha256()
    for key, tensor in model.dil_model.state_dict().items():
        if key.startswith("encoder."):
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def legacy_dil_checksum(model: Naz) -> str:
    digest = hashlib.sha256()
    for tensor in model.dil_model.state_dict().values():
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def checkpoint_training_state(path: Path, device: torch.device | None = None) -> dict:
    checkpoint = load_checkpoint(path, torch.device("cpu") if device is None else device)
    if checkpoint["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format_version={checkpoint.get('format_version')}")
    state = checkpoint["training_state"]
    if state.get("objective") != OBJECTIVE:
        raise ValueError(f"checkpoint objective is not {OBJECTIVE}")
    if state.get("stage") not in STAGES:
        raise ValueError("checkpoint training_state.stage is missing or invalid")
    if not isinstance(state.get("runtime"), dict):
        raise ValueError("checkpoint training_state.runtime is missing or invalid")
    return state


def runtime_training_state(args) -> dict:
    return {field: getattr(args, field) for field in RUNTIME_STATE_FIELDS}


def json_training_state(
    config: NazConfig,
    stage: str,
    step: int,
    metrics: dict,
    checksum: str,
    compile_mode: str,
    init_naz_checkpoint: Path | None,
    runtime: dict,
) -> dict:
    state = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "stage": stage,
        "step": step,
        "metrics": metrics,
        "dil_checksum": checksum,
        "compile_mode": compile_mode,
        "max_surface_pieces_per_unit": getattr(config, "max_surface_pieces_per_unit", 0),
        "latent_size": config.latent_size,
        "semantic_space": "dil_normalized_latent",
        "objective": OBJECTIVE,
        "byte_vocab_size": config.byte_vocab_size,
        "vocab_size": config.vocab_size,
        "pad_token_id": config.pad_token_id,
        "runtime": runtime,
    }
    if init_naz_checkpoint is not None:
        state["init_naz_checkpoint"] = str(init_naz_checkpoint)
    return state


def save_checkpoint(
    output_dir: Path,
    model: Naz,
    optimizer,
    scheduler,
    config: NazConfig,
    stage: str,
    step: int,
    metrics: dict,
    checksum: str,
    compile_mode: str,
    runtime: dict,
    init_naz_checkpoint: Path | None = None,
    checkpoint_name: str = "",
) -> Path:
    checkpoint_dir = output_dir / checkpoint_name if checkpoint_name else output_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(checkpoint_dir)
    state = json_training_state(
        config,
        stage,
        step,
        metrics,
        checksum,
        compile_mode,
        init_naz_checkpoint,
        runtime,
    )
    import os as _os

    tmp_path = checkpoint_dir / "checkpoint.pt.tmp"
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": model.trainable_state_dict(),
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


def restore_checkpoint(path: Path, model: Naz, optimizer, scheduler, device: torch.device) -> tuple[int, dict]:
    checkpoint = load_checkpoint(path, device)
    if checkpoint["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format_version={checkpoint.get('format_version')}")
    state = checkpoint["training_state"]
    if state.get("objective") != OBJECTIVE:
        raise ValueError(f"checkpoint objective is not {OBJECTIVE}")
    if state.get("stage") not in STAGES:
        raise ValueError("checkpoint training_state.stage is missing or invalid")
    if not isinstance(state.get("runtime"), dict):
        raise ValueError("checkpoint training_state.runtime is missing or invalid")
    model.load_trainable_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    restore_rng_state(checkpoint["rng_state"])
    return int(state["step"]), dict(state["metrics"])


def load_init_checkpoint(path: Path, model: Naz, device: torch.device) -> dict:
    checkpoint = load_checkpoint(path, device)
    if checkpoint["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format_version={checkpoint.get('format_version')}")
    state = checkpoint["training_state"]
    if state.get("objective") != OBJECTIVE:
        raise ValueError(f"checkpoint objective is not {OBJECTIVE}")
    if state.get("stage") not in STAGES:
        raise ValueError("checkpoint training_state.stage is missing or invalid")
    model.load_trainable_state_dict(checkpoint["model_state_dict"])
    return state


def is_dataloader_worker_exit(error: RuntimeError) -> bool:
    message = str(error)
    return DATALOADER_WORKER_EXIT in message and "exited unexpectedly" in message


def add_output_metrics(total: dict[str, float], outputs) -> None:
    targets = float(outputs.num_targets.detach().cpu())
    candidate_usage = outputs.candidate_usage.detach().float().cpu()
    candidate_entropy = float(
        -(candidate_usage * candidate_usage.clamp_min(1e-8).log()).sum(dim=-1).mean()
    )
    moe_usage = outputs.moe_usage.detach().float().cpu()
    if moe_usage.numel() == 0:
        moe_entropy = 0.0
        moe_max = 0.0
    else:
        moe_entropy = float(-(moe_usage * moe_usage.clamp_min(1e-8).log()).sum())
        moe_max = float(moe_usage.max())
    total["loss"] += float(outputs.loss.detach().cpu())
    total["reconstruction"] += float(outputs.reconstruction_loss.detach().cpu())
    total["mse"] += float(outputs.mse_loss.detach().cpu())
    total["mixture_nll"] += float(outputs.mixture_nll.detach().cpu())
    total["responsibility"] += float(outputs.responsibility_loss.detach().cpu())
    total["usage_balance"] += float(outputs.usage_balance_loss.detach().cpu())
    total["moe_balance"] += float(outputs.moe_balance_loss.detach().cpu())
    total["min_mse"] += float(outputs.min_mse.detach().cpu()) * targets
    total["chosen_mse"] += float(outputs.chosen_mse.detach().cpu()) * targets
    total["router_entropy"] += float(outputs.router_entropy.detach().cpu()) * targets
    total["cosine_loss"] += float(outputs.cosine_loss.detach().cpu()) * targets
    total["latent_cos"] += float(outputs.latent_cos.detach().cpu()) * targets
    total["candidate_usage_entropy"] += candidate_entropy
    total["candidate_usage_max"] += float(candidate_usage.max())
    total["moe_usage_entropy"] += moe_entropy
    total["moe_usage_max"] += moe_max
    total["targets"] += targets
    total["batches"] += 1


def reduce_output_metrics(total: dict[str, float]) -> dict[str, float]:
    batches = max(total["batches"], 1)
    targets = max(total["targets"], 1.0)
    return {
        "loss": total["loss"] / batches,
        "reconstruction": total["reconstruction"] / batches,
        "mse": total["mse"] / batches,
        "mixture_nll": total["mixture_nll"] / batches,
        "responsibility": total["responsibility"] / batches,
        "usage_balance": total["usage_balance"] / batches,
        "moe_balance": total["moe_balance"] / batches,
        "min_mse": total["min_mse"] / targets,
        "chosen_mse": total["chosen_mse"] / targets,
        "router_entropy": total["router_entropy"] / targets,
        "mse_mean": total["mse"] / targets,
        "cosine_loss": total["cosine_loss"] / targets,
        "latent_cos": total["latent_cos"] / targets,
        "candidate_usage_entropy": total["candidate_usage_entropy"] / batches,
        "candidate_usage_max": total["candidate_usage_max"] / batches,
        "moe_usage_entropy": total["moe_usage_entropy"] / batches,
        "moe_usage_max": total["moe_usage_max"] / batches,
        "targets": total["targets"] / batches,
    }


def semantic_cache_spans(
    token_count: int,
    chunk_tokens: int,
    max_surface_width: int,
    span_width,
):
    start = 0
    while start < token_count:
        high = min(start + chunk_tokens, token_count)
        low = start + 1
        best = 0
        while low <= high:
            mid = (low + high) // 2
            if span_width(start, mid) <= max_surface_width:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        if best <= start:
            width = span_width(start, start + 1)
            raise ValueError(
                f"single semantic cache span width {width} exceeds largest DIL surface bucket {max_surface_width}"
            )
        yield start, best
        start = best


@torch.no_grad()
def build_resident_semantic_cache(
    model: Naz,
    surface_batcher: ResidentNazBatcher,
    chunk_tokens: int,
    autocast_enabled: bool,
):
    model.eval()
    token_count = surface_batcher.token_count
    max_surface_width = max(model.dil_config.surface_bucket_sizes)
    latent_chunks = []
    spans = semantic_cache_spans(
        token_count,
        chunk_tokens,
        max_surface_width,
        surface_batcher.surface_span_width,
    )
    for start, end in spans:
        surface = surface_batcher.surface_slice(start, end)
        unit_mask = torch.ones(surface.unit_lengths.shape, dtype=torch.bool, device=surface.ids.device)
        with autocast_context(autocast_enabled):
            latents = model.encode_sequence_latents(surface, unit_mask)
        latent_chunks.append(latents.squeeze(0).float())
    semantic_states = torch.cat(latent_chunks, dim=0)
    model.train()
    return semantic_states, semantic_states


def semantic_cache_paths(cache_dir: Path, ids_path: Path, checksum: str) -> tuple[Path, Path]:
    key = f"{ids_path.stem}.{checksum[:24]}.semantic"
    return cache_dir / f"{key}.latents.npy", cache_dir / f"{key}.json"


@torch.no_grad()
def build_memmap_semantic_cache(
    model: Naz,
    ids_path: Path,
    lengths_path: Path,
    token_count: int,
    chunk_tokens: int,
    autocast_enabled: bool,
    cache_dir: Path,
    checksum: str,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    latents_path, meta_path = semantic_cache_paths(cache_dir, ids_path, checksum)
    expected_meta = {
        "format_version": SEMANTIC_CACHE_FORMAT_VERSION,
        "token_count": token_count,
        "latent_size": model.config.latent_size,
        "surface_cache": "packed_flat_offsets",
        "dil_checksum": checksum,
        "ids_path": str(ids_path.resolve()),
        "lengths_path": str(lengths_path.resolve()),
    }
    if latents_path.exists() and meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        if all(meta.get(key) == value for key, value in expected_meta.items()):
            return latents_path

    surface_ids = np.load(ids_path, mmap_mode="r")
    surface_offsets = np.load(lengths_path, mmap_mode="r")
    semantic = np.lib.format.open_memmap(
        latents_path,
        mode="w+",
        dtype=np.float32,
        shape=(token_count, model.config.latent_size),
    )

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    max_surface_width = max(model.dil_config.surface_bucket_sizes)
    spans = semantic_cache_spans(
        token_count,
        chunk_tokens,
        max_surface_width,
        lambda start, end: int(surface_offsets[end] - surface_offsets[start]),
    )
    for start, end in spans:
        rows = []
        for token_idx in range(start, end):
            piece_start = int(surface_offsets[token_idx])
            piece_end = int(surface_offsets[token_idx + 1])
            rows.append(np.asarray(surface_ids[piece_start:piece_end], dtype=np.int64).tolist())
        from dilnaz.surface import pack_token_units
        surface = pack_token_units(
            [rows],
            pad_token_id=model.dil_config.pad_token_id,
            bucket_sizes=model.dil_config.surface_bucket_sizes,
            max_pieces_per_unit=model.dil_config.max_surface_pieces_per_unit,
            device=device,
        )
        unit_mask = torch.ones(surface.unit_lengths.shape, dtype=torch.bool, device=device)
        with autocast_context(autocast_enabled):
            chunk_latents = model.encode_sequence_latents(surface, unit_mask)
        semantic[start:end] = chunk_latents.squeeze(0).float().cpu().numpy()
    semantic.flush()
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(expected_meta, handle, indent=2)
    if was_training:
        model.train()
    return latents_path


def run_naz_batch(model: Naz, batch: dict, training_step: int | None = None):
    if "semantic_states" in batch:
        return model.forward_semantic(
            semantic_states=batch["semantic_states"],
            target_latents=batch["target_latents"],
            unit_mask=batch["unit_mask"],
            target_mask=batch["target_mask"],
        )
    return model(**batch, training_step=training_step)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, default="pretrain")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, default=None)
    parser.add_argument("--dil-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--init-naz-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--compile-mode", choices=COMPILE_MODE_CHOICES, default=None)
    parser.add_argument("--data-mode", choices=("streaming", "resident", "cached"), default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--adam-beta1", type=float, default=None)
    parser.add_argument("--adam-beta2", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--text-read-chars", type=int, default=None)
    parser.add_argument("--semantic-cache-chunk-tokens", type=int, default=None)
    parser.add_argument("--token-cache-dir", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--no-cuda-prefetch", action="store_true")
    parser.add_argument("--sync-timing", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
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
    parser.add_argument("--reconstruction-loss-weight", type=float, default=NAZ_MODEL_DEFAULTS["reconstruction_loss_weight"])
    parser.add_argument("--num-semantic-candidates", type=int, default=NAZ_MODEL_DEFAULTS["num_semantic_candidates"])
    parser.add_argument("--mtp-horizons", type=int, default=NAZ_MODEL_DEFAULTS["mtp_horizons"])
    parser.add_argument(
        "--mtp-loss-weights",
        type=float,
        nargs="+",
        default=list(NAZ_MODEL_DEFAULTS["mtp_loss_weights"]),
    )
    parser.add_argument("--mixture-sigma", type=float, default=NAZ_MODEL_DEFAULTS["mixture_sigma"])
    parser.add_argument("--usage-balance-weight", type=float, default=NAZ_MODEL_DEFAULTS["usage_balance_weight"])
    parser.add_argument(
        "--router-responsibility-weight",
        type=float,
        default=NAZ_MODEL_DEFAULTS["router_responsibility_weight"],
    )
    parser.add_argument("--moe-num-experts", type=int, default=NAZ_MODEL_DEFAULTS["moe_num_experts"])
    parser.add_argument("--moe-top-k", type=int, default=NAZ_MODEL_DEFAULTS["moe_top_k"])
    parser.add_argument("--moe-layers", type=int, default=NAZ_MODEL_DEFAULTS["moe_layers"])
    parser.add_argument("--moe-balance-weight", type=float, default=NAZ_MODEL_DEFAULTS["moe_balance_weight"])
    parser.add_argument("--naz-input-jitter-prob", type=float, default=NAZ_MODEL_DEFAULTS["naz_input_jitter_prob"])
    parser.add_argument("--naz-input-jitter-min-cos", type=float, default=NAZ_MODEL_DEFAULTS["naz_input_jitter_min_cos"])
    parser.add_argument("--naz-input-jitter-max-cos", type=float, default=NAZ_MODEL_DEFAULTS["naz_input_jitter_max_cos"])
    args = parser.parse_args(argv)
    args.provided_options = provided_options(argv)
    return args


def apply_stage_defaults(args) -> None:
    defaults = NAZ_FINETUNE_DEFAULTS if args.stage == "finetune" else NAZ_TRAIN_DEFAULTS
    for key, value in defaults.items():
        attr = key.replace("-", "_")
        if hasattr(args, attr) and getattr(args, attr) is None:
            setattr(args, attr, value)
    if args.stage == "finetune" and args.eval_file is None and "--eval-every" not in args.provided_options:
        args.eval_every = 0


def validate_common_args(args) -> None:
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("--batch-size and --eval-batch-size must be > 0")
    if args.sequence_length <= 0:
        raise ValueError("--sequence-length must be > 0")
    if args.text_read_chars <= 0:
        raise ValueError("--text-read-chars must be > 0")
    if args.semantic_cache_chunk_tokens <= 0:
        raise ValueError("--semantic-cache-chunk-tokens must be > 0")
    if args.max_eval_batches <= 0:
        raise ValueError("--max-eval-batches must be > 0")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be > 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be > 0")
    if args.eval_every < 0 or args.checkpoint_every < 0:
        raise ValueError("--eval-every and --checkpoint-every must be >= 0")
    if args.eval_every > 0 and args.eval_file is None:
        raise ValueError("--eval-file is required when --eval-every > 0")
    if args.reconstruction_loss_weight < 0.0:
        raise ValueError("--reconstruction-loss-weight must be >= 0")
    if args.num_semantic_candidates <= 0:
        raise ValueError("--num-semantic-candidates must be > 0")
    if args.mtp_horizons <= 0:
        raise ValueError("--mtp-horizons must be > 0")
    if len(args.mtp_loss_weights) != args.mtp_horizons:
        raise ValueError("--mtp-loss-weights count must equal --mtp-horizons")
    if any(weight <= 0.0 for weight in args.mtp_loss_weights):
        raise ValueError("--mtp-loss-weights must be positive")
    if args.mixture_sigma <= 0.0:
        raise ValueError("--mixture-sigma must be > 0")
    if args.usage_balance_weight < 0.0 or args.router_responsibility_weight < 0.0:
        raise ValueError("--usage-balance-weight and --router-responsibility-weight must be >= 0")
    if args.moe_num_experts <= 0 or args.moe_top_k <= 0:
        raise ValueError("--moe-num-experts and --moe-top-k must be > 0")
    if args.moe_top_k > args.moe_num_experts:
        raise ValueError("--moe-top-k must be <= --moe-num-experts")
    if args.moe_layers < 0:
        raise ValueError("--moe-layers must be >= 0")
    if args.moe_balance_weight < 0.0:
        raise ValueError("--moe-balance-weight must be >= 0")
    if args.naz_input_jitter_prob < 0.0 or args.naz_input_jitter_prob > 1.0:
        raise ValueError("--naz-input-jitter-prob must be inside [0, 1]")
    if args.naz_input_jitter_min_cos <= 0.0 or args.naz_input_jitter_max_cos > 1.0:
        raise ValueError("--naz-input-jitter cosine values must satisfy 0 < cos <= 1")
    if args.naz_input_jitter_min_cos > args.naz_input_jitter_max_cos:
        raise ValueError("--naz-input-jitter-min-cos must be <= --naz-input-jitter-max-cos")
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


def validate_args(args) -> None:
    if args.resume is not None:
        if args.dil_checkpoint_dir is not None or args.init_naz_checkpoint is not None:
            raise ValueError("--resume must be used without --dil-checkpoint-dir or --init-naz-checkpoint")
        state = checkpoint_training_state(args.resume)
        args.stage = state["stage"]
        runtime = state["runtime"]
        for field in RESUME_LOCKED_RUNTIME_FIELDS:
            option = f"--{field.replace('_', '-')}"
            saved_value = runtime[field]
            if option in args.provided_options and getattr(args, field) != saved_value:
                raise ValueError(f"{option} is owned by the checkpoint during --resume")
            setattr(args, field, saved_value)
    elif args.stage == "pretrain":
        if args.dil_checkpoint_dir is None:
            raise ValueError("--dil-checkpoint-dir is required for --stage pretrain")
        if args.init_naz_checkpoint is not None:
            raise ValueError("--init-naz-checkpoint is only valid for --stage finetune")
    else:
        if args.init_naz_checkpoint is None:
            raise ValueError("--init-naz-checkpoint is required for --stage finetune")
        if args.dil_checkpoint_dir is not None:
            raise ValueError("--stage finetune reads Dil path from --init-naz-checkpoint config")

    if args.stage == "finetune" or args.resume is not None:
        forbidden = sorted(args.provided_options & MODEL_OVERRIDE_OPTIONS)
        if forbidden:
            raise ValueError(f"model architecture/objective overrides are not allowed here: {', '.join(forbidden)}")

    apply_stage_defaults(args)
    validate_common_args(args)


def build_config(args, dil_config: DilConfig | None) -> NazConfig:
    if args.resume is not None:
        return NazConfig.from_pretrained(args.resume.parent)
    if args.stage == "finetune":
        return NazConfig.from_pretrained(args.init_naz_checkpoint.parent)
    return NazConfig(
        dil_path=str(args.dil_checkpoint_dir),
        byte_vocab_size=dil_config.byte_vocab_size,
        vocab_size=dil_config.vocab_size,
        pad_token_id=dil_config.pad_token_id,
        eos_token_id=dil_config.eos_token_id,
        latent_size=dil_config.latent_size,
        reconstruction_loss_weight=args.reconstruction_loss_weight,
        num_semantic_candidates=args.num_semantic_candidates,
        mtp_horizons=args.mtp_horizons,
        mtp_loss_weights=tuple(args.mtp_loss_weights),
        mixture_sigma=args.mixture_sigma,
        usage_balance_weight=args.usage_balance_weight,
        router_responsibility_weight=args.router_responsibility_weight,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_layers=args.moe_layers,
        moe_balance_weight=args.moe_balance_weight,
        naz_input_jitter_prob=args.naz_input_jitter_prob,
        naz_input_jitter_min_cos=args.naz_input_jitter_min_cos,
        naz_input_jitter_max_cos=args.naz_input_jitter_max_cos,
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


class NazBaseTrainer(BaseTrainer):
    def __init__(self, args):
        validate_args(args)
        super().__init__(args)
        self.stage = args.stage
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
        self.compile_mode = effective_compile_mode(args.compile_mode, self.device)
        validate_compile_environment(self.compile_mode)
        self.autocast_enabled = bool(args.bf16 and self.device.type == "cuda")
        self.cuda_prefetch = bool(self.device.type == "cuda" and not args.no_cuda_prefetch)
        self.config = build_config(args, self.load_dil_config_for_new_pretrain(args))
        self.dil_checkpoint_dir = Path(self.config.dil_path)
        self.dil_config = DilConfig.from_pretrained(self.dil_checkpoint_dir)
        self.tokenizer = HybridTokenizer.from_file(self.dil_checkpoint_dir / self.dil_config.tokenizer_vocab_file)
        self.model = Naz(self.config).to(self.device)
        self.model.train()
        self.optimizer = AdamW(
            self.optimizer_param_groups(args.weight_decay),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
        )
        self.scheduler = make_scheduler(self.optimizer, args.learning_rate, args.warmup_steps, args.max_steps)
        self.initial_dil_checksum = dil_checksum(self.model)
        self.restore_or_initialize()
        self.model.dil_model.set_compiled_forwards(
            encoder_forward=compile_forward(self.model.dil_model.encoder.forward, self.compile_mode, "DilEncoderCore"),
        )
        self.model.set_compiled_student_forward(
            compile_forward(self.model.student_core.forward, self.compile_mode, "NazStudentCore")
        )
        self.train_iterator = None
        self.eval_loader = None
        self.prepare_data_sources()

    def load_dil_config_for_new_pretrain(self, args) -> DilConfig | None:
        if args.resume is not None or args.stage == "finetune":
            return None
        return DilConfig.from_pretrained(args.dil_checkpoint_dir)

    def restore_or_initialize(self) -> None:
        if self.args.resume is not None:
            self.start_step, self.last_metrics = restore_checkpoint(
                self.args.resume,
                self.model,
                self.optimizer,
                self.scheduler,
                self.device,
            )
            expected = checkpoint_training_state(self.args.resume, self.device)["dil_checksum"]
            checksum = dil_checksum(self.model)
            if checksum != expected:
                print(
                    f"warning: dil_checksum mismatch (expected={expected[:16]}..., got={checksum[:16]}...), "
                    f"resuming anyway",
                    flush=True,
                )
            self.initial_dil_checksum = checksum
        elif self.stage == "finetune":
            init_state = load_init_checkpoint(self.args.init_naz_checkpoint, self.model, self.device)
            init_stage = init_state.get("stage", "unknown")
            print(f"initialized_from={self.args.init_naz_checkpoint} init_stage={init_stage}", flush=True)
            self.initial_dil_checksum = dil_checksum(self.model)

    def prepare_data_sources(self) -> None:
        if self.args.data_mode == "resident":
            self.prepare_resident_sources()
        elif self.args.data_mode == "cached":
            self.prepare_cached_sources()
        else:
            self.prepare_streaming_sources()

    def prepare_resident_sources(self) -> None:
        token_cache_dir = self.args.token_cache_dir or self.args.output_dir / "naz_token_cache"
        train_cache = build_token_cache(
            self.args.train_file,
            self.tokenizer,
            self.dil_config.max_surface_pieces_per_unit,
            self.dil_config.pad_token_id,
            self.args.text_read_chars,
            token_cache_dir,
        )
        train_ids_path, train_lengths_path, train_token_count = train_cache
        train_surface = ResidentNazBatcher(
            train_ids_path,
            train_lengths_path,
            train_token_count,
            self.dil_config,
            self.args.sequence_length,
            self.args.batch_size,
            self.device,
            self.args.seed + self.start_step,
        )
        self.train_iterator = ResidentNazSemanticBatcher(
            *build_resident_semantic_cache(
                self.model,
                train_surface,
                chunk_tokens=self.args.semantic_cache_chunk_tokens,
                autocast_enabled=self.autocast_enabled,
            ),
            sequence_length=self.args.sequence_length,
            batch_size=self.args.batch_size,
            seed=self.args.seed + self.start_step,
            horizons=self.config.mtp_horizons,
        )
        if self.args.eval_every > 0:
            eval_cache = build_token_cache(
                self.args.eval_file,
                self.tokenizer,
                self.dil_config.max_surface_pieces_per_unit,
                self.dil_config.pad_token_id,
                self.args.text_read_chars,
                token_cache_dir,
            )
            eval_ids_path, eval_lengths_path, eval_token_count = eval_cache
            eval_surface = ResidentNazBatcher(
                eval_ids_path,
                eval_lengths_path,
                eval_token_count,
                self.dil_config,
                self.args.sequence_length,
                self.args.eval_batch_size,
                self.device,
                self.args.seed + 1,
            )
            self.eval_loader = ResidentNazSemanticEvalLoader(
                ResidentNazSemanticBatcher(
                    *build_resident_semantic_cache(
                        self.model,
                        eval_surface,
                        chunk_tokens=self.args.semantic_cache_chunk_tokens,
                        autocast_enabled=self.autocast_enabled,
                    ),
                    sequence_length=self.args.sequence_length,
                    batch_size=self.args.eval_batch_size,
                    seed=self.args.seed + 1,
                    horizons=self.config.mtp_horizons,
                ),
                batch_size=self.args.eval_batch_size,
            )

    def prepare_cached_sources(self) -> None:
        token_cache_dir = self.args.token_cache_dir or self.args.output_dir / "naz_token_cache"
        train_ids_path, train_lengths_path, train_token_count = build_token_cache(
            self.args.train_file,
            self.tokenizer,
            self.dil_config.max_surface_pieces_per_unit,
            self.dil_config.pad_token_id,
            self.args.text_read_chars,
            token_cache_dir,
        )
        train_semantic_path = build_memmap_semantic_cache(
            self.model,
            train_ids_path,
            train_lengths_path,
            train_token_count,
            chunk_tokens=self.args.semantic_cache_chunk_tokens,
            autocast_enabled=self.autocast_enabled,
            cache_dir=token_cache_dir,
            checksum=self.initial_dil_checksum,
        )
        self.train_iterator = MemmapNazSemanticBatcher(
            train_semantic_path,
            token_count=train_token_count,
            latent_size=self.config.latent_size,
            sequence_length=self.args.sequence_length,
            batch_size=self.args.batch_size,
            seed=self.args.seed + self.start_step,
            device=self.device,
            horizons=self.config.mtp_horizons,
        )
        if self.args.eval_every > 0:
            eval_ids_path, eval_lengths_path, eval_token_count = build_token_cache(
                self.args.eval_file,
                self.tokenizer,
                self.dil_config.max_surface_pieces_per_unit,
                self.dil_config.pad_token_id,
                self.args.text_read_chars,
                token_cache_dir,
            )
            eval_semantic_path = build_memmap_semantic_cache(
                self.model,
                eval_ids_path,
                eval_lengths_path,
                eval_token_count,
                chunk_tokens=self.args.semantic_cache_chunk_tokens,
                autocast_enabled=self.autocast_enabled,
                cache_dir=token_cache_dir,
                checksum=self.initial_dil_checksum,
            )
            self.eval_loader = MemmapNazSemanticEvalLoader(
                MemmapNazSemanticBatcher(
                    eval_semantic_path,
                    token_count=eval_token_count,
                    latent_size=self.config.latent_size,
                    sequence_length=self.args.sequence_length,
                    batch_size=self.args.eval_batch_size,
                    seed=self.args.seed + 1,
                    device=self.device,
                    horizons=self.config.mtp_horizons,
                ),
                batch_size=self.args.eval_batch_size,
            )

    def prepare_streaming_sources(self) -> None:
        train_loader = make_naz_loader(
            StreamingTextNazDataset(
                self.args.train_file,
                self.tokenizer,
                self.config,
                self.args.sequence_length,
                self.args.batch_size,
                self.args.text_read_chars,
                repeat=True,
            ),
            num_workers=self.args.num_workers,
            pin_memory=self.device.type == "cuda",
            prefetch_factor=self.args.prefetch_factor,
        )
        self.train_iterator = DeviceBatchPrefetcher(train_loader, self.device, self.cuda_prefetch)
        if self.args.eval_every > 0:
            self.eval_loader = make_naz_loader(
                StreamingTextNazDataset(
                    self.args.eval_file,
                    self.tokenizer,
                    self.config,
                    self.args.sequence_length,
                    self.args.eval_batch_size,
                    self.args.text_read_chars,
                    repeat=False,
                ),
                num_workers=self.args.num_workers,
                pin_memory=self.device.type == "cuda",
                prefetch_factor=self.args.prefetch_factor,
            )

    def build_train_iterator(self):
        return self.train_iterator

    def build_eval_iterator(self):
        if self.eval_loader is None:
            return None
        if self.args.data_mode in {"resident", "cached"}:
            return iter(self.eval_loader)
        return DeviceBatchPrefetcher(self.eval_loader, self.device, self.cuda_prefetch)

    def has_eval(self) -> bool:
        return self.eval_loader is not None

    def empty_metric_sums(self) -> dict[str, float]:
        return {
            "loss": 0.0,
            "reconstruction": 0.0,
            "mse": 0.0,
            "mixture_nll": 0.0,
            "responsibility": 0.0,
            "usage_balance": 0.0,
            "moe_balance": 0.0,
            "min_mse": 0.0,
            "chosen_mse": 0.0,
            "router_entropy": 0.0,
            "cosine_loss": 0.0,
            "latent_cos": 0.0,
            "candidate_usage_entropy": 0.0,
            "candidate_usage_max": 0.0,
            "moe_usage_entropy": 0.0,
            "moe_usage_max": 0.0,
            "targets": 0.0,
            "batches": 0,
        }

    def accumulate_metrics(self, total: dict[str, float], result: StepResult) -> None:
        add_output_metrics(total, result.outputs)

    def reduce_metrics(self, total: dict[str, float]) -> dict[str, float]:
        return reduce_output_metrics(total)

    def train_step(self, batch: dict, step: int) -> StepResult:
        outputs = run_naz_batch(self.model, batch, training_step=step)
        return StepResult(
            loss=outputs.loss,
            outputs=outputs,
            token_count=int(outputs.num_targets.detach().cpu()),
        )

    def save_checkpoint(self, checkpoint_name: str, step: int, metrics: dict[str, float]):
        return save_checkpoint(
            self.args.output_dir,
            self.model,
            self.optimizer,
            self.scheduler,
            self.config,
            self.stage,
            step,
            metrics,
            self.initial_dil_checksum,
            self.compile_mode,
            runtime_training_state(self.args),
            self.args.init_naz_checkpoint if self.stage == "finetune" else None,
            checkpoint_name=checkpoint_name,
        )

    def assert_checkpoint_integrity(self) -> None:
        if dil_checksum(self.model) != self.initial_dil_checksum:
            raise RuntimeError("frozen Dil checksum changed during training")

    def is_recoverable_runtime_error(self, error: RuntimeError) -> bool:
        return is_dataloader_worker_exit(error)

    def format_log(self, step: int, metrics: dict[str, float]) -> str:
        fields = [
            f"stage={self.stage}",
            f"step={step}",
            f"loss={metrics['loss']:.4f}",
            f"nll={metrics['mixture_nll']:.4f}",
            f"resp={metrics['responsibility']:.4f}",
            f"usage={metrics['usage_balance']:.4f}",
            f"moe={metrics['moe_balance']:.4f}",
            f"mse_sum={metrics['mse']:.4f}",
            f"mse_mean={metrics['mse_mean']:.4f}",
            f"min_mse={metrics['min_mse']:.4f}",
            f"chosen_mse={metrics['chosen_mse']:.4f}",
            f"router_h={metrics['router_entropy']:.4f}",
            f"candidate_h={metrics['candidate_usage_entropy']:.4f}",
            f"candidate_max={metrics['candidate_usage_max']:.4f}",
            f"moe_h={metrics['moe_usage_entropy']:.4f}",
            f"moe_max={metrics['moe_usage_max']:.4f}",
            f"cosine_loss={metrics['cosine_loss']:.4f}",
            f"latent_cos={metrics['latent_cos']:.4f}",
            f"target_count={metrics['targets']:.1f}",
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

    def run(self) -> None:
        print(
            f"device={self.device.type} bf16={int(self.autocast_enabled)} compile_mode={self.compile_mode} "
            f"stage={self.stage} data_mode={self.args.data_mode} objective={OBJECTIVE} resume_step={self.start_step}",
            flush=True,
        )
        super().run()


class NazPretrainTrainer(NazBaseTrainer):
    pass


class NazFinetuneTrainer(NazBaseTrainer):
    pass


def make_trainer(args) -> NazBaseTrainer:
    if args.resume is not None:
        validate_args(args)
    return NazFinetuneTrainer(args) if args.stage == "finetune" else NazPretrainTrainer(args)


def main(argv: list[str] | None = None):
    trainer = make_trainer(parse_args(argv))
    trainer.run()


if __name__ == "__main__":
    main()
