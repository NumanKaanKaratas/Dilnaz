from __future__ import annotations

import argparse
import shutil
import tempfile
import time

import torch

from dilnaz.models.dil import DilConfig
from dilnaz.models.naz import NazConfig
from dilnaz.models.dil import Dil
from dilnaz.models.naz import Naz


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_dil_config(args) -> DilConfig:
    return DilConfig(
        vocab_size=args.vocab_size,
        pad_token_id=0,
        eos_token_id=1,
        hidden_size=args.dil_hidden_size,
        intermediate_size=args.dil_intermediate_size,
        num_encoder_layers=args.dil_layers,
        latent_size=args.latent_size,
        max_word_bytes=args.max_word_bytes,
        context_radius=args.context_radius,
        dil_dropout=0.0,
    )


def make_naz_config(args, dil_path: Path) -> NazConfig:
    return NazConfig(
        dil_path=str(dil_path),
        vocab_size=args.vocab_size,
        pad_token_id=0,
        eos_token_id=1,
        max_word_bytes=args.max_word_bytes,
        latent_size=args.latent_size,
        hidden_size=args.naz_hidden_size,
        intermediate_size=args.naz_intermediate_size,
        num_hidden_layers=args.naz_layers,
        num_attention_heads=args.naz_heads,
        num_key_value_heads=args.naz_kv_heads,
        head_dim=args.naz_head_dim,
        full_attention_interval=args.full_attention_interval,
        linear_key_head_dim=args.linear_head_dim,
        linear_value_head_dim=args.linear_head_dim,
        linear_num_key_heads=args.linear_heads,
        linear_num_value_heads=args.linear_heads,
        linear_conv_kernel_size=4,
        mtp_horizons=args.horizons,
        mtp_loss_weights=tuple(1.0 / (horizon + 1) for horizon in range(args.horizons)),
        moe_num_experts=args.moe_experts,
        moe_top_k=args.moe_top_k,
        moe_layers=args.moe_layers,
        moe_expert_intermediate_size=args.naz_intermediate_size,
    )


def save_frozen_dil_checkpoint(dil_config: DilConfig, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    dil_config.save_pretrained(path)
    model = Dil(dil_config)
    torch.save(
        {
            "format_version": dil_config.checkpoint_format_version,
            "model_state_dict": model.state_dict(),
        },
        path / "checkpoint.pt",
    )


def synthetic_batch(args, device: torch.device) -> dict[str, torch.Tensor]:
    extended_length = args.seq + args.horizons
    ids = torch.full(
        (args.batch_size, extended_length, args.max_word_bytes),
        0,
        dtype=torch.long,
        device=device,
    )
    lengths = torch.randint(
        1,
        min(args.max_word_bytes, 8) + 1,
        (args.batch_size, extended_length),
        dtype=torch.long,
        device=device,
    )
    random_ids = torch.randint(
        2,
        args.vocab_size,
        ids.shape,
        dtype=torch.long,
        device=device,
    )
    positions = torch.arange(args.max_word_bytes, device=device).view(1, 1, -1)
    masks = positions < lengths.unsqueeze(-1)
    ids[masks] = random_ids[masks]

    horizon_positions = (
        torch.arange(args.seq, device=device).view(1, args.seq, 1)
        + torch.arange(1, args.horizons + 1, device=device).view(1, 1, args.horizons)
    )
    target_input_ids = ids.gather(
        dim=1,
        index=horizon_positions.reshape(1, args.seq * args.horizons, 1).expand(
            args.batch_size,
            -1,
            args.max_word_bytes,
        ),
    ).reshape(args.batch_size, args.seq, args.horizons, args.max_word_bytes)
    target_word_masks = masks.gather(
        dim=1,
        index=horizon_positions.reshape(1, args.seq * args.horizons, 1).expand(
            args.batch_size,
            -1,
            args.max_word_bytes,
        ),
    ).reshape(args.batch_size, args.seq, args.horizons, args.max_word_bytes)
    return {
        "input_ids": ids[:, : args.seq],
        "word_masks": masks[:, : args.seq],
        "target_input_ids": target_input_ids,
        "target_word_masks": target_word_masks,
        "unit_mask": torch.ones(args.batch_size, args.seq, dtype=torch.bool, device=device),
        "target_mask": torch.ones(args.batch_size, args.seq, args.horizons, dtype=torch.bool, device=device),
    }


def run_benchmark(args) -> None:
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float16 if device.type == "cuda" else torch.float32
    temp_root = Path(tempfile.mkdtemp(prefix="naz_pipeline_bench_"))
    try:
        dil_path = temp_root / "Dil"
        dil_config = make_dil_config(args)
        save_frozen_dil_checkpoint(dil_config, dil_path)
        model = Naz(make_naz_config(args, dil_path)).to(device=device)
        model.train()
        batch = synthetic_batch(args, device)

        def run_once() -> None:
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                outputs = model(**batch)
            outputs.loss.backward()
            model.zero_grad(set_to_none=True)

        for _ in range(args.warmup):
            run_once()
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(args.iters):
            run_once()
        synchronize(device)
        elapsed = time.perf_counter() - start
        tokens = args.batch_size * args.seq * args.iters
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024 if device.type == "cuda" else 0.0
        print(
            f"pipeline_train batch={args.batch_size} seq={args.seq} horizons={args.horizons} "
            f"tokens_per_sec={tokens / elapsed:.2f} step_ms={(elapsed / args.iters) * 1000:.2f} "
            f"peak_mb={peak_mb:.1f} dtype={dtype}"
        )
    finally:
        shutil.rmtree(temp_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark live DIL -> NAZ -> loss -> backward pipeline.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq", type=int, default=258)
    parser.add_argument("--horizons", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=778)
    parser.add_argument("--max-word-bytes", type=int, default=32)
    parser.add_argument("--context-radius", type=int, default=2)
    parser.add_argument("--latent-size", type=int, default=512)
    parser.add_argument("--dil-hidden-size", type=int, default=512)
    parser.add_argument("--dil-intermediate-size", type=int, default=1280)
    parser.add_argument("--dil-layers", type=int, default=6)
    parser.add_argument("--naz-hidden-size", type=int, default=512)
    parser.add_argument("--naz-intermediate-size", type=int, default=2752)
    parser.add_argument("--naz-layers", type=int, default=12)
    parser.add_argument("--naz-heads", type=int, default=8)
    parser.add_argument("--naz-kv-heads", type=int, default=2)
    parser.add_argument("--naz-head-dim", type=int, default=64)
    parser.add_argument("--linear-heads", type=int, default=8)
    parser.add_argument("--linear-head-dim", type=int, default=64)
    parser.add_argument("--full-attention-interval", type=int, default=4)
    parser.add_argument("--moe-experts", type=int, default=8)
    parser.add_argument("--moe-top-k", type=int, default=2)
    parser.add_argument("--moe-layers", type=int, default=4)
    run_benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
