from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dilnaz"))

from models.configuration_naz import NazConfig
from models.naz_backbone import NazSemanticBackbone


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def dependency_status() -> dict[str, str]:
    status = {
        name: "ok" if importlib.util.find_spec(name) is not None else "missing"
        for name in ("flash_attn", "causal_conv1d", "fla", "xformers", "triton")
    }
    try:
        status["flash-linear-attention"] = version("flash-linear-attention")
    except PackageNotFoundError:
        status["flash-linear-attention"] = "missing"
    return status


def probe_sdpa_backends(device: torch.device):
    if device.type != "cuda":
        return
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = torch.randn(2, 8, 16, 64, device=device, dtype=torch.float16)
    k = torch.randn(2, 2, 16, 64, device=device, dtype=torch.float16)
    v = torch.randn(2, 2, 16, 64, device=device, dtype=torch.float16)
    mask = torch.ones(2, 1, 16, 16, device=device, dtype=torch.bool)
    for backend in (SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH):
        try:
            with sdpa_kernel(backends=[backend]):
                F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
            synchronize(device)
            print(f"sdpa_backend={backend.name} gqa=ok")
        except Exception as exc:  # pragma: no cover - hardware/backend report
            print(f"sdpa_backend={backend.name} gqa=fail reason={type(exc).__name__}: {str(exc).splitlines()[0]}")


def probe_fla_gated_delta(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda":
        print("fla_gdn=skip reason=cuda_required")
        return
    if importlib.util.find_spec("fla") is None:
        print("fla_gdn=skip reason=fla_missing")
        return
    import torch.nn.functional as F
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

    batch_size, sequence_length, num_heads, head_dim = 2, 64, 4, 32
    value_heads, value_dim = num_heads, head_dim
    q = torch.randn(batch_size, sequence_length, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = F.normalize(
        torch.randn(batch_size, sequence_length, num_heads, head_dim, device=device, dtype=dtype),
        p=2.0,
        dim=-1,
    ).detach().requires_grad_(True)
    v = torch.randn(batch_size, sequence_length, value_heads, value_dim, device=device, dtype=dtype, requires_grad=True)
    g = F.logsigmoid(torch.randn(batch_size, sequence_length, value_heads, device=device, dtype=dtype)).detach().requires_grad_(True)
    beta = torch.rand(batch_size, sequence_length, value_heads, device=device, dtype=dtype, requires_grad=True)

    chunk_output, chunk_state = chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=True)
    loss = chunk_output.float().square().mean()
    loss.backward()
    synchronize(device)

    with torch.no_grad():
        recurrent_output, recurrent_state = fused_recurrent_gated_delta_rule(
            q.detach(),
            k.detach(),
            v.detach(),
            g.detach(),
            beta.detach(),
            output_final_state=True,
        )
    synchronize(device)
    print(
        "fla_gdn=ok "
        f"chunk_output={tuple(chunk_output.shape)} chunk_state={tuple(chunk_state.shape)} "
        f"recurrent_output={tuple(recurrent_output.shape)} recurrent_state={tuple(recurrent_state.shape)} "
        f"chunk_finite={bool(torch.isfinite(chunk_output).all())} "
        f"recurrent_finite={bool(torch.isfinite(recurrent_output).all())}"
    )


def make_config(sequence_length: int) -> NazConfig:
    return NazConfig(
        dil_path="benchmark-only",
        hidden_size=512,
        intermediate_size=2752,
        num_hidden_layers=12,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        full_attention_interval=4,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_num_key_heads=8,
        linear_num_value_heads=8,
        linear_conv_kernel_size=4,
        max_position_embeddings=max(32768, sequence_length),
    )


def make_model(sequence_length: int, device: torch.device, dtype: torch.dtype) -> NazSemanticBackbone:
    model = NazSemanticBackbone(make_config(sequence_length)).to(device=device, dtype=dtype)
    return model


def train_step_benchmark(args, sequence_length: int, device: torch.device, dtype: torch.dtype):
    model = make_model(sequence_length, device, dtype).train()
    unit_mask = torch.ones(args.batch_size, sequence_length, dtype=torch.bool, device=device)

    def run_once():
        inputs = torch.randn(
            args.batch_size,
            sequence_length,
            model.config.hidden_size,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        output = model(inputs_embeds=inputs, attention_mask=unit_mask, use_cache=False).last_hidden_state
        loss = output.float().square().mean()
        loss.backward()
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
    tokens = args.batch_size * sequence_length * args.iters
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024 if device.type == "cuda" else 0.0
    print(
        f"train seq={sequence_length} batch={args.batch_size} tokens_per_sec={tokens / elapsed:.2f} "
        f"step_ms={(elapsed / args.iters) * 1000:.2f} peak_mb={peak_mb:.1f}"
    )


@torch.no_grad()
def cached_generate_benchmark(args, sequence_length: int, device: torch.device, dtype: torch.dtype):
    model = make_model(sequence_length + args.generate_tokens, device, dtype).eval()
    prompt = torch.randn(args.batch_size, sequence_length, model.config.hidden_size, device=device, dtype=dtype)
    prompt_mask = torch.ones(args.batch_size, sequence_length, dtype=torch.bool, device=device)

    def run_once():
        outputs = model(
            inputs_embeds=prompt,
            attention_mask=prompt_mask,
            use_cache=True,
            max_cache_length=sequence_length + args.generate_tokens,
        )
        cache = outputs.past_key_values
        current = torch.randn(args.batch_size, 1, model.config.hidden_size, device=device, dtype=dtype)
        current_mask = torch.ones(args.batch_size, 1, dtype=torch.bool, device=device)
        for _ in range(args.generate_tokens):
            outputs = model(
                inputs_embeds=current,
                attention_mask=current_mask,
                past_key_values=cache,
                use_cache=True,
            )
            cache = outputs.past_key_values
            current = outputs.last_hidden_state[:, -1:].detach()

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
    total_new_tokens = args.batch_size * args.generate_tokens * args.iters
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024 if device.type == "cuda" else 0.0
    print(
        f"generate prompt={sequence_length} new={args.generate_tokens} batch={args.batch_size} "
        f"new_tokens_per_sec={total_new_tokens / elapsed:.2f} token_ms={(elapsed / total_new_tokens) * 1000:.3f} "
        f"peak_mb={peak_mb:.1f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark NAZ SDPA attention and preallocated KV cache.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--generate-tokens", type=int, default=32)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--probe-fla-gdn", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"device={device} dtype={dtype} torch={torch.__version__}")
    if device.type == "cuda":
        print(f"cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(device)} capability={torch.cuda.get_device_capability(device)}")
    print("deps=" + " ".join(f"{name}:{state}" for name, state in dependency_status().items()))
    probe_sdpa_backends(device)
    if args.probe_fla_gdn:
        probe_fla_gated_delta(device, dtype)
    if args.probe_only:
        return

    for sequence_length in args.seq:
        train_step_benchmark(args, sequence_length, device, dtype)
        cached_generate_benchmark(args, sequence_length, device, dtype)


if __name__ == "__main__":
    main()
