"""Benchmark NAZ semantic-latent throughput (writer excluded)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from dilnaz.models.naz import Naz, NazConfig
from dilnaz.models.dil import DilConfig
from dilnaz.tokenization import HybridTokenizer, TokenSegment
from dilnaz.train.interface.interface_naz import make_batch, tokenize_text


def benchmark(
    checkpoint_dir: Path,
    device: torch.device,
    prompt_text: str,
    max_new_tokens: int,
):
    config = NazConfig.from_pretrained(checkpoint_dir)
    checkpoint = torch.load(
        checkpoint_dir / "checkpoint.pt",
        map_location=device,
        weights_only=False,
    )
    model = Naz(config).to(device)
    model.load_trainable_state_dict(checkpoint["model_state_dict"])
    if device.type == "cuda":
        model = model.to(torch.bfloat16)
    print("DEBUG: model.dtype =", model.dtype)
    print("DEBUG: dil_model.embed dtype =", model.dil_model.encoder.embed_tokens.weight.dtype)
    dil_path = Path(config.dil_path)
    if not dil_path.is_absolute():
        cwd_relative = dil_path.resolve()
        checkpoint_relative = (checkpoint_dir / dil_path).resolve()
        dil_path = cwd_relative if cwd_relative.exists() else checkpoint_relative
    dil_config = DilConfig.from_pretrained(dil_path)
    tokenizer = HybridTokenizer.from_file(dil_path / dil_config.tokenizer_vocab_file)
    model.eval()

    segments = tokenize_text(prompt_text, tokenizer)
    surface, unit_mask, _ = make_batch(segments, tokenizer, model.dil_model.config, device)

    prompt_latents = model.encode_sequence_latents(surface, unit_mask).to(model.dtype)
    print("DEBUG: prompt_latents dtype =", prompt_latents.dtype)

    # Warm-up
    with torch.inference_mode():
        for _ in model.generate_stream(
            surface=surface,
            unit_mask=unit_mask,
            max_new_tokens=10,
            min_new_tokens=0,
            repetition_cos_threshold=0.95,
            prompt_latents=prompt_latents,
        ):
            pass

    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()

    step_count = 0
    with torch.inference_mode():
        for _ in model.generate_stream(
            surface=surface,
            unit_mask=unit_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=0,
            repetition_cos_threshold=0.95,
            prompt_latents=prompt_latents,
        ):
            step_count += 1

    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.perf_counter() - start

    print(f"device={device}")
    print(f"steps={step_count}")
    print(f"elapsed={elapsed:.3f}s")
    print(f"tokens/s={step_count / elapsed:.1f}")
    print(f"ms/token={1000 * elapsed / step_count:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompt", type=str, default="Atatürk kimdir ?")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    benchmark(args.checkpoint_dir, device, args.prompt, args.max_new_tokens)
