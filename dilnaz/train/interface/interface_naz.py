import argparse
import sys
from pathlib import Path

import torch

from dilnaz.train.common.runtime import COMPILE_MODE_CHOICES, compile_forward, validate_compile_environment
from dilnaz.models.dil import DilConfig
from dilnaz.models.naz import NazConfig
from dilnaz.models.naz import Naz
from dilnaz.surface import pack_token_units
from dilnaz.train.interface.writer_buffer import UnitWriterBuffer
from dilnaz.tokenization import HybridTokenizer, TokenSegment


CHECKPOINT_FORMAT_VERSION = 27
OBJECTIVE = "semantic_dynamics_moe_mtp_v1"


def tokenize_text(text: str, tokenizer: HybridTokenizer) -> list[TokenSegment]:
    segments = [
        segment
        for segment in tokenizer.encode_segments(text)
        if segment.piece_len > 0
    ]
    if not segments:
        raise ValueError("text produced no tokens")
    return segments


def make_batch(segments: list[TokenSegment], tokenizer: HybridTokenizer, config: DilConfig, device: torch.device):
    rows: list[list[list[int]]] = [[]]
    byte_lengths = []
    for unit_idx, segment in enumerate(segments):
        token_ids = segment.token_ids
        if len(token_ids) > config.max_surface_pieces_per_unit:
            raise ValueError(
                f"token {unit_idx} {tokenizer.decode(token_ids)!r} has {len(token_ids)} pieces; "
                f"max_surface_pieces_per_unit={config.max_surface_pieces_per_unit}"
            )
        rows[0].append(list(token_ids))
        byte_lengths.append(len(token_ids))
    unit_mask = torch.ones((1, len(segments)), dtype=torch.bool, device=device)
    surface = pack_token_units(
        rows,
        pad_token_id=config.pad_token_id,
        bucket_sizes=config.surface_bucket_sizes,
        max_pieces_per_unit=config.max_surface_pieces_per_unit,
        device=device,
    )
    return surface, unit_mask, byte_lengths


def load_model(checkpoint_dir: Path, device: torch.device, compile_mode: str):
    checkpoint_dir = checkpoint_dir.resolve()
    config = NazConfig.from_pretrained(checkpoint_dir)
    checkpoint = torch.load(
        checkpoint_dir / "checkpoint.pt",
        map_location=device,
        weights_only=False,
    )
    if checkpoint["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format_version={checkpoint.get('format_version')}")
    if checkpoint["training_state"].get("objective") != OBJECTIVE:
        raise ValueError(f"checkpoint objective is not {OBJECTIVE}")
    dil_path = Path(config.dil_path)
    if not dil_path.is_absolute():
        cwd_relative = dil_path.resolve()
        checkpoint_relative = (checkpoint_dir / dil_path).resolve()
        dil_path = cwd_relative if cwd_relative.exists() else checkpoint_relative
    config.dil_path = str(dil_path)
    model = Naz(config).to(device)
    model.load_trainable_state_dict(checkpoint["model_state_dict"])
    model.dil_model.set_compiled_forwards(
        encoder_forward=compile_forward(model.dil_model.encoder.forward, compile_mode, "DilEncoderCore"),
        writer_forward=compile_forward(model.dil_model.writer.forward, compile_mode, "DilConditionalWriter"),
    )
    model.eval()
    dil_config = DilConfig.from_pretrained(dil_path)
    tokenizer = HybridTokenizer.from_file(dil_path / dil_config.tokenizer_vocab_file)
    return model, config, tokenizer, checkpoint["training_state"]["step"]


@torch.no_grad()
def stream_text(
    model: Naz,
    config: NazConfig,
    tokenizer: HybridTokenizer,
    prompt_segments: list[TokenSegment],
    device: torch.device,
    max_new_tokens: int,
    min_new_tokens: int,
    repetition_cos_threshold: float,
    writer_microbatch_size: int,
):
    if max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0")
    if writer_microbatch_size <= 0:
        raise ValueError("--writer-microbatch-size must be > 0")
    surface, unit_mask, _ = make_batch(prompt_segments, tokenizer, model.dil_model.config, device)
    prompt_text = "".join(tokenizer.decode(segment.token_ids) for segment in prompt_segments)
    sys.stdout.write(prompt_text)
    sys.stdout.flush()

    writer_buffer = UnitWriterBuffer(model, model.dil_model.config, tokenizer, microbatch_size=writer_microbatch_size)
    prompt_latents = None
    if hasattr(model, "encode_sequence_latents"):
        prompt_latents = model.encode_sequence_latents(surface, unit_mask)
        writer_buffer.seed_prompt(prompt_latents, prompt_segments)

    for step in model.generate_stream(
        surface=surface,
        unit_mask=unit_mask,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        repetition_cos_threshold=repetition_cos_threshold,
        prompt_latents=prompt_latents,
    ):
        writer_buffer.append(
            step.latent,
            bool(step.should_stop[0].detach().cpu()),
        )
        if writer_buffer.flush(force=False):
            break
        if writer_buffer.pending_should_stop and writer_buffer.pending_should_stop[-1]:
            if writer_buffer.flush(force=True):
                break
    else:
        writer_buffer.flush(force=True)
    sys.stdout.write("\n")
    sys.stdout.flush()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--text-file", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--min-new-tokens", type=int, default=None)
    parser.add_argument("--repetition-cos-threshold", type=float, default=None)
    parser.add_argument("--writer-microbatch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--compile-mode", choices=COMPILE_MODE_CHOICES, default="off")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    validate_compile_environment(args.compile_mode)

    model, config, tokenizer, _ = load_model(args.checkpoint_dir, device, args.compile_mode)
    text = args.text_file.read_text(encoding="utf-8") if args.text_file else args.text
    if text is None:
        raise ValueError("--text or --text-file is required")
    segments = tokenize_text(text, tokenizer)
    stream_text(
        model,
        config,
        tokenizer,
        segments,
        device,
        args.max_new_tokens,
        config.min_new_tokens if args.min_new_tokens is None else args.min_new_tokens,
        config.repetition_cos_threshold if args.repetition_cos_threshold is None else args.repetition_cos_threshold,
        args.writer_microbatch_size,
    )


if __name__ == "__main__":
    main()
