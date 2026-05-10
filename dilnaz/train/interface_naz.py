import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from byte_trainer_utils import COMPILE_MODE_CHOICES, compile_forward, validate_compile_environment
from models.configuration_dil import DilConfig
from models.configuration_naz import NazConfig
from models.modeling_naz import Naz
from tokenization import HybridTokenizer, TokenSegment


CHECKPOINT_FORMAT_VERSION = 23
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


def make_batch(segments: list[TokenSegment], tokenizer: HybridTokenizer, config: NazConfig, device: torch.device):
    input_ids = torch.full(
        (1, len(segments), config.max_word_bytes),
        config.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    word_masks = torch.zeros(
        (1, len(segments), config.max_word_bytes),
        dtype=torch.bool,
        device=device,
    )
    byte_lengths = []
    for unit_idx, segment in enumerate(segments):
        token_ids = segment.token_ids
        if len(token_ids) > config.max_word_bytes:
            raise ValueError(
                f"token {unit_idx} {tokenizer.decode(token_ids)!r} has {len(token_ids)} pieces; "
                f"max_word_bytes={config.max_word_bytes}"
            )
        ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        input_ids[0, unit_idx, : ids.numel()] = ids
        word_masks[0, unit_idx, : ids.numel()] = True
        byte_lengths.append(ids.numel())
    unit_mask = torch.ones((1, len(segments)), dtype=torch.bool, device=device)
    return input_ids, word_masks, unit_mask, byte_lengths


def decode_token_ids(tokenizer: HybridTokenizer, token_ids: torch.Tensor, token_mask: torch.Tensor) -> str:
    ids = token_ids[token_mask].detach().cpu().tolist()
    return tokenizer.decode(ids)


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
):
    if max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0")
    input_ids, word_masks, unit_mask, _ = make_batch(prompt_segments, tokenizer, config, device)
    prompt_text = "".join(tokenizer.decode(segment.token_ids) for segment in prompt_segments)
    sys.stdout.write(prompt_text)
    sys.stdout.flush()
    for step in model.generate_stream(
        input_ids=input_ids,
        word_masks=word_masks,
        unit_mask=unit_mask,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        repetition_cos_threshold=repetition_cos_threshold,
    ):
        token_ids, token_masks, lengths = model.dil_model.decode_semantic(step.latent)
        if int(lengths[0].detach().cpu()) == 0:
            break
        token = decode_token_ids(tokenizer, token_ids[0], token_masks[0])
        if not token and bool(step.should_stop[0].detach().cpu()):
            break
        if token:
            sys.stdout.write(token)
            sys.stdout.flush()
        if bool(step.should_stop[0].detach().cpu()):
            break
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
    )


if __name__ == "__main__":
    main()
