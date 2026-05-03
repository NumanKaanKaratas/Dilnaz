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
from tokenization import HybridTokenizer


RIGHT_ATTACHED = set(".,;:!?%)]}")
LEFT_ATTACHED = set("([{")
CHECKPOINT_FORMAT_VERSION = 10


def tokenize_text(text: str, tokenizer: HybridTokenizer) -> list[str]:
    tokens = [
        segment.text
        for segment in tokenizer.encode_segments(text)
        if segment.kind != "space"
    ]
    if not tokens:
        raise ValueError("text produced no tokens")
    return tokens


def join_tokens(tokens: list[str]) -> str:
    text = ""
    for token in tokens:
        if not text:
            text = token
        elif token in RIGHT_ATTACHED:
            text += token
        elif text[-1] in LEFT_ATTACHED:
            text += token
        else:
            text += " " + token
    return text


def format_next_token(current_text: str, token: str) -> str:
    if not current_text:
        return token
    if token in RIGHT_ATTACHED:
        return token
    if current_text[-1] in LEFT_ATTACHED:
        return token
    return " " + token


def make_batch(tokens: list[str], tokenizer: HybridTokenizer, config: NazConfig, device: torch.device):
    input_ids = torch.full(
        (1, len(tokens), config.max_word_bytes),
        config.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    word_masks = torch.zeros(
        (1, len(tokens), config.max_word_bytes),
        dtype=torch.bool,
        device=device,
    )
    byte_lengths = []
    for unit_idx, token in enumerate(tokens):
        segment = tokenizer.encode_segments(token)[0]
        token_ids = segment.token_ids
        if len(token_ids) > config.max_word_bytes:
            raise ValueError(
                f"token {unit_idx} '{token}' has {len(token_ids)} pieces; "
                f"max_word_bytes={config.max_word_bytes}"
            )
        ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        input_ids[0, unit_idx, : ids.numel()] = ids
        word_masks[0, unit_idx, : ids.numel()] = True
        byte_lengths.append(ids.numel())
    unit_mask = torch.ones((1, len(tokens)), dtype=torch.bool, device=device)
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
    dil_path = Path(config.dil_path)
    if not dil_path.is_absolute():
        dil_path = (checkpoint_dir / dil_path).resolve()
    config.dil_path = str(dil_path)
    model = Naz(config).to(device)
    model.load_trainable_state_dict(checkpoint["model_state_dict"])
    model.dil_model.set_compiled_forwards(
        encoder_forward=compile_forward(model.dil_model.encoder.forward, compile_mode, "DilEncoderCore"),
        decode_forward=compile_forward(model.dil_model._decode_from_latents_impl, compile_mode, "DilDecoderRenderer"),
    )
    model.eval()
    dil_config = DilConfig.from_pretrained(dil_path)
    tokenizer = HybridTokenizer.from_file(dil_path / dil_config.tokenizer_vocab_file)
    return model, config, tokenizer, checkpoint["training_state"]["step"]


@torch.no_grad()
def generate_text(
    model: Naz,
    config: NazConfig,
    tokenizer: HybridTokenizer,
    prompt_tokens: list[str],
    device: torch.device,
    max_new_tokens: int,
    num_samples: int,
):
    input_ids, word_masks, unit_mask, _ = make_batch(prompt_tokens, tokenizer, config, device)
    outputs = model.generate(
        input_ids=input_ids,
        word_masks=word_masks,
        unit_mask=unit_mask,
        max_new_tokens=max_new_tokens,
        num_samples=num_samples,
    )
    tokens = [
        decode_token_ids(tokenizer, outputs.sequences[0, idx], outputs.word_masks[0, idx])
        for idx in range(outputs.sequences.shape[1])
    ]
    return join_tokens(tokens)


def parse_flush_schedule(value: str) -> list[int]:
    schedule = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not schedule or any(item <= 0 for item in schedule):
        raise ValueError("--decode-flush-schedule must contain positive integers")
    return schedule


@torch.no_grad()
def stream_text(
    model: Naz,
    config: NazConfig,
    tokenizer: HybridTokenizer,
    prompt_tokens: list[str],
    device: torch.device,
    max_new_tokens: int,
    num_samples: int,
    flush_schedule: list[int],
):
    del num_samples
    input_ids, word_masks, unit_mask, _ = make_batch(prompt_tokens, tokenizer, config, device)
    current_text = join_tokens(prompt_tokens)
    sys.stdout.write(current_text)
    sys.stdout.flush()

    pending_latents = []
    schedule_idx = 0
    flush_size = flush_schedule[schedule_idx]

    def flush_pending():
        nonlocal current_text
        if not pending_latents:
            return
        latents = torch.stack(pending_latents, dim=0).unsqueeze(0)
        token_ids, masks, _ = model.decode_latent_tokens(latents)
        for token_idx in range(token_ids.shape[1]):
            token = decode_token_ids(tokenizer, token_ids[0, token_idx], masks[0, token_idx])
            if not token:
                continue
            piece = format_next_token(current_text, token)
            sys.stdout.write(piece)
            sys.stdout.flush()
            current_text += piece
        pending_latents.clear()

    with torch.inference_mode():
        current_input_embeds = model.semantic_embeddings(input_ids, word_masks, unit_mask)
        past_key_values = None
        for _ in range(max_new_tokens):
            outputs = model.transformer(
                inputs_embeds=current_input_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_mean, next_log_std = model.generative_head.sample_distribution(
                outputs.last_hidden_state[:, -1, :]
            )
            pending_latents.append(next_mean.squeeze(0))
            current_input_embeds = model.student_core.embed_distribution(
                next_mean,
                next_log_std,
            ).unsqueeze(1)

            if len(pending_latents) >= flush_size:
                flush_pending()
                if schedule_idx < len(flush_schedule) - 1:
                    schedule_idx += 1
                    flush_size = flush_schedule[schedule_idx]

        flush_pending()

    sys.stdout.write("\n")
    sys.stdout.flush()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--text-file", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--compile-mode", choices=COMPILE_MODE_CHOICES, default="off")
    parser.add_argument("--decode-flush-schedule", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--no-stream", action="store_true")
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
    tokens = tokenize_text(text, tokenizer)

    if args.no_stream:
        generated_text = generate_text(
            model,
            config,
            tokenizer,
            tokens,
            device,
            args.max_new_tokens,
            args.num_samples,
        )
        print(generated_text)
        return

    stream_text(
        model,
        config,
        tokenizer,
        tokens,
        device,
        args.max_new_tokens,
        args.num_samples,
        parse_flush_schedule(args.decode_flush_schedule),
    )


if __name__ == "__main__":
    main()

