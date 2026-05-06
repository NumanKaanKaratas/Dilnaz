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


CHECKPOINT_FORMAT_VERSION = 13


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


@torch.no_grad()
def delayed_prompt_state(
    model: Naz,
    input_ids: torch.LongTensor,
    word_masks: torch.Tensor,
    unit_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    prompt_embeddings = model.semantic_embeddings(input_ids, word_masks, unit_mask)
    sequence_length = prompt_embeddings.shape[1]
    forced_embeddings = prompt_embeddings[:, sequence_length:sequence_length]
    return prompt_embeddings, forced_embeddings, 0


def generate_latent_steps(
    model: Naz,
    prefill_embeddings: torch.Tensor,
    forced_embeddings: torch.Tensor,
    step_count: int,
):
    outputs = model.transformer(
        inputs_embeds=prefill_embeddings,
        past_key_values=None,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    hidden_state = outputs.last_hidden_state[:, -1, :]

    for step_idx in range(step_count):
        next_mean, next_log_std = model.generative_head.sample_distribution(hidden_state)
        next_mean, next_log_std = model.dil_model.guard_normalized_distribution(next_mean, next_log_std)
        yield next_mean, next_log_std

        if step_idx < forced_embeddings.shape[1]:
            current_input_embeds = forced_embeddings[:, step_idx : step_idx + 1]
        else:
            current_input_embeds = model.student_core.embed_distribution(
                next_mean,
                next_log_std,
            ).unsqueeze(1)

        outputs = model.transformer(
            inputs_embeds=current_input_embeds,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        hidden_state = outputs.last_hidden_state[:, -1, :]


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
        cwd_relative = dil_path.resolve()
        checkpoint_relative = (checkpoint_dir / dil_path).resolve()
        dil_path = cwd_relative if cwd_relative.exists() else checkpoint_relative
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
    prompt_segments: list[TokenSegment],
    device: torch.device,
    max_new_tokens: int,
    num_samples: int,
):
    input_ids, word_masks, unit_mask, _ = make_batch(prompt_segments, tokenizer, config, device)
    del num_samples
    prefill_embeddings, forced_embeddings, warmup_tokens = delayed_prompt_state(
        model,
        input_ids,
        word_masks,
        unit_mask,
    )
    visible_latents = [
        next_mean.squeeze(0)
        for step_idx, (next_mean, _) in enumerate(
            generate_latent_steps(
                model,
                prefill_embeddings,
                forced_embeddings,
                warmup_tokens + max_new_tokens,
            )
        )
        if step_idx >= warmup_tokens
    ]
    if not visible_latents:
        return "".join(tokenizer.decode(segment.token_ids) for segment in prompt_segments)
    latents = torch.stack(visible_latents, dim=0).unsqueeze(0)
    token_ids, masks, _ = model.decode_latent_tokens(latents)
    generated_tokens = [
        decode_token_ids(tokenizer, token_ids[0, token_idx], masks[0, token_idx])
        for token_idx in range(token_ids.shape[1])
    ]
    prompt_text = "".join(tokenizer.decode(segment.token_ids) for segment in prompt_segments)
    return prompt_text + "".join(generated_tokens)


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
    prompt_segments: list[TokenSegment],
    device: torch.device,
    max_new_tokens: int,
    num_samples: int,
    flush_schedule: list[int],
):
    del num_samples
    input_ids, word_masks, unit_mask, _ = make_batch(prompt_segments, tokenizer, config, device)
    current_text = tokenizer.decode([piece.token_id for segment in prompt_segments for piece in segment.pieces])
    sys.stdout.write(current_text)
    sys.stdout.flush()

    pending_latents = []
    schedule_idx = 0
    flush_size = flush_schedule[schedule_idx]
    prefill_embeddings, forced_embeddings, warmup_tokens = delayed_prompt_state(
        model,
        input_ids,
        word_masks,
        unit_mask,
    )

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
            sys.stdout.write(token)
            sys.stdout.flush()
            current_text += token
        pending_latents.clear()

    with torch.inference_mode():
        latent_steps = generate_latent_steps(
            model,
            prefill_embeddings,
            forced_embeddings,
            warmup_tokens + max_new_tokens,
        )
        for step_idx, (next_mean, _) in enumerate(latent_steps):
            if step_idx < warmup_tokens:
                continue
            pending_latents.append(next_mean.squeeze(0))

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
    segments = tokenize_text(text, tokenizer)

    if args.no_stream:
        generated_text = generate_text(
            model,
            config,
            tokenizer,
            segments,
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
        segments,
        device,
        args.max_new_tokens,
        args.num_samples,
        parse_flush_schedule(args.decode_flush_schedule),
    )


if __name__ == "__main__":
    main()

