from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from dilnaz.train.common.runtime import COMPILE_MODE_CHOICES, compile_forward, validate_compile_environment
from dilnaz.train.data.dil_data import align_spans_to_pieces, apply_teacher_centered_add, build_nllb_layer_groups, context_windows
from dilnaz.models.dil import DilConfig, split_factorized_latent
from dilnaz.models.dil import Dil
from dilnaz.surface import pack_token_units
from dilnaz.tokenization import HybridTokenizer, TokenSegment


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
    context_radius = getattr(config, "context_radius", 2)
    windows = context_windows(segments, context_radius)
    rows = []
    byte_lengths = []
    for window in windows:
        row = [list(seg.token_ids) if seg is not None else [] for seg in window]
        rows.append(row)
        byte_lengths.append(len(segments[len(byte_lengths)].token_ids) if len(byte_lengths) < len(segments) else 0)

    return pack_token_units(
        rows,
        pad_token_id=config.pad_token_id,
        bucket_sizes=config.surface_bucket_sizes,
        max_pieces_per_unit=config.max_surface_pieces_per_unit,
        device=device,
    ), byte_lengths


def is_word_token(token: str) -> bool:
    return any(ch.isalnum() or ch == "_" for ch in token)


def find_surface_pair(tokens: list[str], indices: list[int]) -> tuple[int, int] | None:
    seen = {}
    for idx in indices:
        key = tokens[idx].casefold()
        if key in seen:
            return seen[key], idx
        seen[key] = idx
    return None


def find_semantic_pair(indices: list[int], similarities: list[list[float]]) -> tuple[int, int]:
    return max(
        combinations(indices, 2),
        key=lambda pair: similarities[pair[0]][pair[1]],
    )


def next_cycle_index(indices: list[int], after_idx: int, excluded: set[int]) -> int | None:
    candidates = [idx for idx in indices if idx not in excluded]
    for idx in candidates:
        if idx > after_idx:
            return idx
    return candidates[0] if candidates else None


def build_auto_mapping(tokens: list[str], similarities: list[list[float]]) -> dict[int, int]:
    indices = [idx for idx, token in enumerate(tokens) if is_word_token(token)]
    if len(indices) < 2:
        return {}

    pair = find_surface_pair(tokens, indices) or find_semantic_pair(indices, similarities)
    target_idx, source_idx = pair
    third_idx = next_cycle_index(indices, source_idx, {target_idx, source_idx})

    if third_idx is None:
        return {target_idx: source_idx, source_idx: target_idx}

    return {
        target_idx: source_idx,
        source_idx: third_idx,
        third_idx: target_idx,
    }


def load_model(checkpoint_dir: Path, device: torch.device, compile_mode: str):
    config = DilConfig.from_pretrained(checkpoint_dir)
    tokenizer = HybridTokenizer.from_file(checkpoint_dir / config.tokenizer_vocab_file)
    model = Dil(config).to(device)
    checkpoint = torch.load(
        checkpoint_dir / "checkpoint.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.set_compiled_forwards(
        encoder_forward=compile_forward(model.encoder.forward, compile_mode, "DilEncoderCore"),
        writer_forward=compile_forward(model.writer.forward, compile_mode, "DilConditionalWriter"),
    )
    model.eval()
    return model, config, tokenizer, checkpoint.get("training_state", {}).get("step", 0)


@torch.no_grad()
def encode_tokens(model: Dil, surface):
    return model.encode(surface).float()


@torch.no_grad()
def decode_tokens(model: Dil, tokenizer: HybridTokenizer, latents: torch.Tensor) -> list[str]:
    generation = model.decode_semantic(latents)
    tokens = []
    for row_idx in range(generation.token_ids.shape[0]):
        ids = generation.token_ids[row_idx]
        mask = generation.token_mask[row_idx]
        token = tokenizer.decode(ids[mask].detach().cpu().tolist())
        tokens.append(token)
    return tokens


def similarity_matrix(latents: torch.Tensor, config: DilConfig | None = None) -> list[list[float]]:
    if config is not None:
        latents, _ = split_factorized_latent(latents, config.semantic_latent_size, config.surface_latent_size)
    normalized = F.normalize(latents, dim=-1)
    return (normalized @ normalized.T).detach().cpu().tolist()


def print_similarity(tokens: list[str], similarities: list[list[float]], label: str = "self_similarity_matrix"):
    labels = [f"[{idx}]{token}" for idx, token in enumerate(tokens)]
    row_width = max(max(len(label) for label in labels) + 2, 12)
    col_width = max(max(len(label) for label in labels) + 2, 10)

    print(f"{label}:")
    print(" " * row_width + "".join(f"{label:>{col_width}}" for label in labels))
    for label, row in zip(labels, similarities):
        values = "".join(f"{value:>{col_width}.3f}" for value in row)
        print(f"{label:<{row_width}}{values}")


def format_table(headers: list[str], rows: list[list[str]]):
    widths = [
        max(len(headers[col_idx]), *(len(row[col_idx]) for row in rows)) + 2
        for col_idx in range(len(headers))
    ]
    print("".join(f"{header:<{widths[idx]}}" for idx, header in enumerate(headers)).rstrip())
    for row in rows:
        print("".join(f"{value:<{widths[idx]}}" for idx, value in enumerate(row)).rstrip())


@torch.no_grad()
def nllb_similarity(config: DilConfig, text: str, segments: list[TokenSegment], device: torch.device):
    nllb_tokenizer = AutoTokenizer.from_pretrained(config.nllb_model_name)
    if hasattr(nllb_tokenizer, "src_lang"):
        nllb_tokenizer.src_lang = config.nllb_src_lang
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(config.nllb_model_name, dtype=torch.float32).to(device)
    nllb_model.eval()

    starts = [segment.start for segment in segments]
    ends = [segment.end for segment in segments]

    encoded = nllb_tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offsets = encoded.pop("offset_mapping")[0].tolist()
    input_ids = encoded["input_ids"][0].tolist()
    pieces = [
        (piece, int(offset[0]), int(offset[1]), token_idx)
        for token_idx, (piece, offset) in enumerate(zip(nllb_tokenizer.convert_ids_to_tokens(input_ids), offsets))
        if int(offset[0]) != int(offset[1])
    ]
    alignments = align_spans_to_pieces(starts, ends, pieces)
    inputs = {key: value.to(device) for key, value in encoded.items()}

    outputs = nllb_model.get_encoder()(**inputs, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states
    num_encoder_layers = nllb_model.config.encoder_layers if hasattr(nllb_model.config, "encoder_layers") else len(hidden_states) - 1
    layer_groups = build_nllb_layer_groups(num_encoder_layers)

    sample_count = len(segments)
    teacher = torch.zeros((sample_count, len(layer_groups), nllb_model.config.d_model), dtype=torch.float32, device=device)
    teacher_mask = torch.zeros((sample_count,), dtype=torch.bool, device=device)
    for row_idx, positions in enumerate(alignments):
        if not positions or segments[row_idx].text.isspace():
            continue
        teacher_mask[row_idx] = True
        hidden_positions = [pieces[p][3] for p in positions]
        pos_tensor = torch.tensor(hidden_positions, dtype=torch.long, device=device)
        for group_idx, layers in enumerate(layer_groups):
            layer_vectors = [hidden_states[layer][0, pos_tensor].float().mean(dim=0) for layer in layers]
            teacher[row_idx, group_idx] = torch.stack(layer_vectors).mean(dim=0)

    teacher = apply_teacher_centered_add(teacher, teacher_mask)
    teacher_flat = teacher.mean(dim=1)
    normalized = F.normalize(teacher_flat, dim=-1)
    return (normalized @ normalized.T).detach().cpu().tolist()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--text", type=str, default="Dişi aslanın dişi kırıldı.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--compile-mode", choices=COMPILE_MODE_CHOICES, default="off")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    validate_compile_environment(args.compile_mode)

    model, config, tokenizer, _ = load_model(args.checkpoint_dir, device, args.compile_mode)
    segments = tokenize_text(args.text, tokenizer)
    tokens = [segment.text for segment in segments]
    decoded_tokens = [tokenizer.decode(segment.token_ids) for segment in segments]
    surface, byte_lengths = make_batch(segments, tokenizer, config, device)
    latents = encode_tokens(model, surface)
    roundtrip_tokens = decode_tokens(model, tokenizer, latents)
    similarities = similarity_matrix(latents, config)
    mapping = build_auto_mapping(tokens, similarities)

    print(f"tokens={decoded_tokens!r}")
    print()
    nllb_sim = nllb_similarity(config, args.text, segments, device)
    print_similarity(tokens, nllb_sim, label="nllb_teacher_similarity_matrix")
    print()
    print_similarity(tokens, similarities, label="dil_similarity_matrix")
    print()

    print("mapping:")
    if mapping:
        for target_idx, source_idx in mapping.items():
            print(f"[{target_idx}]{tokens[target_idx]} <- [{source_idx}]{tokens[source_idx]}")
    else:
        print("not_available")
    print()

    print("encoder_units:")
    rows = []
    for idx in range(len(tokens)):
        rows.append(
            [
                f"[{idx}]",
                decoded_tokens[idx],
                str(byte_lengths[idx]),
                roundtrip_tokens[idx],
                str(mapping.get(idx, idx)),
            ]
        )
    format_table(
        [
            "index",
            "target",
            "piece_len",
            "writer",
            "mapped_source",
        ],
        rows,
    )


if __name__ == "__main__":
    main()
