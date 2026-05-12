from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from dilnaz.tokenization import HybridTokenizer, default_vocab_path
from dilnaz.train.data.dil_data import align_spans_to_pieces


BASE_SENTENCE = (
    "Dişi aslanın ve kaplanın dişi kırıldı. "
    "Arabamı götürdüler otomobilimi getirdiler ingilizcesinede car dediler."
)

SENTENCES = {
    "base_mix": BASE_SENTENCE,
    "arabam_same": "Arabamı sabah servise bıraktım.",
    "oto_same": "Bugün otomobilimi garajdan çıkardım.",
    "car_same": "İngilizce derste car kelimesini öğrendik.",
    "tooth_same": "Çocuğun sallanan dişi sonunda düştü.",
    "female_same": "Dişi kuş yuvayı dikkatle korudu.",
    "female_limb": "Dişi aslanın kolu yaralandı.",
    "tooth_lion": "Aslanın dişi kırıldı.",
    "same_sentence_homograph": "Dişi aslanın dişi kırıldı.",
}


@dataclass(frozen=True)
class PairSpec:
    name: str
    left_sentence: str
    left_word: str
    left_occurrence: int
    right_sentence: str
    right_word: str
    right_occurrence: int
    kind: str


PAIR_SPECS = (
    PairSpec("same_Arabamı", "base_mix", "Arabamı", 0, "arabam_same", "Arabamı", 0, "same"),
    PairSpec("same_otomobilimi", "base_mix", "otomobilimi", 0, "oto_same", "otomobilimi", 0, "same"),
    PairSpec("same_car", "base_mix", "car", 0, "car_same", "car", 0, "same"),
    PairSpec("same_dişi_tooth", "base_mix", "dişi", 0, "tooth_same", "dişi", 0, "same"),
    PairSpec("same_Dişi_female", "base_mix", "Dişi", 0, "female_same", "Dişi", 0, "same"),
    PairSpec(
        "diff_dişi_female_vs_tooth_cross",
        "female_limb",
        "Dişi",
        0,
        "tooth_lion",
        "dişi",
        0,
        "diff",
    ),
    PairSpec(
        "diff_same_sentence_Dişi_vs_dişi",
        "same_sentence_homograph",
        "Dişi",
        0,
        "same_sentence_homograph",
        "dişi",
        0,
        "diff",
    ),
)


@dataclass
class EncodedSentence:
    labels: list[str]
    raw_vectors: torch.Tensor
    center: torch.Tensor
    alignments: list[list[str]]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def encode_sentence(
    text: str,
    hybrid_tokenizer: HybridTokenizer,
    nllb_tokenizer,
    model,
    device: torch.device,
) -> EncodedSentence:
    segments = [
        segment
        for segment in hybrid_tokenizer.encode_segments(text)
        if segment.kind != "space" and not segment.text.isspace()
    ]
    encoded = nllb_tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    input_ids = encoded["input_ids"][0].tolist()
    piece_tokens = nllb_tokenizer.convert_ids_to_tokens(input_ids)
    pieces = [
        (piece, int(offset[0]), int(offset[1]), token_idx)
        for token_idx, (piece, offset) in enumerate(zip(piece_tokens, offsets))
        if int(offset[0]) != int(offset[1])
    ]
    alignments = align_spans_to_pieces(
        [segment.start for segment in segments],
        [segment.end for segment in segments],
        pieces,
    )
    inputs = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        hidden = model.get_encoder()(**inputs, return_dict=True).last_hidden_state[0].float().cpu()

    labels: list[str] = []
    raw_vectors: list[torch.Tensor] = []
    debug_alignments: list[list[str]] = []
    for segment, positions in zip(segments, alignments):
        if not positions:
            continue
        hidden_positions = [pieces[position][3] for position in positions]
        labels.append(segment.text)
        raw_vectors.append(hidden[hidden_positions].mean(dim=0))
        debug_alignments.append([pieces[position][0] for position in positions])

    raw_tensor = torch.stack(raw_vectors, dim=0)
    center = raw_tensor.mean(dim=0, keepdim=True) if raw_tensor.shape[0] > 1 else raw_tensor.new_zeros((1, raw_tensor.shape[1]))
    return EncodedSentence(labels=labels, raw_vectors=raw_tensor, center=center, alignments=debug_alignments)


def find_word(labels: list[str], word: str, occurrence: int) -> int:
    seen = 0
    for index, label in enumerate(labels):
        if label == word:
            if seen == occurrence:
                return index
            seen += 1
    raise ValueError(f"{word!r} occurrence {occurrence} not found in {labels!r}")


def formula_vector(encoded: EncodedSentence, index: int, mode: str, scale: float) -> torch.Tensor:
    raw = encoded.raw_vectors[index]
    centered = raw - encoded.center[0]
    if mode == "raw":
        return raw
    if mode == "subtract":
        return centered
    if mode == "add":
        return raw + scale * centered
    if mode == "concat":
        return torch.cat((F.normalize(raw, dim=0), scale * F.normalize(centered, dim=0)))
    raise ValueError(f"unknown formula mode: {mode}")


def score_formula(
    encoded: dict[str, EncodedSentence],
    mode: str,
    scale: float,
) -> tuple[list[tuple[str, float, str]], float, float, float, float]:
    rows: list[tuple[str, float, str]] = []
    same_scores: list[float] = []
    diff_scores: list[float] = []
    for pair in PAIR_SPECS:
        left = encoded[pair.left_sentence]
        right = encoded[pair.right_sentence]
        left_index = find_word(left.labels, pair.left_word, pair.left_occurrence)
        right_index = find_word(right.labels, pair.right_word, pair.right_occurrence)
        score = F.cosine_similarity(
            formula_vector(left, left_index, mode, scale),
            formula_vector(right, right_index, mode, scale),
            dim=0,
        ).item()
        rows.append((pair.name, score, pair.kind))
        if pair.kind == "same":
            same_scores.append(score)
        else:
            diff_scores.append(score)

    avg_same = sum(same_scores) / len(same_scores)
    avg_diff = sum(diff_scores) / len(diff_scores)
    min_same = min(same_scores)
    max_diff = max(diff_scores)
    return rows, avg_same, avg_diff, min_same, max_diff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe NLLB teacher vector formulas for Dilnaz.")
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--src-lang", default="tur_Latn")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--add-scales", default="0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--concat-scales", default="0.25,0.5,0.75,1.0,1.5")
    return parser.parse_args()


def parse_scales(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    hybrid_tokenizer = HybridTokenizer.from_file(default_vocab_path())
    nllb_tokenizer = AutoTokenizer.from_pretrained(args.model)
    if hasattr(nllb_tokenizer, "src_lang"):
        nllb_tokenizer.src_lang = args.src_lang
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()

    encoded = {
        name: encode_sentence(sentence, hybrid_tokenizer, nllb_tokenizer, model, device)
        for name, sentence in SENTENCES.items()
    }

    print(f"model={args.model}")
    print(f"device={device} dtype={dtype}")
    print()
    print("Sentences:")
    for name, sentence in SENTENCES.items():
        print(f"{name}: {sentence}")
    print()
    print(f"base_tokens={encoded['base_mix'].labels}")
    print()

    formulas: list[tuple[str, float, str]] = [("raw", 0.0, "raw"), ("subtract", 0.0, "subtract")]
    formulas.extend(("add", scale, f"add:{scale}") for scale in parse_scales(args.add_scales))
    formulas.extend(("concat", scale, f"concat:{scale}") for scale in parse_scales(args.concat_scales))

    for mode, scale, label in formulas:
        rows, avg_same, avg_diff, min_same, max_diff = score_formula(encoded, mode, scale)
        margin = min_same - max_diff
        print(
            f"FORMULA {label} "
            f"avg_same={avg_same:.3f} avg_diff={avg_diff:.3f} "
            f"min_same={min_same:.3f} max_diff={max_diff:.3f} margin={margin:.3f}"
        )
        for name, score, kind in rows:
            print(f"  {name:<34} {score:>7.3f} {kind}")
        print()


if __name__ == "__main__":
    main()
