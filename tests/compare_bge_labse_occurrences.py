from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModel, AutoTokenizer

from dilnaz.train.data.dil_data import align_spans_to_pieces


@dataclass(frozen=True)
class Occurrence:
    key: str
    text: str
    surface: str
    occurrence: int


OCCURRENCES = [
    Occurrence("s1_female_disi", "dişi aslanın dişi kırıldı.", "dişi", 0),
    Occurrence("s1_tooth_disi", "dişi aslanın dişi kırıldı.", "dişi", 1),
    Occurrence("s2_female_disi", "dişi kaplanın dişi parçalandı.", "dişi", 0),
    Occurrence("s2_tooth_disi", "dişi kaplanın dişi parçalandı.", "dişi", 1),
    Occurrence("s1_aslanin", "dişi aslanın dişi kırıldı.", "aslanın", 0),
    Occurrence("s2_kaplanin", "dişi kaplanın dişi parçalandı.", "kaplanın", 0),
    Occurrence("s1_kirildi", "dişi aslanın dişi kırıldı.", "kırıldı", 0),
    Occurrence("s2_parcalandi", "dişi kaplanın dişi parçalandı.", "parçalandı", 0),
    Occurrence("tr_araba", "araba yolda kaldı.", "araba", 0),
    Occurrence("en_car", "the car broke down.", "car", 0),
]

PAIRS = [
    ("s1 içi dişi female/tooth", "s1_female_disi", "s1_tooth_disi"),
    ("s2 içi dişi female/tooth", "s2_female_disi", "s2_tooth_disi"),
    ("female dişi cross same", "s1_female_disi", "s2_female_disi"),
    ("tooth dişi cross same", "s1_tooth_disi", "s2_tooth_disi"),
    ("female vs tooth cross", "s1_female_disi", "s2_tooth_disi"),
    ("tooth vs female cross", "s1_tooth_disi", "s2_female_disi"),
    ("araba vs car", "tr_araba", "en_car"),
    ("aslanın vs kaplanın", "s1_aslanin", "s2_kaplanin"),
    ("kırıldı vs parçalandı", "s1_kirildi", "s2_parcalandi"),
]


def occurrence_span(text: str, surface: str, occurrence: int) -> tuple[int, int]:
    start = -1
    pos = 0
    for _ in range(occurrence + 1):
        start = text.index(surface, pos)
        pos = start + len(surface)
    return start, start + len(surface)


def unique_texts() -> list[str]:
    return list(dict.fromkeys(item.text for item in OCCURRENCES))


def pieces_from_tokenizer(tokenizer, text: str, vector_count: int | None = None):
    encoded = tokenizer(text, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    token_ids = encoded["input_ids"]
    if vector_count is not None:
        offsets = offsets[:vector_count]
        token_ids = token_ids[:vector_count]
    pieces = []
    for idx, (token_id, offset) in enumerate(zip(token_ids, offsets)):
        start, end = int(offset[0]), int(offset[1])
        if start != end:
            pieces.append((tokenizer.convert_ids_to_tokens(int(token_id)), start, end, idx))
    return pieces


def vector_for_occurrence(hidden: torch.Tensor, tokenizer, occurrence: Occurrence) -> torch.Tensor:
    start, end = occurrence_span(occurrence.text, occurrence.surface, occurrence.occurrence)
    pieces = pieces_from_tokenizer(tokenizer, occurrence.text, hidden.shape[0])
    positions = align_spans_to_pieces([start], [end], pieces)[0]
    token_positions = torch.tensor([pieces[pos][3] for pos in positions], dtype=torch.long, device=hidden.device)
    return hidden.index_select(0, token_positions).mean(dim=0)


def bge_vectors(device: str) -> dict[str, torch.Tensor]:
    model_name = "BAAI/bge-m3"
    model = BGEM3FlagModel(model_name, use_fp16=device == "cuda", device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    texts = unique_texts()
    outputs = model.encode(
        texts,
        batch_size=4,
        max_length=256,
        return_dense=False,
        return_sparse=False,
        return_colbert_vecs=True,
    )
    hidden_by_text = {
        text: torch.tensor(vectors, dtype=torch.float32, device=device)
        for text, vectors in zip(texts, outputs["colbert_vecs"])
    }
    return {
        occurrence.key: vector_for_occurrence(hidden_by_text[occurrence.text], tokenizer, occurrence)
        for occurrence in OCCURRENCES
    }


@torch.no_grad()
def labse_vectors(device: str) -> dict[str, torch.Tensor]:
    model_name = "sentence-transformers/LaBSE"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    vectors: dict[str, torch.Tensor] = {}
    for text in unique_texts():
        encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        encoded.pop("offset_mapping")
        inputs = {key: value.to(device) for key, value in encoded.items()}
        hidden = model(**inputs, return_dict=True).last_hidden_state[0].float()
        for occurrence in [item for item in OCCURRENCES if item.text == text]:
            vectors[occurrence.key] = vector_for_occurrence(hidden, tokenizer, occurrence)
    return vectors


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((F.normalize(left, dim=0) * F.normalize(right, dim=0)).sum().detach().cpu())


def print_table(bge: dict[str, torch.Tensor], labse: dict[str, torch.Tensor]):
    print(f"{'pair':38s} {'BGE-M3 ColBERT':>15s} {'LaBSE token':>12s}")
    for label, left, right in PAIRS:
        print(f"{label:38s} {cosine(bge[left], bge[right]):15.3f} {cosine(labse[left], labse[right]):12.3f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    bge = bge_vectors(device)
    labse = labse_vectors(device)
    print_table(bge, labse)


if __name__ == "__main__":
    main()
