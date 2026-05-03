import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dilnaz"))

from dilnaz.tokenization import HybridTokenizer, default_vocab_path
from models.configuration_dil import DilConfig

from dilnaz.train.dil_data import (
    BatchSampleRef,
    HybridDilBatchDataset,
    ResidentDilBatcher,
    apply_teacher_centered_add,
    apply_teacher_centered_add_by_group,
    align_spans_to_pieces,
    split_text_for_nllb,
)


def load_tokenizer():
    return HybridTokenizer.from_file(default_vocab_path())


def test_hybrid_tokenizer_roundtrip_and_core_segments():
    tokenizer = load_tokenizer()
    text = "123. Dişi aslanın dişi_aslan#£ 1453 kırıldı."
    segments = tokenizer.encode_segments(text)
    token_ids = [piece.token_id for segment in segments for piece in segment.pieces]

    assert tokenizer.decode(token_ids) == text

    pairs = [(segment.text, segment.kind) for segment in segments if segment.kind != "space"]
    assert pairs == [
        ("1", "digit"),
        ("2", "digit"),
        ("3", "digit"),
        (".", "dot_numeric"),
        ("Dişi", "word"),
        ("aslanın", "word"),
        ("dişi", "word"),
        ("_", "underscore"),
        ("aslan", "word"),
        ("#", "punct"),
        ("£", "punct"),
        ("1453", "number"),
        ("kırıldı", "word"),
        (".", "dot_sentence"),
    ]

    turkish_initial = [
        (segment.text, segment.kind)
        for segment in tokenizer.encode_segments("ışık çağ öykü")
        if segment.kind != "space"
    ]
    assert turkish_initial == [("ışık", "word"), ("çağ", "word"), ("öykü", "word")]


def test_get_subtoken_alignment_shape():
    starts = [0, 3, 4]
    ends = [3, 4, 12]
    pieces = [
        ("▁get", 0, 3),
        ("_", 3, 4),
        ("sub", 4, 7),
        ("tok", 7, 10),
        ("en", 10, 12),
    ]

    assert align_spans_to_pieces(starts, ends, pieces) == [[0], [1], [2, 3, 4]]


def test_shared_nllb_piece_is_allowed():
    starts = [0, 1, 2, 3]
    ends = [1, 2, 3, 4]
    pieces = [
        ("▁12", 0, 2),
        ("3.", 2, 4),
    ]

    assert align_spans_to_pieces(starts, ends, pieces) == [[0], [0], [1], [1]]


def test_whitespace_surface_tokens_do_not_receive_teacher_distillation():
    tokenizer = load_tokenizer()
    config = DilConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_word_bytes=32,
    )
    dataset = HybridDilBatchDataset(
        train_file=Path("unused.txt"),
        config=config,
        tokenizer=tokenizer,
        batch_size=8,
        read_chars=1024,
    )
    segments = [segment for segment in tokenizer.encode_segments("A\nDişi") if segment.kind != "space"]
    batch = dataset.make_batch(
        ["A\nDişi"],
        [0],
        [segments],
        [BatchSampleRef(0, idx) for idx in range(len(segments))],
    )

    assert [(segment.text, segment.kind) for segment in segments] == [
        ("A", "word"),
        ("\n", "surface"),
        ("Dişi", "word"),
    ]
    assert batch["teacher_distill_mask"].tolist() == [True, False, True]
    assert batch["labels"][1, 0].item() == tokenizer.token_to_id["surface:\n"]
    assert batch["teacher_texts"] == ["A\nDişi"]
    assert batch["teacher_text_indices"].tolist() == [0, 0, 0]


def test_common_word_tokens_only_match_standalone_words():
    tokenizer = load_tokenizer()

    standalone = [
        segment
        for segment in tokenizer.encode_segments("bu da ve bir ile ama çok daha the and for")
        if segment.kind != "space"
    ]
    assert [(segment.text, segment.kind, segment.piece_len) for segment in standalone] == [
        ("bu", "word", 1),
        ("da", "word", 1),
        ("ve", "word", 1),
        ("bir", "word", 1),
        ("ile", "word", 1),
        ("ama", "word", 1),
        ("çok", "word", 1),
        ("daha", "word", 1),
        ("the", "word", 1),
        ("and", "word", 1),
        ("for", "word", 1),
    ]

    embedded = [
        segment
        for segment in tokenizer.encode_segments("buda istanbulda birde")
        if segment.kind != "space"
    ]
    assert [segment.text for segment in embedded] == ["buda", "istanbulda", "birde"]
    assert [segment.piece_len for segment in embedded] == [
        len("buda".encode("utf-8")),
        len("istanbulda".encode("utf-8")),
        len("birde".encode("utf-8")),
    ]


def test_alphanumeric_words_split_at_digit_boundaries():
    tokenizer = load_tokenizer()

    segments = [
        (segment.text, segment.kind)
        for segment in tokenizer.encode_segments("()/345 asd q3421 --- asd1 ")
        if segment.kind != "space"
    ]

    assert segments == [
        ("(", "punct"),
        (")", "punct"),
        ("/", "punct"),
        ("3", "digit"),
        ("4", "digit"),
        ("5", "digit"),
        ("asd", "word"),
        ("q", "word"),
        ("3", "digit"),
        ("4", "digit"),
        ("2", "digit"),
        ("1", "digit"),
        ("---", "surface"),
        ("asd", "word"),
        ("1", "digit"),
    ]


def test_repeating_train_dataset_carries_partial_batch_to_keep_shape(tmp_path):
    train_file = tmp_path / "tiny.txt"
    train_file.write_text("A B C\n", encoding="utf-8")
    tokenizer = load_tokenizer()
    config = DilConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_word_bytes=32,
    )
    dataset = HybridDilBatchDataset(
        train_file=train_file,
        config=config,
        tokenizer=tokenizer,
        batch_size=5,
        read_chars=1024,
        repeat=True,
    )

    batch = next(iter(dataset))

    assert batch["input_ids"].shape[0] == 5


def test_resident_hybrid_batcher_includes_final_partial_batch(tmp_path):
    train_file = tmp_path / "tiny.txt"
    train_file.write_text("A B C\nD E F", encoding="utf-8")
    tokenizer = load_tokenizer()
    config = DilConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_word_bytes=32,
    )
    dataset = HybridDilBatchDataset(
        train_file=train_file,
        config=config,
        tokenizer=tokenizer,
        batch_size=4,
        read_chars=1024,
        repeat=True,
    )

    class Teacher:
        def teacher_layers(self, batch):
            size = batch["labels"].shape[0]
            return torch.zeros(size, 4, 1024), torch.ones(size, dtype=torch.bool)

    batcher = ResidentDilBatcher.from_dataset(
        dataset,
        Teacher(),
        batch_size=4,
        device=torch.device("cpu"),
        seed=1,
    )
    seen = sorted(
        {
            int(line_id)
            for batch in batcher.batches
            for line_id in batch["source_line_ids"].tolist()
        }
    )

    assert [batch["labels"].shape[0] for batch in batcher.batches] == [4, 3]
    assert seen == [0, 1]


def test_teacher_centered_add_uses_only_distilled_tokens():
    teacher = torch.tensor(
        [
            [[1.0, 3.0]],
            [[100.0, 200.0]],
            [[5.0, 7.0]],
        ],
        dtype=torch.float32,
    )
    teacher_mask = torch.tensor([True, False, True])

    result = apply_teacher_centered_add(teacher, teacher_mask)

    center = torch.tensor([[[3.0, 5.0]]])
    expected = teacher.clone()
    expected[0] = teacher[0] + (teacher[0] - center[0]) * 0.5
    expected[2] = teacher[2] + (teacher[2] - center[0]) * 0.5

    assert torch.allclose(result, expected)
    assert torch.equal(result[1], teacher[1])


def test_teacher_centered_add_centers_each_text_group_separately():
    teacher = torch.tensor(
        [
            [[1.0, 1.0]],
            [[3.0, 3.0]],
            [[100.0, 100.0]],
            [[102.0, 102.0]],
        ],
        dtype=torch.float32,
    )
    teacher_mask = torch.tensor([True, True, True, True])
    group_ids = torch.tensor([0, 0, 1, 1])

    result = apply_teacher_centered_add_by_group(teacher, teacher_mask, group_ids)

    expected = torch.tensor(
        [
            [[0.5, 0.5]],
            [[3.5, 3.5]],
            [[99.5, 99.5]],
            [[102.5, 102.5]],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(result, expected)


class FakeNllbTokenizer:
    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": text.split()}


def test_long_teacher_text_splits_on_sentence_boundary_before_nllb_limit():
    text = "a a a. b b b. c c c."

    chunks = split_text_for_nllb(text, FakeNllbTokenizer(), max_tokens=5)

    assert chunks == ["a a a. ", "b b b. ", "c c c."]
