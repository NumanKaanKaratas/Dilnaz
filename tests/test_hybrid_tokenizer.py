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
    trainable_segments,
)
from dilnaz.train.naz_data import stream_token_pieces


def load_tokenizer():
    return HybridTokenizer.from_file(default_vocab_path())


def decode_segments(tokenizer: HybridTokenizer, segments) -> str:
    return "".join(tokenizer.decode(segment.token_ids) for segment in segments)


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


def test_leading_space_token_roundtrips_common_shapes():
    tokenizer = load_tokenizer()

    for text in [
        "1234567890 sayıları bazen çok işe yarar.",
        "1 2 3 4 5",
        "1  2   3",
        "Istanbul'da 1ad47wq çalıştı.",
        "Dişi aslanın dişi kırıldı.",
    ]:
        assert tokenizer.decode(tokenizer.encode(text)) == text
        assert decode_segments(tokenizer, tokenizer.encode_segments(text)) == text


def test_leading_space_token_contract_marks_only_real_boundaries():
    tokenizer = load_tokenizer()

    compact_ids = tokenizer.encode("123")
    spaced_segments = tokenizer.encode_segments("1 2 3")
    spaced_ids = [piece.token_id for segment in spaced_segments for piece in segment.pieces]
    assert compact_ids != spaced_ids
    assert [segment.text for segment in spaced_segments] == ["1", "2", "3"]
    assert tokenizer.is_leading_space_token(spaced_segments[1].token_ids[0])
    assert tokenizer.is_leading_space_token(spaced_segments[2].token_ids[0])
    assert tokenizer.decode(spaced_ids) == "1 2 3"

    multi_space_segments = tokenizer.encode_segments("1  2")
    assert [
        (segment.text, segment.kind, tokenizer.decode(segment.token_ids))
        for segment in multi_space_segments
    ] == [
        ("1", "digit", "1"),
        (" ", "space", " "),
        ("2", "digit", " 2"),
    ]
    assert not tokenizer.is_leading_space_token(multi_space_segments[1].token_ids[0])
    assert tokenizer.is_leading_space_token(multi_space_segments[2].token_ids[0])


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


def test_dil_batch_labels_decode_with_tokenizer_contract():
    tokenizer = load_tokenizer()
    text = "1234567890 sayıları"
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
        batch_size=32,
        read_chars=1024,
    )
    segments = trainable_segments(tokenizer, text, config.max_word_bytes)
    batch = dataset.make_batch(
        [text],
        [0],
        [segments],
        [BatchSampleRef(0, idx) for idx in range(len(segments))],
    )

    decoded_labels = []
    decoded_targets = []
    for row in batch["labels"]:
        ids = [
            int(token_id)
            for token_id in row.tolist()
            if int(token_id) not in (-100, tokenizer.eos_token_id)
        ]
        decoded_labels.append(tokenizer.decode(ids))
    for row in batch["input_ids"][:, config.target_index]:
        ids = [int(token_id) for token_id in row.tolist() if int(token_id) != tokenizer.pad_token_id]
        decoded_targets.append(tokenizer.decode(ids))

    assert "".join(decoded_labels) == text
    assert decoded_targets == decoded_labels
    sayilari = next(segment for segment in segments if segment.text == "sayıları")
    assert tokenizer.is_leading_space_token(sayilari.token_ids[0])
    assert sayilari.start == text.index("sayıları")
    assert batch["teacher_starts"][segments.index(sayilari)].item() == sayilari.start


def test_naz_token_stream_preserves_extra_spaces(tmp_path):
    tokenizer = load_tokenizer()
    train_file = tmp_path / "spacing.txt"
    text = "1  2   3"
    train_file.write_text(text, encoding="utf-8")

    token_ids = list(stream_token_pieces(train_file, tokenizer, max_word_bytes=32, read_chars=2))

    assert token_ids[-1] == [tokenizer.eos_token_id]
    assert "".join(tokenizer.decode(ids) for ids in token_ids[:-1]) == text
    assert tokenizer.is_leading_space_token(token_ids[2][0])
    assert tokenizer.is_leading_space_token(token_ids[-2][0])


def test_interfaces_do_not_keep_manual_join_logic():
    root = Path(__file__).resolve().parents[1]
    for relative_path in ["dilnaz/train/interface_dil.py", "dilnaz/train/interface_naz.py"]:
        source = (root / relative_path).read_text(encoding="utf-8")
        assert "def join_tokens" not in source
        assert "format_next_token" not in source
        assert "RIGHT_ATTACHED" not in source
        assert "LEFT_ATTACHED" not in source


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
