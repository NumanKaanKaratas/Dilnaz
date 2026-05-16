import json
from pathlib import Path

import torch

from dilnaz.models.dil import DilConfig
from dilnaz.surface import PackedSurface
from dilnaz.tokenization import HybridTokenizer
from dilnaz.train.data.dil_data import (
    ContextDilBatchDataset,
    ResidentDilBatcher,
    context_windows,
    segment_piece_ids,
    stream_text_items,
    trainable_segments,
)
from dilnaz.train.data.parallel_dil_data import ParallelDilBatchDataset
from dilnaz.train.dil.train_teacherless_parallel import TeacherlessParallelJsonlDataset


def tiny_tokenizer() -> HybridTokenizer:
    return HybridTokenizer(
        char_tokens=[],
        surface_tokens=["araba", "car", "kirmizi", "red", "."],
        numeric_tokens=[],
        common_word_tokens=[],
        contextual_tokens={},
    )


def tiny_config(tokenizer: HybridTokenizer) -> DilConfig:
    return DilConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.eos_token_id,
        hidden_size=32,
        intermediate_size=64,
        latent_size=16,
        max_surface_pieces_per_unit=16,
        surface_bucket_sizes=(8, 16, 32),
        num_encoder_layers=2,
        writer_num_layers=1,
    )


def test_parallel_dil_dataset_uses_packed_surface(tmp_path: Path):
    data = tmp_path / "pairs.txt"
    data.write_text("eng\ttur\tkirmizi araba\tred car\n", encoding="utf-8")
    tokenizer = tiny_tokenizer()
    config = tiny_config(tokenizer)
    dataset = ParallelDilBatchDataset(data, config, tokenizer, batch_size=2, repeat=False)
    batch = next(dataset.iter_once(worker_id=0, worker_count=1))
    assert isinstance(batch["surface"], PackedSurface)
    assert "labels" not in batch
    assert "writer_labels" not in batch
    assert batch["surface"].batch_size == 2
    assert batch["teacher_starts"].shape == batch["surface"].unit_mask.shape
    assert batch["row_batch_indices"].numel() == int(batch["surface"].unit_mask.sum())
    assert batch["surface"].unit_mask.any()


def test_teacherless_parallel_dataset_uses_packed_surface(tmp_path: Path):
    data = tmp_path / "pairs.jsonl"
    data.write_text(json.dumps({"tr": "kirmizi araba", "en": "red car"}) + "\n", encoding="utf-8")
    tokenizer = tiny_tokenizer()
    config = tiny_config(tokenizer)
    dataset = TeacherlessParallelJsonlDataset(
        data,
        config,
        tokenizer,
        batch_size=1,
        max_segments=4,
        min_segments=1,
        min_length_ratio=0.1,
        max_length_ratio=10.0,
        shuffle_buffer_size=1,
        seed=1,
        repeat=False,
    )
    batch = next(iter(dataset))
    assert isinstance(batch["tr_surface"], PackedSurface)
    assert isinstance(batch["en_surface"], PackedSurface)
    assert "tr_labels" not in batch
    assert "en_labels" not in batch
    assert batch["tr_unit_mask"].dtype == torch.bool
    assert batch["en_unit_mask"].dtype == torch.bool


def test_trainable_segments_filters_by_surface_piece_limit():
    tokenizer = tiny_tokenizer()
    segments = trainable_segments(tokenizer, "kirmizi araba", max_surface_pieces_per_unit=16, add_eos=False)
    assert [segment.text for segment in segments if not segment.text.isspace()] == ["kirmizi", "araba"]


def test_resident_dil_batcher_materializes_one_pass_from_repeating_dataset(tmp_path: Path):
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"text": "kirmizi araba"}) + "\n", encoding="utf-8")
    tokenizer = tiny_tokenizer()
    config = tiny_config(tokenizer)
    dataset = ContextDilBatchDataset(data, config, tokenizer, batch_size=2, read_chars=1024, repeat=True)

    class FakeTeacher:
        def teacher_layers(self, batch, texts=None):
            shape = (*batch["surface"].unit_mask.shape, 1, config.latent_size)
            return torch.zeros(shape), batch["surface"].unit_mask

    batcher = ResidentDilBatcher.from_dataset(
        dataset,
        FakeTeacher(),
        batch_size=2,
        device=torch.device("cpu"),
        seed=1,
    )
    assert len(batcher.batches) > 0
    assert isinstance(next(batcher)["surface"], PackedSurface)


def test_context_dil_batch_size_counts_training_rows(tmp_path: Path):
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"text": "kirmizi araba red car"}) + "\n", encoding="utf-8")
    tokenizer = tiny_tokenizer()
    config = tiny_config(tokenizer)
    dataset = ContextDilBatchDataset(data, config, tokenizer, batch_size=4, read_chars=1024, repeat=True)
    batch = next(dataset.iter_once(worker_id=0, worker_count=1))
    assert batch["surface"].batch_size == 4


def test_context_dil_batch_carries_target_teacher_spans(tmp_path: Path):
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"text": "kirmizi araba"}) + "\n", encoding="utf-8")
    tokenizer = tiny_tokenizer()
    config = tiny_config(tokenizer)
    dataset = ContextDilBatchDataset(data, config, tokenizer, batch_size=2, read_chars=1024, repeat=True)
    batch = next(dataset.iter_once(worker_id=0, worker_count=1))
    assert batch["teacher_texts"] == ["kirmizi araba", "kirmizi araba"]
    assert batch["teacher_text_indices"].tolist() == [0, 1]
    assert batch["teacher_starts"].tolist() == [0, 8]
    assert batch["teacher_ends"].tolist() == [7, 13]
    assert batch["teacher_distill_mask"].tolist() == [True, True]


def test_context_windows_keep_target_at_center():
    tokenizer = tiny_tokenizer()
    segments = trainable_segments(tokenizer, "kirmizi araba", max_surface_pieces_per_unit=16, add_eos=False)
    rows = [[segment_piece_ids(segment) for segment in window] for window in context_windows(segments, context_radius=2)]
    assert rows[0][2] == segment_piece_ids(segments[0])
    assert rows[0][0] == []
    assert rows[0][1] == []


def test_stream_text_items_reads_jsonl_text_field(tmp_path: Path):
    data = tmp_path / "train.jsonl"
    text = "kirmizi araba " * 200
    data.write_text(json.dumps({"text": text}) + "\n", encoding="utf-8")
    assert list(stream_text_items(data, read_chars=16)) == [(0, text, True)]
