import json
from pathlib import Path

import torch

from dilnaz.models.dil import DilConfig
from dilnaz.surface import PackedSurface, PackedWriterTarget
from dilnaz.tokenization import HybridTokenizer
from dilnaz.train.data.dil_data import trainable_segments
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
        num_encoder_layers=2,
        max_surface_pieces_per_unit=16,
        surface_bucket_sizes=(8, 16, 32),
        context_radius=1,
        writer_num_layers=1,
        writer_word_mixer_layers=1,
        writer_word_attention_heads=4,
    )


def test_parallel_dil_dataset_uses_packed_surface(tmp_path: Path):
    data = tmp_path / "pairs.txt"
    data.write_text("eng\ttur\tkirmizi araba\tred car\n", encoding="utf-8")
    tokenizer = tiny_tokenizer()
    config = tiny_config(tokenizer)
    dataset = ParallelDilBatchDataset(data, config, tokenizer, batch_size=2, repeat=False)
    batch = next(dataset.iter_once(worker_id=0, worker_count=1))
    assert isinstance(batch["surface"], PackedSurface)
    assert isinstance(batch["labels"], PackedWriterTarget)
    assert batch["surface"].unit_count == config.context_size
    assert batch["labels"].query.unit_count == 1
    assert batch["labels"].label_mask.any()


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
    assert isinstance(batch["tr_labels"], PackedWriterTarget)
    assert isinstance(batch["en_labels"], PackedWriterTarget)
    assert batch["tr_unit_mask"].dtype == torch.bool
    assert batch["en_unit_mask"].dtype == torch.bool


def test_trainable_segments_filters_by_surface_piece_limit():
    tokenizer = tiny_tokenizer()
    segments = trainable_segments(tokenizer, "kirmizi araba", max_surface_pieces_per_unit=16, add_eos=False)
    assert [segment.text for segment in segments if not segment.text.isspace()] == ["kirmizi", "araba"]
