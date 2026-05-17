import json
from pathlib import Path

import torch

from dilnaz.models.dil import Dil, DilConfig, compose_factorized_latent, normalize_semantic_latents
from dilnaz.surface import PackedSurface, PackedWriterTarget
from dilnaz.tokenization import HybridTokenizer
from dilnaz.train.data.dil_data import (
    ContextDilBatchDataset,
    NllbEncodedText,
    NllbTeacherTextCache,
    ResidentDilBatcher,
    context_windows,
    segment_piece_ids,
    stream_text_items,
    trainable_segments,
)
from dilnaz.train.data.parallel_dil_data import ParallelDilBatchDataset, parallel_alignment_loss
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
        semantic_latent_size=12,
        surface_latent_size=4,
        max_surface_pieces_per_unit=16,
        surface_bucket_sizes=(8, 16, 32),
        num_encoder_layers=2,
        writer_num_layers=1,
    )


def has_no_effective_grad(parameter: torch.nn.Parameter) -> bool:
    return parameter.grad is None or not parameter.grad.abs().gt(0).any()


def test_parallel_dil_dataset_uses_packed_surface(tmp_path: Path):
    data = tmp_path / "pairs.txt"
    data.write_text("eng\ttur\tkirmizi araba\tred car\n", encoding="utf-8")
    tokenizer = tiny_tokenizer()
    config = tiny_config(tokenizer)
    dataset = ParallelDilBatchDataset(data, config, tokenizer, batch_size=2, repeat=False)
    batch = next(dataset.iter_once(worker_id=0, worker_count=1))
    assert isinstance(batch["surface"], PackedSurface)
    assert isinstance(batch["writer_target"], PackedWriterTarget)
    assert batch["writer_target"].label_mask.any()
    assert batch["surface"].batch_size == 2
    assert batch["writer_target"].query.batch_size == batch["surface"].batch_size
    assert batch["teacher_starts"].shape == batch["surface"].unit_mask.shape
    assert batch["row_batch_indices"].numel() == int(batch["surface"].unit_mask.sum())
    assert batch["surface"].unit_mask.any()


def test_parallel_dil_writer_loss_uses_detached_encoder_prior(tmp_path: Path):
    data = tmp_path / "pairs.txt"
    data.write_text("eng\ttur\tkirmizi araba\tred car\n", encoding="utf-8")
    tokenizer = tiny_tokenizer()
    config = tiny_config(tokenizer)
    dataset = ParallelDilBatchDataset(data, config, tokenizer, batch_size=1, repeat=False)
    batch = next(dataset.iter_once(worker_id=0, worker_count=1))
    model = Dil(config)
    output = model(batch["surface"], writer_target=batch["writer_target"])
    output.loss.backward()

    assert has_no_effective_grad(model.encoder.embed_tokens.weight)
    assert has_no_effective_grad(model.encoder.semantic_head.weight)
    assert model.encoder.surface_head.weight.grad is not None
    assert model.writer.token_embeddings.weight.grad is not None
    assert model.writer.encoder_prior_proj.weight.grad is not None
    assert model.writer.encoder_prior_gate.weight.grad is not None


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
    assert isinstance(batch["tr_writer_target"], PackedWriterTarget)
    assert isinstance(batch["en_writer_target"], PackedWriterTarget)
    assert batch["tr_writer_target"].label_mask.any()
    assert batch["en_writer_target"].label_mask.any()
    assert batch["tr_unit_mask"].dtype == torch.bool
    assert batch["en_unit_mask"].dtype == torch.bool


def test_teacherless_writer_targets_use_same_encoder_prior_path(tmp_path: Path):
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
    model = Dil(config)
    latents = model.encode(batch["tr_surface"])
    metrics = model.writer_training_loss_and_metrics(latents, batch["tr_writer_target"], return_metrics=True)
    metrics["loss"].backward()

    assert has_no_effective_grad(model.encoder.embed_tokens.weight)
    assert has_no_effective_grad(model.encoder.semantic_head.weight)
    assert model.encoder.surface_head.weight.grad is not None
    assert model.writer.token_embeddings.weight.grad is not None
    assert model.writer.encoder_prior_proj.weight.grad is not None
    assert model.writer.encoder_prior_gate.weight.grad is not None


def test_parallel_alignment_loss_uses_only_semantic_split():
    tokenizer = tiny_tokenizer()
    config = tiny_config(tokenizer)
    semantic = normalize_semantic_latents(torch.ones(2, config.semantic_latent_size))
    latents = compose_factorized_latent(
        semantic,
        torch.tensor([[-1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0]]),
    )
    batch = {
        "row_batch_indices": torch.zeros(2, dtype=torch.long),
        "row_unit_indices": torch.arange(2, dtype=torch.long),
        "parallel_source_rows": torch.tensor([[0]], dtype=torch.long),
        "parallel_target_rows": torch.tensor([[1]], dtype=torch.long),
        "parallel_source_mask": torch.ones(1, 1, dtype=torch.bool),
        "parallel_target_mask": torch.ones(1, 1, dtype=torch.bool),
    }
    loss = parallel_alignment_loss(
        latents.unsqueeze(0),
        batch,
        config.semantic_latent_size,
        config.surface_latent_size,
    )
    assert torch.isclose(loss, torch.zeros_like(loss), atol=1e-6)


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


def test_nllb_teacher_text_cache_roundtrips_grouped_hidden(tmp_path: Path):
    cache = NllbTeacherTextCache(
        tmp_path,
        {"model_name": "test-nllb", "layer_groups": ((1, 2),), "dtype": "torch.bfloat16"},
    )
    encoded = NllbEncodedText(
        group_hidden=torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4),
        pieces=(("araba", 0, 5, 1), ("lar", 5, 8, 2)),
    )
    cache.put("tur_Latn", "arabalar", encoded)

    loaded = cache.get("tur_Latn", "arabalar")
    assert loaded is not None
    assert torch.equal(loaded.group_hidden, encoded.group_hidden)
    assert loaded.pieces == encoded.pieces

    changed_contract_cache = NllbTeacherTextCache(
        tmp_path,
        {"model_name": "other-nllb", "layer_groups": ((1, 2),), "dtype": "torch.bfloat16"},
    )
    assert changed_contract_cache.get("tur_Latn", "arabalar") is None


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
