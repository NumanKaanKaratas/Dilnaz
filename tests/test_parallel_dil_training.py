import json
import os
from pathlib import Path

import torch

from dilnaz.models.dil import Dil, DilConfig, compose_factorized_latent, normalize_semantic_latents
from dilnaz.surface import PackedSurface, PackedWriterTarget
from dilnaz.tokenization import HybridTokenizer
from dilnaz.train.data.dil_data import (
    ContextDilBatchDataset,
    NllbEncodedText,
    NllbTeacher,
    NllbTeacherTextCache,
    ResidentDilBatcher,
    context_windows,
    merge_punctuation_sentence_chunks,
    nllb_token_count,
    segment_piece_ids,
    sentence_texts_from_chunks,
    split_text_for_nllb,
    stream_text_items,
    trainable_segments,
)
from dilnaz.train.data.parallel_dil_data import ParallelDilBatchDataset, parallel_alignment_loss
from dilnaz.train.data.parallel_dil_data import ParallelNllbTeacher
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


def test_nllb_teacher_text_cache_prunes_old_entries(tmp_path: Path):
    contract = {"model_name": "test-nllb", "layer_groups": ((1, 2),), "dtype": "torch.bfloat16"}
    encoded = NllbEncodedText(
        group_hidden=torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4),
        pieces=(("araba", 0, 5, 1),),
    )
    cache = NllbTeacherTextCache(tmp_path, contract)
    cache.put("tur_Latn", "birinci", encoded)
    first_path = cache.path_for(cache.key("tur_Latn", "birinci"))
    first_size = first_path.stat().st_size
    os.utime(first_path, (1, 1))

    cache.max_disk_bytes = first_size + first_size // 2
    cache.put("tur_Latn", "ikinci", encoded)

    second_path = cache.path_for(cache.key("tur_Latn", "ikinci"))
    assert not first_path.exists()
    assert second_path.exists()
    assert cache.get("tur_Latn", "birinci") is None
    assert cache.get("tur_Latn", "ikinci") is not None
    assert cache.disk_bytes <= cache.max_disk_bytes


class FakeNllbTokenizer:
    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(
        self,
        texts,
        add_special_tokens=True,
        return_offsets_mapping=False,
        padding=False,
        return_tensors=None,
    ):
        single = isinstance(texts, str)
        text_list = [texts] if single else list(texts)
        rows = []
        offset_rows = []
        for text in text_list:
            spans = []
            cursor = 0
            for piece in text.split():
                start = text.index(piece, cursor)
                end = start + len(piece)
                spans.append((start, end))
                cursor = end
            ids = list(range(10, 10 + len(spans)))
            offsets = list(spans)
            if add_special_tokens:
                ids = [1, *ids, 2]
                offsets = [(0, 0), *offsets, (0, 0)]
            rows.append(ids)
            offset_rows.append(offsets)
        width = max(len(row) for row in rows)
        padded_ids = []
        padded_offsets = []
        attention = []
        for ids, offsets in zip(rows, offset_rows):
            pad = width - len(ids)
            padded_ids.append([*ids, *([0] * pad)])
            padded_offsets.append([*offsets, *([(0, 0)] * pad)])
            attention.append([*(1 for _ in ids), *([0] * pad)])
        if return_tensors == "pt":
            output = {"input_ids": torch.tensor(padded_ids), "attention_mask": torch.tensor(attention)}
            if return_offsets_mapping:
                output["offset_mapping"] = torch.tensor(padded_offsets)
            return output
        output = {"input_ids": rows[0] if single else padded_ids}
        if return_offsets_mapping:
            output["offset_mapping"] = offset_rows[0] if single else padded_offsets
        return output

    def convert_ids_to_tokens(self, input_ids):
        return [f"tok{int(token_id)}" for token_id in input_ids]


class FakeNllbEncoder:
    def __init__(self, model):
        self.model = model

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=True, return_dict=True):
        self.model.max_seen_width = max(self.model.max_seen_width, int(input_ids.shape[1]))
        batch, width = input_ids.shape
        values = torch.arange(batch * width * self.model.config.d_model, dtype=torch.float32).reshape(
            batch,
            width,
            self.model.config.d_model,
        )
        return type("Output", (), {"hidden_states": (values,)})()


class FakeNllbModel:
    def __init__(self):
        self.config = type("Config", (), {"d_model": 4})()
        self.max_seen_width = 0

    def get_encoder(self):
        return FakeNllbEncoder(self)


def test_nllb_teacher_chunks_texts_that_exceed_encoder_limit():
    text = " ".join(f"w{i}" for i in range(5))
    tokenizer = FakeNllbTokenizer()
    chunks = split_text_for_nllb(text, tokenizer, max_encoder_tokens=4)
    assert len(chunks) == 3
    assert all(nllb_token_count(tokenizer, chunk.text) <= 4 for chunk in chunks)

    teacher = object.__new__(NllbTeacher)
    teacher.tokenizer = tokenizer
    teacher.model = FakeNllbModel()
    teacher.device = torch.device("cpu")
    teacher.dtype = torch.float32
    teacher.batch_size = 8
    teacher.max_encoder_tokens = 4
    teacher.layer_groups = ((0,),)

    stats = {
        "tokenize_seconds": 0.0,
        "forward_seconds": 0.0,
        "input_tokens": 0.0,
        "padded_tokens": 0.0,
    }
    encoded = teacher.encode_missing_texts([text], stats)[text]

    assert teacher.model.max_seen_width <= 4
    assert len(encoded.pieces) == 5
    assert [piece[1] for piece in encoded.pieces] == [text.index(f"w{i}") for i in range(5)]
    assert encoded.group_hidden.shape[0] == 1


def test_nllb_text_chunking_prefers_previous_sentence_boundary():
    text = "w0 w1 . w2 w3 w4 w5"
    tokenizer = FakeNllbTokenizer()

    chunks = split_text_for_nllb(text, tokenizer, max_encoder_tokens=6)

    assert [chunk.text for chunk in chunks] == ["w0 w1 .", "w2 w3 w4 w5"]
    assert [piece for chunk in chunks for piece in chunk.text.split()] == text.split()
    assert chunks[1].start == text.index("w2")
    assert all(nllb_token_count(tokenizer, chunk.text) <= 6 for chunk in chunks)


def test_parallel_nllb_teacher_chunks_texts_that_exceed_encoder_limit():
    text = " ".join(f"w{i}" for i in range(5))
    teacher = object.__new__(ParallelNllbTeacher)
    teacher.tokenizer = FakeNllbTokenizer()
    teacher.model = FakeNllbModel()
    teacher.source_lang = "tur_Latn"
    teacher.target_lang = "eng_Latn"
    teacher.device = torch.device("cpu")
    teacher.dtype = torch.float32
    teacher.batch_size = 8
    teacher.align_layer = 0
    teacher.max_encoder_tokens = 4
    teacher.set_lang = lambda _lang: None

    encoded = teacher.encode_texts([0], [text], "tur_Latn")[0]

    assert teacher.model.max_seen_width <= 4
    assert len(encoded.pieces) == 5
    assert encoded.hidden_states[0].shape == encoded.align_hidden.shape


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


def test_sentence_chunk_helpers_preserve_text_coverage():
    text = "Yapacak misin bunu? ... Gidiyorsun. A. B. Moliere geldi."
    chunks = ["Yapacak misin bunu? ", "... ", "Gidiyorsun. ", "A. B. Moliere geldi."]
    merged = merge_punctuation_sentence_chunks(chunks)

    assert merged == ["Yapacak misin bunu? ... ", "Gidiyorsun. ", "A. B. Moliere geldi."]
    assert sentence_texts_from_chunks(text, merged) == [
        "Yapacak misin bunu? ...",
        "Gidiyorsun.",
        "A. B. Moliere geldi.",
    ]


def test_context_dil_sentence_split_counts_sentence_rows(tmp_path: Path):
    class FakeSentenceSplitter:
        def split(self, _text: str) -> list[str]:
            return ["kirmizi araba", "red car"]

    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"text": "kirmizi araba. red car."}) + "\n", encoding="utf-8")
    tokenizer = tiny_tokenizer()
    config = tiny_config(tokenizer)
    dataset = ContextDilBatchDataset(
        data,
        config,
        tokenizer,
        batch_size=6,
        read_chars=1024,
        repeat=True,
        sentence_split=True,
    )
    dataset._sentence_splitter = FakeSentenceSplitter()

    batch = next(dataset.iter_once(worker_id=0, worker_count=1))

    assert batch["source_row_count"] == 2
    assert batch["target_unit_count"] == 6
    assert batch["teacher_texts"][:3] == ["kirmizi araba", "kirmizi araba", "kirmizi araba"]
    assert batch["teacher_texts"][3:] == ["red car", "red car", "red car"]
