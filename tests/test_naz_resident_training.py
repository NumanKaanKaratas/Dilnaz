from pathlib import Path
from types import SimpleNamespace

import torch

from dilnaz.models.dil import DilConfig
from dilnaz.models.naz import NazConfig
from dilnaz.train.naz import train as naz_train


def test_semantic_cache_spans_respect_surface_bucket_width():
    offsets = [0, 1000, 2000, 7000, 9000, 10000]

    spans = list(
        naz_train.semantic_cache_spans(
            token_count=5,
            chunk_tokens=5,
            max_surface_width=8192,
            span_width=lambda start, end: offsets[end] - offsets[start],
        )
    )

    assert spans == [(0, 3), (3, 5)]


def test_resident_source_uses_dil_surface_config(monkeypatch, tmp_path: Path):
    dil_config = DilConfig(
        vocab_size=32,
        pad_token_id=0,
        eos_token_id=1,
        hidden_size=32,
        intermediate_size=64,
        latent_size=16,
        max_surface_pieces_per_unit=16,
        surface_bucket_sizes=(8, 32),
        encoder_context_layers=2,
        encoder_layer_pattern=("sliding", "global"),
        encoder_attention_heads=4,
        encoder_key_value_heads=2,
        encoder_head_dim=8,
        encoder_intermediate_size=64,
        encoder_attention_window=4,
        writer_num_layers=1,
        writer_word_mixer_layers=1,
        writer_word_attention_heads=4,
    )
    naz_config = NazConfig(latent_size=dil_config.latent_size, mtp_horizons=2)
    seen = {}

    def fake_build_token_cache(*args, **kwargs):
        return tmp_path / "ids.npy", tmp_path / "offsets.npy", 12

    class FakeResidentNazBatcher:
        def __init__(self, ids_path, offsets_path, token_count, config, *args):
            seen["config"] = config
            self.token_count = token_count

    class FakeSemanticBatcher:
        def __init__(self, semantic_states, target_latents, *args, **kwargs):
            self.semantic_states = semantic_states
            self.target_latents = target_latents

    def fake_build_resident_semantic_cache(*args, **kwargs):
        semantic = torch.zeros(12, dil_config.latent_size)
        target = torch.zeros(12, naz_config.mtp_horizons, dil_config.latent_size)
        return semantic, target

    monkeypatch.setattr(naz_train, "build_token_cache", fake_build_token_cache)
    monkeypatch.setattr(naz_train, "ResidentNazBatcher", FakeResidentNazBatcher)
    monkeypatch.setattr(naz_train, "ResidentNazSemanticBatcher", FakeSemanticBatcher)
    monkeypatch.setattr(naz_train, "build_resident_semantic_cache", fake_build_resident_semantic_cache)

    trainer = SimpleNamespace(
        args=SimpleNamespace(
            token_cache_dir=None,
            output_dir=tmp_path,
            train_file=tmp_path / "train.jsonl",
            text_read_chars=1024,
            sequence_length=4,
            batch_size=2,
            seed=1,
            semantic_cache_chunk_tokens=8,
            eval_every=0,
        ),
        tokenizer=object(),
        dil_config=dil_config,
        config=naz_config,
        model=object(),
        autocast_enabled=False,
        device=torch.device("cpu"),
        start_step=0,
        train_iterator=None,
        eval_loader=None,
    )

    naz_train.NazBaseTrainer.prepare_resident_sources(trainer)

    assert seen["config"] is dil_config
    assert trainer.train_iterator is not None
