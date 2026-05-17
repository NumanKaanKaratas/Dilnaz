from pathlib import Path
from types import SimpleNamespace

import torch

from dilnaz.models.dil import DilConfig, compose_factorized_latent, normalize_semantic_latents
from dilnaz.models.naz import Naz, NazConfig
from dilnaz.models.naz.outputs import NazDynamicsOutput
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
        semantic_latent_size=12,
        surface_latent_size=4,
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
    )
    naz_config = NazConfig(
        latent_size=dil_config.latent_size,
        semantic_latent_size=dil_config.semantic_latent_size,
        surface_latent_size=dil_config.surface_latent_size,
        mtp_horizons=2,
    )
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
        target = torch.zeros(12, dil_config.latent_size)
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


def test_naz_mixture_loss_splits_semantic_and_surface_terms():
    config = NazConfig(
        latent_size=16,
        semantic_latent_size=12,
        surface_latent_size=4,
        mtp_horizons=1,
        mtp_loss_weights=(1.0,),
        num_semantic_candidates=1,
    )
    semantic = normalize_semantic_latents(torch.ones(1, 1, 1, config.semantic_latent_size))
    target = compose_factorized_latent(
        semantic,
        torch.zeros(1, 1, 1, config.surface_latent_size),
    )
    selected = compose_factorized_latent(
        semantic,
        torch.ones(1, 1, 1, config.surface_latent_size),
    )
    dynamics = NazDynamicsOutput(
        candidate_latents=selected.unsqueeze(3),
        selected_latents=selected,
        router_logits=torch.zeros(1, 1, 1, 1),
        selected_indices=torch.zeros(1, 1, 1, dtype=torch.long),
    )

    class Harness:
        def horizon_loss_weights(self, device, dtype):
            return Naz.horizon_loss_weights(self, device, dtype)

    harness = Harness()
    harness.config = config
    harness.mixture_sigma = torch.ones(config.mtp_horizons)
    losses = Naz.semantic_mixture_losses(
        harness,
        dynamics,
        target,
        torch.ones(1, 1, 1, dtype=torch.bool),
    )

    assert losses["chosen_mse"].item() == 0.0
    assert losses["surface_mse"].item() == config.surface_latent_size


def test_naz_repetition_cosine_ignores_surface_tail():
    config = NazConfig(
        latent_size=16,
        semantic_latent_size=12,
        surface_latent_size=4,
        mtp_horizons=1,
        mtp_loss_weights=(1.0,),
    )
    semantic = normalize_semantic_latents(torch.ones(1, config.semantic_latent_size))
    previous = compose_factorized_latent(semantic, -torch.ones(1, config.surface_latent_size))
    predicted = compose_factorized_latent(semantic, torch.ones(1, config.surface_latent_size))

    class FakeStudent:
        def embed_semantic_states(self, states):
            return states

    class FakeTransformer:
        def __call__(self, inputs_embeds, attention_mask, past_key_values, use_cache, max_cache_length):
            return SimpleNamespace(last_hidden_state=inputs_embeds, past_key_values=None)

    class FakeHead:
        def __call__(self, hidden_states):
            return NazDynamicsOutput(
                selected_latents=predicted.view(1, 1, 1, -1),
                selected_indices=torch.zeros(1, 1, 1, dtype=torch.long),
            )

    class Harness:
        pass

    harness = Harness()
    harness.config = config
    harness.min_new_tokens = 0
    harness.repetition_cos_threshold = 0.99
    harness.student_core = FakeStudent()
    harness.transformer = FakeTransformer()
    harness.semantic_head = FakeHead()

    step = next(
        Naz._generate_stream_from_semantic_states(
            harness,
            previous.view(1, 1, -1),
            torch.ones(1, 1, dtype=torch.bool),
            max_new_tokens=1,
            min_new_tokens=0,
            repetition_cos_threshold=0.99,
        )
    )

    assert step.latent_cos_to_previous.item() > 0.999
    assert step.should_stop.item()
