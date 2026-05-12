import sys
import shutil
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dilnaz"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dilnaz" / "train"))

from dil_data import context_offsets
from models.configuration_dil import DilConfig
from models.configuration_naz import NazConfig
from models.modeling_dil import (
    Dil,
    DilByteConvStem,
    DilGatedMLP,
    DilRMSNorm,
    angular_noise_like,
    normalize_semantic_latents,
)
from models.modeling_naz import Naz
from models.naz_backbone import SemanticDeltaMixer, SemanticGlobalAttention, SparseMoEFeedForward, ZeroCenteredRMSNorm
from naz_data import (
    MemmapNazSemanticBatcher,
    ResidentNazBatcher,
    ResidentNazSemanticBatcher,
    StreamingTextNazDataset,
)
from train_naz import (
    NazFinetuneTrainer,
    NazPretrainTrainer,
    build_resident_semantic_cache,
    make_trainer,
    parse_args as parse_naz_args,
    validate_args as validate_naz_args,
)
from train_dil_writer import (
    build_future_latents,
    calibrate_emit_logits,
    freeze_for_writer_only,
    resolve_future_mode,
    sample_diffusion_step,
    writer_only_forward,
    writer_only_metrics,
)
from interface_naz import SlidingWriterBuffer, stream_text as stream_naz_text


def grad_abs_sum(parameter: torch.nn.Parameter) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().abs().sum())


def fixture_tokenizer():
    return __import__("tokenization").HybridTokenizer.from_file(
        Path(__file__).resolve().parents[1] / "dilnaz" / "tokenization" / "hybrid_surface_vocab.json"
    )


def fixture_tokenizer_path() -> Path:
    return Path(__file__).resolve().parents[1] / "dilnaz" / "tokenization" / "hybrid_surface_vocab.json"


class FakeGeneratedDil:
    def __init__(self, tokenizer, max_word_bytes: int, decoded_steps: list[str | None], commit_score_values: list[float] | None = None):
        self.tokenizer = tokenizer
        self.max_word_bytes = max_word_bytes
        self.decoded_steps = decoded_steps
        self.commit_score_values = [] if commit_score_values is None else commit_score_values
        self.calls = 0
        self.cursor = 0
        self.writer_kwargs = []
        self.config = DilConfig(
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            hidden_size=8,
            intermediate_size=16,
            num_encoder_layers=2,
            latent_size=1,
            max_word_bytes=max_word_bytes,
            context_radius=1,
        )

    def decode_semantic(self, latent: torch.Tensor):
        self.calls += 1
        batch_size = latent.shape[0]
        token_ids = torch.full((batch_size, self.max_word_bytes), self.tokenizer.pad_token_id, dtype=torch.long)
        token_masks = torch.zeros_like(token_ids, dtype=torch.bool)
        lengths = torch.zeros((batch_size,), dtype=torch.long)
        for row_idx in range(batch_size):
            value = self.decoded_steps[self.cursor]
            self.cursor += 1
            if value is None:
                continue
            if value == "<eos>":
                ids = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
            else:
                segment = next(segment for segment in self.tokenizer.encode_segments(value) if segment.piece_len > 0)
                ids = torch.tensor(segment.token_ids, dtype=torch.long)
            token_ids[row_idx, : ids.numel()] = ids
            token_masks[row_idx, : ids.numel()] = True
            lengths[row_idx] = ids.numel()
        return token_ids, token_masks, lengths

    def decode_semantic_window(self, semantic: torch.Tensor, **writer_kwargs):
        call_idx = len(self.writer_kwargs)
        self.writer_kwargs.append(
            {
                key: value.detach().clone() if torch.is_tensor(value) else value
                for key, value in writer_kwargs.items()
            }
        )
        self.calls += 1
        batch_size, window_size = semantic.shape[:2]
        token_ids = torch.full((batch_size, window_size, self.max_word_bytes), self.tokenizer.pad_token_id, dtype=torch.long)
        token_masks = torch.zeros_like(token_ids, dtype=torch.bool)
        lengths = torch.zeros((batch_size, window_size), dtype=torch.long)
        score_value = self.commit_score_values[call_idx] if call_idx < len(self.commit_score_values) else 1.0
        commit_scores = torch.full(
            (batch_size, window_size, self.config.writer_max_positions),
            score_value,
            dtype=torch.float32,
        )
        for row_idx in range(batch_size):
            for slot_idx in range(self.config.writer_left_frozen, window_size):
                if self.cursor >= len(self.decoded_steps):
                    break
                value = self.decoded_steps[self.cursor]
                self.cursor += 1
                if value is None:
                    continue
                if value == "<eos>":
                    ids = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
                else:
                    segment = next(segment for segment in self.tokenizer.encode_segments(value) if segment.piece_len > 0)
                    ids = torch.tensor(segment.token_ids, dtype=torch.long)
                token_ids[row_idx, slot_idx, : ids.numel()] = ids
                token_masks[row_idx, slot_idx, : ids.numel()] = True
                lengths[row_idx, slot_idx] = ids.numel()
        return token_ids, token_masks, lengths, commit_scores


class FakeGeneratedNaz:
    def __init__(self, tokenizer, max_word_bytes: int, decoded_steps: list[str | None], commit_score_values: list[float] | None = None):
        self.dil_model = FakeGeneratedDil(tokenizer, max_word_bytes, decoded_steps, commit_score_values)
        self.encode_calls = 0
        self.generated_prompt_latents = None

    def eval(self):
        return self

    def train(self):
        return self

    def encode_sequence_latents(self, input_ids, word_masks, unit_mask):
        del word_masks, unit_mask
        self.encode_calls += 1
        values = torch.arange(input_ids.shape[1], dtype=torch.float32, device=input_ids.device).view(1, -1, 1)
        return values.expand(input_ids.shape[0], -1, self.dil_model.config.latent_size)

    def generate_stream(
        self,
        input_ids,
        word_masks,
        unit_mask,
        max_new_tokens,
        min_new_tokens,
        repetition_cos_threshold,
        prompt_latents=None,
    ):
        del word_masks, unit_mask, min_new_tokens, repetition_cos_threshold
        self.generated_prompt_latents = prompt_latents
        batch_size = input_ids.shape[0]
        for _ in range(max_new_tokens):
            yield SimpleNamespace(
                latent=torch.zeros((batch_size, 1), dtype=torch.float32),
                future_latents=torch.zeros((batch_size, 2, 1), dtype=torch.float32),
                should_stop=torch.zeros((batch_size,), dtype=torch.bool),
            )


def tiny_config() -> DilConfig:
    return DilConfig(
        vocab_size=64,
        pad_token_id=0,
        eos_token_id=1,
        hidden_size=32,
        intermediate_size=64,
        num_encoder_layers=2,
        latent_size=16,
        max_word_bytes=4,
        context_radius=2,
        dil_dropout=0.0,
    )


def tiny_naz_config(tmp_path, dil_config: DilConfig) -> NazConfig:
    return NazConfig(
        dil_path=str(tmp_path),
        vocab_size=dil_config.vocab_size,
        pad_token_id=dil_config.pad_token_id,
        eos_token_id=dil_config.eos_token_id,
        max_word_bytes=dil_config.max_word_bytes,
        latent_size=dil_config.latent_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        full_attention_interval=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_conv_kernel_size=4,
        reconstruction_loss_weight=1.0,
    )


def save_tiny_trainer_dil_checkpoint(checkpoint_dir: Path) -> DilConfig:
    tokenizer = fixture_tokenizer()
    config = DilConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        hidden_size=16,
        intermediate_size=32,
        num_encoder_layers=2,
        latent_size=8,
        max_word_bytes=8,
        context_radius=1,
        byte_conv_layers=1,
        dil_dropout=0.0,
        writer_num_layers=1,
        writer_dropout=0.0,
    )
    model = Dil(config)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(checkpoint_dir)
    shutil.copyfile(fixture_tokenizer_path(), checkpoint_dir / config.tokenizer_vocab_file)
    torch.save(
        {
            "format_version": config.checkpoint_format_version,
            "model_state_dict": model.state_dict(),
        },
        checkpoint_dir / "checkpoint.pt",
    )
    return config


def tiny_pretrain_args(tmp_path: Path, train_file: Path, dil_dir: Path, output_name: str = "naz_pretrain"):
    return parse_naz_args(
        [
            "--stage",
            "pretrain",
            "--train-file",
            str(train_file),
            "--dil-checkpoint-dir",
            str(dil_dir),
            "--output-dir",
            str(tmp_path / output_name),
            "--compile-mode",
            "off",
            "--data-mode",
            "streaming",
            "--max-steps",
            "1",
            "--batch-size",
            "1",
            "--eval-batch-size",
            "1",
            "--sequence-length",
            "2",
            "--learning-rate",
            "1e-4",
            "--weight-decay",
            "0.0",
            "--warmup-steps",
            "0",
            "--max-grad-norm",
            "1.0",
            "--log-every",
            "1",
            "--checkpoint-every",
            "0",
            "--eval-every",
            "0",
            "--max-eval-batches",
            "1",
            "--text-read-chars",
            "256",
            "--num-workers",
            "0",
            "--prefetch-factor",
            "1",
            "--seed",
            "1",
            "--hidden-size",
            "32",
            "--intermediate-size",
            "64",
            "--num-hidden-layers",
            "1",
            "--num-attention-heads",
            "4",
            "--num-key-value-heads",
            "2",
            "--head-dim",
            "8",
            "--full-attention-interval",
            "1",
            "--linear-key-head-dim",
            "8",
            "--linear-value-head-dim",
            "8",
            "--linear-num-key-heads",
            "4",
            "--linear-num-value-heads",
            "4",
            "--linear-conv-kernel-size",
            "4",
            "--num-semantic-candidates",
            "2",
            "--mtp-horizons",
            "2",
            "--mtp-loss-weights",
            "1.0",
            "0.3",
            "--mixture-sigma",
            "0.55",
            "--usage-balance-weight",
            "0.05",
            "--router-responsibility-weight",
            "1.0",
            "--moe-num-experts",
            "2",
            "--moe-top-k",
            "1",
            "--moe-layers",
            "1",
            "--moe-balance-weight",
            "0.01",
            "--naz-input-jitter-prob",
            "0.0",
        ]
    )


def tiny_finetune_args(tmp_path: Path, train_file: Path, init_checkpoint: Path):
    return parse_naz_args(
        [
            "--stage",
            "finetune",
            "--train-file",
            str(train_file),
            "--init-naz-checkpoint",
            str(init_checkpoint),
            "--output-dir",
            str(tmp_path / "naz_finetune"),
            "--compile-mode",
            "off",
            "--data-mode",
            "streaming",
            "--max-steps",
            "1",
            "--batch-size",
            "1",
            "--eval-batch-size",
            "1",
            "--sequence-length",
            "2",
            "--learning-rate",
            "3e-5",
            "--weight-decay",
            "0.0",
            "--warmup-steps",
            "0",
            "--max-grad-norm",
            "1.0",
            "--log-every",
            "1",
            "--checkpoint-every",
            "0",
            "--eval-every",
            "0",
            "--max-eval-batches",
            "1",
            "--text-read-chars",
            "256",
            "--num-workers",
            "0",
            "--prefetch-factor",
            "1",
            "--seed",
            "1",
        ]
    )


def write_naz_trainer_text(path: Path) -> None:
    path.write_text(
        "2 + 2 = 4\n3 + 1 = 4\n5 + 5 = 10\n7 + 8 = 15\n",
        encoding="utf-8",
    )


def test_dil_config_uses_left_context_contract():
    config = tiny_config()

    assert config.context_radius == 2
    assert config.context_size == 5
    assert config.target_index == 2
    assert list(context_offsets(2)) == [-2, -1, 0, 1, 2]
    assert config.checkpoint_format_version == 24
    assert config.writer_max_positions == config.max_word_bytes + 1
    assert config.writer_stop_token_id == config.vocab_size
    assert config.writer_vocab_size == config.vocab_size + 1
    assert not hasattr(config, "context_left_radius")


def test_dil_rejects_pre_parallel_writer_checkpoint_family():
    config = tiny_config()
    config.checkpoint_format_version = 13

    try:
        Dil(config)
    except ValueError as error:
        assert "checkpoint_format_version=24" in str(error)
    else:
        raise AssertionError("Dil accepted stale checkpoint_format_version")


def test_dil_native_semantic_has_no_post_fit_normalizer_symbols():
    config = tiny_config()
    model = Dil(config)

    assert not hasattr(model, "semantic_normalizer")
    assert not hasattr(model, "encode_raw")
    assert not hasattr(model, "denormalize")


def test_dil_encode_returns_native_normalized_semantic_contract():
    config = tiny_config()
    model = Dil(config).eval()
    input_ids = torch.tensor(
        [
            [[2, 0, 0, 0], [3, 0, 0, 0], [4, 5, 0, 0], [6, 0, 0, 0], [7, 0, 0, 0]],
        ],
        dtype=torch.long,
    )
    word_masks = input_ids.ne(config.pad_token_id)

    normalized = model.encode(input_ids, word_masks)
    normalized_with_layers, layer_vectors = model.encode(input_ids, word_masks, output_hidden_states=True)
    norms = normalized.float().norm(dim=-1)

    assert torch.allclose(norms, torch.full_like(norms, config.latent_size**0.5), atol=1e-5)
    assert torch.allclose(normalized_with_layers, normalized)
    assert len(layer_vectors) == config.num_encoder_layers


def test_normalize_semantic_latents_maps_to_fixed_radius():
    config = tiny_config()
    latents = torch.randn(6, config.latent_size)
    latents[0].zero_()
    normalized = normalize_semantic_latents(latents)

    assert torch.allclose(
        normalized.norm(dim=-1),
        torch.full((6,), config.latent_size**0.5),
        atol=1e-5,
    )
    assert normalized[0, 0] > 0.0


def test_angular_noise_preserves_norm_and_cosine_range():
    config = tiny_config()
    latents = normalize_semantic_latents(torch.randn(8, config.latent_size))
    min_cos = torch.full((8,), 0.985)
    max_cos = torch.full((8,), 0.995)

    noised = angular_noise_like(latents, min_cos, max_cos)
    cosine = torch.nn.functional.cosine_similarity(latents, noised, dim=-1)

    assert torch.allclose(noised.norm(dim=-1), latents.norm(dim=-1), atol=1e-5)
    assert torch.all(cosine >= min_cos - 1e-5)
    assert torch.all(cosine <= max_cos + 1e-5)


def test_writer_training_semantic_noise_is_training_only_after_warmup():
    config = tiny_config()
    config.writer_noise_warmup_steps = 0
    config.writer_noise_clean_ratio = 0.0
    config.writer_noise_easy_ratio = 1.0
    config.writer_noise_mid_ratio = 0.0
    config.writer_noise_hard_ratio = 0.0
    model = Dil(config)
    semantic = normalize_semantic_latents(torch.randn(4, config.latent_size))

    model.eval()
    assert torch.equal(model.writer_training_semantic(semantic, training_step=1), semantic)

    model.train()
    noised = model.writer_training_semantic(semantic, training_step=1)
    cosine = torch.nn.functional.cosine_similarity(semantic, noised, dim=-1)

    assert not torch.equal(noised, semantic)
    assert torch.all(cosine >= config.writer_noise_easy_min_cos - 1e-5)
    assert torch.all(cosine <= config.writer_noise_easy_max_cos + 1e-5)


def test_writer_zone_noise_downgrades_left_and_upgrades_right():
    config = tiny_config()
    config.writer_noise_warmup_steps = 0
    config.writer_noise_clean_ratio = 0.0
    config.writer_noise_easy_ratio = 1.0
    config.writer_noise_mid_ratio = 0.0
    config.writer_noise_hard_ratio = 0.0
    model = Dil(config).train()
    semantic = normalize_semantic_latents(torch.randn(1, 3, config.latent_size))
    zone_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    window_mask = torch.ones((1, 3), dtype=torch.bool)

    noised = model.writer_training_semantic(semantic, training_step=1, zone_ids=zone_ids, window_mask=window_mask)
    cosine = torch.nn.functional.cosine_similarity(semantic, noised, dim=-1)

    assert torch.allclose(noised[:, 0], semantic[:, 0])
    assert torch.all(cosine[:, 1] >= config.writer_noise_easy_min_cos - 1e-5)
    assert torch.all(cosine[:, 1] <= config.writer_noise_easy_max_cos + 1e-5)
    assert torch.all(cosine[:, 2] >= config.writer_noise_mid_min_cos - 1e-5)
    assert torch.all(cosine[:, 2] <= config.writer_noise_mid_max_cos + 1e-5)


def test_dil_byte_conv_stem_preserves_shape_and_padding_mask():
    config = tiny_config()
    stem = DilByteConvStem(config).eval()
    hidden_states = torch.randn(2, config.context_size, config.max_word_bytes, config.hidden_size)
    word_masks = torch.ones(2, config.context_size, config.max_word_bytes, dtype=torch.bool)
    word_masks[:, :, -1] = False

    output = stem(hidden_states, word_masks)

    assert output.shape == hidden_states.shape
    assert torch.equal(output[:, :, -1].abs().sum(dim=-1), torch.zeros(2, config.context_size))


def test_dil_context_attention_keeps_context_activations_read_only():
    config = tiny_config()
    model = Dil(config)
    token_states = torch.randn(2, config.context_size, config.hidden_size, requires_grad=True)
    token_mask = torch.ones(2, config.context_size, dtype=torch.bool)

    model.encoder.target_conditioned_by_context(token_states, token_mask).sum().backward()

    context_grad = token_states.grad.index_select(1, model.encoder.context_indices)
    target_grad = token_states.grad[:, config.target_index]
    assert float(context_grad.abs().sum()) == 0.0
    assert float(target_grad.abs().sum()) > 0.0


def test_dil_parallel_writer_outputs_surface_stop_logits():
    config = tiny_config()
    model = Dil(config)
    semantic = torch.randn(3, config.latent_size)

    token_logits = model.writer_outputs(semantic)
    token_ids, token_masks, lengths = model.decode_semantic(semantic)

    assert token_logits.shape == (3, config.writer_max_positions, config.writer_vocab_size)
    assert token_ids.shape == (3, config.max_word_bytes)
    assert token_masks.shape == (3, config.max_word_bytes)
    assert lengths.shape == (3,)


def test_dil_sliding_writer_outputs_token_state_and_emit_logits():
    config = tiny_config()
    model = Dil(config)
    semantic = torch.randn(2, config.writer_sliding_window_size, config.latent_size)
    surface_state = torch.full(
        (2, config.writer_sliding_window_size, config.writer_max_positions),
        -100,
        dtype=torch.long,
    )
    frozen_mask = torch.zeros_like(surface_state, dtype=torch.bool)
    zone_ids = torch.full((2, config.writer_sliding_window_size), 1, dtype=torch.long)
    zone_ids[:, : config.writer_left_frozen] = 0
    zone_ids[:, config.writer_left_frozen + config.writer_active_size :] = 2
    output = model.writer_transition_outputs(
        semantic,
        surface_state=surface_state,
        frozen_mask=frozen_mask,
        zone_ids=zone_ids,
    )

    assert output.token_logits.shape == (
        2,
        config.writer_sliding_window_size,
        config.writer_max_positions,
        config.writer_vocab_size,
    )
    assert output.state_valid_logits.shape == (2, config.writer_sliding_window_size, config.writer_max_positions)
    assert output.emit_logits.shape == (2, config.writer_sliding_window_size, config.writer_max_positions)
    assert config.writer_empty_token_id != config.writer_stop_token_id
    assert config.writer_state_vocab_size == config.writer_vocab_size + 1


def test_dil_refinement_steps_default_matches_single_step():
    config = tiny_config()
    model = Dil(config).eval()
    semantic = torch.randn(2, config.writer_sliding_window_size, config.latent_size)
    surface_state = torch.full(
        (2, config.writer_sliding_window_size, config.writer_max_positions),
        -100,
        dtype=torch.long,
    )
    frozen_mask = torch.zeros_like(surface_state, dtype=torch.bool)
    frozen_mask[:, : config.writer_left_frozen] = True
    zone_ids = torch.full((2, config.writer_sliding_window_size), 1, dtype=torch.long)
    zone_ids[:, : config.writer_left_frozen] = 0
    zone_ids[:, config.writer_left_frozen + config.writer_active_size :] = 2
    out1 = model.writer_transition_outputs(
        semantic, surface_state=surface_state, frozen_mask=frozen_mask, zone_ids=zone_ids
    )
    out_explicit = model.writer_transition_outputs(
        semantic, surface_state=surface_state, frozen_mask=frozen_mask, zone_ids=zone_ids, refinement_steps=1
    )
    assert torch.allclose(out1.token_logits, out_explicit.token_logits)


def test_dil_refinement_changes_output_with_frozen_preserved():
    config = tiny_config()
    config.writer_refinement_steps = 2
    model = Dil(config).eval()
    semantic = torch.randn(2, config.writer_sliding_window_size, config.latent_size)
    surface_state = torch.full(
        (2, config.writer_sliding_window_size, config.writer_max_positions),
        -100,
        dtype=torch.long,
    )
    surface_state[:, 0, 0] = 5
    frozen_mask = torch.zeros_like(surface_state, dtype=torch.bool)
    frozen_mask[:, 0] = True
    zone_ids = torch.full((2, config.writer_sliding_window_size), 1, dtype=torch.long)
    zone_ids[:, 0] = 0
    zone_ids[:, config.writer_left_frozen + config.writer_active_size :] = 2
    out1 = model.writer_transition_outputs(
        semantic, surface_state=surface_state, frozen_mask=frozen_mask, zone_ids=zone_ids, refinement_steps=1
    )
    out2 = model.writer_transition_outputs(
        semantic, surface_state=surface_state, frozen_mask=frozen_mask, zone_ids=zone_ids, refinement_steps=2
    )
    assert not torch.allclose(out1.token_logits, out2.token_logits)


def test_dil_position_age_changes_writer_transition_output():
    config = tiny_config()
    model = Dil(config).eval()
    semantic = torch.randn(1, config.writer_sliding_window_size, config.latent_size)
    surface_state = torch.full(
        (1, config.writer_sliding_window_size, config.writer_max_positions),
        -100,
        dtype=torch.long,
    )
    zone_ids = torch.full((1, config.writer_sliding_window_size), 1, dtype=torch.long)
    fresh_age = torch.zeros((1, config.writer_sliding_window_size), dtype=torch.long)
    stale_age = torch.full((1, config.writer_sliding_window_size), config.writer_max_position_age, dtype=torch.long)

    fresh = model.writer_transition_outputs(
        semantic,
        surface_state=surface_state,
        zone_ids=zone_ids,
        position_age=fresh_age,
    )
    stale = model.writer_transition_outputs(
        semantic,
        surface_state=surface_state,
        zone_ids=zone_ids,
        position_age=stale_age,
    )

    assert not torch.allclose(fresh.token_logits, stale.token_logits)


def test_dil_zone_condition_changes_writer_transition_output():
    config = tiny_config()
    model = Dil(config).eval()
    semantic = torch.randn(1, config.writer_sliding_window_size, config.latent_size)
    surface_state = torch.full(
        (1, config.writer_sliding_window_size, config.writer_max_positions),
        -100,
        dtype=torch.long,
    )
    active_zone = torch.full((1, config.writer_sliding_window_size), model.writer.ZONE_ACTIVE, dtype=torch.long)
    right_zone = torch.full((1, config.writer_sliding_window_size), model.writer.ZONE_RIGHT, dtype=torch.long)

    active = model.writer_transition_outputs(semantic, surface_state=surface_state, zone_ids=active_zone)
    right = model.writer_transition_outputs(semantic, surface_state=surface_state, zone_ids=right_zone)

    assert not torch.allclose(active.token_logits, right.token_logits)


def test_dil_byte_state_cross_attention_handles_empty_state_without_nan():
    config = tiny_config()
    model = Dil(config).eval()
    semantic = torch.randn(2, config.writer_sliding_window_size, config.latent_size)
    surface_state = torch.full(
        (2, config.writer_sliding_window_size, config.writer_max_positions),
        -100,
        dtype=torch.long,
    )
    surface_state_mask = torch.zeros_like(surface_state)
    frozen_mask = torch.zeros_like(surface_state, dtype=torch.bool)

    output = model.writer_transition_outputs(
        semantic,
        surface_state=surface_state,
        surface_state_mask=surface_state_mask,
        frozen_mask=frozen_mask,
    )

    assert torch.isfinite(output.token_logits).all()
    assert torch.isfinite(output.state_valid_logits).all()
    assert torch.isfinite(output.emit_logits).all()


def test_dil_known_byte_state_changes_writer_transition_output():
    config = tiny_config()
    model = Dil(config).eval()
    semantic = torch.randn(1, config.writer_sliding_window_size, config.latent_size)
    state_a = torch.full(
        (1, config.writer_sliding_window_size, config.writer_max_positions),
        -100,
        dtype=torch.long,
    )
    state_b = state_a.clone()
    state_a[:, config.writer_left_frozen, 0] = 5
    state_b[:, config.writer_left_frozen, 0] = 6
    surface_state_mask = torch.zeros_like(state_a)
    surface_state_mask[:, config.writer_left_frozen, 0] = model.writer.STATE_KNOWN
    frozen_mask = surface_state_mask.gt(0)

    output_a = model.writer_transition_outputs(
        semantic,
        surface_state=state_a,
        surface_state_mask=surface_state_mask,
        frozen_mask=frozen_mask,
    )
    output_b = model.writer_transition_outputs(
        semantic,
        surface_state=state_b,
        surface_state_mask=surface_state_mask,
        frozen_mask=frozen_mask,
    )

    assert not torch.allclose(output_a.token_logits, output_b.token_logits)


def test_dil_future_attention_accepts_short_horizon_inputs():
    config = tiny_config()
    model = Dil(config).eval()
    semantic = torch.randn(1, config.writer_sliding_window_size, config.latent_size)
    future_latents = torch.randn(1, config.writer_sliding_window_size, 2, config.latent_size)

    output = model.writer_transition_outputs(semantic, future_latents=future_latents)

    assert output.token_logits.shape == (
        1,
        config.writer_sliding_window_size,
        config.writer_max_positions,
        config.writer_vocab_size,
    )
    assert torch.isfinite(output.token_logits).all()


def test_dil_forward_keeps_target_latent_shape():
    config = tiny_config()
    model = Dil(config)
    input_ids = torch.tensor(
        [
            [[2, 0, 0, 0], [3, 4, 0, 0], [5, 6, 7, 0], [14, 0, 0, 0], [15, 0, 0, 0]],
            [[8, 0, 0, 0], [9, 10, 0, 0], [11, 12, 13, 0], [16, 0, 0, 0], [17, 0, 0, 0]],
        ],
        dtype=torch.long,
    )
    word_masks = input_ids.ne(config.pad_token_id)
    teacher_layers = torch.randn(input_ids.shape[0], 4, config.latent_size)
    teacher_mask = torch.ones(input_ids.shape[0], dtype=torch.bool)

    outputs = model(
        input_ids=input_ids,
        word_masks=word_masks,
        teacher_layers=teacher_layers,
        teacher_mask=teacher_mask,
    )

    assert outputs.semantic.shape == (2, config.latent_size)
    assert isinstance(model.encoder.encoder_layers[0].mlp, DilGatedMLP)
    assert isinstance(model.encoder.encoder_layers[0].layernorm, DilRMSNorm)
    assert torch.equal(model.encoder.encoder_layers[0].layernorm.weight, torch.zeros_like(model.encoder.encoder_layers[0].layernorm.weight))
    assert hasattr(model, "writer")
    assert torch.isfinite(outputs.loss)


def test_dil_encoder_conditions_target_with_left_context():
    config = tiny_config()
    model = Dil(config).eval()
    target = torch.tensor([5, 6, 7, 0], dtype=torch.long)
    input_ids = torch.stack(
        [
            torch.stack(
                [
                    torch.tensor([2, 0, 0, 0]),
                    torch.tensor([3, 4, 0, 0]),
                    target,
                    torch.tensor([18, 0, 0, 0]),
                    torch.tensor([19, 0, 0, 0]),
                ]
            ),
            torch.stack(
                [
                    torch.tensor([16, 0, 0, 0]),
                    torch.tensor([17, 0, 0, 0]),
                    target,
                    torch.tensor([20, 0, 0, 0]),
                    torch.tensor([21, 0, 0, 0]),
                ]
            ),
        ]
    )
    word_masks = input_ids.ne(config.pad_token_id)

    with torch.no_grad():
        semantic = model.encoder(input_ids=input_ids, word_masks=word_masks)

    assert not torch.allclose(semantic[0], semantic[1])


def test_dil_encoder_uses_offset_order_for_context():
    config = tiny_config()
    model = Dil(config).eval()
    target = torch.tensor([5, 6, 7, 0], dtype=torch.long)
    input_ids = torch.stack(
        [
            torch.stack(
                [
                    torch.tensor([2, 0, 0, 0]),
                    torch.tensor([3, 0, 0, 0]),
                    target,
                    torch.tensor([4, 0, 0, 0]),
                    torch.tensor([6, 0, 0, 0]),
                ]
            ),
            torch.stack(
                [
                    torch.tensor([3, 0, 0, 0]),
                    torch.tensor([2, 0, 0, 0]),
                    target,
                    torch.tensor([6, 0, 0, 0]),
                    torch.tensor([4, 0, 0, 0]),
                ]
            ),
        ]
    )
    word_masks = input_ids.ne(config.pad_token_id)

    with torch.no_grad():
        semantic = model.encoder(input_ids=input_ids, word_masks=word_masks)

    assert not torch.allclose(semantic[0], semantic[1])


def test_semantic_losses_update_semantic_encoder():
    config = tiny_config()
    model = Dil(config)
    input_ids = torch.tensor(
        [
            [[2, 0, 0, 0], [3, 4, 0, 0], [5, 6, 7, 0], [14, 0, 0, 0], [15, 0, 0, 0]],
            [[8, 0, 0, 0], [9, 10, 0, 0], [11, 12, 13, 0], [16, 0, 0, 0], [17, 0, 0, 0]],
        ],
        dtype=torch.long,
    )
    word_masks = input_ids.ne(config.pad_token_id)
    teacher_layers = torch.randn(input_ids.shape[0], 4, config.latent_size)
    teacher_mask = torch.ones(input_ids.shape[0], dtype=torch.bool)

    outputs = model(
        input_ids=input_ids,
        word_masks=word_masks,
        teacher_layers=teacher_layers,
        teacher_mask=teacher_mask,
    )
    outputs.loss.backward()

    assert model.encoder.embed_tokens.weight.grad is not None
    assert model.encoder.context_q_proj.weight.grad is not None
    assert model.encoder.context_gate.weight.grad is not None
    assert hasattr(model, "writer")


def test_dil_writer_only_step_freezes_encoder():
    config = tiny_config()
    model = Dil(config)
    freeze_for_writer_only(model)
    input_ids = torch.tensor(
        [
            [[2, 0, 0, 0], [3, 4, 0, 0], [5, 6, 7, 0], [14, 0, 0, 0], [15, 0, 0, 0]],
            [[8, 0, 0, 0], [9, 10, 0, 0], [11, 12, 13, 0], [16, 0, 0, 0], [17, 0, 0, 0]],
        ],
        dtype=torch.long,
    )
    labels = torch.tensor(
        [
            [5, 6, 7, config.writer_stop_token_id, -100],
            [11, 12, 13, config.writer_stop_token_id, -100],
        ],
        dtype=torch.long,
    )
    batch = {
        "input_ids": input_ids,
        "word_masks": input_ids.ne(config.pad_token_id),
        "labels": labels,
    }

    loss, byte_acc, token_exact, stop_acc = writer_only_forward(model, batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(byte_acc)
    assert torch.isfinite(token_exact)
    assert torch.isfinite(stop_acc)
    assert grad_abs_sum(model.encoder.embed_tokens.weight) == 0.0
    assert grad_abs_sum(model.encoder.hidden_to_semantic.weight) == 0.0
    assert not hasattr(model, "semantic_normalizer")
    assert model.writer.token_head.weight.grad is not None


def test_sliding_writer_only_step_keeps_online_encoder_frozen():
    config = tiny_config()
    model = Dil(config)
    freeze_for_writer_only(model)
    batch_size = 1
    window_size = config.writer_sliding_window_size
    input_ids = torch.full(
        (batch_size, window_size, config.context_size, config.max_word_bytes),
        config.pad_token_id,
        dtype=torch.long,
    )
    input_ids[:, :, config.target_index, 0] = 2
    word_masks = input_ids.ne(config.pad_token_id)
    labels = torch.full((batch_size, window_size, config.writer_max_positions), -100, dtype=torch.long)
    labels[:, :, 0] = 2
    labels[:, :, 1] = config.writer_stop_token_id
    zone_ids = torch.full((batch_size, window_size), 1, dtype=torch.long)
    zone_ids[:, : config.writer_left_frozen] = 0
    zone_ids[:, config.writer_left_frozen + config.writer_active_size :] = 2
    window_mask = torch.ones((batch_size, window_size), dtype=torch.bool)
    batch = {
        "input_ids": input_ids,
        "word_masks": word_masks,
        "labels": labels,
        "zone_ids": zone_ids,
        "window_mask": window_mask,
    }

    loss, byte_acc, token_exact, stop_acc = writer_only_forward(model, batch, training_step=1)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(byte_acc)
    assert torch.isfinite(token_exact)
    assert torch.isfinite(stop_acc)
    assert grad_abs_sum(model.encoder.embed_tokens.weight) == 0.0
    assert grad_abs_sum(model.encoder.hidden_to_semantic.weight) == 0.0
    assert model.writer.token_head.weight.grad is not None
    assert model.writer.state_valid_head.weight.grad is not None
    assert model.writer.emit_head.weight.grad is not None


def test_sliding_writer_metrics_include_state_quality_ratios():
    config = tiny_config()
    model = Dil(config)
    freeze_for_writer_only(model)
    batch_size = 1
    window_size = config.writer_sliding_window_size
    input_ids = torch.full(
        (batch_size, window_size, config.context_size, config.max_word_bytes),
        config.pad_token_id,
        dtype=torch.long,
    )
    input_ids[:, :, config.target_index, 0] = 2
    labels = torch.full((batch_size, window_size, config.writer_max_positions), -100, dtype=torch.long)
    labels[:, :, 0] = 2
    labels[:, :, 1] = config.writer_stop_token_id
    zone_ids = torch.full((batch_size, window_size), model.writer.ZONE_ACTIVE, dtype=torch.long)
    zone_ids[:, : config.writer_left_frozen] = model.writer.ZONE_LEFT
    zone_ids[:, config.writer_left_frozen + config.writer_active_size :] = model.writer.ZONE_RIGHT
    batch = {
        "input_ids": input_ids,
        "word_masks": input_ids.ne(config.pad_token_id),
        "labels": labels,
        "zone_ids": zone_ids,
        "window_mask": torch.ones((batch_size, window_size), dtype=torch.bool),
    }

    metrics = writer_only_metrics(model, batch, training_step=1)

    for key in ("empty_ratio", "draft_ratio", "known_ratio", "frozen_ratio"):
        assert key in metrics
        assert torch.isfinite(metrics[key])
        assert 0.0 <= float(metrics[key]) <= 1.0


def test_writer_eval_diffusion_step_uses_low_noise_endpoint():
    config = tiny_config()
    config.writer_diffusion_steps = 4
    config.writer_diffusion_min_mask_ratio = 0.05
    config.writer_diffusion_max_mask_ratio = 0.95
    step, mask_ratio = sample_diffusion_step(config, torch.device("cpu"), training_step=None)

    assert step == 3
    assert abs(mask_ratio - 0.05) < 1e-6


def test_emit_calibration_prefers_precision_constrained_threshold():
    logits = torch.tensor([4.0, 3.0, -1.0, -2.0])
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0])

    calibration = calibrate_emit_logits(logits, targets, min_precision=0.95)

    assert calibration["precision"] >= 0.95
    assert calibration["recall"] >= 0.99
    assert 0.0 <= calibration["threshold"] <= 1.0
    assert calibration["temperature"] > 0.0


def test_predicted_future_curriculum_pads_short_naz_horizons():
    class ShortHorizonPredictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def predict_semantic_dynamics(self, semantic_states, unit_mask, attention_mask=None):
            batch_size, window_size, latent_size = semantic_states.shape
            selected = torch.ones((batch_size, window_size, 2, latent_size), device=semantic_states.device)
            return SimpleNamespace(selected_latents=selected)

    config = tiny_config()
    config.writer_future_mix_ratio = 1.0
    true_future = torch.full((1, 3, 4, config.latent_size), 7.0)
    semantic = torch.zeros((1, 3, config.latent_size))
    window_mask = torch.ones((1, 3), dtype=torch.bool)

    future_latents, mode_id = build_future_latents(
        config,
        true_future,
        semantic,
        window_mask,
        ShortHorizonPredictor(),
        "mixed",
    )

    assert mode_id == 4.0
    assert future_latents.shape == true_future.shape
    assert torch.allclose(future_latents[:, :, :2], torch.ones_like(future_latents[:, :, :2]))
    assert torch.allclose(future_latents[:, :, 2:], torch.zeros_like(future_latents[:, :, 2:]))


def test_future_curriculum_has_true_noised_predicted_mixed_phases():
    config = tiny_config()
    config.writer_future_noised_start_step = 2
    config.writer_future_predicted_start_step = 4
    config.writer_future_mixed_start_step = 6
    predictor = object()

    assert resolve_future_mode(config, 1, predictor, "curriculum") == "true"
    assert resolve_future_mode(config, 2, predictor, "curriculum") == "noised"
    assert resolve_future_mode(config, 4, predictor, "curriculum") == "predicted"
    assert resolve_future_mode(config, 6, predictor, "curriculum") == "mixed"
    assert resolve_future_mode(config, 6, None, "curriculum") == "noised"


def test_naz_encodes_active_dil_latents(tmp_path):
    dil_config = tiny_config()
    dil_model = Dil(dil_config)
    dil_config.save_pretrained(tmp_path)
    torch.save(
        {
            "format_version": dil_config.checkpoint_format_version,
            "model_state_dict": dil_model.state_dict(),
        },
        tmp_path / "checkpoint.pt",
    )
    naz_config = NazConfig(
        dil_path=str(tmp_path),
        vocab_size=dil_config.vocab_size,
        pad_token_id=dil_config.pad_token_id,
        eos_token_id=dil_config.eos_token_id,
        max_word_bytes=dil_config.max_word_bytes,
        latent_size=dil_config.latent_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        full_attention_interval=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_conv_kernel_size=4,
    )
    model = Naz(naz_config)
    target_input_ids = torch.tensor(
        [[[2, 0, 0, 0], [3, 4, 0, 0], [5, 6, 7, 0]]],
        dtype=torch.long,
    )
    target_word_masks = target_input_ids.ne(dil_config.pad_token_id)
    unit_mask = torch.ones(target_input_ids.shape[:2], dtype=torch.bool)

    target_latents = model.encode_active_dil_latents(target_input_ids, target_word_masks, unit_mask)

    assert target_latents.shape == (3, dil_config.latent_size)


def test_naz_embeds_frozen_dil_sequence_latents(tmp_path):
    dil_config = tiny_config()
    dil_model = Dil(dil_config)
    dil_config.save_pretrained(tmp_path)
    torch.save(
        {
            "format_version": dil_config.checkpoint_format_version,
            "model_state_dict": dil_model.state_dict(),
        },
        tmp_path / "checkpoint.pt",
    )
    naz_config = NazConfig(
        dil_path=str(tmp_path),
        vocab_size=dil_config.vocab_size,
        pad_token_id=dil_config.pad_token_id,
        eos_token_id=dil_config.eos_token_id,
        max_word_bytes=dil_config.max_word_bytes,
        latent_size=dil_config.latent_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        full_attention_interval=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_conv_kernel_size=4,
    )
    model = Naz(naz_config)
    input_ids = torch.tensor(
        [[[2, 0, 0, 0], [3, 4, 0, 0], [5, 6, 7, 0]]],
        dtype=torch.long,
    )
    word_masks = input_ids.ne(dil_config.pad_token_id)
    unit_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool)

    semantic_states = model.encode_sequence_latents(input_ids, word_masks, unit_mask)
    embeddings = model.embed_sequence_latents(input_ids, word_masks, unit_mask)
    embeddings.sum().backward()

    assert semantic_states.shape == (1, 3, dil_config.latent_size)
    assert embeddings.shape == (1, 3, naz_config.hidden_size)
    assert model.student_core.semantic_embed_proj[-2].weight.grad is not None
    assert grad_abs_sum(model.dil_model.encoder.embed_tokens.weight) == 0.0
    assert not hasattr(model, "byte_embed_tokens")


def test_naz_semantic_dynamics_head_predicts_mtp_candidates(tmp_path):
    dil_config = tiny_config()
    dil_model = Dil(dil_config)
    dil_config.save_pretrained(tmp_path)
    torch.save(
        {
            "format_version": dil_config.checkpoint_format_version,
            "model_state_dict": dil_model.state_dict(),
        },
        tmp_path / "checkpoint.pt",
    )
    naz_config = tiny_naz_config(tmp_path, dil_config)
    model = Naz(naz_config).eval()
    hidden_states = torch.randn(2, 3, naz_config.hidden_size)

    dynamics = model.semantic_head(hidden_states)

    assert dynamics.candidate_latents.shape == (
        2,
        3,
        naz_config.mtp_horizons,
        naz_config.num_semantic_candidates,
        dil_config.latent_size,
    )
    assert dynamics.router_logits.shape == (2, 3, naz_config.mtp_horizons, naz_config.num_semantic_candidates)
    assert dynamics.selected_latents.shape == (2, 3, naz_config.mtp_horizons, dil_config.latent_size)
    assert dynamics.selected_indices.shape == (2, 3, naz_config.mtp_horizons)
    expected_norm = dil_config.latent_size**0.5
    assert torch.allclose(
        dynamics.candidate_latents.float().norm(dim=-1),
        torch.full(dynamics.candidate_latents.shape[:-1], expected_norm),
        atol=1e-5,
    )
    assert torch.allclose(
        dynamics.selected_latents.float().norm(dim=-1),
        torch.full(dynamics.selected_latents.shape[:-1], expected_norm),
        atol=1e-5,
    )
    assert model.student_core.semantic_embed_proj[0].in_features == dil_config.latent_size
    assert model.student_core.semantic_embed_proj[0].out_features == 2 * naz_config.hidden_size
    assert not hasattr(model, "latent_head")
    assert not hasattr(model, "generative_head")
    assert not hasattr(model, "mean_head")
    assert not hasattr(model, "writer_logits")
    assert hasattr(model.dil_model, "writer")


def test_naz_training_jitter_only_perturbs_unmasked_inputs(tmp_path):
    dil_config = tiny_config()
    dil_model = Dil(dil_config)
    dil_config.save_pretrained(tmp_path)
    torch.save(
        {
            "format_version": dil_config.checkpoint_format_version,
            "model_state_dict": dil_model.state_dict(),
        },
        tmp_path / "checkpoint.pt",
    )
    naz_config = tiny_naz_config(tmp_path, dil_config)
    naz_config.naz_input_jitter_prob = 1.0
    model = Naz(naz_config).train()
    semantic_states = normalize_semantic_latents(torch.randn(1, 3, dil_config.latent_size))
    unit_mask = torch.tensor([[True, False, True]])

    noised = model.jitter_semantic_states_for_training(semantic_states, unit_mask)
    cosine = torch.nn.functional.cosine_similarity(semantic_states, noised, dim=-1)

    assert torch.allclose(noised.float().norm(dim=-1), semantic_states.float().norm(dim=-1), atol=1e-5)
    assert torch.all(cosine[unit_mask] >= naz_config.naz_input_jitter_min_cos - 1e-5)
    assert torch.all(cosine[unit_mask] <= naz_config.naz_input_jitter_max_cos + 1e-5)
    assert torch.equal(noised[~unit_mask], semantic_states[~unit_mask])


def test_naz_hybrid_backbone_uses_native_layer_pattern(tmp_path):
    dil_config = tiny_config()
    dil_model = Dil(dil_config)
    dil_config.save_pretrained(tmp_path)
    torch.save(
        {
            "format_version": dil_config.checkpoint_format_version,
            "model_state_dict": dil_model.state_dict(),
        },
        tmp_path / "checkpoint.pt",
    )
    naz_config = tiny_naz_config(tmp_path, dil_config)
    model = Naz(naz_config).eval()

    assert model.transformer.layer_types == (
        "delta",
        "delta",
        "delta",
        "global",
    )
    assert isinstance(model.transformer.layers[0].mixer, SemanticDeltaMixer)
    assert isinstance(model.transformer.layers[3].mixer, SemanticGlobalAttention)
    assert isinstance(model.transformer.layers[0].input_norm, ZeroCenteredRMSNorm)
    assert model.transformer.layers[3].mixer.num_key_value_groups == 2
    assert model.transformer.layers[3].mixer.rotary.partial_dim == 2
    assert all(layer.uses_moe for layer in model.transformer.layers[-naz_config.moe_layers :])
    assert isinstance(model.transformer.layers[-1].feedforward, SparseMoEFeedForward)


def test_sparse_moe_selected_expert_dispatch_matches_dense_reference():
    torch.manual_seed(7)
    moe = SparseMoEFeedForward(
        hidden_size=6,
        shared_intermediate_size=10,
        expert_intermediate_size=8,
        num_experts=5,
        top_k=2,
    ).eval()
    hidden_states = torch.randn(3, 4, 6, requires_grad=True)

    output, balance_loss, usage = moe(hidden_states)

    flat_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    router_probs = torch.softmax(moe.router(flat_states).float(), dim=-1).to(hidden_states.dtype)
    top_weights, top_indices = torch.topk(router_probs, k=moe.top_k, dim=-1)
    top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(top_weights.dtype).eps)
    route_weights = flat_states.new_zeros(flat_states.shape[0], moe.num_experts)
    route_weights.scatter_add_(dim=1, index=top_indices, src=top_weights)
    expert_gate = torch.einsum("th,eih->tei", flat_states, moe.expert_gate_weight)
    expert_up = torch.einsum("th,eih->tei", flat_states, moe.expert_up_weight)
    expert_hidden = torch.nn.functional.silu(expert_gate) * expert_up
    expert_output = torch.einsum("tei,ehi->teh", expert_hidden, moe.expert_down_weight)
    dense_routed = torch.sum(expert_output * route_weights.unsqueeze(-1), dim=1)
    expected = moe.shared(hidden_states) + dense_routed.reshape_as(hidden_states)

    assert torch.allclose(output, expected, atol=1e-6)
    assert torch.isfinite(balance_loss)
    assert usage.shape == (moe.num_experts,)


def test_naz_global_attention_uses_sdpa_without_expanding_gqa_cache():
    root = Path(__file__).resolve().parents[1]
    source = (root / "dilnaz" / "models" / "naz_backbone" / "attention.py").read_text(encoding="utf-8")

    assert "scaled_dot_product_attention" in source
    assert "enable_gqa=self.num_key_value_groups > 1" in source
    assert "torch.cat((cache.key" not in source
    assert "repeat_interleave" not in source


def test_naz_delta_mixer_uses_fla_gated_delta_kernels():
    root = Path(__file__).resolve().parents[1]
    source = (root / "dilnaz" / "models" / "naz_backbone" / "delta.py").read_text(encoding="utf-8")

    assert "ShortConvolution" in source
    assert "chunk_gated_delta_rule" in source
    assert "fused_recurrent_gated_delta_rule" in source
    assert "updates.cumsum" not in source


def test_naz_global_attention_sdpa_shape_and_dtype(tmp_path):
    dil_config = tiny_config()
    naz_config = tiny_naz_config(tmp_path, dil_config)
    attention = SemanticGlobalAttention(naz_config).eval()
    hidden_states = torch.randn(2, 5, naz_config.hidden_size)
    attention_mask = torch.ones(2, 5, dtype=torch.bool)
    attention_mask[1, 4] = False
    position_ids = torch.arange(5).reshape(1, 5).expand(2, 5)

    output = attention(hidden_states, attention_mask, position_ids)

    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype


def test_naz_global_attention_cuda_cudnn_gqa_smoke(tmp_path):
    if not torch.cuda.is_available():
        return
    from torch.nn.attention import SDPBackend, sdpa_kernel

    naz_config = NazConfig(
        dil_path=str(tmp_path),
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
    )
    attention = SemanticGlobalAttention(naz_config).cuda().half().eval()
    hidden_states = torch.randn(2, 4, naz_config.hidden_size, device="cuda", dtype=torch.float16)
    attention_mask = torch.ones(2, 4, dtype=torch.bool, device="cuda")
    position_ids = torch.arange(4, device="cuda").reshape(1, 4).expand(2, 4)

    with sdpa_kernel(backends=[SDPBackend.CUDNN_ATTENTION]):
        output = attention(hidden_states, attention_mask, position_ids)
    torch.cuda.synchronize()

    assert output.shape == hidden_states.shape
    assert output.dtype == torch.float16


def test_naz_hybrid_backbone_cache_matches_full_forward(tmp_path):
    dil_config = tiny_config()
    dil_model = Dil(dil_config)
    dil_config.save_pretrained(tmp_path)
    torch.save(
        {
            "format_version": dil_config.checkpoint_format_version,
            "model_state_dict": dil_model.state_dict(),
        },
        tmp_path / "checkpoint.pt",
    )
    naz_config = tiny_naz_config(tmp_path, dil_config)
    model = Naz(naz_config).eval()
    inputs_embeds = torch.randn(1, 3, naz_config.hidden_size)

    full = model.transformer(inputs_embeds=inputs_embeds, use_cache=False).last_hidden_state
    cached_prefix = model.transformer(inputs_embeds=inputs_embeds[:, :2], use_cache=True, max_cache_length=3)
    assert cached_prefix.past_key_values.position == 2
    global_cache = cached_prefix.past_key_values.layers[3]
    assert global_cache.key.shape[1] == 3
    cached_output = model.transformer(
        inputs_embeds=inputs_embeds[:, 2:],
        past_key_values=cached_prefix.past_key_values,
        use_cache=True,
    )
    cached_last = cached_output.last_hidden_state
    assert cached_output.past_key_values.position == 3
    assert cached_output.past_key_values.layers[3].key.shape[1] == 3

    assert torch.allclose(full[:, -1], cached_last[:, -1], atol=1e-5, rtol=1e-5)


def test_dil_naz_code_has_no_external_backbone_imports():
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "dilnaz" / "models" / "modeling_dil.py",
        root / "dilnaz" / "models" / "configuration_dil.py",
        root / "dilnaz" / "models" / "modeling_naz.py",
        root / "dilnaz" / "models" / "configuration_naz.py",
        *(root / "dilnaz" / "models" / "naz_backbone").glob("*.py"),
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    external_names = ("Q" + "wen", "q" + "wen", "L" + "lama", "l" + "lama")
    assert not any(name in text for name in external_names)


def test_naz_interface_has_no_flush_schedule_cli():
    root = Path(__file__).resolve().parents[1]
    text = (root / "dilnaz" / "train" / "interface_naz.py").read_text(encoding="utf-8")

    assert "--decode-flush-schedule" not in text
    assert "--no-stream" not in text
    assert "--temperature" not in text
    assert "--num-samples" not in text
    assert "NazLatentWriter" not in text
    assert "def stream_text" in text
    assert "model.generate_stream" in text
    assert "decode_semantic" in text
    assert "model.generate(" not in text


def test_naz_forward_optimizes_semantic_dynamics_moe_mtp_objective(tmp_path):
    dil_config = tiny_config()
    dil_model = Dil(dil_config)
    dil_config.save_pretrained(tmp_path)
    torch.save(
        {
            "format_version": dil_config.checkpoint_format_version,
            "model_state_dict": dil_model.state_dict(),
        },
        tmp_path / "checkpoint.pt",
    )
    naz_config = tiny_naz_config(tmp_path, dil_config)
    model = Naz(naz_config)
    input_ids = torch.tensor(
        [[[2, 0, 0, 0], [3, 4, 0, 0], [5, 6, 7, 0]]],
        dtype=torch.long,
    )
    word_masks = input_ids.ne(dil_config.pad_token_id)
    target_input_ids = torch.tensor(
        [
            [
                [[3, 4, 0, 0], [5, 6, 7, 0], [8, 9, 0, 0]],
                [[5, 6, 7, 0], [8, 9, 0, 0], [10, 0, 0, 0]],
                [[8, 9, 0, 0], [10, 0, 0, 0], [11, 12, 0, 0]],
            ]
        ],
        dtype=torch.long,
    )
    target_word_masks = target_input_ids.ne(dil_config.pad_token_id)
    unit_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool)
    target_mask = torch.ones(target_input_ids.shape[:3], dtype=torch.bool)

    outputs = model(
        input_ids=input_ids,
        word_masks=word_masks,
        target_input_ids=target_input_ids,
        target_word_masks=target_word_masks,
        unit_mask=unit_mask,
        target_mask=target_mask,
    )

    expected_loss = (
        outputs.mixture_nll
        + naz_config.router_responsibility_weight * outputs.responsibility_loss
        + naz_config.usage_balance_weight * outputs.usage_balance_loss
        + naz_config.moe_balance_weight * outputs.moe_balance_loss
    )
    assert outputs.latent_predictions.shape == (1, 3, dil_config.latent_size)
    assert outputs.predicted_latents.shape == (9, dil_config.latent_size)
    assert outputs.target_latents.shape == (9, dil_config.latent_size)
    assert outputs.candidate_usage.shape == (naz_config.mtp_horizons, naz_config.num_semantic_candidates)
    assert int(outputs.num_targets) == 9
    assert torch.isfinite(outputs.loss)
    assert torch.allclose(outputs.loss, expected_loss)
    assert torch.allclose(outputs.reconstruction_loss, outputs.mixture_nll)
    assert torch.allclose(outputs.mse_mean, outputs.chosen_mse)
    assert torch.isfinite(outputs.reconstruction_loss)
    assert torch.isfinite(outputs.mse_loss)
    assert torch.isfinite(outputs.mse_mean)
    assert torch.isfinite(outputs.mixture_nll)
    assert torch.isfinite(outputs.responsibility_loss)
    assert torch.isfinite(outputs.usage_balance_loss)
    assert torch.isfinite(outputs.moe_balance_loss)
    assert torch.isfinite(outputs.min_mse)
    assert torch.isfinite(outputs.chosen_mse)
    assert torch.isfinite(outputs.router_entropy)
    assert torch.isfinite(outputs.cosine_loss)
    assert torch.isfinite(outputs.latent_cos)
    assert not hasattr(outputs, "energy_loss")
    assert not hasattr(outputs, "writer_loss")
    assert not hasattr(outputs, "byte_acc")


def test_naz_forward_semantic_empty_target_mask_stays_finite(tmp_path):
    dil_config = tiny_config()
    dil_model = Dil(dil_config)
    dil_config.save_pretrained(tmp_path)
    torch.save(
        {
            "format_version": dil_config.checkpoint_format_version,
            "model_state_dict": dil_model.state_dict(),
        },
        tmp_path / "checkpoint.pt",
    )
    naz_config = tiny_naz_config(tmp_path, dil_config)
    model = Naz(naz_config)
    semantic_states = torch.randn(1, 3, naz_config.latent_size)
    target_latents = torch.randn(1, 3, naz_config.mtp_horizons, naz_config.latent_size)
    unit_mask = torch.ones((1, 3), dtype=torch.bool)
    target_mask = torch.zeros((1, 3, naz_config.mtp_horizons), dtype=torch.bool)

    outputs = model.forward_semantic(
        semantic_states=semantic_states,
        target_latents=target_latents,
        unit_mask=unit_mask,
        target_mask=target_mask,
    )

    assert int(outputs.num_targets) == 0
    assert outputs.predicted_latents.shape == (0, naz_config.latent_size)
    assert torch.isfinite(outputs.loss)
    assert torch.isfinite(outputs.latent_cos)
    assert torch.isfinite(outputs.cosine_loss)
    assert outputs.latent_cos.item() == 0.0
    assert outputs.cosine_loss.item() == 0.0


def test_naz_learnable_sigma_mixture_loss_matches_gaussian_constant(tmp_path):
    dil_config = tiny_config()
    dil_model = Dil(dil_config)
    dil_config.save_pretrained(tmp_path)
    torch.save(
        {
            "format_version": dil_config.checkpoint_format_version,
            "model_state_dict": dil_model.state_dict(),
        },
        tmp_path / "checkpoint.pt",
    )
    naz_config = tiny_naz_config(tmp_path, dil_config)
    model = Naz(naz_config)
    candidates = torch.zeros(
        1,
        1,
        naz_config.mtp_horizons,
        naz_config.num_semantic_candidates,
        naz_config.latent_size,
    )
    logits = torch.zeros(1, 1, naz_config.mtp_horizons, naz_config.num_semantic_candidates)
    target = torch.zeros(1, 1, naz_config.mtp_horizons, naz_config.latent_size)
    target_mask = torch.zeros(1, 1, naz_config.mtp_horizons, dtype=torch.bool)
    target_mask[0, 0, 0] = True

    losses = model.semantic_mixture_losses(
        SimpleNamespace(candidate_latents=candidates, router_logits=logits, selected_latents=candidates[:, :, :, 0]),
        target,
        target_mask,
    )

    sigma = model.mixture_sigma.detach()[0]
    expected_nll = 0.5 * naz_config.latent_size * torch.log(torch.tensor(2.0 * torch.pi) * sigma.square())
    expected_entropy = torch.log(torch.tensor(float(naz_config.num_semantic_candidates)))
    assert torch.allclose(losses["mixture_nll"], expected_nll)
    assert torch.allclose(losses["responsibility_loss"], expected_entropy)
    assert torch.allclose(losses["router_entropy"], expected_entropy)
    assert torch.allclose(losses["usage_balance_loss"], torch.zeros(()), atol=1e-6)
    assert torch.allclose(losses["min_mse"], torch.zeros(()))
    assert torch.allclose(losses["chosen_mse"], torch.zeros(()))
    losses["mixture_nll"].backward()
    assert model.mixture_sigma.shape == (naz_config.mtp_horizons,)
    assert model.mixture_sigma_logit.grad is not None
    assert model.mixture_sigma_logit.grad.shape == (naz_config.mtp_horizons,)
    assert model.mixture_sigma_logit.grad[0].abs() > 0.0


def test_naz_has_no_owned_latent_normalizer(tmp_path):
    dil_config = tiny_config()
    dil_model = Dil(dil_config)
    dil_config.save_pretrained(tmp_path)
    torch.save(
        {
            "format_version": dil_config.checkpoint_format_version,
            "model_state_dict": dil_model.state_dict(),
        },
        tmp_path / "checkpoint.pt",
    )
    model = Naz(tiny_naz_config(tmp_path, dil_config))

    assert not hasattr(model, "latent_normalizer")
    assert not hasattr(model, "normalize_latents")
    assert not hasattr(model, "denormalize_latents")


def test_naz_generation_feeds_predicted_latents_directly(tmp_path):
    dil_config = tiny_config()
    dil_model = Dil(dil_config)
    dil_config.save_pretrained(tmp_path)
    torch.save(
        {
            "format_version": dil_config.checkpoint_format_version,
            "model_state_dict": dil_model.state_dict(),
        },
        tmp_path / "checkpoint.pt",
    )
    naz_config = tiny_naz_config(tmp_path, dil_config)
    model = Naz(naz_config).eval()
    input_ids = torch.tensor(
        [[[2, 0, 0, 0], [3, 4, 0, 0]]],
        dtype=torch.long,
    )
    word_masks = input_ids.ne(dil_config.pad_token_id)
    unit_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool)
    call_count = 0
    original_forward = model.dil_model.encoder.forward

    def counted_forward(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_forward(*args, **kwargs)

    model.dil_model.encoder.forward = counted_forward

    outputs = model.generate(
        input_ids=input_ids,
        word_masks=word_masks,
        unit_mask=unit_mask,
        max_new_tokens=4,
        min_new_tokens=4,
    )

    assert call_count == 1
    assert outputs.prompt_latents.shape == (1, 2, dil_config.latent_size)
    assert outputs.generated_latents.shape == (1, 4, dil_config.latent_size)


def test_naz_generate_stream_yields_latent_steps(tmp_path):
    dil_config = tiny_config()
    dil_model = Dil(dil_config)
    dil_config.save_pretrained(tmp_path)
    torch.save(
        {
            "format_version": dil_config.checkpoint_format_version,
            "model_state_dict": dil_model.state_dict(),
        },
        tmp_path / "checkpoint.pt",
    )
    naz_config = tiny_naz_config(tmp_path, dil_config)
    model = Naz(naz_config).eval()
    input_ids = torch.tensor(
        [[[2, 0, 0, 0], [3, 4, 0, 0]]],
        dtype=torch.long,
    )
    word_masks = input_ids.ne(dil_config.pad_token_id)
    unit_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool)
    steps = list(
        model.generate_stream(
            input_ids=input_ids,
            word_masks=word_masks,
            unit_mask=unit_mask,
            max_new_tokens=3,
            min_new_tokens=3,
        )
    )

    assert len(steps) == 3
    for step in steps:
        assert step.latent.shape == (1, dil_config.latent_size)
        assert step.future_latents.shape == (1, model.config.mtp_horizons - 1, dil_config.latent_size)
        assert step.latent_cos_to_previous.shape == (1,)
        assert step.should_stop.shape == (1,)


def test_naz_generate_stream_yields_normalized_output_latents(tmp_path):
    class ConstantBackbone(torch.nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.hidden_size = hidden_size

        def forward(
            self,
            inputs_embeds,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            max_cache_length=None,
        ):
            del max_cache_length
            return SimpleNamespace(
                last_hidden_state=torch.ones((*inputs_embeds.shape[:2], self.hidden_size), dtype=inputs_embeds.dtype),
                past_key_values=None,
            )

    dil_config = tiny_config()
    dil_model = Dil(dil_config)
    dil_config.save_pretrained(tmp_path)
    torch.save(
        {
            "format_version": dil_config.checkpoint_format_version,
            "model_state_dict": dil_model.state_dict(),
        },
        tmp_path / "checkpoint.pt",
    )
    model = Naz(tiny_naz_config(tmp_path, dil_config)).eval()
    model.student_core.backbone = ConstantBackbone(model.config.hidden_size)
    for parameter in model.semantic_head.parameters():
        parameter.data.zero_()
    input_ids = torch.tensor([[[2, 0, 0, 0]]], dtype=torch.long)
    word_masks = input_ids.ne(dil_config.pad_token_id)
    unit_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool)

    step = next(
        model.generate_stream(
            input_ids=input_ids,
            word_masks=word_masks,
            unit_mask=unit_mask,
            max_new_tokens=1,
            min_new_tokens=1,
        )
    )

    assert torch.allclose(step.latent.norm(dim=-1), torch.full((1,), dil_config.latent_size**0.5), atol=1e-5)
    assert torch.equal(step.candidate_index, torch.zeros_like(step.candidate_index))


def test_naz_generate_stream_rejects_padded_prompts(tmp_path):
    dil_config = tiny_config()
    dil_model = Dil(dil_config)
    dil_config.save_pretrained(tmp_path)
    torch.save(
        {
            "format_version": dil_config.checkpoint_format_version,
            "model_state_dict": dil_model.state_dict(),
        },
        tmp_path / "checkpoint.pt",
    )
    model = Naz(tiny_naz_config(tmp_path, dil_config)).eval()
    input_ids = torch.tensor(
        [[[2, 0, 0, 0], [0, 0, 0, 0]]],
        dtype=torch.long,
    )
    word_masks = input_ids.ne(dil_config.pad_token_id)
    unit_mask = torch.tensor([[True, False]])

    try:
        list(
            model.generate_stream(
                input_ids=input_ids,
                word_masks=word_masks,
                unit_mask=unit_mask,
                max_new_tokens=1,
            )
        )
    except ValueError as error:
        assert "packed prompts" in str(error)
    else:
        raise AssertionError("Naz accepted padded prompt generation")


def test_naz_generate_returns_prompt_and_generated_latents(tmp_path):
    dil_config = tiny_config()
    dil_model = Dil(dil_config)
    dil_config.save_pretrained(tmp_path)
    torch.save(
        {
            "format_version": dil_config.checkpoint_format_version,
            "model_state_dict": dil_model.state_dict(),
        },
        tmp_path / "checkpoint.pt",
    )
    model = Naz(tiny_naz_config(tmp_path, dil_config)).eval()
    input_ids = torch.tensor(
        [[[2, 0, 0, 0], [3, 4, 0, 0]]],
        dtype=torch.long,
    )
    word_masks = input_ids.ne(dil_config.pad_token_id)
    unit_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool)

    outputs = model.generate(
        input_ids=input_ids,
        word_masks=word_masks,
        unit_mask=unit_mask,
        max_new_tokens=2,
        min_new_tokens=2,
    )

    assert outputs.prompt_latents.shape == (1, 2, dil_config.latent_size)
    assert outputs.generated_latents.shape == (1, 2, dil_config.latent_size)


def test_resident_semantic_cache_matches_full_symmetric_context_pass(tmp_path):
    dil_config = tiny_config()
    dil_model = Dil(dil_config)
    dil_config.save_pretrained(tmp_path)
    torch.save(
        {
            "format_version": dil_config.checkpoint_format_version,
            "model_state_dict": dil_model.state_dict(),
        },
        tmp_path / "checkpoint.pt",
    )
    naz_config = tiny_naz_config(tmp_path, dil_config)
    model = Naz(naz_config).eval()
    byte_ids = torch.tensor(
        [
            [2, 0, 0, 0],
            [3, 4, 0, 0],
            [5, 6, 7, 0],
            [8, 0, 0, 0],
            [9, 10, 0, 0],
            [11, 12, 13, 0],
        ],
        dtype=torch.long,
    )
    lengths = byte_ids.ne(dil_config.pad_token_id).sum(dim=-1)
    ids_path = tmp_path / "ids.npy"
    lengths_path = tmp_path / "lengths.npy"
    np.save(ids_path, byte_ids.numpy())
    np.save(lengths_path, lengths.numpy())
    batcher = ResidentNazBatcher(
        ids_path,
        lengths_path,
        token_count=byte_ids.shape[0],
        config=dil_config,
        sequence_length=3,
        batch_size=1,
        device=torch.device("cpu"),
        seed=1,
    )

    semantic_states, latent_cache, cached_byte_ids, cached_lengths = build_resident_semantic_cache(
        model,
        batcher,
        chunk_tokens=2,
        autocast_enabled=False,
    )
    positions = torch.arange(dil_config.max_word_bytes).reshape(1, 1, -1)
    masks = positions < lengths.reshape(1, -1, 1)
    unit_mask = torch.ones((1, byte_ids.shape[0]), dtype=torch.bool)
    full_latents = model.encode_active_dil_latents(byte_ids.unsqueeze(0), masks, unit_mask)

    assert torch.allclose(latent_cache, full_latents.reshape(byte_ids.shape[0], -1), atol=1e-6, rtol=1e-5)
    assert torch.allclose(semantic_states, latent_cache)
    assert torch.equal(cached_byte_ids.cpu(), byte_ids)
    assert torch.equal(cached_lengths.cpu(), lengths)


def test_resident_naz_semantic_batcher_surfaces_next_latents():
    semantic_states = torch.randn(6, 4)
    target_latents = torch.randn(6, 4)
    byte_ids = torch.tensor(
        [
            [2, 0, 0, 0],
            [3, 4, 0, 0],
            [5, 6, 7, 0],
            [8, 0, 0, 0],
            [9, 10, 0, 0],
            [11, 12, 13, 0],
        ],
        dtype=torch.long,
    )
    lengths = byte_ids.ne(0).sum(dim=-1)
    batcher = ResidentNazSemanticBatcher(
        semantic_states,
        target_latents,
        byte_ids,
        lengths,
        sequence_length=3,
        batch_size=2,
        seed=1,
        horizons=3,
    )

    batch = batcher.make_batch(torch.tensor([[0], [2]]))

    assert batch["semantic_states"].shape == (2, 3, 4)
    assert batch["target_latents"].shape == (2, 3, 3, 4)
    assert batch["target_mask"].shape == (2, 3, 3)
    assert batch["unit_mask"].dtype == torch.bool
    assert torch.equal(batch["semantic_states"][0, 0], semantic_states[0])
    assert torch.equal(batch["target_latents"][0, 0, 0], target_latents[1])
    assert torch.equal(batch["target_latents"][0, 0, 1], target_latents[2])
    assert torch.equal(batch["target_latents"][0, 0, 2], target_latents[3])
    assert torch.equal(batch["semantic_states"][1, 0], semantic_states[2])
    assert torch.equal(batch["target_latents"][1, 0, 0], target_latents[3])


def test_memmap_naz_semantic_batcher_surfaces_next_latents(tmp_path):
    semantic_states = np.arange(24, dtype=np.float32).reshape(6, 4)
    semantic_path = tmp_path / "semantic.npy"
    np.save(semantic_path, semantic_states)
    batcher = MemmapNazSemanticBatcher(
        semantic_path,
        token_count=6,
        latent_size=4,
        sequence_length=3,
        batch_size=2,
        seed=1,
        device=torch.device("cpu"),
        horizons=3,
    )

    batch = batcher.make_batch(np.asarray([[0], [2]], dtype=np.int64))

    assert batch["semantic_states"].shape == (2, 3, 4)
    assert batch["target_latents"].shape == (2, 3, 3, 4)
    assert batch["unit_mask"].dtype == torch.bool
    assert batch["target_mask"].dtype == torch.bool
    assert torch.equal(batch["semantic_states"][0, 0], torch.from_numpy(semantic_states[0]))
    assert torch.equal(batch["target_latents"][0, 0, 0], torch.from_numpy(semantic_states[1]))
    assert torch.equal(batch["target_latents"][0, 0, 2], torch.from_numpy(semantic_states[3]))
    assert torch.equal(batch["semantic_states"][1, 0], torch.from_numpy(semantic_states[2]))


def test_streaming_text_naz_dataset_reads_plain_text_without_cache(tmp_path):
    data_file = tmp_path / "math.txt"
    data_file.write_text("2 + 2 = 4\n3 + 1 = 4\n", encoding="utf-8")
    tokenizer = __import__("tokenization").HybridTokenizer.from_file(
        Path(__file__).resolve().parents[1] / "dilnaz" / "tokenization" / "hybrid_surface_vocab.json"
    )
    config = DilConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        hidden_size=8,
        intermediate_size=16,
        num_encoder_layers=2,
        latent_size=4,
        max_word_bytes=8,
        context_radius=1,
        dil_dropout=0.0,
    )
    dataset = StreamingTextNazDataset(
        data_file,
        tokenizer,
        config,
        sequence_length=3,
        batch_size=2,
        read_chars=16,
        repeat=False,
    )

    batch = next(iter(dataset))

    assert batch["input_ids"].shape == (2, 3, config.max_word_bytes)
    assert batch["target_input_ids"].shape == (2, 3, 3, config.max_word_bytes)
    assert batch["word_masks"].dtype == torch.bool
    assert batch["target_word_masks"].dtype == torch.bool
    assert batch["target_mask"].shape == (2, 3, 3)
    assert batch["unit_mask"].all()
    assert batch["attention_mask"] is None
    assert not any(tmp_path.glob("*.npy"))


def test_naz_pretrain_trainer_runs_one_small_step(tmp_path):
    data_file = tmp_path / "trainer.txt"
    write_naz_trainer_text(data_file)
    dil_dir = tmp_path / "Dil"
    save_tiny_trainer_dil_checkpoint(dil_dir)

    trainer = NazPretrainTrainer(tiny_pretrain_args(tmp_path, data_file, dil_dir))
    batch = next(trainer.build_train_iterator())
    result = trainer.train_step(batch, 1)
    result.loss.backward()

    assert trainer.stage == "pretrain"
    assert torch.isfinite(result.loss)
    assert result.token_count > 0
    assert grad_abs_sum(next(trainer.model.student_core.parameters())) > 0.0


def test_naz_cached_pretrain_trainer_runs_one_small_step(tmp_path):
    data_file = tmp_path / "trainer.txt"
    write_naz_trainer_text(data_file)
    dil_dir = tmp_path / "Dil"
    save_tiny_trainer_dil_checkpoint(dil_dir)
    args = tiny_pretrain_args(tmp_path, data_file, dil_dir, output_name="naz_cached_pretrain")
    args.data_mode = "cached"

    trainer = NazPretrainTrainer(args)
    batch = next(trainer.build_train_iterator())
    result = trainer.train_step(batch, 1)

    assert trainer.stage == "pretrain"
    assert torch.isfinite(result.loss)
    assert result.token_count > 0
    assert any((args.output_dir / "naz_token_cache").glob("*.semantic.latents.npy"))


def test_naz_finetune_trainer_initializes_from_pretrain_checkpoint_and_runs_step(tmp_path):
    data_file = tmp_path / "trainer.txt"
    write_naz_trainer_text(data_file)
    dil_dir = tmp_path / "Dil"
    save_tiny_trainer_dil_checkpoint(dil_dir)
    pretrain_trainer = NazPretrainTrainer(tiny_pretrain_args(tmp_path, data_file, dil_dir))
    checkpoint_dir = pretrain_trainer.save_checkpoint("checkpoint-1", 1, {"loss": 1.0})
    init_checkpoint = checkpoint_dir / "checkpoint.pt"

    trainer = NazFinetuneTrainer(tiny_finetune_args(tmp_path, data_file, init_checkpoint))
    batch = next(trainer.build_train_iterator())
    result = trainer.train_step(batch, 1)

    assert trainer.stage == "finetune"
    assert trainer.config.hidden_size == pretrain_trainer.config.hidden_size
    assert trainer.config.mtp_horizons == pretrain_trainer.config.mtp_horizons
    assert trainer.scheduler.get_last_lr()[0] == 3e-5
    assert torch.isfinite(result.loss)
    assert result.token_count > 0


def test_naz_finetune_rejects_architecture_override(tmp_path):
    args = parse_naz_args(
        [
            "--stage",
            "finetune",
            "--train-file",
            str(tmp_path / "trainer.txt"),
            "--init-naz-checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--output-dir",
            str(tmp_path / "out"),
            "--hidden-size",
            "64",
        ]
    )

    try:
        validate_naz_args(args)
    except ValueError as error:
        assert "model architecture/objective overrides" in str(error)
        assert "--hidden-size" in str(error)
    else:
        raise AssertionError("Naz finetune accepted architecture override")


def test_naz_resume_uses_checkpoint_stage_and_locked_runtime(tmp_path):
    data_file = tmp_path / "trainer.txt"
    write_naz_trainer_text(data_file)
    dil_dir = tmp_path / "Dil"
    save_tiny_trainer_dil_checkpoint(dil_dir)
    pretrain_trainer = NazPretrainTrainer(tiny_pretrain_args(tmp_path, data_file, dil_dir))
    checkpoint_dir = pretrain_trainer.save_checkpoint("checkpoint-1", 1, {"loss": 1.0})

    args = parse_naz_args(
        [
            "--train-file",
            str(data_file),
            "--resume",
            str(checkpoint_dir / "checkpoint.pt"),
            "--output-dir",
            str(tmp_path / "resumed"),
            "--compile-mode",
            "off",
            "--max-steps",
            "2",
            "--batch-size",
            "1",
            "--eval-batch-size",
            "1",
            "--log-every",
            "1",
            "--checkpoint-every",
            "0",
            "--eval-every",
            "0",
            "--max-eval-batches",
            "1",
            "--text-read-chars",
            "256",
            "--num-workers",
            "0",
            "--prefetch-factor",
            "1",
        ]
    )
    trainer = make_trainer(args)

    assert isinstance(trainer, NazPretrainTrainer)
    assert trainer.stage == "pretrain"
    assert trainer.start_step == 1
    assert trainer.args.sequence_length == 2
    assert trainer.args.data_mode == "streaming"


def test_naz_trainer_detects_frozen_dil_checksum_change(tmp_path):
    data_file = tmp_path / "trainer.txt"
    write_naz_trainer_text(data_file)
    dil_dir = tmp_path / "Dil"
    save_tiny_trainer_dil_checkpoint(dil_dir)
    trainer = NazPretrainTrainer(tiny_pretrain_args(tmp_path, data_file, dil_dir))

    with torch.no_grad():
        next(trainer.model.dil_model.parameters()).add_(1.0)

    try:
        trainer.assert_checkpoint_integrity()
    except RuntimeError as error:
        assert "frozen Dil checksum changed" in str(error)
    else:
        raise AssertionError("Naz trainer accepted a changed frozen Dil checksum")


def test_train_naz_unified_entrypoint_excludes_sft_path():
    source = (Path(__file__).resolve().parents[1] / "dilnaz" / "train" / "train_naz.py").read_text(encoding="utf-8")

    assert "PromptAnswerNazDataset" not in source
    assert "masked_sft_forward" not in source
    assert "exact_answer_accuracy" not in source
    assert not (Path(__file__).resolve().parents[1] / "dilnaz" / "train" / "train_naz_finetune.py").exists()


def test_naz_training_batch_latents_use_single_extended_dil_pass(tmp_path):
    dil_config = tiny_config()
    dil_model = Dil(dil_config)
    dil_config.save_pretrained(tmp_path)
    torch.save(
        {
            "format_version": dil_config.checkpoint_format_version,
            "model_state_dict": dil_model.state_dict(),
        },
        tmp_path / "checkpoint.pt",
    )
    model = Naz(tiny_naz_config(tmp_path, dil_config))
    extended_ids = torch.tensor(
        [[[2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0], [6, 0, 0, 0], [1, 0, 0, 0]]],
        dtype=torch.long,
    )
    horizons = model.config.mtp_horizons
    sequence_length = 3
    input_ids = extended_ids[:, :sequence_length]
    word_masks = input_ids.ne(dil_config.pad_token_id)
    target_positions = torch.arange(sequence_length).view(1, sequence_length, 1) + torch.arange(
        1,
        horizons + 1,
    ).view(1, 1, horizons)
    target_input_ids = extended_ids.gather(
        dim=1,
        index=target_positions.reshape(1, sequence_length * horizons, 1).expand(
            1,
            -1,
            dil_config.max_word_bytes,
        ),
    ).reshape(1, sequence_length, horizons, dil_config.max_word_bytes)
    target_word_masks = target_input_ids.ne(dil_config.pad_token_id)
    unit_mask = torch.ones(1, sequence_length, dtype=torch.bool)
    target_mask = torch.ones(1, sequence_length, horizons, dtype=torch.bool)

    semantic_states, target_latents = model.encode_training_batch_latents(
        input_ids,
        word_masks,
        target_input_ids,
        target_word_masks,
        unit_mask,
        target_mask,
    )
    expected_source = model.encode_sequence_latents(
        input_ids,
        word_masks,
        unit_mask,
    )
    expected_targets = model.encode_sequence_latents(
        extended_ids,
        extended_ids.ne(dil_config.pad_token_id),
        torch.ones(1, sequence_length + horizons, dtype=torch.bool),
    )

    assert torch.allclose(semantic_states, expected_source, atol=1e-6, rtol=1e-5)
    assert torch.allclose(target_latents[:, :, 0], expected_targets[:, 1 : sequence_length + 1], atol=1e-6, rtol=1e-5)
    assert torch.allclose(
        target_latents[:, :, -1],
        expected_targets[:, horizons : sequence_length + horizons],
        atol=1e-6,
        rtol=1e-5,
    )


def test_naz_interface_stream_stops_at_writer_eos(capsys):
    tokenizer = fixture_tokenizer()
    prompt = "15 + 4241 ="
    prompt_segments = [segment for segment in tokenizer.encode_segments(prompt) if segment.piece_len > 0]
    config = NazConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_word_bytes=8,
        latent_size=1,
        min_new_tokens=3,
        repetition_cos_threshold=1.1,
    )
    model = FakeGeneratedNaz(tokenizer, config.max_word_bytes, ["4", None, "9"])

    stream_naz_text(
        model,
        config,
        tokenizer,
        prompt_segments,
        torch.device("cpu"),
        max_new_tokens=3,
        min_new_tokens=3,
        repetition_cos_threshold=1.1,
        writer_microbatch_size=8,
    )

    captured = capsys.readouterr().out
    assert captured == f"{prompt}4\n"
    assert model.dil_model.calls == 1
    seed_count = min(model.dil_model.config.writer_left_frozen, len(prompt_segments))
    left_start = model.dil_model.config.writer_left_frozen - seed_count
    surface_state_mask = model.dil_model.writer_kwargs[0]["surface_state_mask"]
    frozen_mask = model.dil_model.writer_kwargs[0]["frozen_mask"]
    assert surface_state_mask[0, left_start : model.dil_model.config.writer_left_frozen].gt(0).any(dim=-1).all()
    assert frozen_mask[0, left_start : model.dil_model.config.writer_left_frozen].any(dim=-1).all()


def test_naz_interface_stops_when_writer_decodes_semantic_eos(capsys):
    tokenizer = fixture_tokenizer()
    prompt = "15 + 4241 ="
    prompt_segments = [segment for segment in tokenizer.encode_segments(prompt) if segment.piece_len > 0]
    config = NazConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_word_bytes=8,
        latent_size=1,
        min_new_tokens=1,
        repetition_cos_threshold=1.1,
    )
    model = FakeGeneratedNaz(tokenizer, config.max_word_bytes, ["<eos>", "9"])

    stream_naz_text(
        model,
        config,
        tokenizer,
        prompt_segments,
        torch.device("cpu"),
        max_new_tokens=1,
        min_new_tokens=1,
        repetition_cos_threshold=1.1,
        writer_microbatch_size=8,
    )

    captured = capsys.readouterr().out
    assert captured == f"{prompt}\n"
    assert model.dil_model.calls == 1
    assert not model.dil_model.writer_kwargs[0]["surface_state_mask"][0, model.dil_model.config.writer_left_frozen].any()


def test_sliding_writer_buffer_caches_pending_surface_state(capsys):
    tokenizer = fixture_tokenizer()
    model = FakeGeneratedNaz(
        tokenizer,
        max_word_bytes=4,
        decoded_steps=["1", "2", "3", "4", "5", "6"],
        commit_score_values=[0.0, 1.0],
    )
    config = model.dil_model.config
    config.writer_sliding_window_size = 4
    config.writer_left_frozen = 1
    config.writer_active_size = 2
    config.writer_right_guard = 1
    buffer = SlidingWriterBuffer(model, config, tokenizer, commit_threshold=0.5)

    for _ in range(3):
        buffer.append(
            torch.zeros((1, config.latent_size), dtype=torch.float32),
            torch.zeros((1, config.latent_size), dtype=torch.float32),
            False,
        )

    assert not buffer.flush(force=False)
    assert model.dil_model.calls == 1
    assert all(surface is not None for surface in buffer.pending_surfaces)

    assert not buffer.flush(force=False)
    capsys.readouterr()
    surface_state_mask = model.dil_model.writer_kwargs[1]["surface_state_mask"]
    assert surface_state_mask[0, 1:4].gt(0).any(dim=-1).all()


def test_sliding_writer_buffer_passes_position_age(capsys):
    tokenizer = fixture_tokenizer()
    model = FakeGeneratedNaz(
        tokenizer,
        max_word_bytes=4,
        decoded_steps=["1", "2", "3", "4", "5", "6"],
        commit_score_values=[0.0, 0.0],
    )
    config = model.dil_model.config
    config.writer_sliding_window_size = 4
    config.writer_left_frozen = 1
    config.writer_active_size = 2
    config.writer_right_guard = 1
    buffer = SlidingWriterBuffer(model, config, tokenizer, commit_threshold=0.5)
    for _ in range(3):
        buffer.append(
            torch.zeros((1, config.latent_size), dtype=torch.float32),
            torch.zeros((1, config.latent_size), dtype=torch.float32),
            False,
        )

    assert not buffer.flush(force=False)
    assert not buffer.flush(force=False)
    capsys.readouterr()

    first_age = model.dil_model.writer_kwargs[0]["position_age"]
    second_age = model.dil_model.writer_kwargs[1]["position_age"]
    assert torch.equal(first_age[0, 1:4], torch.zeros(3, dtype=torch.long))
    assert torch.equal(second_age[0, 1:4], torch.ones(3, dtype=torch.long))
