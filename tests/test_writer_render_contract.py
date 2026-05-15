from pathlib import Path

import torch

from dilnaz.models.dil import Dil, DilConfig
from dilnaz.surface import pack_token_units
from dilnaz.tokenization import HybridTokenizer
from dilnaz.train.interface.interface_dil import decode_tokens, make_batch, tokenize_text
from dilnaz.train.interface.writer_render import render_latents_with_sliding_writer


def tiny_config():
    return DilConfig(
        vocab_size=64,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=1,
        hidden_size=32,
        intermediate_size=64,
        latent_size=16,
        max_surface_pieces_per_unit=8,
        surface_bucket_sizes=(8, 16, 32, 64),
        encoder_context_layers=2,
        encoder_layer_pattern=("sliding", "global"),
        encoder_attention_heads=4,
        encoder_key_value_heads=2,
        encoder_head_dim=8,
        encoder_intermediate_size=64,
        encoder_attention_window=4,
        byte_conv_layers=1,
        writer_num_layers=1,
        writer_word_mixer_layers=1,
        writer_word_attention_heads=4,
        writer_max_window_size=5,
        writer_sliding_window_size=5,
        writer_left_frozen=1,
        writer_active_size=3,
        writer_right_guard=1,
        writer_stride=3,
    )


def test_render_does_not_use_direct_decode_semantic():
    cfg = tiny_config()
    model = Dil(cfg).eval()
    surface = pack_token_units(
        [[[2], [3, 4], [5]]],
        pad_token_id=cfg.pad_token_id,
        bucket_sizes=cfg.surface_bucket_sizes,
        max_pieces_per_unit=cfg.max_surface_pieces_per_unit,
    )
    latents = model.encode(surface).float()

    class DirectDecodePatch:
        def __init__(self, original):
            self.original = original
            self.called = False

        def __call__(self, semantic):
            self.called = True
            return self.original(semantic)

    original_decode = model.decode_semantic
    patched = DirectDecodePatch(original_decode)
    model.decode_semantic = patched

    dummy_tokenizer = HybridTokenizer(
        char_tokens=[],
        surface_tokens=[],
        numeric_tokens=[],
        common_word_tokens=[],
        contextual_tokens={},
    )
    try:
        render_latents_with_sliding_writer(model, dummy_tokenizer, latents)
    except Exception:
        pass
    assert not patched.called, "render_latents_with_sliding_writer must not call decode_semantic (direct decode)"


def test_render_short_sequence_returns_expected_count():
    cfg = tiny_config()
    model = Dil(cfg).eval()
    surface = pack_token_units(
        [[[2], [3, 4], [5]]],
        pad_token_id=cfg.pad_token_id,
        bucket_sizes=cfg.surface_bucket_sizes,
        max_pieces_per_unit=cfg.max_surface_pieces_per_unit,
    )
    latents = model.encode(surface).float()

    dummy_tokenizer = HybridTokenizer(
        char_tokens=[],
        surface_tokens=[],
        numeric_tokens=[],
        common_word_tokens=[],
        contextual_tokens={},
    )
    output = render_latents_with_sliding_writer(model, dummy_tokenizer, latents)
    assert len(output) == 3, f"expected 3 tokens for 3-unit sequence, got {len(output)}"


def test_render_respects_unit_mask():
    cfg = tiny_config()
    model = Dil(cfg).eval()
    surface = pack_token_units(
        [[[2], [3, 4], [5]]],
        pad_token_id=cfg.pad_token_id,
        bucket_sizes=cfg.surface_bucket_sizes,
        max_pieces_per_unit=cfg.max_surface_pieces_per_unit,
    )
    latents = model.encode(surface).float()
    unit_mask = torch.tensor([True, False, True], dtype=torch.bool)

    dummy_tokenizer = HybridTokenizer(
        char_tokens=[],
        surface_tokens=[],
        numeric_tokens=[],
        common_word_tokens=[],
        contextual_tokens={},
    )
    output = render_latents_with_sliding_writer(model, dummy_tokenizer, latents, unit_mask=unit_mask)
    assert len(output) == 2, f"expected 2 tokens for 3-unit sequence with unit_mask excluding index 1, got {len(output)}"


def test_render_long_sequence_uses_multiple_windows():
    cfg = tiny_config()
    model = Dil(cfg).eval()
    unit_count = 8
    rows = [[[i + 2] for i in range(unit_count)]]
    surface = pack_token_units(
        rows,
        pad_token_id=cfg.pad_token_id,
        bucket_sizes=cfg.surface_bucket_sizes,
        max_pieces_per_unit=cfg.max_surface_pieces_per_unit,
    )
    latents = model.encode(surface).float()
    assert latents.shape[1] == unit_count

    dummy_tokenizer = HybridTokenizer(
        char_tokens=[],
        surface_tokens=[],
        numeric_tokens=[],
        common_word_tokens=[],
        contextual_tokens={},
    )
    output = render_latents_with_sliding_writer(model, dummy_tokenizer, latents)
    assert len(output) <= unit_count, f"expected at most {unit_count} tokens (empty positions are filtered), got {len(output)}"
    assert len(output) >= unit_count // 2, f"expected at least {unit_count // 2} non-empty tokens, got {len(output)}"


def test_render_output_is_finite():
    cfg = tiny_config()
    model = Dil(cfg).eval()
    surface = pack_token_units(
        [[[2], [3, 4], [5]]],
        pad_token_id=cfg.pad_token_id,
        bucket_sizes=cfg.surface_bucket_sizes,
        max_pieces_per_unit=cfg.max_surface_pieces_per_unit,
    )
    latents = model.encode(surface).float()
    assert torch.isfinite(latents).all(), "latents must be finite"

    dummy_tokenizer = HybridTokenizer(
        char_tokens=[],
        surface_tokens=[],
        numeric_tokens=[],
        common_word_tokens=[],
        contextual_tokens={},
    )
    output = render_latents_with_sliding_writer(model, dummy_tokenizer, latents)
    assert isinstance(output, list), "output must be a list of strings"


def test_interface_dil_decode_tokens_does_not_call_direct_decode_semantic():
    """interface_dil.decode_tokens must use the sliding-window helper, not direct decode_semantic."""
    import dilnaz.train.interface.interface_dil as interface

    source = Path(interface.__file__).read_text(encoding="utf-8")
    assert "decode_semantic(latents.unsqueeze" not in source, (
        "interface_dil must not call decode_semantic with unsqueeze directly"
    )
    assert "render_latents_with_sliding_writer" in source, (
        "interface_dil.decode_tokens must import and use render_latents_with_sliding_writer"
    )


def test_zone_ids_match_training_contract():
    cfg = tiny_config()
    zone_ids_template = torch.full((cfg.writer_sliding_window_size,), 1, dtype=torch.long)
    zone_ids_template[: cfg.writer_left_frozen] = 0
    zone_ids_template[cfg.writer_left_frozen + cfg.writer_active_size :] = 2

    expected = torch.tensor([0, 1, 1, 1, 2])
    assert cfg.writer_sliding_window_size == 5
    assert cfg.writer_left_frozen == 1
    assert cfg.writer_active_size == 3
    assert cfg.writer_right_guard == 1
    assert torch.equal(zone_ids_template, expected), f"zone_ids {zone_ids_template} must match training contract {expected}"
