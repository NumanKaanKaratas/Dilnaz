import ast
from pathlib import Path

import torch

from dilnaz.models.dil import Dil, DilConfig
from dilnaz.models.dil.writer import DilConditionalWriter
from dilnaz.surface import (
    empty_surface_state,
    pack_token_units,
    pack_writer_targets,
    synthetic_state_from_targets,
)


def tiny_config() -> DilConfig:
    return DilConfig(
        vocab_size=64,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=1,
        hidden_size=32,
        intermediate_size=64,
        latent_size=16,
        num_encoder_layers=2,
        max_surface_pieces_per_unit=16,
        surface_bucket_sizes=(8, 16, 32, 64, 128),
        context_radius=1,
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


def test_writer_targets_use_exact_lengths():
    cfg = tiny_config()
    comma = pack_writer_targets(
        [[[7]]],
        pad_token_id=cfg.pad_token_id,
        stop_token_id=cfg.writer_stop_token_id,
        surface_bucket_sizes=cfg.surface_bucket_sizes,
        max_pieces_per_unit=cfg.max_surface_pieces_per_unit,
    )
    araba = pack_writer_targets(
        [[[2, 3, 4, 5, 6]]],
        pad_token_id=cfg.pad_token_id,
        stop_token_id=cfg.writer_stop_token_id,
        surface_bucket_sizes=cfg.surface_bucket_sizes,
        max_pieces_per_unit=cfg.max_surface_pieces_per_unit,
    )
    long_word = pack_writer_targets(
        [[list(range(2, 27))]],
        pad_token_id=cfg.pad_token_id,
        stop_token_id=cfg.writer_stop_token_id,
        surface_bucket_sizes=cfg.surface_bucket_sizes,
        max_pieces_per_unit=32,
    )
    padded = pack_writer_targets(
        [[[2, 3], []]],
        pad_token_id=cfg.pad_token_id,
        stop_token_id=cfg.writer_stop_token_id,
        surface_bucket_sizes=cfg.surface_bucket_sizes,
        max_pieces_per_unit=cfg.max_surface_pieces_per_unit,
    )
    assert comma.query.unit_lengths.item() == 2
    assert araba.query.unit_lengths.item() == 6
    assert long_word.query.unit_lengths.item() == 26
    assert padded.query.unit_lengths.tolist() == [[3, 0]]
    assert padded.query.unit_mask.tolist() == [[True, False]]
    assert padded.label_mask.sum().item() == 3
    assert comma.labels[0, 0].item() == 7
    assert comma.labels[0, 1].item() == cfg.writer_stop_token_id


def test_dil_packed_encoder_output_shape():
    cfg = tiny_config()
    model = Dil(cfg)
    surface = pack_token_units(
        [[[2], [3, 4], [5]]],
        pad_token_id=cfg.pad_token_id,
        bucket_sizes=cfg.surface_bucket_sizes,
        max_pieces_per_unit=cfg.max_surface_pieces_per_unit,
    )
    semantic, layers = model.encode(surface, output_hidden_states=True)
    assert semantic.shape == (1, cfg.latent_size)
    assert len(layers) == cfg.num_encoder_layers
    assert torch.isfinite(semantic).all()


def test_writer_packed_logits_and_all_empty_state_no_nan():
    cfg = tiny_config()
    model = Dil(cfg)
    semantic = torch.randn(2, cfg.writer_sliding_window_size, cfg.latent_size)
    target = pack_writer_targets(
        [
            [[2], [3, 4], [5], [6], [7]],
            [[8], [9], [10, 11, 12], [13], [14]],
        ],
        pad_token_id=cfg.pad_token_id,
        stop_token_id=cfg.writer_stop_token_id,
        surface_bucket_sizes=cfg.surface_bucket_sizes,
        max_pieces_per_unit=cfg.max_surface_pieces_per_unit,
    )
    state = empty_surface_state(target.query, cfg.writer_empty_token_id)
    output = model.writer.transition(semantic, query_surface=target.query, surface_state=state)
    assert output.token_logits.shape == (2, target.query.surface_width, cfg.writer_vocab_size)
    assert output.state_valid_logits.shape == (2, target.query.surface_width)
    assert output.emit_logits.shape == (2, target.query.surface_width)
    assert not hasattr(output, "length_bucket_logits")
    assert torch.isfinite(output.token_logits).all()


def test_known_frozen_state_changes_writer_output():
    cfg = tiny_config()
    model = Dil(cfg)
    semantic = torch.randn(1, cfg.writer_sliding_window_size, cfg.latent_size)
    target = pack_writer_targets(
        [[[2], [3, 4], [5], [6], [7]]],
        pad_token_id=cfg.pad_token_id,
        stop_token_id=cfg.writer_stop_token_id,
        surface_bucket_sizes=cfg.surface_bucket_sizes,
        max_pieces_per_unit=cfg.max_surface_pieces_per_unit,
    )
    zone_ids = torch.tensor([[0, 1, 1, 1, 2]])
    window_mask = torch.ones(1, cfg.writer_sliding_window_size, dtype=torch.bool)
    empty = empty_surface_state(target.query, cfg.writer_empty_token_id)
    known = synthetic_state_from_targets(
        target,
        zone_ids=zone_ids,
        window_mask=window_mask,
        empty_token_id=cfg.writer_empty_token_id,
        vocab_size=cfg.writer_vocab_size,
        mask_ratio=1.0,
        draft_max_ratio=0.0,
    )
    out_empty = model.writer.transition(semantic, query_surface=target.query, surface_state=empty, zone_ids=zone_ids).token_logits
    out_known = model.writer.transition(semantic, query_surface=target.query, surface_state=known, zone_ids=zone_ids).token_logits
    assert not torch.allclose(out_empty, out_known)


def test_writer_transition_loss_is_exact_length_token_loss():
    cfg = tiny_config()
    model = Dil(cfg)
    semantic = torch.randn(1, cfg.writer_sliding_window_size, cfg.latent_size)
    target = pack_writer_targets(
        [[[2], [3, 4], [5], [6], [7]]],
        pad_token_id=cfg.pad_token_id,
        stop_token_id=cfg.writer_stop_token_id,
        surface_bucket_sizes=cfg.surface_bucket_sizes,
        max_pieces_per_unit=cfg.max_surface_pieces_per_unit,
    )
    zone_ids = torch.tensor([[0, 1, 1, 1, 2]])
    window_mask = torch.ones(1, cfg.writer_sliding_window_size, dtype=torch.bool)
    state = synthetic_state_from_targets(
        target,
        zone_ids=zone_ids,
        window_mask=window_mask,
        empty_token_id=cfg.writer_empty_token_id,
        vocab_size=cfg.writer_vocab_size,
        mask_ratio=0.5,
        draft_max_ratio=0.2,
    )
    metrics = model.writer_transition_loss_and_metrics(
        semantic,
        target,
        state,
        zone_ids,
        window_mask,
        return_metrics=True,
    )
    assert "length_bucket_loss" not in metrics
    assert torch.isfinite(metrics["loss"])


def test_writer_decode_public_tuple_contract_uses_exact_stop_limit():
    cfg = tiny_config()
    model = Dil(cfg)
    semantic = torch.randn(2, cfg.latent_size)
    token_ids, token_mask, lengths = model.decode_semantic(semantic)
    assert token_ids.shape == (2, cfg.max_surface_pieces_per_unit)
    assert token_mask.shape == token_ids.shape
    assert lengths.shape == (2,)

    semantic_window = torch.randn(1, cfg.writer_sliding_window_size, cfg.latent_size)
    window_ids, window_mask, window_lengths, commit_scores = model.decode_semantic_window(semantic_window)
    assert window_ids.shape == (1, cfg.writer_sliding_window_size, cfg.max_surface_pieces_per_unit)
    assert window_mask.shape == window_ids.shape
    assert window_lengths.shape == (1, cfg.writer_sliding_window_size)
    assert commit_scores.shape == (
        1,
        cfg.writer_sliding_window_size,
        cfg.max_surface_pieces_per_unit + 1,
    )


def test_naz_forward_is_semantic_only():
    tree = ast.parse(Path("dilnaz/models/naz/model.py").read_text(encoding="utf-8"))
    forward = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    parameters = [arg.arg for arg in forward.args.args]
    assert "semantic_states" in parameters
    assert "target_latents" in parameters
    assert "input_ids" not in parameters
    assert "word_masks" not in parameters
