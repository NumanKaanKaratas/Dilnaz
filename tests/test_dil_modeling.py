import ast
from pathlib import Path
from types import MethodType

import torch

from dilnaz.models.dil import Dil, DilConfig
from dilnaz.models.dil.writer import DilWriterOutput
from dilnaz.surface import pack_token_units, pack_writer_targets
from dilnaz.train.configs.defaults import DIL_MODEL_DEFAULTS


def tiny_config() -> DilConfig:
    return DilConfig(
        vocab_size=64,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=1,
        hidden_size=32,
        intermediate_size=64,
        latent_size=16,
        max_surface_pieces_per_unit=16,
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
    )


def test_dil_config_sequence_limit_matches_training_default():
    assert DilConfig().max_sequence_units == DIL_MODEL_DEFAULTS["max_sequence_units"]


def make_writer_target(cfg: DilConfig, rows):
    return pack_writer_targets(
        rows,
        pad_token_id=cfg.pad_token_id,
        bos_token_id=cfg.decoder_start_token_id,
        stop_token_id=cfg.eos_token_id,
        surface_bucket_sizes=cfg.surface_bucket_sizes,
        max_pieces_per_unit=cfg.max_surface_pieces_per_unit,
    )


def test_shared_embedding_contract():
    cfg = tiny_config()
    model = Dil(cfg)
    assert model.encoder.embed_tokens is model.shared_token_embeddings
    assert model.writer.token_embeddings is model.shared_token_embeddings
    assert "_".join(("surface", "input", "embeddings")) not in dict(model.writer.named_modules())
    assert not hasattr(model.writer, "_".join(("token", "head")))


def test_writer_targets_use_single_vocab_causal_inputs_and_eos_stop():
    cfg = tiny_config()
    comma = make_writer_target(cfg, [[[7]]])
    araba = make_writer_target(cfg, [[[2, 3, 4, 5, 6]]])
    eos = make_writer_target(cfg, [[[cfg.eos_token_id]]])
    padded = make_writer_target(cfg, [[[2, 3], []]])
    assert comma.query.unit_lengths.item() == 2
    assert araba.query.unit_lengths.item() == 6
    assert padded.query.unit_lengths.tolist() == [[3, 0]]
    assert padded.query.unit_mask.tolist() == [[True, False]]
    assert comma.query.ids[0, 0].item() == cfg.decoder_start_token_id
    assert comma.query.ids[0, 1].item() == 7
    assert comma.labels[0, 0].item() == 7
    assert comma.labels[0, 1].item() == cfg.eos_token_id
    assert eos.query.ids[0, 0].item() == cfg.decoder_start_token_id
    assert eos.query.ids[0, 1].item() == cfg.eos_token_id
    assert eos.labels[0, 0].item() == cfg.eos_token_id
    assert eos.labels[0, 1].item() == cfg.eos_token_id


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
    assert semantic.shape == (1, 3, cfg.latent_size)
    assert len(layers) == cfg.encoder_context_layers
    assert torch.isfinite(semantic).all()
    assert torch.allclose(
        semantic.norm(dim=-1),
        torch.full((1, 3), cfg.latent_size**0.5),
        atol=1e-4,
        rtol=1e-4,
    )


def test_writer_packed_logits_no_nan():
    cfg = tiny_config()
    model = Dil(cfg)
    semantic = torch.randn(2, cfg.latent_size)
    target = make_writer_target(cfg, [[[2]], [[3, 4]]])
    output = model.writer.transition(semantic, query_surface=target.query)
    assert output.token_logits.shape == (2, target.query.surface_width, cfg.vocab_size)
    assert torch.isfinite(output.token_logits).all()


def test_writer_causal_surface_path_does_not_see_future_inputs():
    cfg = tiny_config()
    model = Dil(cfg).eval()
    semantic = torch.randn(1, cfg.latent_size)
    left = make_writer_target(cfg, [[[2, 3, 4]]])
    right = make_writer_target(cfg, [[[2, 9, 9]]])
    with torch.no_grad():
        left_logits = model.writer.transition(semantic, query_surface=left.query).token_logits
        right_logits = model.writer.transition(semantic, query_surface=right.query).token_logits
    assert torch.allclose(left_logits[:, 0], right_logits[:, 0], atol=1e-6, rtol=1e-5)
    assert torch.allclose(left_logits[:, 1], right_logits[:, 1], atol=1e-6, rtol=1e-5)


def test_writer_loss_is_unit_local_token_loss():
    cfg = tiny_config()
    model = Dil(cfg)
    semantic = torch.randn(1, cfg.latent_size)
    target = make_writer_target(cfg, [[[2, 3, 4]]])
    metrics = model.writer_loss_and_metrics(semantic, target, return_metrics=True)
    assert torch.isfinite(metrics["loss"])
    assert torch.equal(metrics["loss"], metrics["token_loss"])


def test_writer_generation_result_stops_or_hits_surface_limit():
    cfg = tiny_config()
    model = Dil(cfg)
    semantic = torch.randn(2, cfg.latent_size)
    generation = model.decode_semantic(semantic)
    assert generation.token_ids.shape == (2, cfg.max_surface_pieces_per_unit)
    assert generation.token_mask.shape == generation.token_ids.shape
    assert generation.lengths.shape == (2,)
    assert generation.stopped.shape == (2,)


def test_writer_generation_stops_on_eos_stop_token():
    cfg = tiny_config()
    model = Dil(cfg)

    def stop_transition(self, semantic, query_surface=None):
        logits = torch.zeros(
            query_surface.ids.shape[0],
            query_surface.surface_width,
            self.vocab_size,
            device=semantic.device,
        )
        logits[..., self.stop_token_id] = 1.0
        return DilWriterOutput(token_logits=logits, query_surface=query_surface)

    model.writer.transition = MethodType(stop_transition, model.writer)
    generation = model.decode_semantic(torch.randn(2, cfg.latent_size))
    assert generation.stopped.tolist() == [True, True]
    assert generation.lengths.tolist() == [0, 0]
    assert not generation.token_mask.any()


def test_writer_generation_caps_when_stop_is_missing():
    cfg = tiny_config()
    model = Dil(cfg)

    def token_transition(self, semantic, query_surface=None):
        logits = torch.zeros(
            query_surface.ids.shape[0],
            query_surface.surface_width,
            self.vocab_size,
            device=semantic.device,
        )
        logits[..., 2] = 1.0
        return DilWriterOutput(token_logits=logits, query_surface=query_surface)

    model.writer.transition = MethodType(token_transition, model.writer)
    generation = model.decode_semantic(torch.randn(2, cfg.latent_size))
    assert generation.stopped.tolist() == [False, False]
    assert generation.lengths.tolist() == [cfg.max_surface_pieces_per_unit, cfg.max_surface_pieces_per_unit]
    assert generation.token_mask.all()


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
