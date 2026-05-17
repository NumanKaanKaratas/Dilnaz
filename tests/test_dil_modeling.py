import ast
from pathlib import Path

import pytest
import torch
from torch.optim import AdamW

from dilnaz.models.dil import Dil, DilConfig
from dilnaz.models.common.norms import DilRMSNorm
from dilnaz.surface import pack_token_units, pack_writer_targets
from dilnaz.train.configs.defaults import DIL_MODEL_DEFAULTS
from dilnaz.train.common.objectives import WRITER_OBJECTIVE
from dilnaz.train.common.runtime import rng_state
from dilnaz.train.common.trainer_core import make_scheduler
from dilnaz.train.dil.train import prepare_writer_for_surface_training, restore_checkpoint
from dilnaz.train.writer.train import (
    WRITER_METRIC_KEYS,
    WriterContextDataset,
    freeze_for_writer_only,
    load_model_checkpoint,
    writer_only_metrics,
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
        max_surface_pieces_per_unit=16,
        surface_bucket_sizes=(8, 16, 32, 64),
        byte_conv_layers=1,
        num_encoder_layers=2,
        writer_num_layers=1,
    )


def test_encoder_bf16_runtime_is_scoped_to_encoder_reductions():
    cfg = tiny_config()
    model = Dil(cfg)
    model.set_encoder_bf16_runtime(True)

    encoder_norms = [module for module in model.encoder.modules() if isinstance(module, DilRMSNorm)]
    writer_norms = [module for module in model.writer.modules() if isinstance(module, DilRMSNorm)]

    assert encoder_norms
    assert writer_norms
    assert {module.reduction_dtype for module in encoder_norms} == {None}
    assert {module.reduction_dtype for module in writer_norms} == {torch.float32}
    assert model.encoder.semantic_norm_reduction_dtype is None


def test_dil_config_sequence_limit_matches_training_default():
    assert DilConfig().max_sequence_units == DIL_MODEL_DEFAULTS["max_sequence_units"]


def test_dil_defaults_expose_only_context_radius():
    assert "context_radius" in DIL_MODEL_DEFAULTS
    assert "context_size" not in DIL_MODEL_DEFAULTS
    assert "target_index" not in DIL_MODEL_DEFAULTS
    assert "encoder_context_layers" not in DIL_MODEL_DEFAULTS


def test_center_conditioned_context_keeps_context_states_detached():
    cfg = tiny_config()
    model = Dil(cfg)
    token_states = torch.randn(2, cfg.context_size, cfg.hidden_size, requires_grad=True)
    token_mask = torch.ones(2, cfg.context_size, dtype=torch.bool)

    output = model.encoder.target_conditioned_by_context(token_states, token_mask)
    output.square().mean().backward()

    context_grad = token_states.grad.index_select(1, model.encoder.context_indices)
    assert token_states.grad[:, cfg.target_index].abs().gt(0).any()
    assert not context_grad.abs().gt(0).any()


def make_writer_target(cfg: DilConfig, rows):
    return pack_writer_targets(
        rows,
        pad_token_id=cfg.pad_token_id,
        bos_token_id=cfg.decoder_start_token_id,
        stop_token_id=cfg.writer_stop_token_id,
        surface_bucket_sizes=cfg.surface_bucket_sizes,
        max_pieces_per_unit=cfg.max_surface_pieces_per_unit,
    )


def test_writer_targets_use_semantic_position_queries_and_eos_stop():
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
    assert comma.labels[0, 1].item() == cfg.writer_stop_token_id
    assert eos.query.ids[0, 0].item() == cfg.decoder_start_token_id
    assert eos.query.ids[0, 1].item() == cfg.eos_token_id
    assert eos.labels[0, 0].item() == cfg.eos_token_id
    assert eos.labels[0, 1].item() == cfg.writer_stop_token_id


def test_writer_targets_are_autoregressive_next_token_pairs():
    cfg = tiny_config()
    target = pack_writer_targets(
        [[[2, 3, 4]]],
        pad_token_id=cfg.pad_token_id,
        bos_token_id=cfg.decoder_start_token_id,
        stop_token_id=cfg.writer_stop_token_id,
        surface_bucket_sizes=cfg.surface_bucket_sizes,
        max_pieces_per_unit=cfg.max_surface_pieces_per_unit,
    )
    assert target.query.unit_lengths.tolist() == [[4]]
    assert target.query.ids[0, :4].tolist() == [cfg.decoder_start_token_id, 2, 3, 4]
    assert target.labels[0, :4].tolist() == [2, 3, 4, cfg.writer_stop_token_id]
    assert target.label_mask[0, :4].tolist() == [True, True, True, True]
    assert not target.label_mask[0, 4:].any()


def test_dil_packed_encoder_output_shape():
    cfg = tiny_config()
    model = Dil(cfg)
    surface = pack_token_units(
        [[[2], [3, 4], [5]]],
        pad_token_id=cfg.pad_token_id,
        bucket_sizes=cfg.surface_bucket_sizes,
        max_pieces_per_unit=cfg.max_surface_pieces_per_unit,
    )
    semantic, layers = model.encode(surface, output_hidden_states=True, return_all=True)
    assert semantic.shape == (1, 3, cfg.latent_size)
    assert len(layers) == cfg.num_encoder_layers // 2
    assert torch.isfinite(semantic).all()

    window_surface = pack_token_units(
        [[[2], [3, 4], [5], [6], [7]]],
        pad_token_id=cfg.pad_token_id,
        bucket_sizes=cfg.surface_bucket_sizes,
        max_pieces_per_unit=cfg.max_surface_pieces_per_unit,
    )
    semantic_pooled = model.encode(window_surface)
    assert semantic_pooled.shape == (1, cfg.latent_size), f"shape={semantic_pooled.shape}"
    assert torch.allclose(
        semantic_pooled.norm(dim=-1),
        torch.full((1,), cfg.latent_size**0.5),
        atol=1e-4,
        rtol=1e-4,
    )


def test_writer_packed_logits_no_nan():
    cfg = tiny_config()
    model = Dil(cfg)
    semantic = torch.randn(2, cfg.latent_size)
    target = make_writer_target(cfg, [[[2]], [[3, 4]]])
    output = model.writer.transition(
        semantic,
        query_surface=target.query,
        encoder_embedding_weight=model.writer_encoder_embedding_weight(),
    )
    assert output.token_logits.shape == (2, target.query.surface_width, cfg.writer_vocab_size)
    assert torch.isfinite(output.token_logits).all()


def test_writer_accepts_unit_semantics_with_packed_query():
    cfg = tiny_config()
    model = Dil(cfg)
    semantic = torch.randn(1, 2, cfg.latent_size)
    target = make_writer_target(cfg, [[[2], [3, 4]]])
    output = model.writer.transition(
        semantic,
        query_surface=target.query,
        encoder_embedding_weight=model.writer_encoder_embedding_weight(),
    )
    assert output.token_logits.shape == (1, target.query.surface_width, cfg.writer_vocab_size)
    assert torch.isfinite(output.token_logits).all()


def test_writer_causal_surface_path_does_not_see_future_inputs():
    cfg = tiny_config()
    model = Dil(cfg).eval()
    semantic = torch.randn(1, cfg.latent_size)
    left = make_writer_target(cfg, [[[2, 3, 4]]])
    right = make_writer_target(cfg, [[[2, 9, 9]]])
    with torch.no_grad():
        encoder_weight = model.writer_encoder_embedding_weight()
        left_logits = model.writer.transition(
            semantic,
            query_surface=left.query,
            encoder_embedding_weight=encoder_weight,
        ).token_logits
        right_logits = model.writer.transition(
            semantic,
            query_surface=right.query,
            encoder_embedding_weight=encoder_weight,
        ).token_logits
    assert torch.allclose(left_logits[:, 0], right_logits[:, 0], atol=1e-6, rtol=1e-5)
    assert torch.allclose(left_logits[:, 1], right_logits[:, 1], atol=1e-6, rtol=1e-5)


def test_writer_incremental_step_matches_full_causal_forward():
    cfg = tiny_config()
    model = Dil(cfg).eval()
    semantic = torch.randn(1, cfg.latent_size)
    target = make_writer_target(cfg, [[[2, 3, 4]]])
    with torch.no_grad():
        encoder_weight = model.writer_encoder_embedding_weight()
        full_logits = model.writer.transition(
            semantic,
            query_surface=target.query,
            encoder_embedding_weight=encoder_weight,
        ).token_logits
        caches = [None for _ in model.writer.blocks]
        stepped = []
        for position in range(target.query.surface_width):
            logits, caches = model.writer.step(
                semantic,
                target.query.ids[:, position],
                target.query.pos_in_unit[:, position],
                caches,
                encoder_weight,
            )
            stepped.append(logits)
    step_logits = torch.stack(stepped, dim=1)
    assert torch.allclose(
        step_logits[target.query.mask],
        full_logits[target.query.mask],
        atol=1e-5,
        rtol=1e-4,
    )


def test_writer_encoder_prior_changes_logits_without_sharing_parameters():
    cfg = tiny_config()
    model = Dil(cfg).eval()
    semantic = torch.randn(1, cfg.latent_size)
    target = make_writer_target(cfg, [[[2, 3, 4]]])
    encoder_weight = model.writer_encoder_embedding_weight()
    altered_weight = encoder_weight.clone()
    altered_weight[1:5] = altered_weight[1:5] + 5.0
    with torch.no_grad():
        base_logits = model.writer.transition(
            semantic,
            query_surface=target.query,
            encoder_embedding_weight=encoder_weight,
        ).token_logits
        altered_logits = model.writer.transition(
            semantic,
            query_surface=target.query,
            encoder_embedding_weight=altered_weight,
        ).token_logits
    assert not torch.allclose(base_logits[target.query.mask], altered_logits[target.query.mask])


def test_writer_stop_token_has_zero_encoder_prior():
    cfg = tiny_config()
    model = Dil(cfg).eval()
    token_ids = torch.tensor([cfg.writer_stop_token_id])
    encoder_weight = model.writer_encoder_embedding_weight()
    with torch.no_grad():
        token_state = model.writer.token_condition(token_ids, encoder_weight)
        writer_only = model.writer.token_embeddings(token_ids)
    assert torch.allclose(token_state, writer_only, atol=1e-6, rtol=1e-6)


def test_writer_loss_is_unit_local_token_loss():
    cfg = tiny_config()
    model = Dil(cfg)
    semantic = torch.randn(1, cfg.latent_size)
    target = make_writer_target(cfg, [[[2, 3, 4]]])
    metrics = model.writer_training_loss_and_metrics(semantic, target, return_metrics=True)
    assert torch.isfinite(metrics["loss"])
    assert torch.equal(metrics["loss"], metrics["token_loss"])


def test_writer_loss_uses_detached_semantic_and_encoder_prior():
    cfg = tiny_config()
    model = Dil(cfg).eval()
    surface = pack_token_units(
        [[[2], [3], [4], [5], [6]]],
        pad_token_id=cfg.pad_token_id,
        bucket_sizes=cfg.surface_bucket_sizes,
        max_pieces_per_unit=cfg.max_surface_pieces_per_unit,
    )
    target = make_writer_target(cfg, [[[4]]])
    output = model(surface, writer_target=target)
    output.loss.backward()

    def has_no_effective_grad(parameter: torch.nn.Parameter) -> bool:
        return parameter.grad is None or not parameter.grad.abs().gt(0).any()

    assert has_no_effective_grad(model.encoder.hidden_to_semantic.weight)
    assert has_no_effective_grad(model.encoder.embed_tokens.weight)
    assert model.writer.token_embeddings.weight.grad is not None
    assert model.writer.encoder_prior_proj.weight.grad is not None
    assert model.writer.encoder_prior_gate.weight.grad is not None
    assert model.writer.semantic_proj.weight.grad is not None
    assert model.writer.token_embeddings.weight.grad.abs().gt(0).any()
    assert model.writer.encoder_prior_proj.weight.grad.abs().gt(0).any()
    assert model.writer.encoder_prior_gate.weight.grad.abs().gt(0).any()
    assert model.writer.semantic_proj.weight.grad.abs().gt(0).any()


def test_writer_loss_weight_zero_freezes_writer_training():
    cfg = tiny_config()
    cfg.writer_loss_weight = 0.0
    model = Dil(cfg)
    prepare_writer_for_surface_training(model)
    assert all(not param.requires_grad for param in model.writer.parameters())

    target = make_writer_target(cfg, [[[4]]])
    surface = pack_token_units(
        [[[2], [3], [4], [5], [6]]],
        pad_token_id=cfg.pad_token_id,
        bucket_sizes=cfg.surface_bucket_sizes,
        max_pieces_per_unit=cfg.max_surface_pieces_per_unit,
    )
    output = model(surface, writer_target=target)
    assert output.surface_loss.item() == 0.0
    assert output.writer_loss.item() == 0.0


def test_dil_forward_and_writer_only_share_writer_loss_contract():
    cfg = tiny_config()
    model = Dil(cfg).eval()
    surface = pack_token_units(
        [[[2], [3], [4], [5], [6]]],
        pad_token_id=cfg.pad_token_id,
        bucket_sizes=cfg.surface_bucket_sizes,
        max_pieces_per_unit=cfg.max_surface_pieces_per_unit,
    )
    target = make_writer_target(cfg, [[[4]]])

    output = model(surface, writer_target=target)
    metrics = writer_only_metrics(model, {"surface": surface, "writer_target": target})

    assert torch.allclose(output.surface_loss, metrics["loss"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(output.writer_token_loss, metrics["token_loss"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(output.token_exact, metrics["token_exact"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(output.stop_acc, metrics["stop_acc"], atol=1e-6, rtol=1e-6)


def test_writer_trainer_accepts_base_dil_checkpoint_format(tmp_path: Path):
    cfg = tiny_config()
    model = Dil(cfg)
    cfg.save_pretrained(tmp_path)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "format_version": cfg.checkpoint_format_version,
            "model_state_dict": model.state_dict(),
            "training_state": {"step": 0},
        },
        checkpoint_path,
    )
    loaded, loaded_config, checkpoint = load_model_checkpoint(checkpoint_path, torch.device("cpu"))
    assert isinstance(loaded, Dil)
    assert loaded_config.checkpoint_format_version == cfg.checkpoint_format_version
    assert checkpoint["format_version"] == cfg.checkpoint_format_version


def test_writer_trainer_rejects_previous_dil_checkpoint_format(tmp_path: Path):
    cfg = tiny_config()
    model = Dil(cfg)
    cfg.save_pretrained(tmp_path)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "format_version": 31,
            "model_state_dict": model.state_dict(),
            "training_state": {"step": 0},
        },
        checkpoint_path,
    )
    with pytest.raises(ValueError, match="unsupported Dil checkpoint format_version=31"):
        load_model_checkpoint(checkpoint_path, torch.device("cpu"))


def test_dil_restore_accepts_writer_checkpoint_with_dil_optimizer_state(tmp_path: Path):
    cfg = tiny_config()
    source_model = Dil(cfg)
    source_optimizer = AdamW(source_model.parameters(), lr=1e-3)
    source_scheduler = make_scheduler(source_optimizer, 1e-3, warmup_steps=0, max_steps=20)
    cfg.save_pretrained(tmp_path)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "format_version": cfg.checkpoint_format_version,
            "model_state_dict": source_model.state_dict(),
            "optimizer_state_dict": source_optimizer.state_dict(),
            "scheduler_state_dict": source_scheduler.state_dict(),
            "training_state": {
                "objective": WRITER_OBJECTIVE,
                "step": 11,
                "metrics": {"loss": 9.0},
                "source_dil_step": 7,
                "source_dil_metrics": {"loss": 1.25},
            },
            "rng_state": rng_state(),
        },
        checkpoint_path,
    )

    model = Dil(cfg)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = make_scheduler(optimizer, 1e-3, warmup_steps=0, max_steps=20)
    step, metrics = restore_checkpoint(checkpoint_path, model, optimizer, scheduler, torch.device("cpu"))

    assert step == 7
    assert metrics == {"loss": 1.25}
    for key, value in source_model.state_dict().items():
        assert torch.equal(model.state_dict()[key], value)


def test_dil_restore_accepts_writer_checkpoint_without_dil_optimizer_state(tmp_path: Path):
    cfg = tiny_config()
    source_model = Dil(cfg)
    cfg.save_pretrained(tmp_path)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "format_version": cfg.checkpoint_format_version,
            "model_state_dict": source_model.state_dict(),
            "training_state": {
                "objective": WRITER_OBJECTIVE,
                "step": 11,
                "metrics": {"loss": 9.0},
            },
            "rng_state": rng_state(),
        },
        checkpoint_path,
    )

    model = Dil(cfg)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = make_scheduler(optimizer, 1e-3, warmup_steps=0, max_steps=20)
    step, metrics = restore_checkpoint(checkpoint_path, model, optimizer, scheduler, torch.device("cpu"))

    assert step == 0
    assert metrics == {}
    for key, value in source_model.state_dict().items():
        assert torch.equal(model.state_dict()[key], value)


def test_writer_only_metrics_are_tensors():
    cfg = tiny_config()
    model = Dil(cfg)
    surface = pack_token_units(
        [[[2], [3], [4], [5], [6]]],
        pad_token_id=cfg.pad_token_id,
        bucket_sizes=cfg.surface_bucket_sizes,
        max_pieces_per_unit=cfg.max_surface_pieces_per_unit,
    )
    target = make_writer_target(cfg, [[[4]]])
    metrics = writer_only_metrics(model, {"surface": surface, "writer_target": target})
    assert all(hasattr(metrics[key], "detach") for key in WRITER_METRIC_KEYS)


def test_writer_only_training_does_not_update_encoder_embedding():
    cfg = tiny_config()
    model = Dil(cfg)
    freeze_for_writer_only(model)
    surface = pack_token_units(
        [[[2], [3], [4], [5], [6]]],
        pad_token_id=cfg.pad_token_id,
        bucket_sizes=cfg.surface_bucket_sizes,
        max_pieces_per_unit=cfg.max_surface_pieces_per_unit,
    )
    target = make_writer_target(cfg, [[[4]]])
    metrics = writer_only_metrics(model, {"surface": surface, "writer_target": target})
    metrics["loss"].backward()

    assert model.encoder.embed_tokens.weight.grad is None
    assert model.writer.token_embeddings.weight.grad is not None
    assert model.writer.encoder_prior_proj.weight.grad is not None
    assert model.writer.encoder_prior_gate.weight.grad is not None


def test_writer_context_dataset_uses_writer_stop_token(tmp_path: Path):
    cfg = tiny_config()
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"abc"}\n', encoding="utf-8")

    class TinyTokenizer:
        eos_token_id = cfg.eos_token_id

        def encode_segments(self, text):
            from dilnaz.tokenization.hybrid_tokenizer import TokenPiece, TokenSegment

            return [
                TokenSegment(
                    text="abc",
                    start=0,
                    end=3,
                    kind="surface",
                    pieces=(TokenPiece(token_id=2, text="abc", start=0, end=3, kind="surface"),),
                )
            ]

    dataset = WriterContextDataset(data, cfg, TinyTokenizer(), batch_size=8, read_chars=1024, repeat=True)
    batch = next(dataset.iter_once(0, 1))
    labels = batch["writer_target"].labels
    assert labels.eq(cfg.writer_stop_token_id).any()


def test_writer_context_dataset_aligns_center_surface_with_writer_target(tmp_path: Path):
    cfg = tiny_config()
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"abc def"}\n', encoding="utf-8")

    class TinyTokenizer:
        eos_token_id = cfg.eos_token_id

        def encode_segments(self, text):
            from dilnaz.tokenization.hybrid_tokenizer import TokenPiece, TokenSegment

            return [
                TokenSegment(
                    text="abc",
                    start=0,
                    end=3,
                    kind="surface",
                    pieces=(TokenPiece(token_id=2, text="abc", start=0, end=3, kind="surface"),),
                ),
                TokenSegment(
                    text=" ",
                    start=3,
                    end=4,
                    kind="space",
                    pieces=(TokenPiece(token_id=3, text=" ", start=3, end=4, kind="space"),),
                ),
                TokenSegment(
                    text="def",
                    start=4,
                    end=7,
                    kind="surface",
                    pieces=(TokenPiece(token_id=4, text="def", start=4, end=7, kind="surface"),),
                ),
            ]

    dataset = WriterContextDataset(data, cfg, TinyTokenizer(), batch_size=8, read_chars=1024, repeat=True)
    batch = next(dataset.iter_once(0, 1))
    surface = batch["surface"]
    labels = batch["writer_target"].labels
    assert surface.unit_lengths[0].tolist() == [0, 0, 1, 1, 1]
    assert surface.ids[0, surface.unit_offsets[0, cfg.target_index]].item() == 2
    assert labels[0, 0].item() == 2
    assert labels[1, 0].item() == 3
    assert labels[2, 0].item() == 4


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

    def stop_step(self, semantic, token_ids, positions, caches, encoder_embedding_weight):
        del encoder_embedding_weight
        logits = torch.zeros(semantic.shape[0], self.vocab_size, device=semantic.device)
        logits[:, self.stop_token_id] = 1.0
        return logits, caches

    from types import MethodType
    model.writer.step = MethodType(stop_step, model.writer)
    generation = model.decode_semantic(torch.randn(2, cfg.latent_size))
    assert generation.stopped.tolist() == [True, True]
    assert generation.lengths.tolist() == [0, 0]
    assert not generation.token_mask.any()


def test_writer_generation_caps_when_stop_is_missing():
    cfg = tiny_config()
    model = Dil(cfg)

    def token_step(self, semantic, token_ids, positions, caches, encoder_embedding_weight):
        del encoder_embedding_weight
        logits = torch.zeros(semantic.shape[0], self.vocab_size, device=semantic.device)
        logits[:, 2] = 1.0
        return logits, caches

    from types import MethodType
    model.writer.step = MethodType(token_step, model.writer)
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
