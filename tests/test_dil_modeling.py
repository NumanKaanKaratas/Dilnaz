import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dilnaz"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dilnaz" / "train"))

from models.configuration_dil import DilConfig
from models.configuration_naz import NazConfig
from models.modeling_dil import Dil, DilGatedMLP, DilRMSNorm
from models.modeling_naz import Naz
from models.naz_backbone import SemanticDeltaMixer, SemanticGlobalAttention, ZeroCenteredRMSNorm
from naz_data import ResidentNazBatcher
from train_naz import build_resident_semantic_cache


def grad_abs_sum(parameter: torch.nn.Parameter) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().abs().sum())


def tiny_config() -> DilConfig:
    return DilConfig(
        vocab_size=64,
        pad_token_id=0,
        eos_token_id=1,
        hidden_size=32,
        intermediate_size=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
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
        num_mlp_layers=1,
        num_samples=2,
        energy_target_samples=4,
        decode_chunk_size=2,
    )


def test_dil_config_uses_symmetric_context_contract():
    config = tiny_config()

    assert config.context_radius == 2
    assert config.context_size == 5
    assert config.target_index == 2
    assert config.checkpoint_format_version == 9
    assert not hasattr(config, "context_left_radius")


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
    labels = torch.tensor(
        [
            [5, 6, 7, -100],
            [11, 12, 13, -100],
        ],
        dtype=torch.long,
    )
    teacher_layers = torch.randn(input_ids.shape[0], 4, config.latent_size)
    teacher_mask = torch.ones(input_ids.shape[0], dtype=torch.bool)

    outputs = model(
        input_ids=input_ids,
        word_masks=word_masks,
        labels=labels,
        teacher_layers=teacher_layers,
        teacher_mask=teacher_mask,
    )

    assert outputs.mean.shape == (2, config.latent_size)
    assert outputs.log_std.shape == (2, config.latent_size)
    assert outputs.logits.shape == (2, config.max_word_bytes, config.vocab_size)
    assert isinstance(model.encoder.encoder_layers[0].mlp, DilGatedMLP)
    assert isinstance(model.encoder.encoder_layers[0].layernorm, DilRMSNorm)
    assert model.decoder.lm_head_weight is model.encoder.embed_tokens.weight
    assert torch.isfinite(outputs.loss)


def test_dil_encoder_conditions_target_with_symmetric_context():
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
                    torch.tensor([14, 0, 0, 0]),
                    torch.tensor([15, 0, 0, 0]),
                ]
            ),
            torch.stack(
                [
                    torch.tensor([2, 0, 0, 0]),
                    torch.tensor([3, 4, 0, 0]),
                    target,
                    torch.tensor([16, 0, 0, 0]),
                    torch.tensor([17, 0, 0, 0]),
                ]
            ),
        ]
    )
    word_masks = input_ids.ne(config.pad_token_id)

    with torch.no_grad():
        latent_states = model.encoder(input_ids=input_ids, word_masks=word_masks)
        mean, _ = torch.chunk(latent_states, 2, dim=-1)

    assert not torch.allclose(mean[0], mean[1])


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
                    torch.tensor([5, 0, 0, 0]),
                ]
            ),
            torch.stack(
                [
                    torch.tensor([5, 0, 0, 0]),
                    torch.tensor([4, 0, 0, 0]),
                    target,
                    torch.tensor([3, 0, 0, 0]),
                    torch.tensor([2, 0, 0, 0]),
                ]
            ),
        ]
    )
    word_masks = input_ids.ne(config.pad_token_id)

    with torch.no_grad():
        mean, _ = torch.chunk(model.encoder(input_ids=input_ids, word_masks=word_masks), 2, dim=-1)

    assert not torch.allclose(mean[0], mean[1])


def test_reconstruction_loss_updates_vae_encoder():
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
    labels = torch.tensor(
        [
            [5, 6, 7, -100],
            [11, 12, 13, -100],
        ],
        dtype=torch.long,
    )

    outputs = model(input_ids=input_ids, word_masks=word_masks, labels=labels)
    (outputs.ce_loss + outputs.length_loss).backward()

    assert grad_abs_sum(model.encoder.embed_tokens.weight) > 0.0
    assert grad_abs_sum(model.encoder.hidden_to_latent.weight) > 0.0
    assert model.decoder.latent_to_hidden.weight.grad is not None
    assert model.length_head[-1].weight.grad is not None


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
    (outputs.kl_loss * config.kl_weight + outputs.distill_loss).backward()

    assert model.encoder.embed_tokens.weight.grad is not None
    assert model.encoder.context_q_proj.weight.grad is not None
    assert model.encoder.context_gate.weight.grad is not None
    assert model.decoder.latent_to_hidden.weight.grad is None


def test_naz_uses_dil_encoder_target_log_std(tmp_path):
    dil_config = tiny_config()
    dil_model = Dil(dil_config)
    with torch.no_grad():
        dil_model.encoder.hidden_to_latent.weight.zero_()
        dil_model.encoder.hidden_to_latent.bias.zero_()
        dil_model.encoder.hidden_to_latent.bias[dil_config.latent_size :].fill_(-1.25)
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
        num_mlp_layers=1,
        num_samples=2,
        energy_target_samples=4,
    )
    model = Naz(naz_config)
    target_input_ids = torch.tensor(
        [[[2, 0, 0, 0], [3, 4, 0, 0], [5, 6, 7, 0]]],
        dtype=torch.long,
    )
    target_word_masks = target_input_ids.ne(dil_config.pad_token_id)
    unit_mask = torch.ones(target_input_ids.shape[:2], dtype=torch.bool)

    _, target_log_std = model.target_distribution(target_input_ids, target_word_masks, unit_mask)

    assert torch.allclose(target_log_std, torch.full_like(target_log_std, -1.25))


def test_naz_input_uses_frozen_dil_semantic_embeddings(tmp_path):
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
        num_mlp_layers=1,
        num_samples=2,
        energy_target_samples=4,
    )
    model = Naz(naz_config)
    input_ids = torch.tensor(
        [[[2, 0, 0, 0], [3, 4, 0, 0], [5, 6, 7, 0]]],
        dtype=torch.long,
    )
    word_masks = input_ids.ne(dil_config.pad_token_id)
    unit_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool)

    embeddings = model.semantic_embeddings(input_ids, word_masks, unit_mask)
    embeddings.sum().backward()

    assert embeddings.shape == (1, 3, naz_config.hidden_size)
    assert model.student_core.semantic_embed_proj[-2].weight.grad is not None
    assert grad_abs_sum(model.dil_model.encoder.embed_tokens.weight) == 0.0
    assert not hasattr(model, "byte_embed_tokens")


def test_naz_distribution_head_outputs_mean_and_log_std(tmp_path):
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

    pred_mean, pred_log_std = model.generative_head.sample_distribution(hidden_states)

    assert pred_mean.shape == (2, 3, dil_config.latent_size)
    assert pred_log_std.shape == (2, 3, dil_config.latent_size)
    assert float(pred_log_std.detach().min()) >= naz_config.pred_log_std_min
    assert float(pred_log_std.detach().max()) <= naz_config.pred_log_std_max


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
    cached_prefix = model.transformer(inputs_embeds=inputs_embeds[:, :2], use_cache=True)
    cached_last = model.transformer(
        inputs_embeds=inputs_embeds[:, 2:],
        past_key_values=cached_prefix.past_key_values,
        use_cache=True,
    ).last_hidden_state

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


def test_naz_forward_samples_energy_from_predicted_distribution(tmp_path):
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
        [[[3, 4, 0, 0], [5, 6, 7, 0], [8, 9, 0, 0]]],
        dtype=torch.long,
    )
    target_word_masks = target_input_ids.ne(dil_config.pad_token_id)
    unit_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool)

    outputs = model(
        input_ids=input_ids,
        word_masks=word_masks,
        target_input_ids=target_input_ids,
        target_word_masks=target_word_masks,
        unit_mask=unit_mask,
    )

    assert outputs.latent_predictions.shape == (naz_config.num_samples, 3, dil_config.latent_size)
    assert outputs.predicted_mean.shape == (3, dil_config.latent_size)
    assert outputs.predicted_log_std.shape == (3, dil_config.latent_size)
    assert torch.isfinite(outputs.loss)
    assert torch.isfinite(outputs.log_std_loss)


def test_naz_semantic_loop_generation_does_not_reencode_generated_tokens(tmp_path):
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
    )

    assert call_count == 1
    assert outputs.sequences.shape == (1, 6, dil_config.max_word_bytes)
    assert outputs.generated_mean.shape == (1, 4, dil_config.latent_size)
    assert outputs.generated_log_std.shape == (1, 4, dil_config.latent_size)
    assert outputs.generated_lengths.shape == (1, 4)


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

    semantic_states, mean_cache, log_std_cache = build_resident_semantic_cache(
        model,
        batcher,
        chunk_tokens=2,
        autocast_enabled=False,
    )
    positions = torch.arange(dil_config.max_word_bytes).reshape(1, 1, -1)
    masks = positions < lengths.reshape(1, -1, 1)
    unit_mask = torch.ones((1, byte_ids.shape[0]), dtype=torch.bool)
    full_mean, full_log_std = model.latent_distribution(byte_ids.unsqueeze(0), masks, unit_mask)

    assert torch.allclose(mean_cache, full_mean.reshape(byte_ids.shape[0], -1))
    assert torch.allclose(log_std_cache, full_log_std.reshape(byte_ids.shape[0], -1))
    assert torch.allclose(semantic_states, torch.cat((mean_cache, log_std_cache), dim=-1))


def test_naz_decode_latent_tokens_uses_chunked_batch_shape(tmp_path):
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
    latents = torch.randn(1, 5, dil_config.latent_size)

    token_ids, masks, lengths = model.decode_latent_tokens(latents, chunk_size=2)

    assert token_ids.shape == (1, 5, dil_config.max_word_bytes)
    assert masks.shape == (1, 5, dil_config.max_word_bytes)
    assert lengths.shape == (1, 5)
