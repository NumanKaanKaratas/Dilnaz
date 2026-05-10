import sys
from types import SimpleNamespace
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
from naz_data import PromptAnswerNazDataset, ResidentNazBatcher, ResidentNazSemanticBatcher, StreamingTextNazDataset
from train_naz import build_resident_semantic_cache
from train_dil_writer import freeze_for_writer_only, writer_only_forward
from interface_naz import stream_text as stream_naz_text
from train_naz_finetune import exact_answer_accuracy, masked_sft_forward


def grad_abs_sum(parameter: torch.nn.Parameter) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().abs().sum())


def fixture_tokenizer():
    return __import__("tokenization").HybridTokenizer.from_file(
        Path(__file__).resolve().parents[1] / "dilnaz" / "tokenization" / "hybrid_surface_vocab.json"
    )


class FakeGeneratedDil:
    def __init__(self, tokenizer, max_word_bytes: int, decoded_steps: list[str | None]):
        self.tokenizer = tokenizer
        self.max_word_bytes = max_word_bytes
        self.decoded_steps = decoded_steps
        self.calls = 0

    def decode_semantic(self, latent: torch.Tensor):
        value = self.decoded_steps[self.calls]
        self.calls += 1
        batch_size = latent.shape[0]
        token_ids = torch.full((batch_size, self.max_word_bytes), self.tokenizer.pad_token_id, dtype=torch.long)
        token_masks = torch.zeros_like(token_ids, dtype=torch.bool)
        lengths = torch.zeros((batch_size,), dtype=torch.long)
        if value is None:
            return token_ids, token_masks, lengths
        segment = next(segment for segment in self.tokenizer.encode_segments(value) if segment.piece_len > 0)
        ids = torch.tensor(segment.token_ids, dtype=torch.long)
        token_ids[:, : ids.numel()] = ids
        token_masks[:, : ids.numel()] = True
        lengths[:] = ids.numel()
        return token_ids, token_masks, lengths


class FakeGeneratedNaz:
    def __init__(self, tokenizer, max_word_bytes: int, decoded_steps: list[str | None]):
        self.dil_model = FakeGeneratedDil(tokenizer, max_word_bytes, decoded_steps)

    def eval(self):
        return self

    def train(self):
        return self

    def generate_stream(self, input_ids, word_masks, unit_mask, max_new_tokens, min_new_tokens, repetition_cos_threshold):
        batch_size = input_ids.shape[0]
        for _ in range(max_new_tokens):
            yield SimpleNamespace(
                latent=torch.zeros((batch_size, 1), dtype=torch.float32),
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


def test_dil_config_uses_left_context_contract():
    config = tiny_config()

    assert config.context_radius == 2
    assert config.context_size == 3
    assert config.target_index == 2
    assert config.checkpoint_format_version == 17
    assert not hasattr(config, "context_left_radius")


def test_dil_rejects_pre_parallel_writer_checkpoint_family():
    config = tiny_config()
    config.checkpoint_format_version = 13

    try:
        Dil(config)
    except ValueError as error:
        assert "checkpoint_format_version=17" in str(error)
    else:
        raise AssertionError("Dil accepted stale checkpoint_format_version")


def test_dil_forward_keeps_target_latent_shape():
    config = tiny_config()
    model = Dil(config)
    input_ids = torch.tensor(
        [
            [[2, 0, 0, 0], [3, 4, 0, 0], [5, 6, 7, 0]],
            [[8, 0, 0, 0], [9, 10, 0, 0], [11, 12, 13, 0]],
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
                ]
            ),
            torch.stack(
                [
                    torch.tensor([16, 0, 0, 0]),
                    torch.tensor([17, 0, 0, 0]),
                    target,
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
                ]
            ),
            torch.stack(
                [
                    torch.tensor([3, 0, 0, 0]),
                    torch.tensor([2, 0, 0, 0]),
                    target,
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
            [[2, 0, 0, 0], [3, 4, 0, 0], [5, 6, 7, 0]],
            [[8, 0, 0, 0], [9, 10, 0, 0], [11, 12, 13, 0]],
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
            [[2, 0, 0, 0], [3, 4, 0, 0], [5, 6, 7, 0]],
            [[8, 0, 0, 0], [9, 10, 0, 0], [11, 12, 13, 0]],
        ],
        dtype=torch.long,
    )
    labels = torch.tensor(
        [
            [5, 6, 7, 1],
            [11, 12, 13, 1],
        ],
        dtype=torch.long,
    )
    batch = {
        "input_ids": input_ids,
        "word_masks": input_ids.ne(config.pad_token_id),
        "labels": labels,
    }

    loss, byte_acc, token_exact = writer_only_forward(model, batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(byte_acc)
    assert torch.isfinite(token_exact)
    assert grad_abs_sum(model.encoder.embed_tokens.weight) == 0.0
    assert grad_abs_sum(model.encoder.hidden_to_semantic.weight) == 0.0
    assert model.writer.token_embeddings.weight.grad is not None


def test_naz_uses_dil_encoder_semantic_with_zero_uncertainty(tmp_path):
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

    target_mean, target_log_std = model.target_distribution(target_input_ids, target_word_masks, unit_mask)

    assert target_mean.shape == (3, dil_config.latent_size)
    assert torch.equal(target_log_std, torch.zeros_like(target_log_std))


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
    )
    model = Naz(naz_config)
    input_ids = torch.tensor(
        [[[2, 0, 0, 0], [3, 4, 0, 0], [5, 6, 7, 0]]],
        dtype=torch.long,
    )
    word_masks = input_ids.ne(dil_config.pad_token_id)
    unit_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool)

    semantic_states = model.semantic_states(input_ids, word_masks, unit_mask)
    embeddings = model.semantic_embeddings(input_ids, word_masks, unit_mask)
    embeddings.sum().backward()

    assert semantic_states.shape == (1, 3, dil_config.latent_size)
    assert embeddings.shape == (1, 3, naz_config.hidden_size)
    assert model.student_core.semantic_embed_proj[-2].weight.grad is not None
    assert grad_abs_sum(model.dil_model.encoder.embed_tokens.weight) == 0.0
    assert not hasattr(model, "byte_embed_tokens")


def test_naz_latent_head_predicts_single_next_latent(tmp_path):
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

    latents = model.latent_head(hidden_states)

    assert latents.shape == (2, 3, dil_config.latent_size)
    assert model.student_core.semantic_embed_proj[0].in_features == dil_config.latent_size
    assert model.student_core.semantic_embed_proj[0].out_features == 2 * naz_config.hidden_size
    assert not hasattr(model, "generative_head")
    assert not hasattr(model, "mean_head")
    assert not hasattr(model, "writer_logits")
    assert hasattr(model.dil_model, "writer")


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


def test_naz_forward_optimizes_lcm_next_latent_objective(tmp_path):
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

    expected_loss = outputs.reconstruction_loss
    assert outputs.latent_predictions.shape == (1, 3, dil_config.latent_size)
    assert outputs.predicted_latents.shape == (3, dil_config.latent_size)
    assert outputs.target_latents.shape == (3, dil_config.latent_size)
    assert int(outputs.num_targets) == 3
    assert torch.isfinite(outputs.loss)
    assert torch.allclose(outputs.loss, expected_loss)
    assert torch.allclose(outputs.loss, outputs.mse_loss)
    assert torch.allclose(outputs.mse_loss, outputs.mse_mean * outputs.num_targets.to(outputs.mse_mean.dtype))
    assert torch.isfinite(outputs.reconstruction_loss)
    assert torch.isfinite(outputs.mse_loss)
    assert torch.isfinite(outputs.mse_mean)
    assert torch.isfinite(outputs.cosine_loss)
    assert torch.isfinite(outputs.latent_cos)
    assert not hasattr(outputs, "energy_loss")
    assert not hasattr(outputs, "writer_loss")
    assert not hasattr(outputs, "byte_acc")


def test_naz_normalizer_fits_latent_distribution(tmp_path):
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
    latents = torch.randn(16, dil_config.latent_size) * 3.0 + 2.0

    model.latent_normalizer.fit(latents)
    normalized = model.latent_normalizer.normalize(latents)

    assert torch.allclose(normalized.mean(dim=0), torch.zeros(dil_config.latent_size), atol=1e-5, rtol=1e-5)
    assert torch.all(model.latent_normalizer.scale > 0)


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
        assert step.latent_cos_to_previous.shape == (1,)
        assert step.should_stop.shape == (1,)


def test_naz_generate_stream_denormalizes_output_latents(tmp_path):
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
    for parameter in model.latent_head.parameters():
        parameter.data.zero_()
    model.latent_normalizer.mean.fill_(5.0)
    model.latent_normalizer.scale.fill_(2.0)
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

    assert torch.allclose(step.latent, torch.full_like(step.latent, 5.0))


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


def test_resident_semantic_cache_matches_full_left_context_pass(tmp_path):
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

    semantic_states, mean_cache, cached_byte_ids, cached_lengths = build_resident_semantic_cache(
        model,
        batcher,
        chunk_tokens=2,
        autocast_enabled=False,
    )
    positions = torch.arange(dil_config.max_word_bytes).reshape(1, 1, -1)
    masks = positions < lengths.reshape(1, -1, 1)
    unit_mask = torch.ones((1, byte_ids.shape[0]), dtype=torch.bool)
    full_mean, _ = model.latent_distribution(byte_ids.unsqueeze(0), masks, unit_mask)

    assert torch.allclose(mean_cache, full_mean.reshape(byte_ids.shape[0], -1), atol=1e-6, rtol=1e-5)
    assert torch.allclose(semantic_states, mean_cache)
    assert torch.equal(cached_byte_ids.cpu(), byte_ids)
    assert torch.equal(cached_lengths.cpu(), lengths)


def test_resident_naz_semantic_batcher_surfaces_next_latents():
    semantic_states = torch.randn(6, 4)
    target_mean = torch.randn(6, 4)
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
        target_mean,
        byte_ids,
        lengths,
        sequence_length=3,
        batch_size=2,
        seed=1,
    )

    batch = batcher.make_batch(torch.tensor([[0], [2]]))

    assert batch["semantic_states"].shape == (2, 3, 4)
    assert batch["target_mean"].shape == (2, 3, 4)
    assert batch["unit_mask"].dtype == torch.bool
    assert torch.equal(batch["semantic_states"][0, 0], semantic_states[0])
    assert torch.equal(batch["target_mean"][0, 0], target_mean[1])
    assert torch.equal(batch["semantic_states"][1, 0], semantic_states[2])
    assert torch.equal(batch["target_mean"][1, 0], target_mean[3])


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
    assert batch["target_input_ids"].shape == (2, 3, config.max_word_bytes)
    assert batch["word_masks"].dtype == torch.bool
    assert batch["target_word_masks"].dtype == torch.bool
    assert batch["unit_mask"].all()
    assert not any(tmp_path.glob("*.npy"))


def test_prompt_answer_naz_dataset_masks_only_answer_targets(tmp_path):
    data_file = tmp_path / "math.tsv"
    data_file.write_text("15 + 4241 =\t4256\n", encoding="utf-8")
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
    dataset = PromptAnswerNazDataset(
        data_file,
        tokenizer,
        config,
        sequence_length=32,
        batch_size=1,
        read_chars=16,
        repeat=False,
    )

    batch = next(iter(dataset))

    assert batch["unit_mask"].sum() > batch["loss_mask"].sum()
    assert batch["loss_mask"].sum() >= 2
    assert not batch["loss_mask"][0, 0]
    assert batch["loss_mask"][0].nonzero()[0].item() > 0


def test_naz_sft_forward_uses_answer_loss_mask(tmp_path):
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
    input_ids = torch.tensor(
        [[[2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0]]],
        dtype=torch.long,
    )
    target_ids = torch.tensor(
        [[[3, 0, 0, 0], [4, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]],
        dtype=torch.long,
    )
    unit_mask = torch.tensor([[True, True, True, False]])
    loss_mask = torch.tensor([[False, False, True, False]])
    batch = {
        "input_ids": input_ids,
        "word_masks": input_ids.ne(dil_config.pad_token_id) & unit_mask.unsqueeze(-1),
        "target_input_ids": target_ids,
        "target_word_masks": target_ids.ne(dil_config.pad_token_id) & unit_mask.unsqueeze(-1),
        "unit_mask": unit_mask,
        "loss_mask": loss_mask,
    }

    outputs = masked_sft_forward(model, batch)

    assert torch.isfinite(outputs.loss)
    assert int(outputs.num_targets) == 1
    assert outputs.predicted_latents.shape == (1, dil_config.latent_size)


def test_naz_sft_exact_eval_stops_at_writer_eos(tmp_path, capsys):
    data_file = tmp_path / "math.tsv"
    data_file.write_text("15 + 4241 =\t4\n", encoding="utf-8")
    tokenizer = fixture_tokenizer()
    config = NazConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_word_bytes=8,
        latent_size=1,
    )
    model = FakeGeneratedNaz(tokenizer, config.max_word_bytes, ["4", None, "9"])

    metrics = exact_answer_accuracy(
        model,
        data_file,
        tokenizer,
        config,
        torch.device("cpu"),
        read_chars=32,
        max_examples=1,
        max_new_tokens=3,
        print_examples=1,
    )

    captured = capsys.readouterr().out
    assert metrics["eval_exact_answer_acc"] == 1.0
    assert "predicted='4'" in captured
    assert model.dil_model.calls == 2


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
    )

    captured = capsys.readouterr().out
    assert captured == f"{prompt}4\n"
    assert model.dil_model.calls == 2
