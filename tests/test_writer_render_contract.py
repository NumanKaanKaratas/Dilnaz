from pathlib import Path
from types import SimpleNamespace

import torch

from dilnaz.models.dil import Dil, DilConfig
from dilnaz.surface import pack_token_units
from dilnaz.tokenization import HybridTokenizer
from dilnaz.train.interface.interface_dil import decode_tokens, make_batch, tokenize_text
from dilnaz.train.interface.writer_render import render_latents_with_unit_writer
from dilnaz.train.writer.train import HybridDilUnitWriterDataset


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
    )


def constant_decode(token_id: int = 65):
    def decode(semantic: torch.Tensor):
        batch_size = semantic.shape[0]
        token_ids = torch.full((batch_size, 4), 0, dtype=torch.long, device=semantic.device)
        token_mask = torch.zeros((batch_size, 4), dtype=torch.bool, device=semantic.device)
        token_ids[:, 0] = token_id
        token_mask[:, 0] = True
        return SimpleNamespace(
            token_ids=token_ids,
            token_mask=token_mask,
            lengths=torch.ones((batch_size,), dtype=torch.long, device=semantic.device),
            stopped=torch.ones((batch_size,), dtype=torch.bool, device=semantic.device),
        )

    return decode


def test_render_uses_unit_microbatches():
    cfg = tiny_config()
    model = Dil(cfg).eval()
    surface = pack_token_units(
        [[[2], [3, 4], [5]]],
        pad_token_id=cfg.pad_token_id,
        bucket_sizes=cfg.surface_bucket_sizes,
        max_pieces_per_unit=cfg.max_surface_pieces_per_unit,
    )
    latents = model.encode(surface).float()

    class DecodePatch:
        def __init__(self):
            self.calls = 0

        def __call__(self, semantic):
            assert semantic.dim() == 2
            self.calls += 1
            return constant_decode()(semantic)

    patched = DecodePatch()
    model.decode_semantic = patched

    dummy_tokenizer = HybridTokenizer(
        char_tokens=[],
        surface_tokens=[],
        numeric_tokens=[],
        common_word_tokens=[],
        contextual_tokens={},
    )
    render_latents_with_unit_writer(model, dummy_tokenizer, latents, microbatch_size=2)
    assert patched.calls == 2


def test_render_short_sequence_returns_expected_count():
    cfg = tiny_config()
    model = Dil(cfg).eval()
    model.decode_semantic = constant_decode()
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
    output = render_latents_with_unit_writer(model, dummy_tokenizer, latents)
    assert len(output) == 3, f"expected 3 tokens for 3-unit sequence, got {len(output)}"


def test_render_respects_unit_mask():
    cfg = tiny_config()
    model = Dil(cfg).eval()
    model.decode_semantic = constant_decode()
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
    output = render_latents_with_unit_writer(model, dummy_tokenizer, latents, unit_mask=unit_mask)
    assert len(output) == 2, f"expected 2 tokens for 3-unit sequence with unit_mask excluding index 1, got {len(output)}"


def test_render_long_sequence_uses_unit_batches():
    cfg = tiny_config()
    model = Dil(cfg).eval()
    model.decode_semantic = constant_decode()
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
    output = render_latents_with_unit_writer(model, dummy_tokenizer, latents, microbatch_size=3)
    assert len(output) <= unit_count, f"expected at most {unit_count} tokens (empty positions are filtered), got {len(output)}"
    assert len(output) >= unit_count // 2, f"expected at least {unit_count // 2} non-empty tokens, got {len(output)}"


def test_render_output_is_finite():
    cfg = tiny_config()
    model = Dil(cfg).eval()
    model.decode_semantic = constant_decode()
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
    output = render_latents_with_unit_writer(model, dummy_tokenizer, latents)
    assert isinstance(output, list), "output must be a list of strings"


def test_interface_dil_decode_tokens_does_not_call_direct_decode_semantic():
    """interface_dil.decode_tokens must use the unit writer helper, not an ad-hoc direct decode."""
    import dilnaz.train.interface.interface_dil as interface

    source = Path(interface.__file__).read_text(encoding="utf-8")
    assert "decode_semantic(latents.unsqueeze" not in source, (
        "interface_dil must not call decode_semantic with unsqueeze directly"
    )
    assert "render_latents_with_unit_writer" in source, (
        "interface_dil.decode_tokens must import and use render_latents_with_unit_writer"
    )


def test_writer_contract_has_no_context_window_fields():
    cfg = tiny_config()
    for name in (
        "_".join(("writer", "sliding", "window", "size")),
        "_".join(("writer", "left", "frozen")),
        "_".join(("writer", "active", "size")),
        "_".join(("writer", "right", "guard")),
        "_".join(("writer", "stride")),
    ):
        assert not hasattr(cfg, name)


def test_unit_writer_dataset_encodes_full_sequence_and_gathers_writer_units(tmp_path: Path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"araba car"}\n', encoding="utf-8")
    tokenizer = HybridTokenizer(
        char_tokens=[],
        surface_tokens=["araba", "car"],
        numeric_tokens=[],
        common_word_tokens=[],
        contextual_tokens={},
    )
    cfg = tiny_config()
    cfg.vocab_size = tokenizer.vocab_size
    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.eos_token_id = tokenizer.eos_token_id
    dataset = HybridDilUnitWriterDataset(
        data,
        cfg,
        tokenizer,
        batch_size=8,
        read_chars=1024,
        repeat=False,
        context_aug_max_units=0,
    )
    batch = next(dataset.iter_once(worker_id=0, worker_count=1))

    assert "labels" not in batch
    assert "window_mask" not in batch
    assert batch["surface"].batch_size == 1
    assert batch["writer_source_rows"].shape[0] == batch["writer_labels"].true_lengths.shape[0]
    assert batch["writer_labels"].true_lengths.shape[0] == int(batch["surface"].unit_mask.sum())
    assert int(batch["writer_unit_indices"].max()) < batch["surface"].unit_count


def test_unit_writer_dataset_adds_short_context_spans(tmp_path: Path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"araba car araba car araba car araba car araba car"}\n', encoding="utf-8")
    tokenizer = HybridTokenizer(
        char_tokens=[],
        surface_tokens=["araba", "car"],
        numeric_tokens=[],
        common_word_tokens=[],
        contextual_tokens={},
    )
    cfg = tiny_config()
    cfg.vocab_size = tokenizer.vocab_size
    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.eos_token_id = tokenizer.eos_token_id
    dataset = HybridDilUnitWriterDataset(
        data,
        cfg,
        tokenizer,
        batch_size=128,
        read_chars=1024,
        repeat=False,
        context_aug_max_units=4,
        context_aug_stride=2,
    )
    batch = next(dataset.iter_once(worker_id=0, worker_count=1))
    row_lengths = batch["surface"].unit_mask.sum(dim=1)

    assert batch["surface"].batch_size > 1
    assert row_lengths.max().item() > 4
    assert row_lengths.eq(1).any()
    assert row_lengths.eq(2).any()
    assert row_lengths.eq(4).any()
    assert batch["writer_labels"].true_lengths.shape[0] == int(batch["surface"].unit_mask.sum())
