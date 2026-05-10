import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dilnaz"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dilnaz" / "train"))

from dil_data import load_hybrid_tokenizer
from models.configuration_dil import DilConfig
from models.modeling_dil import Dil
from parallel_dil_data import (
    ParallelAlignmentGroup,
    ParallelDilBatchDataset,
    alignment_groups_to_tensors,
    apply_one_to_one_shared_teacher,
    parse_parallel_line,
    parallel_alignment_loss,
    parallel_total_loss,
)
from train_dil import model_inputs


def tiny_parallel_config(tokenizer) -> DilConfig:
    return DilConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        hidden_size=8,
        intermediate_size=16,
        num_encoder_layers=2,
        latent_size=4,
        max_word_bytes=32,
        context_radius=1,
        dil_dropout=0.0,
    )


def decoded_labels(tokenizer, labels: torch.Tensor) -> list[str]:
    rows = []
    for row in labels:
        ids = [
            int(token_id)
            for token_id in row.tolist()
            if int(token_id) not in (-100, tokenizer.eos_token_id)
        ]
        rows.append(tokenizer.decode(ids))
    return rows


def decoded_context_slot(tokenizer, input_ids: torch.Tensor, slot: int, pad_token_id: int) -> list[str]:
    rows = []
    for row in input_ids[:, slot]:
        ids = [int(token_id) for token_id in row.tolist() if int(token_id) != pad_token_id]
        rows.append(tokenizer.decode(ids))
    return rows


def test_parse_parallel_tr_en_line():
    parsed = parse_parallel_line("eng\ttur\tThe lioness's tooth broke.\tDişi aslanın dişi kırıldı.\n")

    assert parsed == ("Dişi aslanın dişi kırıldı.", "The lioness's tooth broke.")


def test_parallel_batch_keeps_pair_rows_together_and_decodes_surfaces(tmp_path):
    data_file = tmp_path / "tr-en.txt"
    data_file.write_text(
        "eng\ttur\tThe lioness's tooth broke.\tDişi aslanın dişi kırıldı.\n",
        encoding="utf-8",
    )
    tokenizer = load_hybrid_tokenizer()
    dataset = ParallelDilBatchDataset(
        data_file,
        tiny_parallel_config(tokenizer),
        tokenizer,
        batch_size=64,
        repeat=False,
    )

    batch = next(dataset.iter_once(worker_id=0, worker_count=1))
    surfaces = decoded_labels(tokenizer, batch["labels"])
    target_slot_surfaces = decoded_context_slot(
        tokenizer,
        batch["input_ids"],
        tiny_parallel_config(tokenizer).target_index,
        tokenizer.pad_token_id,
    )

    assert "Dişi" in surfaces
    assert " aslanın" in surfaces
    assert " lioness" in "".join(surfaces) or "lioness" in "".join(surfaces)
    assert target_slot_surfaces == surfaces
    assert batch["row_pair_indices"].unique().tolist() == [0]
    assert set(batch["row_side_ids"].tolist()) == {0, 1}


def test_parallel_batches_fill_fixed_training_shape(tmp_path):
    data_file = tmp_path / "tr-en.txt"
    data_file.write_text(
        "\n".join(
            [
                "eng\ttur\tcar\taraba",
                "eng\ttur\thouse\tev",
                "eng\ttur\troad\tyol",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    tokenizer = load_hybrid_tokenizer()
    config = tiny_parallel_config(tokenizer)
    dataset = ParallelDilBatchDataset(
        data_file,
        config,
        tokenizer,
        batch_size=3,
        repeat=False,
    )

    batches = list(dataset.iter_once(worker_id=0, worker_count=1))

    assert [batch["labels"].shape[0] for batch in batches] == [3, 3]
    assert all(batch["input_ids"].shape[:2] == (3, config.context_size) for batch in batches)


def test_one_to_one_alignment_shares_teacher_vector():
    teacher = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)
    teacher_mask = torch.ones(3, dtype=torch.bool)
    groups = [ParallelAlignmentGroup((0,), (1,), 1.0)]

    shared = apply_one_to_one_shared_teacher(teacher, teacher_mask, groups)

    expected = (teacher[0] + teacher[1]) * 0.5
    assert torch.equal(shared[0], expected)
    assert torch.equal(shared[1], expected)
    assert torch.equal(shared[2], teacher[2])


def test_phrase_alignment_loss_uses_group_mean_not_individual_tokens():
    mean = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ],
        requires_grad=True,
    )
    batch = alignment_groups_to_tensors(
        [ParallelAlignmentGroup((0, 1), (2,), 1.0)],
        torch.device("cpu"),
    )

    loss = parallel_alignment_loss(mean, batch)
    loss.backward()

    assert float(loss.detach()) < 1e-6
    assert mean.grad is not None


def test_parallel_total_loss_adds_weighted_alignment_loss():
    mean = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    batch = alignment_groups_to_tensors(
        [ParallelAlignmentGroup((0,), (1,), 1.0)],
        torch.device("cpu"),
    )
    outputs = SimpleNamespace(loss=torch.tensor(2.0), semantic=mean)

    total, alignment = parallel_total_loss(outputs, batch, 0.25)

    assert torch.allclose(total, outputs.loss + alignment * 0.25)


def test_parallel_dil_mock_teacher_forward_backward(tmp_path):
    data_file = tmp_path / "tr-en.txt"
    data_file.write_text("eng\ttur\tcar\taraba\n", encoding="utf-8")
    tokenizer = load_hybrid_tokenizer()
    config = tiny_parallel_config(tokenizer)
    dataset = ParallelDilBatchDataset(data_file, config, tokenizer, batch_size=16, repeat=False)
    batch = next(dataset.iter_once(worker_id=0, worker_count=1))
    row_count = batch["labels"].shape[0]
    batch["teacher_layers"] = torch.randn(row_count, 4, 8)
    batch["teacher_mask"] = torch.ones(row_count, dtype=torch.bool)
    batch.update(
        alignment_groups_to_tensors(
            [ParallelAlignmentGroup((0,), (1,), 1.0)],
            torch.device("cpu"),
        )
    )
    model = Dil(config)

    outputs = model(**model_inputs(batch))
    loss, alignment = parallel_total_loss(outputs, batch, 1.0)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(alignment)
    assert outputs.writer_loss is not None
    assert outputs.distill_loss is not None
    assert not hasattr(outputs, "ce_loss")
    assert model.encoder.embed_tokens.weight.grad is not None


def test_dil_checkpoint_format_matches_encoder_only_family():
    tokenizer = load_hybrid_tokenizer()
    config = tiny_parallel_config(tokenizer)
    model = Dil(config)

    assert config.checkpoint_format_version == 17
    assert hasattr(model, "writer")
    assert any(key.startswith("writer.") for key in model.state_dict())
