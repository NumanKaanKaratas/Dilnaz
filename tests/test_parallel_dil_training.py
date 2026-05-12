import sys
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dilnaz"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dilnaz" / "train"))

from dil_data import DILNAZ_READY_FORMAT, NLLB_LAYER_GROUPS, file_sha256, load_hybrid_tokenizer
from models.dil import DilConfig
from models.dil import Dil
from parallel_dil_data import (
    ParallelAlignmentGroup,
    ParallelDilBatchDataset,
    alignment_groups_to_tensors,
    apply_one_to_one_shared_teacher,
    parse_parallel_line,
    parallel_alignment_loss,
    parallel_total_loss,
)
from tokenization import default_vocab_path
from train_dil import (
    DilPretrainTrainer,
    make_trainer as make_dil_trainer,
    model_inputs,
    parse_args as parse_dil_args,
)
from train_dil_teacherless_parallel import (
    TeacherlessParallelDilTrainer,
    TeacherlessParallelJsonlDataset,
    batch_token_set_targets,
    make_trainer as make_teacherless_trainer,
    parse_args as parse_teacherless_args,
)


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


def tiny_teacherless_config(tokenizer) -> DilConfig:
    return DilConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        hidden_size=8,
        intermediate_size=16,
        num_encoder_layers=2,
        latent_size=8,
        max_word_bytes=16,
        context_radius=1,
        byte_conv_layers=0,
        byte_conv_expansion=1,
        dil_dropout=0.0,
        distillation_weight=0.0,
        mean_geometry_weight=0.0,
        variance_weight=0.0,
        writer_loss_weight=1.0,
        writer_num_layers=1,
        writer_conv_expansion=1,
        writer_dropout=0.0,
    )


def tiny_trainer_args(tmp_path: Path, train_file: Path, output_name: str = "dil_out"):
    return parse_dil_args(
        [
            "--train-file",
            str(train_file),
            "--output-dir",
            str(tmp_path / output_name),
            "--tokenizer-vocab",
            str(default_vocab_path()),
            "--compile-mode",
            "off",
            "--data-mode",
            "streaming",
            "--max-steps",
            "1",
            "--batch-size",
            "2",
            "--eval-batch-size",
            "2",
            "--nllb-batch-size",
            "1",
            "--max-batch-reuse",
            "1",
            "--text-read-chars",
            "256",
            "--prefetch-factor",
            "1",
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
            "--num-workers",
            "0",
            "--seed",
            "1",
            "--hidden-size",
            "8",
            "--intermediate-size",
            "16",
            "--num-encoder-layers",
            "2",
            "--latent-size",
            "4",
            "--max-word-bytes",
            "8",
            "--context-radius",
            "1",
            "--byte-conv-layers",
            "0",
            "--byte-conv-kernel-size",
            "3",
            "--byte-conv-expansion",
            "1",
            "--dil-dropout",
            "0.0",
            "--distillation-weight",
            "1.0",
            "--mean-geometry-weight",
            "1.0",
            "--variance-weight",
            "0.0",
            "--writer-loss-weight",
            "0.0",
            "--writer-num-layers",
            "0",
            "--writer-conv-kernel-size",
            "3",
            "--writer-conv-expansion",
            "1",
            "--writer-dropout",
            "0.0",
        ]
    )


def tiny_teacherless_args(tmp_path: Path, train_file: Path, output_name: str = "teacherless_out"):
    return parse_teacherless_args(
        [
            "--train-file",
            str(train_file),
            "--output-dir",
            str(tmp_path / output_name),
            "--tokenizer-vocab",
            str(default_vocab_path()),
            "--compile-mode",
            "off",
            "--max-steps",
            "1",
            "--batch-size",
            "2",
            "--eval-batch-size",
            "2",
            "--max-segments",
            "6",
            "--min-segments",
            "1",
            "--shuffle-buffer-size",
            "2",
            "--prefetch-factor",
            "1",
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
            "--num-workers",
            "0",
            "--seed",
            "1",
            "--hidden-size",
            "8",
            "--intermediate-size",
            "16",
            "--num-encoder-layers",
            "2",
            "--latent-size",
            "8",
            "--max-word-bytes",
            "16",
            "--context-radius",
            "1",
            "--byte-conv-layers",
            "0",
            "--byte-conv-kernel-size",
            "3",
            "--byte-conv-expansion",
            "1",
            "--dil-dropout",
            "0.0",
            "--writer-num-layers",
            "1",
            "--writer-conv-kernel-size",
            "3",
            "--writer-conv-expansion",
            "1",
            "--writer-dropout",
            "0.0",
            "--token-set-start-step",
            "1",
            "--token-set-ramp-steps",
            "1",
            "--token-balance-start-step",
            "1",
            "--token-balance-ramp-steps",
            "1",
            "--covariance-start-step",
            "1",
        ]
    )


def write_tiny_ready_parquet(path: Path, config: DilConfig) -> None:
    rows = 4
    input_rows = []
    mask_rows = []
    label_rows = []
    teacher_rows = []
    for row_idx in range(rows):
        input_ids = [config.pad_token_id] * (config.context_size * config.max_word_bytes)
        word_masks = [False] * (config.context_size * config.max_word_bytes)
        for context_idx in range(config.context_size):
            offset = context_idx * config.max_word_bytes
            input_ids[offset] = 2 + row_idx + context_idx
            word_masks[offset] = True
        labels = [-100] * config.writer_max_positions
        labels[0] = 2 + row_idx
        labels[1] = config.writer_stop_token_id
        teacher_layers = torch.randn(len(NLLB_LAYER_GROUPS), 1024, generator=torch.Generator().manual_seed(row_idx))
        input_rows.append(input_ids)
        mask_rows.append(word_masks)
        label_rows.append(labels)
        teacher_rows.append(teacher_layers.reshape(-1).tolist())
    table = pa.table(
        {
            "input_ids": input_rows,
            "word_masks": mask_rows,
            "labels": label_rows,
            "teacher_layers": teacher_rows,
            "teacher_mask": [True] * rows,
        }
    )
    metadata = {
        "format": DILNAZ_READY_FORMAT,
        "tokenizer_vocab_size": str(config.vocab_size),
        "pad_token_id": str(config.pad_token_id),
        "eos_token_id": str(config.eos_token_id),
        "max_word_bytes": str(config.max_word_bytes),
        "context_radius": str(config.context_radius),
        "context_size": str(config.context_size),
        "target_index": str(config.target_index),
        "teacher_dim": "1024",
        "teacher_layer_count": str(len(NLLB_LAYER_GROUPS)),
        "tokenizer_vocab_sha256": file_sha256(default_vocab_path()),
        "teacher_formula": "centered_add_w050_grouped",
    }
    pq.write_table(table.replace_schema_metadata(metadata), path)


def tiny_trainer_config(args, tokenizer) -> DilConfig:
    return DilConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_encoder_layers=args.num_encoder_layers,
        latent_size=args.latent_size,
        max_word_bytes=args.max_word_bytes,
        context_radius=args.context_radius,
        byte_conv_layers=args.byte_conv_layers,
        byte_conv_kernel_size=args.byte_conv_kernel_size,
        byte_conv_expansion=args.byte_conv_expansion,
        dil_dropout=args.dil_dropout,
        distillation_weight=args.distillation_weight,
        mean_geometry_weight=args.mean_geometry_weight,
        variance_weight=args.variance_weight,
        writer_loss_weight=args.writer_loss_weight,
        writer_num_layers=args.writer_num_layers,
        writer_conv_kernel_size=args.writer_conv_kernel_size,
        writer_conv_expansion=args.writer_conv_expansion,
        writer_dropout=args.writer_dropout,
        tokenizer_vocab_file=args.tokenizer_vocab.name,
    )


def decoded_labels(tokenizer, labels: torch.Tensor) -> list[str]:
    rows = []
    for row in labels:
        ids = [
            int(token_id)
            for token_id in row.tolist()
            if int(token_id) not in (-100, tokenizer.eos_token_id) and int(token_id) < tokenizer.vocab_size
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


def test_teacherless_token_set_targets_are_batch_local_multihot():
    vocab_ids, targets = batch_token_set_targets(
        torch.tensor(
            [
                [[2, 3, 99, -100], [0, 99, -100, -100]],
                [[3, 4, 99, -100], [-100, -100, -100, -100]],
            ]
        ),
        writer_stop_token_id=99,
        pad_token_id=0,
        eos_token_id=1,
    )

    assert vocab_ids.tolist() == [2, 3, 4]
    assert targets.tolist() == [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]


def test_teacherless_parallel_jsonl_batch_keeps_sentence_pairs(tmp_path):
    data_file = tmp_path / "opus.jsonl"
    rows = [
        {"tr": "araba geldi", "en": "car came"},
        {"tr": "ev buyuk", "en": "house big"},
    ]
    data_file.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    tokenizer = load_hybrid_tokenizer()
    config = tiny_teacherless_config(tokenizer)
    dataset = TeacherlessParallelJsonlDataset(
        data_file,
        config,
        tokenizer,
        batch_size=2,
        max_segments=6,
        min_segments=1,
        min_length_ratio=0.25,
        max_length_ratio=4.0,
        shuffle_buffer_size=2,
        seed=1,
        repeat=False,
    )

    batch = next(iter(dataset))
    tr_surfaces = decoded_labels(tokenizer, batch["tr_labels"][0, batch["tr_unit_mask"][0]])
    en_surfaces = decoded_labels(tokenizer, batch["en_labels"][0, batch["en_unit_mask"][0]])

    assert batch["tr_input_ids"].shape == (2, 6, config.context_size, config.max_word_bytes)
    assert batch["en_input_ids"].shape == (2, 6, config.context_size, config.max_word_bytes)
    assert batch["tr_unit_mask"].sum().item() > 0
    assert batch["en_unit_mask"].sum().item() > 0
    assert "araba" in "".join(tr_surfaces) or "ev" in "".join(tr_surfaces)
    assert "car" in "".join(en_surfaces) or "house" in "".join(en_surfaces)


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


def test_dil_pretrain_trainer_runs_one_ready_parquet_step(tmp_path):
    tokenizer = load_hybrid_tokenizer()
    args = tiny_trainer_args(tmp_path, tmp_path / "ready.parquet")
    write_tiny_ready_parquet(args.train_file, tiny_trainer_config(args, tokenizer))

    trainer = DilPretrainTrainer(args)
    batch = next(trainer.build_train_iterator())
    result = trainer.train_step(batch, 1)
    result.loss.backward()

    assert torch.isfinite(result.loss)
    assert result.token_count == 4
    assert result.window_count == 2
    assert trainer.teacher is None
    assert trainer.train_is_parquet
    assert trainer.model.encoder.embed_tokens.weight.grad is not None


def test_teacherless_parallel_trainer_runs_one_step_and_resumes(tmp_path):
    data_file = tmp_path / "opus.jsonl"
    rows = [
        {"tr": "araba geldi", "en": "car came"},
        {"tr": "ev buyuk", "en": "house big"},
    ]
    data_file.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    args = tiny_teacherless_args(tmp_path, data_file)
    trainer = TeacherlessParallelDilTrainer(args)
    batch = next(trainer.build_train_iterator())
    result = trainer.train_step(batch, 1)
    result.loss.backward()
    checkpoint_dir = trainer.save_checkpoint("checkpoint-1", 1, {"loss": float(result.loss.detach())})

    resume_args = parse_teacherless_args(
        [
            "--train-file",
            str(data_file),
            "--output-dir",
            str(tmp_path / "teacherless_resumed"),
            "--resume",
            str(checkpoint_dir / "checkpoint.pt"),
            "--compile-mode",
            "off",
            "--max-steps",
            "2",
            "--batch-size",
            "2",
            "--eval-batch-size",
            "2",
            "--max-segments",
            "6",
            "--min-segments",
            "1",
            "--shuffle-buffer-size",
            "2",
            "--prefetch-factor",
            "1",
            "--log-every",
            "1",
            "--checkpoint-every",
            "0",
            "--eval-every",
            "0",
            "--max-eval-batches",
            "1",
            "--num-workers",
            "0",
        ]
    )
    resumed = make_teacherless_trainer(resume_args)

    assert torch.isfinite(result.loss)
    assert result.token_count > 0
    assert result.window_count > 0
    assert trainer.model.encoder.embed_tokens.weight.grad is not None
    assert resumed.start_step == 1
    assert resumed.config.hidden_size == 8
    assert resumed.config.latent_size == 8


def test_dil_trainer_resume_restores_step_and_config(tmp_path):
    tokenizer = load_hybrid_tokenizer()
    args = tiny_trainer_args(tmp_path, tmp_path / "ready.parquet")
    write_tiny_ready_parquet(args.train_file, tiny_trainer_config(args, tokenizer))
    trainer = DilPretrainTrainer(args)
    checkpoint_dir = trainer.save_checkpoint("checkpoint-1", 1, {"loss": 1.0})

    resume_args = parse_dil_args(
        [
            "--train-file",
            str(args.train_file),
            "--output-dir",
            str(tmp_path / "resumed"),
            "--resume",
            str(checkpoint_dir / "checkpoint.pt"),
            "--compile-mode",
            "off",
            "--data-mode",
            "streaming",
            "--max-steps",
            "2",
            "--batch-size",
            "2",
            "--eval-batch-size",
            "2",
            "--nllb-batch-size",
            "1",
            "--max-batch-reuse",
            "1",
            "--prefetch-factor",
            "1",
            "--log-every",
            "1",
            "--checkpoint-every",
            "0",
            "--eval-every",
            "0",
            "--max-eval-batches",
            "1",
            "--num-workers",
            "0",
        ]
    )
    resumed = make_dil_trainer(resume_args)

    assert isinstance(resumed, DilPretrainTrainer)
    assert resumed.start_step == 1
    assert resumed.config.hidden_size == 8
    assert resumed.config.context_radius == 1


def test_dil_checkpoint_format_matches_encoder_only_family():
    tokenizer = load_hybrid_tokenizer()
    config = tiny_parallel_config(tokenizer)
    model = Dil(config)

    assert config.checkpoint_format_version == 24
    assert hasattr(model, "writer")
    assert any(key.startswith("writer.") for key in model.state_dict())
