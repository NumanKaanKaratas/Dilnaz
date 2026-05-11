import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import IterableDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from byte_trainer_utils import (  # noqa: E402
    COMPILE_MODE_CHOICES,
    DeviceBatchPrefetcher,
    autocast_context,
    compile_forward,
    cudagraph_step_begin,
    cuda_sync,
    effective_compile_mode,
    load_checkpoint,
    restore_rng_state,
    rng_state,
    validate_compile_environment,
)
from dil_data import (  # noqa: E402
    ResidentDilBatcher,
    ResidentDilEvalLoader,
    context_offsets,
    load_hybrid_tokenizer,
    make_dil_batch_loader,
    segment_piece_ids,
    stream_teacher_text_items,
    trainable_segments,
)
from dilnaz_config import DIL_TRAIN_DEFAULTS  # noqa: E402
from models.configuration_dil import DilConfig  # noqa: E402
from models.modeling_dil import Dil  # noqa: E402
from trainer_core import make_adamw_param_groups, make_scheduler  # noqa: E402


CHECKPOINT_FORMAT_VERSION = 23
WRITER_OBJECTIVE = "sliding_block_writer_online_encoder"
WRITER_METRIC_KEYS = (
    "loss",
    "token_loss",
    "active_token_loss",
    "right_guard_token_loss",
    "left_consistency_loss",
    "commit_loss",
    "byte_acc",
    "token_exact",
    "stop_acc",
    "right_guard_byte_acc",
    "right_guard_token_exact",
    "right_guard_stop_acc",
    "commit_precision",
    "commit_recall",
    "commit_f1",
    "false_commit_rate",
    "mean_commit_score",
    "step0_byte_acc",
    "step0_token_exact",
    "step0_stop_acc",
    "stepT_byte_acc",
    "stepT_token_exact",
    "stepT_stop_acc",
    "self_conditioning_ratio",
    "mean_mask_ratio",
    "future_horizons",
)


class HybridDilSlidingWindowDataset(IterableDataset):
    def __init__(
        self,
        train_file: Path,
        config: DilConfig,
        tokenizer,
        batch_size: int,
        read_chars: int,
        repeat: bool = True,
        max_samples: int = 0,
        window_size: int | None = None,
        left_frozen: int | None = None,
        active_size: int | None = None,
        right_guard: int | None = None,
        stride: int | None = None,
    ):
        super().__init__()
        self.train_file = train_file
        self.config = config
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.read_chars = read_chars
        self.repeat = repeat
        self.max_samples = max_samples
        self.window_size = config.writer_sliding_window_size if window_size is None else window_size
        self.left_frozen = config.writer_left_frozen if left_frozen is None else left_frozen
        self.active_size = config.writer_active_size if active_size is None else active_size
        self.right_guard = config.writer_right_guard if right_guard is None else right_guard
        self.stride = config.writer_stride if stride is None else stride
        if self.left_frozen + self.active_size + self.right_guard != self.window_size:
            raise ValueError("writer window zones must sum to window_size")
        if self.stride <= 0 or self.stride > self.active_size:
            raise ValueError("stride must be in 1..active_size")
        self.zone_template = torch.full((self.window_size,), 1, dtype=torch.long)
        self.zone_template[: self.left_frozen] = 0
        self.zone_template[self.left_frozen + self.active_size :] = 2
        self._carry_texts: list[str] = []
        self._carry_line_ids: list[int] = []
        self._carry_segments = []
        self._carry_refs: list[tuple[int, int]] = []
        self._produced = 0

    def write_segment(self, target: np.ndarray, mask: np.ndarray, context_idx: int, segment):
        piece_ids = np.asarray(segment_piece_ids(segment), dtype=np.int64)
        width = piece_ids.shape[0]
        if width <= 0 or width > self.config.max_word_bytes:
            return
        target[context_idx, :width] = piece_ids
        mask[context_idx, :width] = True

    def make_batch(self, texts: list[str], line_ids: list[int], segments_by_text: list, refs: list[tuple[int, int]]):
        batch_size = len(refs)
        input_ids = np.full(
            (batch_size, self.window_size, self.config.context_size, self.config.max_word_bytes),
            self.config.pad_token_id,
            dtype=np.int64,
        )
        word_masks = np.zeros(
            (batch_size, self.window_size, self.config.context_size, self.config.max_word_bytes),
            dtype=np.bool_,
        )
        labels = np.full(
            (batch_size, self.window_size, self.config.writer_max_positions),
            -100,
            dtype=np.int64,
        )
        window_mask = np.zeros((batch_size, self.window_size), dtype=np.bool_)
        source_line_ids = np.zeros((batch_size,), dtype=np.int64)

        for batch_idx, (text_idx, active_start) in enumerate(refs):
            segments = segments_by_text[text_idx]
            source_line_ids[batch_idx] = line_ids[text_idx]
            window_start = active_start - self.left_frozen
            for window_idx in range(self.window_size):
                token_idx = window_start + window_idx
                if token_idx < 0 or token_idx >= len(segments):
                    continue
                window_mask[batch_idx, window_idx] = True
                segment = segments[token_idx]
                for context_idx, offset in enumerate(context_offsets(self.config.context_radius)):
                    source_idx = token_idx + offset
                    if 0 <= source_idx < len(segments):
                        self.write_segment(
                            input_ids[batch_idx, window_idx],
                            word_masks[batch_idx, window_idx],
                            context_idx,
                            segments[source_idx],
                        )
                piece_ids = np.asarray(segment_piece_ids(segment), dtype=np.int64)
                labels[batch_idx, window_idx, : piece_ids.shape[0]] = piece_ids
                labels[batch_idx, window_idx, piece_ids.shape[0]] = self.config.writer_stop_token_id

        return {
            "input_ids": torch.from_numpy(input_ids),
            "word_masks": torch.from_numpy(word_masks),
            "labels": torch.from_numpy(labels),
            "zone_ids": self.zone_template.unsqueeze(0).expand(batch_size, -1).clone(),
            "window_mask": torch.from_numpy(window_mask),
            "source_line_ids": torch.from_numpy(source_line_ids),
        }

    def carry_batch(self):
        if not self._carry_refs:
            return None
        batch = self.make_batch(self._carry_texts, self._carry_line_ids, self._carry_segments, self._carry_refs)
        self._carry_texts = []
        self._carry_line_ids = []
        self._carry_segments = []
        self._carry_refs = []
        return batch

    def iter_once(self, worker_id: int, worker_count: int):
        texts = self._carry_texts
        line_ids = self._carry_line_ids
        segments_by_text = self._carry_segments
        refs = self._carry_refs
        self._carry_texts = []
        self._carry_line_ids = []
        self._carry_segments = []
        self._carry_refs = []

        for text_idx, (source_line_id, text) in enumerate(stream_teacher_text_items(self.train_file, self.read_chars)):
            if text_idx % worker_count != worker_id:
                continue
            segments = trainable_segments(self.tokenizer, text, self.config.max_word_bytes)
            if not segments:
                continue
            local_text_idx = len(texts)
            texts.append(text)
            line_ids.append(source_line_id)
            segments_by_text.append(segments)
            for active_start in range(0, len(segments), self.stride):
                refs.append((local_text_idx, active_start))
                self._produced += 1
                if len(refs) == self.batch_size:
                    yield self.make_batch(texts, line_ids, segments_by_text, refs)
                    texts, line_ids, segments_by_text, refs = [], [], [], []
                    if active_start + self.stride < len(segments):
                        local_text_idx = 0
                        texts.append(text)
                        line_ids.append(source_line_id)
                        segments_by_text.append(segments)
                if self.max_samples > 0 and self._produced >= self.max_samples:
                    if refs:
                        yield self.make_batch(texts, line_ids, segments_by_text, refs)
                    return

        if refs and not self.repeat:
            yield self.make_batch(texts, line_ids, segments_by_text, refs)
        elif refs:
            self._carry_texts = texts
            self._carry_line_ids = line_ids
            self._carry_segments = segments_by_text
            self._carry_refs = refs

    def __iter__(self):
        from torch.utils.data import get_worker_info

        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        while True:
            yielded = False
            for batch in self.iter_once(worker_id, worker_count):
                yielded = True
                yield batch
            if not yielded and not self._carry_refs:
                raise ValueError(f"{self.train_file} produced no sliding writer windows")
            if not self.repeat:
                return


def freeze_for_writer_only(model: Dil):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.writer.parameters():
        param.requires_grad = True
    model.encoder.eval()
    model.writer.train()


def self_conditioning_probability(config: DilConfig, training_step: int | None) -> float:
    if training_step is None:
        return 0.0
    start = float(config.writer_self_conditioning_start)
    final = float(config.writer_self_conditioning_final)
    if training_step <= 1000:
        return start
    if training_step >= 10000:
        return final
    ratio = (training_step - 1000) / 9000.0
    return start + ratio * (final - start)


def synthetic_surface_state(
    config: DilConfig,
    labels: torch.Tensor,
    zone_ids: torch.Tensor,
    window_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = labels.device
    valid = labels.ne(-100) & window_mask.unsqueeze(-1)
    left = zone_ids.eq(0).unsqueeze(-1) & valid
    surface_state = torch.full_like(labels, -100)
    surface_state_mask = torch.zeros_like(labels)
    frozen_mask = torch.zeros_like(labels, dtype=torch.bool)

    draw = torch.rand(labels.shape, device=device)
    draft = valid & draw.lt(0.55)
    known = valid & draw.ge(0.55) & draw.lt(0.75)
    corrupt = draft & torch.rand(labels.shape, device=device).lt(0.30)
    random_tokens = torch.randint(config.writer_vocab_size, labels.shape, device=device, dtype=torch.long)
    surface_state[draft | known] = labels[draft | known]
    surface_state[corrupt] = random_tokens[corrupt]
    surface_state[left] = labels[left]
    surface_state_mask[draft] = 1
    surface_state_mask[known | left] = 2
    frozen_mask[known | left] = True
    return surface_state, surface_state_mask, frozen_mask


def synthetic_position_age(config: DilConfig, labels: torch.Tensor, zone_ids: torch.Tensor, window_mask: torch.Tensor) -> torch.Tensor:
    valid_words = labels.ne(-100).any(dim=-1) & window_mask
    active = zone_ids.eq(1) & valid_words
    left = zone_ids.eq(0) & valid_words
    age = torch.zeros(zone_ids.shape, device=labels.device, dtype=torch.long)
    random_age = torch.randint(config.writer_max_position_age + 1, zone_ids.shape, device=labels.device, dtype=torch.long)
    age = torch.where(active, random_age, age)
    age = torch.where(left, torch.full_like(age, config.writer_max_position_age), age)
    return age


@torch.no_grad()
def self_conditioned_surface_state(
    model: Dil,
    semantic: torch.Tensor,
    labels: torch.Tensor,
    zone_ids: torch.Tensor,
    window_mask: torch.Tensor,
    future_latents: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output = model.writer_transition_outputs(
        semantic,
        zone_ids=zone_ids,
        window_mask=window_mask,
        future_latents=future_latents,
    )
    valid = labels.ne(-100) & window_mask.unsqueeze(-1)
    left = zone_ids.eq(0).unsqueeze(-1) & valid
    surface_state = output.token_logits.argmax(dim=-1).masked_fill(~valid, -100)
    surface_state[left] = labels[left]
    surface_state_mask = torch.where(valid, torch.ones_like(labels), torch.zeros_like(labels))
    surface_state_mask[left] = 2
    frozen_mask = torch.zeros_like(labels, dtype=torch.bool)
    frozen_mask[left] = True
    return surface_state, surface_state_mask, frozen_mask


def sliding_future_latents(semantic: torch.Tensor, window_mask: torch.Tensor, horizons: int) -> torch.Tensor | None:
    if horizons <= 0:
        return None
    batch_size, window_size, latent_size = semantic.shape
    future = semantic.new_zeros((batch_size, window_size, horizons, latent_size))
    for horizon_idx in range(horizons):
        offset = horizon_idx + 1
        if offset >= window_size:
            break
        future[:, :-offset, horizon_idx] = (
            semantic[:, offset:] * window_mask[:, offset:].unsqueeze(-1).to(semantic.dtype)
        )
    return future


def sliding_writer_metrics(
    model: Dil,
    batch: dict,
    training_step: int | None = None,
    use_future_latents: bool = True,
    use_persistent_state: bool = True,
) -> dict[str, torch.Tensor]:
    input_ids = batch["input_ids"]
    labels = batch["labels"].to(input_ids.device)
    word_masks = batch["word_masks"]
    zone_ids = batch["zone_ids"].to(input_ids.device)
    window_mask = batch["window_mask"].to(input_ids.device, dtype=torch.bool)
    batch_size, window_size, context_size, byte_width = input_ids.shape
    with torch.no_grad():
        flat_semantic = model.encode(
            input_ids.reshape(batch_size * window_size, context_size, byte_width),
            word_masks.reshape(batch_size * window_size, context_size, byte_width),
        ).float()
        semantic = flat_semantic.reshape(batch_size, window_size, -1)

    future_latents = None
    if use_future_latents:
        future_latents = sliding_future_latents(
            semantic,
            window_mask,
            min(model.config.writer_right_guard, max(window_size - 1, 0)),
        )
    probability = self_conditioning_probability(model.config, training_step)
    self_conditioned = (
        use_persistent_state
        and model.training
        and torch.rand((), device=input_ids.device).item() < probability
    )
    if self_conditioned:
        surface_state, surface_state_mask, frozen_mask = self_conditioned_surface_state(
            model,
            semantic,
            labels,
            zone_ids,
            window_mask,
            future_latents=future_latents,
        )
    else:
        surface_state, surface_state_mask, frozen_mask = synthetic_surface_state(
            model.config,
            labels,
            zone_ids,
            window_mask,
        )
        if not use_persistent_state:
            surface_state = torch.full_like(labels, -100)
            surface_state_mask = torch.zeros_like(labels)
            frozen_mask = torch.zeros_like(labels, dtype=torch.bool)
    position_age = synthetic_position_age(model.config, labels, zone_ids, window_mask)
    metrics = model.writer_transition_loss_and_metrics(
        semantic.detach(),
        labels,
        surface_state,
        surface_state_mask,
        frozen_mask,
        zone_ids,
        window_mask,
        future_latents=future_latents,
        position_age=position_age,
        training_step=training_step,
        return_metrics=True,
    )
    valid = labels.ne(-100) & window_mask.unsqueeze(-1)
    filled = surface_state_mask.gt(0) & valid
    metrics["self_conditioning_ratio"] = metrics["loss"].new_tensor(float(self_conditioned))
    metrics["mean_mask_ratio"] = 1.0 - filled.sum().to(metrics["loss"].dtype) / valid.sum().clamp_min(1).to(metrics["loss"].dtype)
    metrics["future_horizons"] = metrics["loss"].new_tensor(0.0 if future_latents is None else float(future_latents.shape[2]))
    return metrics


def writer_only_metrics(
    model: Dil,
    batch: dict,
    training_step: int | None = None,
    use_future_latents: bool = True,
    use_persistent_state: bool = True,
) -> dict[str, torch.Tensor]:
    if batch["input_ids"].dim() == 4:
        return sliding_writer_metrics(
            model,
            batch,
            training_step,
            use_future_latents=use_future_latents,
            use_persistent_state=use_persistent_state,
        )
    labels = batch["labels"].to(batch["input_ids"].device)
    loss, token_loss, byte_acc, token_exact, stop_acc = model.writer_loss_and_metrics(
        model.encode(batch["input_ids"], batch["word_masks"]).detach(),
        labels,
        training_step=training_step,
    )
    zero = loss.new_zeros(())
    return {
        "loss": loss,
        "token_loss": token_loss,
        "active_token_loss": token_loss,
        "right_guard_token_loss": zero,
        "left_consistency_loss": zero,
        "commit_loss": zero,
        "byte_acc": byte_acc,
        "token_exact": token_exact,
        "stop_acc": stop_acc,
        "right_guard_byte_acc": zero,
        "right_guard_token_exact": zero,
        "right_guard_stop_acc": zero,
        "commit_precision": zero,
        "commit_recall": zero,
        "commit_f1": zero,
        "false_commit_rate": zero,
        "mean_commit_score": zero,
        "step0_byte_acc": byte_acc,
        "step0_token_exact": token_exact,
        "step0_stop_acc": stop_acc,
        "stepT_byte_acc": byte_acc,
        "stepT_token_exact": token_exact,
        "stepT_stop_acc": stop_acc,
        "self_conditioning_ratio": zero,
        "mean_mask_ratio": zero,
        "future_horizons": zero,
    }


def writer_only_forward(model: Dil, batch: dict, training_step: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    metrics = writer_only_metrics(model, batch, training_step)
    return metrics["loss"], metrics["byte_acc"], metrics["token_exact"], metrics["stop_acc"]


def materialize_writer_batches(dataset, device: torch.device, batch_size: int, seed: int):
    batches = [
        {
            key: value.detach().cpu()
            for key, value in batch.items()
            if key in ("input_ids", "word_masks", "labels", "source_line_ids", "zone_ids", "window_mask")
        }
        for batch in dataset.iter_once(worker_id=0, worker_count=1)
    ]
    carry_batch = dataset.carry_batch() if hasattr(dataset, "carry_batch") else None
    if carry_batch is not None:
        batches.append(
            {
                key: value.detach().cpu()
                for key, value in carry_batch.items()
                if key in ("input_ids", "word_masks", "labels", "source_line_ids", "zone_ids", "window_mask")
            }
        )
    return ResidentDilBatcher(batches, batch_size=batch_size, device=device, seed=seed)


def save_checkpoint(
    output_dir: Path,
    model: Dil,
    optimizer,
    scheduler,
    config: DilConfig,
    tokenizer_vocab_path: Path,
    step: int,
    metrics: dict,
    compile_mode: str,
    checkpoint_name: str = "",
):
    checkpoint_dir = output_dir / checkpoint_name if checkpoint_name else output_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(checkpoint_dir)
    dst_vocab = checkpoint_dir / config.tokenizer_vocab_file
    if tokenizer_vocab_path.resolve() != dst_vocab.resolve():
        shutil.copyfile(tokenizer_vocab_path, dst_vocab)
    training_state = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "objective": WRITER_OBJECTIVE,
        "step": step,
        "metrics": metrics,
        "compile_mode": compile_mode,
        "vocab_size": config.vocab_size,
        "pad_token_id": config.pad_token_id,
        "eos_token_id": config.eos_token_id,
        "max_word_bytes": config.max_word_bytes,
        "context_radius": config.context_radius,
        "target_index": config.target_index,
        "latent_size": config.latent_size,
        "writer_window_size": config.writer_sliding_window_size,
        "writer_left_frozen": config.writer_left_frozen,
        "writer_active_size": config.writer_active_size,
        "writer_right_guard": config.writer_right_guard,
        "writer_stride": config.writer_stride,
        "writer_refinement_steps": config.writer_refinement_steps,
        "writer_use_step_embedding": config.writer_use_step_embedding,
        "writer_use_zone_noise": config.writer_use_zone_noise,
        "writer_gradient_checkpointing": config.writer_gradient_checkpointing,
        "writer_commit_temperature": config.writer_commit_temperature,
        "writer_commit_threshold": config.writer_commit_threshold,
    }
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": model.state_dict(),
            "writer_optimizer_state_dict": optimizer.state_dict(),
            "writer_scheduler_state_dict": scheduler.state_dict(),
            "training_state": training_state,
            "rng_state": rng_state(),
        },
        checkpoint_dir / "checkpoint.pt",
    )
    with (checkpoint_dir / "training_state.json").open("w", encoding="utf-8") as handle:
        json.dump(training_state, handle, indent=2)
    return checkpoint_dir


def load_model_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[Dil, DilConfig, dict]:
    config = DilConfig.from_pretrained(checkpoint_path.parent)
    model = Dil(config).to(device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"unsupported Dil checkpoint format_version={checkpoint.get('format_version')}")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config, checkpoint


@torch.no_grad()
def evaluate(
    model,
    eval_loader,
    device,
    compile_mode: str,
    autocast_enabled: bool,
    cuda_prefetch: bool,
    max_batches: int,
    use_future_latents: bool,
    use_persistent_state: bool,
):
    model.eval()
    model.encoder.eval()
    total = {key: 0.0 for key in WRITER_METRIC_KEYS}
    total["batches"] = 0
    for batch_idx, batch in enumerate(DeviceBatchPrefetcher(eval_loader, device, cuda_prefetch), start=1):
        cudagraph_step_begin(device, compile_mode)
        with autocast_context(autocast_enabled):
            metrics = writer_only_metrics(
                model,
                batch,
                use_future_latents=use_future_latents,
                use_persistent_state=use_persistent_state,
            )
        for key in WRITER_METRIC_KEYS:
            total[key] += float(metrics[key].detach().cpu())
        total["batches"] += 1
        if batch_idx >= max_batches:
            break
    model.train()
    model.encoder.eval()
    batches = max(total.pop("batches"), 1)
    return {f"eval_{key}": value / batches for key, value in total.items()}


def format_log(step: int, metrics: dict) -> str:
    fields = [
        f"step={step}",
        f"loss={metrics['loss']:.4f}",
        f"tok={metrics['token_loss']:.4f}",
        f"act={metrics['active_token_loss']:.4f}",
        f"guard={metrics['right_guard_token_loss']:.4f}",
        f"left={metrics['left_consistency_loss']:.4f}",
        f"commit={metrics['commit_loss']:.4f}",
        f"byte_acc={metrics['byte_acc']:.4f}",
        f"token_exact={metrics['token_exact']:.4f}",
        f"stop_acc={metrics['stop_acc']:.4f}",
        f"guard_acc={metrics['right_guard_byte_acc']:.4f}",
        f"commit_p={metrics['commit_precision']:.4f}",
        f"commit_r={metrics['commit_recall']:.4f}",
        f"commit_f1={metrics['commit_f1']:.4f}",
        f"false_commit={metrics['false_commit_rate']:.4f}",
        f"commit_score={metrics['mean_commit_score']:.4f}",
        f"step0_acc={metrics['step0_byte_acc']:.4f}",
        f"stepT_acc={metrics['stepT_byte_acc']:.4f}",
        f"step0_exact={metrics['step0_token_exact']:.4f}",
        f"stepT_exact={metrics['stepT_token_exact']:.4f}",
        f"self_cond={metrics['self_conditioning_ratio']:.4f}",
        f"mask={metrics['mean_mask_ratio']:.4f}",
        f"future_h={metrics['future_horizons']:.1f}",
        f"lr={metrics['lr']:.2e}",
        f"data_s={metrics['data_seconds']:.4f}",
        f"compute_s={metrics['compute_seconds']:.4f}",
        f"t/s={metrics['tokens_per_second']:.1f}",
        f"w/s={metrics['windows_per_second']:.1f}",
        f"step/s={metrics['steps_per_second']:.2f}",
    ]
    if "source_lines_seen" in metrics:
        fields.append(f"total/row={int(metrics['source_lines_seen'])}")
    for key in sorted(k for k in metrics if k.startswith("eval_")):
        fields.append(f"{key}={metrics[key]:.4f}")
    return " ".join(fields)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compile-mode", choices=COMPILE_MODE_CHOICES, default=None)
    parser.add_argument("--data-mode", choices=("streaming", "resident"), default=DIL_TRAIN_DEFAULTS["data_mode"])
    parser.add_argument("--max-steps", type=int, default=DIL_TRAIN_DEFAULTS["max_steps"])
    parser.add_argument("--batch-size", type=int, default=DIL_TRAIN_DEFAULTS["batch_size"])
    parser.add_argument("--eval-batch-size", type=int, default=DIL_TRAIN_DEFAULTS["eval_batch_size"])
    parser.add_argument("--text-read-chars", type=int, default=DIL_TRAIN_DEFAULTS["text_read_chars"])
    parser.add_argument("--prefetch-factor", type=int, default=DIL_TRAIN_DEFAULTS["prefetch_factor"])
    parser.add_argument("--no-cuda-prefetch", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=DIL_TRAIN_DEFAULTS["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=DIL_TRAIN_DEFAULTS["weight_decay"])
    parser.add_argument("--adam-beta1", type=float, default=DIL_TRAIN_DEFAULTS["adam_beta1"])
    parser.add_argument("--adam-beta2", type=float, default=DIL_TRAIN_DEFAULTS["adam_beta2"])
    parser.add_argument("--warmup-steps", type=int, default=DIL_TRAIN_DEFAULTS["warmup_steps"])
    parser.add_argument("--max-grad-norm", type=float, default=DIL_TRAIN_DEFAULTS["max_grad_norm"])
    parser.add_argument("--log-every", type=int, default=DIL_TRAIN_DEFAULTS["log_every"])
    parser.add_argument("--checkpoint-every", type=int, default=DIL_TRAIN_DEFAULTS["checkpoint_every"])
    parser.add_argument("--eval-every", type=int, default=DIL_TRAIN_DEFAULTS["eval_every"])
    parser.add_argument("--max-eval-batches", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=DIL_TRAIN_DEFAULTS["num_workers"])
    parser.add_argument("--seed", type=int, default=DIL_TRAIN_DEFAULTS["seed"])
    parser.add_argument("--max-samples", type=int, default=DIL_TRAIN_DEFAULTS["max_samples"])
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--left-frozen", type=int, default=8)
    parser.add_argument("--active-size", type=int, default=20)
    parser.add_argument("--right-guard", type=int, default=4)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--right-guard-loss-weight", type=float, default=0.2)
    parser.add_argument("--left-consistency-weight", type=float, default=0.5)
    parser.add_argument("--commit-loss-weight", type=float, default=0.25)
    parser.add_argument("--self-conditioning-start", type=float, default=0.2)
    parser.add_argument("--self-conditioning-final", type=float, default=0.6)
    parser.add_argument("--writer-refinement-steps", type=int, default=None)
    parser.add_argument("--writer-commit-temperature", type=float, default=1.0)
    parser.add_argument("--writer-commit-threshold", type=float, default=0.5)
    parser.add_argument("--writer-gradient-checkpointing", action="store_true")
    parser.add_argument("--disable-refinement", action="store_true")
    parser.add_argument("--disable-step-embedding", action="store_true")
    parser.add_argument("--disable-future-latents", action="store_true")
    parser.add_argument("--disable-zone-noise", action="store_true")
    parser.add_argument("--disable-persistent-state", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("--batch-size and --eval-batch-size must be > 0")
    if args.text_read_chars <= 0:
        raise ValueError("--text-read-chars must be > 0")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be > 0")
    if args.eval_every < 0 or args.checkpoint_every < 0:
        raise ValueError("--eval-every and --checkpoint-every must be >= 0")
    if args.eval_every > 0 and args.eval_file is None:
        raise ValueError("--eval-file is required when --eval-every > 0")
    if args.max_eval_batches <= 0:
        raise ValueError("--max-eval-batches must be > 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.data_mode == "resident" and args.max_samples > 0:
        raise ValueError("--max-samples is not supported with --data-mode resident")
    if args.left_frozen + args.active_size + args.right_guard != args.window_size:
        raise ValueError("--left-frozen + --active-size + --right-guard must equal --window-size")
    if args.stride <= 0 or args.stride > args.active_size:
        raise ValueError("--stride must be in 1..--active-size")
    if min(args.right_guard_loss_weight, args.left_consistency_weight, args.commit_loss_weight) < 0.0:
        raise ValueError("writer loss weights must be >= 0")
    if not (0.0 <= args.self_conditioning_start <= 1.0 and 0.0 <= args.self_conditioning_final <= 1.0):
        raise ValueError("self-conditioning rates must be in [0, 1]")
    if args.writer_refinement_steps is not None and args.writer_refinement_steps <= 0:
        raise ValueError("--writer-refinement-steps must be > 0")
    if args.writer_commit_temperature <= 0.0:
        raise ValueError("--writer-commit-temperature must be > 0")
    if not (0.0 <= args.writer_commit_threshold <= 1.0):
        raise ValueError("--writer-commit-threshold must be in [0, 1]")


def sync_writer_runtime_config(model: Dil, config: DilConfig) -> None:
    model.writer.writer_refinement_steps = config.writer_refinement_steps
    model.writer.use_step_embedding = config.writer_use_step_embedding
    model.writer.gradient_checkpointing = config.writer_gradient_checkpointing
    model.writer.commit_temperature = config.writer_commit_temperature


def main():
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    compile_mode = effective_compile_mode(args.compile_mode, device)
    validate_compile_environment(compile_mode)
    autocast_enabled = bool(args.bf16 and device.type == "cuda")
    cuda_prefetch = bool(device.type == "cuda" and not args.no_cuda_prefetch)

    model, config, checkpoint = load_model_checkpoint(args.checkpoint, device)
    if args.window_size > config.writer_max_window_size:
        raise ValueError("--window-size must be <= config.writer_max_window_size")
    config.writer_sliding_window_size = args.window_size
    config.writer_left_frozen = args.left_frozen
    config.writer_active_size = args.active_size
    config.writer_right_guard = args.right_guard
    config.writer_stride = args.stride
    config.writer_right_guard_loss_weight = args.right_guard_loss_weight
    config.writer_left_consistency_weight = args.left_consistency_weight
    config.writer_commit_loss_weight = args.commit_loss_weight
    config.writer_self_conditioning_start = args.self_conditioning_start
    config.writer_self_conditioning_final = args.self_conditioning_final
    config.writer_refinement_steps = 1 if args.disable_refinement else (
        config.writer_refinement_steps if args.writer_refinement_steps is None else args.writer_refinement_steps
    )
    config.writer_use_step_embedding = not args.disable_step_embedding
    config.writer_use_zone_noise = not args.disable_zone_noise
    config.writer_gradient_checkpointing = bool(args.writer_gradient_checkpointing)
    config.writer_commit_temperature = args.writer_commit_temperature
    config.writer_commit_threshold = args.writer_commit_threshold
    sync_writer_runtime_config(model, config)
    tokenizer_vocab_path = args.checkpoint.parent / config.tokenizer_vocab_file
    tokenizer = load_hybrid_tokenizer(tokenizer_vocab_path)
    freeze_for_writer_only(model)
    model.set_compiled_forwards(
        encoder_forward=compile_forward(model.encoder.forward, compile_mode, "DilEncoderCore"),
        writer_forward=compile_forward(model.writer.forward, compile_mode, "DilConditionalWriter"),
        transition_forward=compile_forward(model.writer.transition, compile_mode, "DilConditionalWriterTransition"),
    )
    writer_named_parameters = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad and name.startswith("writer.")
    ]
    optimizer = AdamW(
        make_adamw_param_groups(writer_named_parameters, args.weight_decay),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
    )
    scheduler = make_scheduler(optimizer, args.learning_rate, args.warmup_steps, args.max_steps)
    if checkpoint.get("training_state", {}).get("objective") == WRITER_OBJECTIVE:
        optimizer.load_state_dict(checkpoint["writer_optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["writer_scheduler_state_dict"])
        restore_rng_state(checkpoint["rng_state"])

    train_dataset = HybridDilSlidingWindowDataset(
        args.train_file,
        config,
        tokenizer,
        batch_size=args.batch_size,
        read_chars=args.text_read_chars,
        repeat=True,
        max_samples=args.max_samples,
        window_size=args.window_size,
        left_frozen=args.left_frozen,
        active_size=args.active_size,
        right_guard=args.right_guard,
        stride=args.stride,
    )
    eval_dataset = None
    if args.eval_every > 0:
        eval_dataset = HybridDilSlidingWindowDataset(
            args.eval_file,
            config,
            tokenizer,
            batch_size=args.eval_batch_size,
            read_chars=args.text_read_chars,
            repeat=False,
            window_size=args.window_size,
            left_frozen=args.left_frozen,
            active_size=args.active_size,
            right_guard=args.right_guard,
            stride=args.stride,
        )

    if args.data_mode == "resident":
        print("resident_writer_data_prepare_start=1", flush=True)
        train_iter = materialize_writer_batches(train_dataset, device, args.batch_size, args.seed)
        print(f"resident_writer_data_prepare_done=1 batches={len(train_iter.batches)}", flush=True)
        eval_loader = None
        if eval_dataset is not None:
            print("resident_writer_eval_prepare_start=1", flush=True)
            eval_loader = ResidentDilEvalLoader(
                materialize_writer_batches(eval_dataset, device, args.eval_batch_size, args.seed + 1)
            )
            print(f"resident_writer_eval_prepare_done=1 batches={len(eval_loader.batches)}", flush=True)
    else:
        train_loader = make_dil_batch_loader(
            train_dataset,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            prefetch_factor=args.prefetch_factor,
        )
        train_iter = DeviceBatchPrefetcher(train_loader, device, cuda_prefetch)
        eval_loader = None
        if eval_dataset is not None:
            eval_loader = make_dil_batch_loader(
                eval_dataset,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                prefetch_factor=args.prefetch_factor,
            )

    print(
        f"device={device.type} bf16={int(autocast_enabled)} compile_mode={compile_mode} "
        f"data_mode={args.data_mode} objective={WRITER_OBJECTIVE} "
        f"vocab_size={config.vocab_size} latent_size={config.latent_size} hidden_size={config.hidden_size} "
        f"window={args.window_size} zones={args.left_frozen}|{args.active_size}|{args.right_guard} stride={args.stride} "
        f"refine_steps={config.writer_refinement_steps} step_embed={int(config.writer_use_step_embedding)} "
        f"future_latents={int(not args.disable_future_latents)} zone_noise={int(config.writer_use_zone_noise)} "
        f"persistent_state={int(not args.disable_persistent_state)} commit_temp={config.writer_commit_temperature:.3f} "
        f"commit_threshold={config.writer_commit_threshold:.3f}",
        flush=True,
    )

    log_start = time.perf_counter()
    log_tokens = 0
    log_windows = 0
    log_steps = 0
    data_seconds = 0.0
    compute_seconds = 0.0
    source_lines_seen: set[int] = set()
    metric_sums = {key: 0.0 for key in WRITER_METRIC_KEYS}
    last_metrics = {}
    completed_step = 0

    def save_current(checkpoint_name: str = ""):
        return save_checkpoint(
            args.output_dir,
            model,
            optimizer,
            scheduler,
            config,
            tokenizer_vocab_path,
            completed_step,
            last_metrics,
            compile_mode,
            checkpoint_name=checkpoint_name,
        )

    try:
        for step in range(1, args.max_steps + 1):
            data_start = time.perf_counter()
            batch = next(train_iter)
            data_seconds += time.perf_counter() - data_start

            compute_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            cudagraph_step_begin(device, compile_mode)
            with autocast_context(autocast_enabled):
                metrics = writer_only_metrics(
                    model,
                    batch,
                    step,
                    use_future_latents=not args.disable_future_latents,
                    use_persistent_state=not args.disable_persistent_state,
                )
                loss = metrics["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.writer.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            cuda_sync(device)
            compute_seconds += time.perf_counter() - compute_start
            completed_step = step

            real_tokens = int(batch["labels"].ne(-100).sum().detach().cpu())
            log_tokens += real_tokens
            log_windows += int(batch["labels"].shape[0])
            log_steps += 1
            if "source_line_ids" in batch:
                source_lines_seen.update(int(line_id) for line_id in batch["source_line_ids"].detach().cpu().tolist())
            for key in WRITER_METRIC_KEYS:
                metric_sums[key] += float(metrics[key].detach().cpu())

            should_log = step % args.log_every == 0 or step == 1 or step == args.max_steps
            should_eval = eval_loader is not None and args.eval_every > 0 and step % args.eval_every == 0
            if should_log or should_eval:
                elapsed = max(time.perf_counter() - log_start, 1e-9)
                averaged = {key: value / max(log_steps, 1) for key, value in metric_sums.items()}
                averaged["lr"] = scheduler.get_last_lr()[0]
                averaged["data_seconds"] = data_seconds / max(log_steps, 1)
                averaged["compute_seconds"] = compute_seconds / max(log_steps, 1)
                averaged["tokens_per_second"] = log_tokens / elapsed
                averaged["windows_per_second"] = log_windows / elapsed
                averaged["steps_per_second"] = log_steps / elapsed
                if source_lines_seen:
                    averaged["source_lines_seen"] = len(source_lines_seen)
                if should_eval:
                    averaged.update(
                        evaluate(
                            model,
                            eval_loader,
                            device,
                            compile_mode,
                            autocast_enabled,
                            cuda_prefetch,
                            args.max_eval_batches,
                            not args.disable_future_latents,
                            not args.disable_persistent_state,
                        )
                    )
                print(format_log(step, averaged), flush=True)
                last_metrics = averaged
                log_start = time.perf_counter()
                log_tokens = 0
                log_windows = 0
                log_steps = 0
                data_seconds = 0.0
                compute_seconds = 0.0
                for key in metric_sums:
                    metric_sums[key] = 0.0

            if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                save_current(checkpoint_name=f"checkpoint-{step}")
    except KeyboardInterrupt:
        interrupted_dir = save_current()
        print(f"interrupted_saved={interrupted_dir}", flush=True)
        return

    final_dir = save_current()
    print(f"saved={final_dir}", flush=True)


if __name__ == "__main__":
    main()
