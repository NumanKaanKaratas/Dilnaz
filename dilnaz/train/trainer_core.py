from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch.optim.lr_scheduler import LambdaLR

from byte_trainer_utils import autocast_context, cuda_sync, cudagraph_step_begin


@dataclass
class StepResult:
    loss: torch.Tensor
    outputs: Any
    token_count: int
    window_count: int = 0
    batch: Any = None


class BatchTimingSource:
    def __init__(self, iterator):
        self.iterator = iterator
        self.last_data_seconds = 0.0
        self.last_transfer_seconds = 0.0

    def __iter__(self):
        return self

    def __next__(self):
        start = time.perf_counter()
        batch = next(self.iterator)
        self.last_data_seconds = getattr(self.iterator, "last_data_seconds", time.perf_counter() - start)
        self.last_transfer_seconds = getattr(self.iterator, "last_transfer_seconds", 0.0)
        return batch


def make_scheduler(optimizer, learning_rate: float, warmup_steps: int, max_steps: int | None = None):
    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return min(1.0, float(step + 1) / float(warmup_steps))
        if max_steps is None or max_steps <= warmup_steps:
            return 1.0
        progress = min(1.0, float(step - warmup_steps) / float(max_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def make_adamw_param_groups(named_parameters, weight_decay: float):
    decay_params = []
    no_decay_params = []
    for _, param in named_parameters:
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    groups = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return groups


class BaseTrainer:
    def __init__(self, args):
        self.args = args
        self.start_step = 0
        self.completed_step = 0
        self.last_metrics: dict[str, float] = {}
        self._trainable_param_list = None

    def trainable_parameters(self):
        if self._trainable_param_list is None:
            self._trainable_param_list = [param for param in self.model.parameters() if param.requires_grad]
        return self._trainable_param_list

    def optimizer_param_groups(self, weight_decay: float):
        named_parameters = [
            (name, param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        ]
        self._trainable_param_list = [param for _, param in named_parameters]
        return make_adamw_param_groups(named_parameters, weight_decay)

    def build_train_iterator(self):
        raise NotImplementedError

    def build_eval_iterator(self):
        return None

    def has_eval(self) -> bool:
        return False

    def empty_metric_sums(self) -> dict[str, float]:
        raise NotImplementedError

    def accumulate_metrics(self, total: dict[str, float], result: StepResult) -> None:
        raise NotImplementedError

    def reduce_metrics(self, total: dict[str, float]) -> dict[str, float]:
        raise NotImplementedError

    def train_step(self, batch: dict, step: int) -> StepResult:
        raise NotImplementedError

    def eval_step(self, batch: dict) -> StepResult:
        return self.train_step(batch, self.completed_step)

    def save_checkpoint(self, checkpoint_name: str, step: int, metrics: dict[str, float]):
        raise NotImplementedError

    def format_log(self, step: int, metrics: dict[str, float]) -> str:
        raise NotImplementedError

    def assert_checkpoint_integrity(self) -> None:
        return

    def is_recoverable_runtime_error(self, error: RuntimeError) -> bool:
        del error
        return False

    def close(self) -> None:
        return

    @torch.no_grad()
    def evaluate(self, max_batches: int) -> dict[str, float]:
        eval_iterator = self.build_eval_iterator()
        if eval_iterator is None:
            return {}
        self.model.eval()
        total = self.empty_metric_sums()
        for batch_idx, batch in enumerate(BatchTimingSource(eval_iterator), start=1):
            cudagraph_step_begin(self.device, self.compile_mode)
            with autocast_context(self.autocast_enabled):
                result = self.eval_step(batch)
            self.accumulate_metrics(total, result)
            if batch_idx >= max_batches:
                break
        self.model.train()
        return {f"eval_{key}": value for key, value in self.reduce_metrics(total).items()}

    def save_interrupted(self) -> None:
        self.assert_checkpoint_integrity()
        checkpoint_dir = self.save_checkpoint("", self.completed_step, self.last_metrics)
        print(f"interrupted_saved={checkpoint_dir}", flush=True)

    def run(self) -> None:
        train_iterator = BatchTimingSource(self.build_train_iterator())
        metric_sums = self.empty_metric_sums()
        log_start = time.perf_counter()
        log_tokens = 0
        log_windows = 0
        log_steps = 0
        data_seconds = 0.0
        transfer_seconds = 0.0
        compute_seconds = 0.0

        try:
            for step in range(self.start_step + 1, self.args.max_steps + 1):
                batch = next(train_iterator)
                data_seconds += train_iterator.last_data_seconds
                transfer_seconds += train_iterator.last_transfer_seconds

                sync_timing = bool(getattr(self.args, "sync_timing", False))
                if sync_timing:
                    cuda_sync(self.device)
                compute_start = time.perf_counter()
                self.optimizer.zero_grad(set_to_none=True)
                cudagraph_step_begin(self.device, self.compile_mode)
                with autocast_context(self.autocast_enabled):
                    result = self.train_step(batch, step)

                result.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.trainable_parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                if sync_timing:
                    cuda_sync(self.device)
                compute_seconds += time.perf_counter() - compute_start
                self.completed_step = step

                log_tokens += result.token_count
                log_windows += result.window_count
                log_steps += 1
                self.accumulate_metrics(metric_sums, result)

                should_log = step % self.args.log_every == 0 or step == self.start_step + 1 or step == self.args.max_steps
                should_eval = self.has_eval() and step % self.args.eval_every == 0
                if should_log or should_eval:
                    elapsed = max(time.perf_counter() - log_start, 1e-9)
                    averaged = self.reduce_metrics(metric_sums)
                    averaged["lr"] = self.scheduler.get_last_lr()[0]
                    averaged["data_seconds"] = data_seconds / max(log_steps, 1)
                    averaged["transfer_seconds"] = transfer_seconds / max(log_steps, 1)
                    averaged["compute_seconds"] = compute_seconds / max(log_steps, 1)
                    averaged["tokens_per_second"] = log_tokens / elapsed
                    averaged["windows_per_second"] = log_windows / elapsed
                    averaged["steps_per_second"] = log_steps / elapsed
                    if should_eval:
                        averaged.update(self.evaluate(self.args.max_eval_batches))
                    print(self.format_log(step, averaged), flush=True)
                    self.last_metrics = averaged
                    log_start = time.perf_counter()
                    log_tokens = 0
                    log_windows = 0
                    log_steps = 0
                    data_seconds = 0.0
                    transfer_seconds = 0.0
                    compute_seconds = 0.0
                    metric_sums = self.empty_metric_sums()

                if self.args.checkpoint_every > 0 and step % self.args.checkpoint_every == 0:
                    self.assert_checkpoint_integrity()
                    self.save_checkpoint(f"checkpoint-{step}", step, self.last_metrics)
        except KeyboardInterrupt:
            self.save_interrupted()
            return
        except RuntimeError as error:
            if not self.is_recoverable_runtime_error(error):
                raise
            self.save_interrupted()
            return
        finally:
            self.close()

        self.assert_checkpoint_integrity()
        final_dir = self.save_checkpoint("", self.args.max_steps, self.last_metrics)
        print(f"saved={final_dir}", flush=True)
