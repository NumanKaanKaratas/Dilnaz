import random
import os
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from dataclasses import is_dataclass
from typing import Any

import numpy as np
import torch


COMPILE_MODE_CHOICES = ("off", "default", "reduce-overhead", "max-autotune")


def effective_compile_mode(requested: str | None, device: torch.device) -> str:
    if requested is not None:
        return requested
    return "reduce-overhead" if device.type == "cuda" else "off"


def compile_forward(forward, compile_mode: str, name: str):
    if compile_mode == "off":
        return None
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build")
    print(f"compiled={name} mode={compile_mode}", flush=True)
    if compile_mode == "default":
        return torch.compile(forward, backend="inductor")
    return torch.compile(forward, backend="inductor", mode=compile_mode)


def validate_compile_environment(compile_mode: str):
    if compile_mode == "off":
        return
    if os.environ.get("CC"):
        return
    if any(shutil.which(name) for name in ("cl", "clang", "gcc", "cc")):
        return
    raise RuntimeError(
        "torch.compile with Inductor/Triton requires a C compiler. "
        "Install Visual Studio Build Tools or set the CC environment variable, "
        "or run with --compile-mode off."
    )


def autocast_context(enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return torch.cuda.amp.autocast(dtype=torch.bfloat16)


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if hasattr(value, "to") and is_dataclass(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def record_stream(value, stream):
    if isinstance(value, torch.Tensor):
        value.record_stream(stream)
    elif is_dataclass(value):
        for item in value.__dict__.values():
            record_stream(item, stream)
    elif isinstance(value, dict):
        for item in value.values():
            record_stream(item, stream)
    elif isinstance(value, (tuple, list)):
        for item in value:
            record_stream(item, stream)


def batch_to_device(batch, device):
    return move_to_device(batch, device)


def cuda_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cudagraph_step_begin(device: torch.device, compile_mode: str):
    if device.type == "cuda" and compile_mode != "off":
        torch.compiler.cudagraph_mark_step_begin()


class DeviceBatchPrefetcher:
    def __init__(self, loader, device: torch.device, enabled: bool):
        self.loader_iter = iter(loader)
        self.device = device
        self.enabled = enabled and device.type == "cuda"
        self.next_batch = None
        self.stream = None
        self.device_index = None
        self.last_data_seconds = 0.0
        self.last_transfer_seconds = 0.0
        if self.enabled:
            self.device_index = device.index if device.index is not None else torch.cuda.current_device()
            self.stream = torch.cuda.Stream(device=self.device_index)
        self.preload()

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration

        if not self.enabled:
            batch = self.next_batch
            self.preload()
            return batch

        current_stream = torch.cuda.current_stream(self.device_index)
        current_stream.wait_stream(self.stream)
        batch = self.next_batch
        record_stream(batch, current_stream)
        self.preload()
        return batch

    def preload(self):
        data_start = time.perf_counter()
        try:
            batch = next(self.loader_iter)
        except StopIteration:
            self.next_batch = None
            self.last_data_seconds = time.perf_counter() - data_start
            self.last_transfer_seconds = 0.0
            return
        self.last_data_seconds = time.perf_counter() - data_start

        transfer_start = time.perf_counter()
        if not self.enabled:
            self.next_batch = batch_to_device(batch, self.device)
            self.last_transfer_seconds = time.perf_counter() - transfer_start
            return

        with torch.cuda.stream(self.stream):
            self.next_batch = batch_to_device(batch, self.device)
        self.last_transfer_seconds = time.perf_counter() - transfer_start


def rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def load_checkpoint(path: Path, device: torch.device):
    del device
    return torch.load(path, map_location="cpu", weights_only=False)
