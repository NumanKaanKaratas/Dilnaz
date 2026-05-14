import sys

import torch

from dilnaz.models.dil import DilConfig
from dilnaz.models.naz import Naz
from dilnaz.tokenization import HybridTokenizer, TokenSegment


def decode_token_ids(tokenizer: HybridTokenizer, token_ids: torch.Tensor, token_mask: torch.Tensor) -> str:
    ids = token_ids[token_mask].detach().cpu().tolist()
    return tokenizer.decode(ids)


class SlidingWriterBuffer:
    def __init__(self, model: Naz, config: DilConfig, tokenizer: HybridTokenizer):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.window_size = config.writer_sliding_window_size
        self.left_frozen = config.writer_left_frozen
        self.active_size = config.writer_active_size
        self.right_guard = config.writer_right_guard
        self.latent_size = config.latent_size
        self.max_position_age = config.writer_max_position_age
        self.left_latents: list[torch.Tensor] = []
        self.pending_latents: list[torch.Tensor] = []
        self.pending_futures: list[torch.Tensor | None] = []
        self.pending_ages: list[int] = []
        self.pending_should_stop: list[bool] = []
        self._tensor_cache = {}
        self.zone_ids = torch.full((self.window_size,), 1, dtype=torch.long)
        self.zone_ids[: self.left_frozen] = 0
        self.zone_ids[self.left_frozen + self.active_size :] = 2

    def seed_prompt(self, prompt_latents: torch.Tensor, prompt_segments: list[TokenSegment]):
        if self.left_frozen <= 0 or prompt_latents.numel() == 0 or not prompt_segments:
            return
        if prompt_latents.dim() != 3 or prompt_latents.shape[0] != 1:
            raise ValueError("prompt_latents must be shaped [1, units, latent_size]")
        seed_count = min(self.left_frozen, prompt_latents.shape[1], len(prompt_segments))
        if seed_count <= 0:
            return
        latents = prompt_latents[0, -seed_count:]
        for latent in latents:
            self.left_latents.append(latent.detach())
        self.left_latents = self.left_latents[-self.left_frozen :]

    def append(self, latent: torch.Tensor, future_latents: torch.Tensor | None, should_stop: bool):
        self.pending_latents.append(latent.squeeze(0).detach())
        self.pending_futures.append(None if future_latents is None else future_latents.squeeze(0).detach())
        self.pending_ages.append(0)
        self.pending_should_stop.append(should_stop)

    def _future_horizons(self) -> int:
        for future in self.pending_futures:
            if future is not None and future.numel() > 0:
                return int(future.shape[0])
        return 0

    def _cached_window_tensors(self, device: torch.device, dtype: torch.dtype, future_horizons: int):
        key = (device.type, device.index, str(dtype), future_horizons)
        cached = self._tensor_cache.get(key)
        if cached is None:
            semantic = torch.zeros((1, self.window_size, self.latent_size), dtype=dtype, device=device)
            window_mask = torch.empty((1, self.window_size), dtype=torch.bool, device=device)
            future_tensor = None
            if future_horizons > 0:
                future_tensor = torch.zeros((1, self.window_size, future_horizons, self.latent_size), dtype=dtype, device=device)
            position_age = torch.empty((1, self.window_size), dtype=torch.long, device=device)
            cached = semantic, window_mask, future_tensor, position_age
            self._tensor_cache[key] = cached
        semantic, window_mask, future_tensor, position_age = cached
        semantic.zero_()
        window_mask.zero_()
        position_age.zero_()
        if future_tensor is not None:
            future_tensor.zero_()
        return cached

    def _window_tensors(self):
        device = self.pending_latents[0].device
        dtype = self.pending_latents[0].dtype
        future_horizons = self._future_horizons()
        semantic, window_mask, future_tensor, position_age = self._cached_window_tensors(device, dtype, future_horizons)
        zone_ids = self.zone_ids.to(device=device).unsqueeze(0)

        left_count = min(self.left_frozen, len(self.left_latents))
        left_start = self.left_frozen - left_count
        for idx in range(left_count):
            slot = left_start + idx
            semantic[0, slot] = self.left_latents[-left_count + idx].to(device=device, dtype=dtype)
            window_mask[0, slot] = True
            position_age[0, slot] = self.max_position_age

        pending_count = min(len(self.pending_latents), self.window_size - self.left_frozen)
        for idx in range(pending_count):
            slot = self.left_frozen + idx
            semantic[0, slot] = self.pending_latents[idx].to(device=device, dtype=dtype)
            window_mask[0, slot] = True
            position_age[0, slot] = min(self.pending_ages[idx], self.max_position_age)
            future = self.pending_futures[idx]
            if future_tensor is not None and future is not None and future.numel() > 0:
                copy_count = min(future_horizons, future.shape[0])
                future_tensor[0, slot, :copy_count] = future[:copy_count].to(device=device, dtype=dtype)
        return semantic, zone_ids, window_mask, future_tensor, position_age

    def _drop_pending_prefix(self, count: int):
        del self.pending_latents[:count]
        del self.pending_futures[:count]
        del self.pending_ages[:count]
        del self.pending_should_stop[:count]

    def _bump_pending_ages(self):
        self.pending_ages = [min(age + 1, self.max_position_age) for age in self.pending_ages]

    def _emission_limit(self, force: bool) -> int:
        if not self.pending_latents:
            return 0
        if force:
            return min(len(self.pending_latents), self.active_size)
        if len(self.pending_latents) < self.active_size + self.right_guard:
            return 0
        return min(self.active_size, len(self.pending_latents) - self.right_guard)

    def flush(self, force: bool = False) -> bool:
        stop_after_flush = False
        while self.pending_latents:
            emission_limit = self._emission_limit(force)
            if emission_limit <= 0:
                break
            tensors = self._window_tensors()
            generation = self.model.dil_model.decode_semantic_window(
                tensors[0],
                zone_ids=tensors[1],
                window_mask=tensors[2],
                future_latents=tensors[3],
                position_age=tensors[4],
            )
            token_ids = generation.token_ids
            token_masks = generation.token_mask
            lengths = generation.lengths
            emitted = 0
            slots = torch.arange(
                self.left_frozen,
                self.left_frozen + emission_limit,
                device=lengths.device,
            )
            slot_lengths = lengths[0, slots].clamp_max(self.config.max_surface_pieces_per_unit)
            eos_tensor = token_masks[0, slots, 0] & token_ids[0, slots, 0].eq(self.tokenizer.eos_token_id)
            length_values = slot_lengths.detach().cpu().tolist()
            eos_values = eos_tensor.detach().cpu().tolist()
            for local_idx in range(emission_limit):
                slot = self.left_frozen + local_idx
                length = int(length_values[local_idx])
                if bool(eos_values[local_idx]):
                    self._drop_pending_prefix(1)
                    stop_after_flush = True
                    emitted += 1
                    break
                token = decode_token_ids(self.tokenizer, token_ids[0, slot], token_masks[0, slot])
                if length == 0 or (not token and self.pending_should_stop[0]):
                    self._drop_pending_prefix(1)
                    stop_after_flush = True
                    emitted += 1
                    break
                if token:
                    sys.stdout.write(token)
                    sys.stdout.flush()
                self.left_latents.append(self.pending_latents[0])
                self.left_latents = self.left_latents[-self.left_frozen :]
                should_stop = self.pending_should_stop[0]
                emitted += 1
                self._drop_pending_prefix(1)
                if should_stop:
                    stop_after_flush = True
                    break
            if self.pending_latents:
                self._bump_pending_ages()
            if emitted == 0:
                break
            if stop_after_flush:
                break
            if not force:
                break
        return stop_after_flush
