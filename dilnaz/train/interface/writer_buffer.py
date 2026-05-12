import sys

import torch

from dilnaz.models.dil import DilConfig
from dilnaz.models.naz import Naz
from dilnaz.surface import PackedSurfaceState, writer_query_from_lengths
from dilnaz.tokenization import HybridTokenizer, TokenSegment


def decode_token_ids(tokenizer: HybridTokenizer, token_ids: torch.Tensor, token_mask: torch.Tensor) -> str:
    ids = token_ids[token_mask].detach().cpu().tolist()
    return tokenizer.decode(ids)


def decoded_surface_state(token_ids: torch.Tensor, token_mask: torch.Tensor, config: DilConfig) -> list[int]:
    ids = token_ids[token_mask].detach().cpu().tolist()
    return ids


def segment_surface_state(segment: TokenSegment, config: DilConfig, device: torch.device) -> list[int]:
    token_ids = list(segment.token_ids)
    if len(token_ids) > config.max_surface_pieces_per_unit:
        raise ValueError(
            f"prompt token has {len(token_ids)} pieces; max_surface_pieces_per_unit={config.max_surface_pieces_per_unit}"
        )
    return token_ids


class SlidingWriterBuffer:
    def __init__(self, model: Naz, config: DilConfig, tokenizer: HybridTokenizer, commit_threshold: float | None = None):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.commit_threshold = config.writer_commit_threshold if commit_threshold is None else commit_threshold
        self.window_size = config.writer_sliding_window_size
        self.left_frozen = config.writer_left_frozen
        self.active_size = config.writer_active_size
        self.right_guard = config.writer_right_guard
        self.latent_size = config.latent_size
        self.max_position_age = config.writer_max_position_age
        self.left_latents: list[torch.Tensor] = []
        self.left_surfaces: list[list[int]] = []
        self.pending_latents: list[torch.Tensor] = []
        self.pending_futures: list[torch.Tensor | None] = []
        self.pending_surfaces: list[list[int] | None] = []
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
        segments = prompt_segments[-seed_count:]
        device = latents.device
        for latent, segment in zip(latents, segments):
            self.left_latents.append(latent.detach())
            self.left_surfaces.append(segment_surface_state(segment, self.config, device))
        self.left_latents = self.left_latents[-self.left_frozen :]
        self.left_surfaces = self.left_surfaces[-self.left_frozen :]

    def append(self, latent: torch.Tensor, future_latents: torch.Tensor | None, should_stop: bool):
        self.pending_latents.append(latent.squeeze(0).detach())
        self.pending_futures.append(None if future_latents is None else future_latents.squeeze(0).detach())
        self.pending_surfaces.append(None)
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

    def _packed_state(self, rows: list[list[int]], kinds: list[int], frozen_flags: list[bool], device: torch.device) -> PackedSurfaceState:
        lengths = torch.tensor([[len(row) for row in rows]], dtype=torch.long, device=device)
        query = writer_query_from_lengths(
            lengths,
            pad_token_id=self.config.pad_token_id,
            surface_bucket_sizes=self.config.surface_bucket_sizes,
        )
        ids = torch.full_like(query.ids, self.config.writer_empty_token_id)
        state_kind = torch.zeros_like(query.ids)
        frozen = torch.zeros_like(query.mask)
        for unit_idx, row in enumerate(rows):
            if not row:
                continue
            start = int(query.unit_offsets[0, unit_idx].detach().cpu())
            end = start + len(row)
            ids[0, start:end] = torch.tensor(row, dtype=torch.long, device=device)
            state_kind[0, start:end] = int(kinds[unit_idx])
            frozen[0, start:end] = bool(frozen_flags[unit_idx])
        return PackedSurfaceState(
            ids=ids,
            state_kind=state_kind,
            frozen=frozen,
            mask=query.mask,
            unit_ids=query.unit_ids,
            pos_in_unit=query.pos_in_unit,
            unit_lengths=query.unit_lengths,
            unit_offsets=query.unit_offsets,
            unit_mask=query.unit_mask,
        )

    def _window_tensors(self):
        device = self.pending_latents[0].device
        dtype = self.pending_latents[0].dtype
        future_horizons = self._future_horizons()
        semantic, window_mask, future_tensor, position_age = self._cached_window_tensors(device, dtype, future_horizons)
        zone_ids = self.zone_ids.to(device=device).unsqueeze(0)
        state_rows = [[] for _ in range(self.window_size)]
        state_kinds = [0 for _ in range(self.window_size)]
        state_frozen = [False for _ in range(self.window_size)]

        left_count = min(self.left_frozen, len(self.left_latents))
        left_start = self.left_frozen - left_count
        for idx in range(left_count):
            slot = left_start + idx
            semantic[0, slot] = self.left_latents[-left_count + idx].to(device=device, dtype=dtype)
            state_rows[slot] = self.left_surfaces[-left_count + idx] + [self.config.writer_stop_token_id]
            state_kinds[slot] = 2
            state_frozen[slot] = True
            window_mask[0, slot] = True
            position_age[0, slot] = self.max_position_age

        pending_count = min(len(self.pending_latents), self.window_size - self.left_frozen)
        for idx in range(pending_count):
            slot = self.left_frozen + idx
            semantic[0, slot] = self.pending_latents[idx].to(device=device, dtype=dtype)
            window_mask[0, slot] = True
            position_age[0, slot] = min(self.pending_ages[idx], self.max_position_age)
            pending_surface = self.pending_surfaces[idx]
            if pending_surface is not None:
                state_rows[slot] = pending_surface + [self.config.writer_stop_token_id]
                state_kinds[slot] = 1
            future = self.pending_futures[idx]
            if future_tensor is not None and future is not None and future.numel() > 0:
                copy_count = min(future_horizons, future.shape[0])
                future_tensor[0, slot, :copy_count] = future[:copy_count].to(device=device, dtype=dtype)
        surface_state = self._packed_state(state_rows, state_kinds, state_frozen, device)
        return semantic, surface_state, zone_ids, window_mask, future_tensor, position_age

    def _cache_pending_surfaces(self, token_ids: torch.Tensor, token_masks: torch.Tensor, pending_count: int):
        for idx in range(pending_count):
            slot = self.left_frozen + idx
            self.pending_surfaces[idx] = decoded_surface_state(
                token_ids[0, slot],
                token_masks[0, slot],
                self.config,
            )

    def _drop_pending_prefix(self, count: int):
        del self.pending_latents[:count]
        del self.pending_futures[:count]
        del self.pending_surfaces[:count]
        del self.pending_ages[:count]
        del self.pending_should_stop[:count]

    def _bump_pending_ages(self):
        self.pending_ages = [min(age + 1, self.max_position_age) for age in self.pending_ages]

    def _commit_limit(self, force: bool) -> int:
        if not self.pending_latents:
            return 0
        if force:
            return min(len(self.pending_latents), self.active_size)
        if len(self.pending_latents) < self.active_size + self.right_guard:
            return 0
        return min(self.active_size, len(self.pending_latents) - self.right_guard)

    def _ready_values(self, ready_tensor: torch.Tensor, commit_limit: int, force: bool) -> list[bool]:
        if force:
            return [True] * commit_limit
        ready_values = ready_tensor.detach().cpu().tolist()
        for idx in range(commit_limit):
            if self.pending_ages[idx] >= self.max_position_age:
                ready_values[idx] = True
        return ready_values

    def flush(self, force: bool = False) -> bool:
        stop_after_flush = False
        while self.pending_latents:
            commit_limit = self._commit_limit(force)
            if commit_limit <= 0:
                break
            tensors = self._window_tensors()
            token_ids, token_masks, lengths, commit_scores = self.model.dil_model.decode_semantic_window(
                tensors[0],
                surface_state=tensors[1],
                zone_ids=tensors[2],
                window_mask=tensors[3],
                future_latents=tensors[4],
                position_age=tensors[5],
            )
            emitted = 0
            pending_count = min(len(self.pending_latents), self.window_size - self.left_frozen)
            self._cache_pending_surfaces(token_ids, token_masks, pending_count)
            slots = torch.arange(
                self.left_frozen,
                self.left_frozen + commit_limit,
                device=lengths.device,
            )
            slot_lengths = lengths[0, slots].clamp_max(self.config.max_surface_pieces_per_unit)
            positions = torch.arange(commit_scores.shape[-1], device=lengths.device).unsqueeze(0)
            valid_commit_positions = positions < (slot_lengths.unsqueeze(1) + 1).clamp_max(commit_scores.shape[-1])
            ready_tensor = (
                commit_scores[0, slots].ge(self.commit_threshold) | ~valid_commit_positions
            ).all(dim=1)
            eos_tensor = token_masks[0, slots, 0] & token_ids[0, slots, 0].eq(self.tokenizer.eos_token_id)
            length_values = slot_lengths.detach().cpu().tolist()
            ready_values = self._ready_values(ready_tensor, commit_limit, force)
            eos_values = eos_tensor.detach().cpu().tolist()
            for local_idx in range(commit_limit):
                slot = self.left_frozen + local_idx
                length = int(length_values[local_idx])
                ready = bool(ready_values[local_idx])
                if not force and not ready:
                    break
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
                surface_state = decoded_surface_state(token_ids[0, slot], token_masks[0, slot], self.config)
                self.left_latents.append(self.pending_latents[0])
                self.left_surfaces.append(surface_state)
                self.left_latents = self.left_latents[-self.left_frozen :]
                self.left_surfaces = self.left_surfaces[-self.left_frozen :]
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
