import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from byte_trainer_utils import COMPILE_MODE_CHOICES, compile_forward, validate_compile_environment
from models.dil import DilConfig
from models.naz import NazConfig
from models.naz import Naz
from tokenization import HybridTokenizer, TokenSegment


CHECKPOINT_FORMAT_VERSION = 26
OBJECTIVE = "semantic_dynamics_moe_mtp_v1"


def tokenize_text(text: str, tokenizer: HybridTokenizer) -> list[TokenSegment]:
    segments = [
        segment
        for segment in tokenizer.encode_segments(text)
        if segment.piece_len > 0
    ]
    if not segments:
        raise ValueError("text produced no tokens")
    return segments


def make_batch(segments: list[TokenSegment], tokenizer: HybridTokenizer, config: NazConfig, device: torch.device):
    input_ids = torch.full(
        (1, len(segments), config.max_word_bytes),
        config.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    word_masks = torch.zeros(
        (1, len(segments), config.max_word_bytes),
        dtype=torch.bool,
        device=device,
    )
    byte_lengths = []
    for unit_idx, segment in enumerate(segments):
        token_ids = segment.token_ids
        if len(token_ids) > config.max_word_bytes:
            raise ValueError(
                f"token {unit_idx} {tokenizer.decode(token_ids)!r} has {len(token_ids)} pieces; "
                f"max_word_bytes={config.max_word_bytes}"
            )
        ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        input_ids[0, unit_idx, : ids.numel()] = ids
        word_masks[0, unit_idx, : ids.numel()] = True
        byte_lengths.append(ids.numel())
    unit_mask = torch.ones((1, len(segments)), dtype=torch.bool, device=device)
    return input_ids, word_masks, unit_mask, byte_lengths


def decode_token_ids(tokenizer: HybridTokenizer, token_ids: torch.Tensor, token_mask: torch.Tensor) -> str:
    ids = token_ids[token_mask].detach().cpu().tolist()
    return tokenizer.decode(ids)


def decoded_surface_state(token_ids: torch.Tensor, token_mask: torch.Tensor, config: DilConfig) -> torch.Tensor:
    state = torch.full((config.writer_max_positions,), -100, dtype=torch.long, device=token_ids.device)
    length = int(token_mask.sum().detach().cpu())
    if length > 0:
        state[:length] = token_ids[:length]
    if length < config.writer_max_positions:
        state[length] = config.writer_stop_token_id
    return state


def segment_surface_state(segment: TokenSegment, config: DilConfig, device: torch.device) -> torch.Tensor:
    token_ids = torch.tensor(segment.token_ids, dtype=torch.long, device=device)
    if token_ids.numel() > config.writer_max_positions - 1:
        raise ValueError(
            f"prompt token has {token_ids.numel()} pieces; max_word_bytes={config.writer_max_positions - 1}"
        )
    token_mask = torch.ones((token_ids.numel(),), dtype=torch.bool, device=device)
    return decoded_surface_state(token_ids, token_mask, config)


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
        self.left_surfaces: list[torch.Tensor] = []
        self.pending_latents: list[torch.Tensor] = []
        self.pending_futures: list[torch.Tensor | None] = []
        self.pending_surfaces: list[torch.Tensor | None] = []
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
            self.left_surfaces.append(segment_surface_state(segment, self.config, device).detach())
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
            surface_state = torch.empty(
                (1, self.window_size, self.config.writer_max_positions),
                dtype=torch.long,
                device=device,
            )
            surface_state_mask = torch.empty_like(surface_state)
            frozen_mask = torch.empty_like(surface_state, dtype=torch.bool)
            window_mask = torch.empty((1, self.window_size), dtype=torch.bool, device=device)
            future_tensor = None
            if future_horizons > 0:
                future_tensor = torch.zeros((1, self.window_size, future_horizons, self.latent_size), dtype=dtype, device=device)
            position_age = torch.empty((1, self.window_size), dtype=torch.long, device=device)
            cached = semantic, surface_state, surface_state_mask, frozen_mask, window_mask, future_tensor, position_age
            self._tensor_cache[key] = cached
        semantic, surface_state, surface_state_mask, frozen_mask, window_mask, future_tensor, position_age = cached
        semantic.zero_()
        surface_state.fill_(-100)
        surface_state_mask.zero_()
        frozen_mask.zero_()
        window_mask.zero_()
        position_age.zero_()
        if future_tensor is not None:
            future_tensor.zero_()
        return cached

    def _window_tensors(self):
        device = self.pending_latents[0].device
        dtype = self.pending_latents[0].dtype
        future_horizons = self._future_horizons()
        semantic, surface_state, surface_state_mask, frozen_mask, window_mask, future_tensor, position_age = (
            self._cached_window_tensors(device, dtype, future_horizons)
        )
        zone_ids = self.zone_ids.to(device=device).unsqueeze(0)

        left_count = min(self.left_frozen, len(self.left_latents))
        left_start = self.left_frozen - left_count
        for idx in range(left_count):
            slot = left_start + idx
            semantic[0, slot] = self.left_latents[-left_count + idx].to(device=device, dtype=dtype)
            surface_state[0, slot] = self.left_surfaces[-left_count + idx].to(device=device)
            surface_state_mask[0, slot] = torch.where(surface_state[0, slot].ge(0), 2, 0)
            frozen_mask[0, slot] = surface_state[0, slot].ge(0)
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
                surface_state[0, slot] = pending_surface.to(device=device)
                surface_state_mask[0, slot] = torch.where(surface_state[0, slot].ge(0), 1, 0)
            future = self.pending_futures[idx]
            if future_tensor is not None and future is not None and future.numel() > 0:
                copy_count = min(future_horizons, future.shape[0])
                future_tensor[0, slot, :copy_count] = future[:copy_count].to(device=device, dtype=dtype)
        return semantic, surface_state, surface_state_mask, frozen_mask, zone_ids, window_mask, future_tensor, position_age

    def _cache_pending_surfaces(self, token_ids: torch.Tensor, token_masks: torch.Tensor, pending_count: int):
        for idx in range(pending_count):
            slot = self.left_frozen + idx
            self.pending_surfaces[idx] = decoded_surface_state(
                token_ids[0, slot],
                token_masks[0, slot],
                self.config,
            ).detach()

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
                surface_state_mask=tensors[2],
                frozen_mask=tensors[3],
                zone_ids=tensors[4],
                window_mask=tensors[5],
                future_latents=tensors[6],
                position_age=tensors[7],
            )
            emitted = 0
            pending_count = min(len(self.pending_latents), self.window_size - self.left_frozen)
            self._cache_pending_surfaces(token_ids, token_masks, pending_count)
            slots = torch.arange(
                self.left_frozen,
                self.left_frozen + commit_limit,
                device=lengths.device,
            )
            slot_lengths = lengths[0, slots].clamp_max(self.config.writer_max_positions - 1)
            positions = torch.arange(self.config.writer_max_positions, device=lengths.device).unsqueeze(0)
            valid_commit_positions = positions < (slot_lengths.unsqueeze(1) + 1).clamp_max(self.config.writer_max_positions)
            ready_tensor = (
                commit_scores[0, slots].ge(self.commit_threshold) | ~valid_commit_positions
            ).all(dim=1)
            eos_tensor = token_masks[0, slots, 0] & token_ids[0, slots, 0].eq(self.tokenizer.eos_token_id)
            length_values = slot_lengths.detach().cpu().tolist()
            ready_values = [True] * commit_limit if force else ready_tensor.detach().cpu().tolist()
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


def load_model(checkpoint_dir: Path, device: torch.device, compile_mode: str):
    checkpoint_dir = checkpoint_dir.resolve()
    config = NazConfig.from_pretrained(checkpoint_dir)
    checkpoint = torch.load(
        checkpoint_dir / "checkpoint.pt",
        map_location=device,
        weights_only=False,
    )
    if checkpoint["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format_version={checkpoint.get('format_version')}")
    if checkpoint["training_state"].get("objective") != OBJECTIVE:
        raise ValueError(f"checkpoint objective is not {OBJECTIVE}")
    dil_path = Path(config.dil_path)
    if not dil_path.is_absolute():
        cwd_relative = dil_path.resolve()
        checkpoint_relative = (checkpoint_dir / dil_path).resolve()
        dil_path = cwd_relative if cwd_relative.exists() else checkpoint_relative
    config.dil_path = str(dil_path)
    model = Naz(config).to(device)
    model.load_trainable_state_dict(checkpoint["model_state_dict"])
    model.dil_model.set_compiled_forwards(
        encoder_forward=compile_forward(model.dil_model.encoder.forward, compile_mode, "DilEncoderCore"),
        writer_forward=compile_forward(model.dil_model.writer.forward, compile_mode, "DilConditionalWriter"),
        transition_forward=compile_forward(model.dil_model.writer.transition, compile_mode, "DilConditionalWriterTransition"),
    )
    model.eval()
    dil_config = DilConfig.from_pretrained(dil_path)
    tokenizer = HybridTokenizer.from_file(dil_path / dil_config.tokenizer_vocab_file)
    return model, config, tokenizer, checkpoint["training_state"]["step"]


@torch.no_grad()
def stream_text(
    model: Naz,
    config: NazConfig,
    tokenizer: HybridTokenizer,
    prompt_segments: list[TokenSegment],
    device: torch.device,
    max_new_tokens: int,
    min_new_tokens: int,
    repetition_cos_threshold: float,
    writer_microbatch_size: int,
):
    if max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0")
    if writer_microbatch_size <= 0:
        raise ValueError("--writer-microbatch-size must be > 0")
    input_ids, word_masks, unit_mask, _ = make_batch(prompt_segments, tokenizer, config, device)
    prompt_text = "".join(tokenizer.decode(segment.token_ids) for segment in prompt_segments)
    sys.stdout.write(prompt_text)
    sys.stdout.flush()

    writer_buffer = SlidingWriterBuffer(model, model.dil_model.config, tokenizer)
    prompt_latents = None
    if hasattr(model, "encode_sequence_latents"):
        prompt_latents = model.encode_sequence_latents(input_ids, word_masks, unit_mask)
        writer_buffer.seed_prompt(prompt_latents, prompt_segments)

    for step in model.generate_stream(
        input_ids=input_ids,
        word_masks=word_masks,
        unit_mask=unit_mask,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        repetition_cos_threshold=repetition_cos_threshold,
        prompt_latents=prompt_latents,
    ):
        writer_buffer.append(
            step.latent,
            getattr(step, "future_latents", None),
            bool(step.should_stop[0].detach().cpu()),
        )
        if writer_buffer.flush(force=False):
            break
        if writer_buffer.pending_should_stop and writer_buffer.pending_should_stop[-1]:
            if writer_buffer.flush(force=True):
                break
    else:
        writer_buffer.flush(force=True)
    sys.stdout.write("\n")
    sys.stdout.flush()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--text-file", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--min-new-tokens", type=int, default=None)
    parser.add_argument("--repetition-cos-threshold", type=float, default=None)
    parser.add_argument("--writer-microbatch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--compile-mode", choices=COMPILE_MODE_CHOICES, default="off")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    validate_compile_environment(args.compile_mode)

    model, config, tokenizer, _ = load_model(args.checkpoint_dir, device, args.compile_mode)
    text = args.text_file.read_text(encoding="utf-8") if args.text_file else args.text
    if text is None:
        raise ValueError("--text or --text-file is required")
    segments = tokenize_text(text, tokenizer)
    stream_text(
        model,
        config,
        tokenizer,
        segments,
        device,
        args.max_new_tokens,
        config.min_new_tokens if args.min_new_tokens is None else args.min_new_tokens,
        config.repetition_cos_threshold if args.repetition_cos_threshold is None else args.repetition_cos_threshold,
        args.writer_microbatch_size,
    )


if __name__ == "__main__":
    main()
