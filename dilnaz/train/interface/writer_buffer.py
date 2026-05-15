import sys

import torch

from dilnaz.models.dil import DilConfig
from dilnaz.models.naz import Naz
from dilnaz.tokenization import HybridTokenizer, TokenSegment


def decode_token_ids(tokenizer: HybridTokenizer, token_ids: torch.Tensor, token_mask: torch.Tensor) -> str:
    ids = token_ids[token_mask].detach().cpu().tolist()
    return tokenizer.decode(ids)


class UnitWriterBuffer:
    def __init__(self, model: Naz, config: DilConfig, tokenizer: HybridTokenizer, microbatch_size: int = 1):
        if microbatch_size <= 0:
            raise ValueError("microbatch_size must be > 0")
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.latent_size = config.latent_size
        self.microbatch_size = microbatch_size
        self.pending_latents: list[torch.Tensor] = []
        self.pending_should_stop: list[bool] = []

    def seed_prompt(self, prompt_latents: torch.Tensor, prompt_segments: list[TokenSegment]):
        del prompt_latents, prompt_segments

    def append(self, latent: torch.Tensor, should_stop: bool):
        self.pending_latents.append(latent.squeeze(0).detach())
        self.pending_should_stop.append(should_stop)

    def _drop_pending_prefix(self, count: int):
        del self.pending_latents[:count]
        del self.pending_should_stop[:count]

    def _emission_limit(self, force: bool) -> int:
        del force
        return min(len(self.pending_latents), self.microbatch_size)

    def _decode_pending_prefix(self, count: int):
        latent = torch.stack(self.pending_latents[:count], dim=0)
        return self.model.dil_model.decode_semantic(latent)

    def flush(self, force: bool = False) -> bool:
        del force
        stop_after_flush = False
        while self.pending_latents:
            emission_limit = self._emission_limit(force=True)
            if emission_limit <= 0:
                break
            generation = self._decode_pending_prefix(emission_limit)
            emitted = 0
            for row_idx in range(emission_limit):
                token_ids = generation.token_ids[row_idx]
                token_mask = generation.token_mask[row_idx]
                length = int(generation.lengths[row_idx].detach().cpu())
                is_global_eos = bool(token_mask[0].detach().cpu()) and int(token_ids[0].detach().cpu()) == self.tokenizer.eos_token_id
                if is_global_eos:
                    self._drop_pending_prefix(1)
                    stop_after_flush = True
                    emitted += 1
                    break
                token = decode_token_ids(self.tokenizer, token_ids, token_mask)
                if length == 0 or (not token and self.pending_should_stop[0]):
                    self._drop_pending_prefix(1)
                    stop_after_flush = True
                    emitted += 1
                    break
                if token:
                    sys.stdout.write(token)
                    sys.stdout.flush()
                should_stop = self.pending_should_stop[0]
                emitted += 1
                self._drop_pending_prefix(1)
                if should_stop:
                    stop_after_flush = True
                    break
            if emitted == 0 or stop_after_flush:
                break
            if len(self.pending_latents) < self.microbatch_size:
                break
        return stop_after_flush
