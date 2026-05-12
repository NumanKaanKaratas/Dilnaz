import torch
import torch.nn.functional as F
from torch import nn

from ..common.latents import normalize_semantic_latents
from .configuration import NazConfig
from .outputs import NazDynamicsOutput


class SemanticDynamicsMixtureHead(nn.Module):
    def __init__(self, config: NazConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.latent_size = config.latent_size
        self.num_candidates = config.num_semantic_candidates
        self.horizons = config.mtp_horizons
        expert_size = config.hidden_size
        self.in_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.shared = nn.Sequential(
            nn.Linear(config.hidden_size, 2 * config.hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(2 * config.hidden_size, config.hidden_size, bias=True),
            nn.SiLU(),
        )
        self.base = nn.Linear(config.hidden_size, self.horizons * self.latent_size, bias=True)
        self.expert_up = nn.Linear(config.hidden_size, self.num_candidates * expert_size, bias=True)
        self.expert_gate = nn.Linear(config.hidden_size, self.num_candidates * expert_size, bias=True)
        self.expert_down = nn.Parameter(
            torch.empty(self.num_candidates, expert_size, self.horizons * self.latent_size)
        )
        self.expert_down_bias = nn.Parameter(torch.zeros(self.num_candidates, self.horizons * self.latent_size))
        self.router = nn.Linear(config.hidden_size, self.horizons * self.num_candidates, bias=True)
        self.offset_gate = nn.Linear(config.hidden_size, self.horizons * self.num_candidates, bias=True)
        self.reset_parameters(config.initializer_range)

    def reset_parameters(self, initializer_range: float) -> None:
        nn.init.normal_(self.expert_down, mean=0.0, std=initializer_range)
        nn.init.normal_(self.router.weight, mean=0.0, std=initializer_range)
        nn.init.zeros_(self.router.bias)

    def forward(self, hidden_states: torch.Tensor) -> NazDynamicsOutput:
        batch_size, sequence_length, _ = hidden_states.shape
        shared = self.shared(self.in_norm(hidden_states))
        base = self.base(shared).view(batch_size, sequence_length, self.horizons, self.latent_size)
        expert_up = self.expert_up(shared).view(batch_size, sequence_length, self.num_candidates, -1)
        expert_gate = self.expert_gate(shared).view(batch_size, sequence_length, self.num_candidates, -1)
        expert_hidden = F.silu(expert_gate) * expert_up
        offsets = torch.einsum("btke,keh->btkh", expert_hidden, self.expert_down)
        offsets = offsets + self.expert_down_bias.view(1, 1, self.num_candidates, -1)
        offsets = offsets.view(
            batch_size,
            sequence_length,
            self.num_candidates,
            self.horizons,
            self.latent_size,
        ).permute(0, 1, 3, 2, 4)
        offset_gate = torch.sigmoid(
            self.offset_gate(shared).view(batch_size, sequence_length, self.horizons, self.num_candidates)
        ).unsqueeze(-1)
        candidate_latents = normalize_semantic_latents(base.unsqueeze(3) + offset_gate * offsets)
        router_logits = self.router(shared).view(batch_size, sequence_length, self.horizons, self.num_candidates)
        selected_indices = router_logits.argmax(dim=-1)
        gather_index = selected_indices.unsqueeze(-1).unsqueeze(-1).expand(
            batch_size,
            sequence_length,
            self.horizons,
            1,
            self.latent_size,
        )
        selected_latents = candidate_latents.gather(dim=3, index=gather_index).squeeze(3)
        return NazDynamicsOutput(
            candidate_latents=candidate_latents.float(),
            router_logits=router_logits.float(),
            selected_latents=selected_latents.float(),
            selected_indices=selected_indices,
        )
