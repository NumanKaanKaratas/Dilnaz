from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SyntheticLatentSet:
    context: torch.Tensor
    target: torch.Tensor
    mode: torch.Tensor


@dataclass(frozen=True)
class SyntheticWorld:
    style_to_context: torch.Tensor
    mode_to_context: torch.Tensor
    prototypes: torch.Tensor
    mode_offsets: torch.Tensor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


def orthogonal_prototypes(num_modes: int, latent_dim: int, radius: float, device: torch.device) -> torch.Tensor:
    raw = torch.randn(latent_dim, num_modes, device=device)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return q.T.contiguous() * radius


def build_synthetic_world(
    *,
    num_modes: int,
    context_dim: int,
    style_dim: int,
    latent_dim: int,
    prototype_radius: float,
    offset_scale: float,
    device: torch.device,
) -> SyntheticWorld:
    style_to_context = torch.randn(style_dim, context_dim, device=device) / math.sqrt(style_dim)
    mode_to_context = torch.randn(num_modes, context_dim, device=device) / math.sqrt(num_modes)
    prototypes = orthogonal_prototypes(num_modes, latent_dim, prototype_radius, device)
    mode_offsets = torch.randn(num_modes, style_dim, latent_dim, device=device)
    mode_offsets = mode_offsets / math.sqrt(style_dim) * offset_scale
    return SyntheticWorld(
        style_to_context=style_to_context,
        mode_to_context=mode_to_context,
        prototypes=prototypes,
        mode_offsets=mode_offsets,
    )


def sample_synthetic_latents(
    *,
    world: SyntheticWorld,
    n: int,
    mode_hint: float,
    noise_std: float,
    device: torch.device,
) -> SyntheticLatentSet:
    style_dim = world.style_to_context.shape[0]
    context_dim = world.style_to_context.shape[1]
    latent_dim = world.prototypes.shape[1]
    num_modes = world.prototypes.shape[0]

    style = torch.randn(n, style_dim, device=device)
    mode = torch.randint(0, num_modes, (n,), device=device)
    mode_one_hot = F.one_hot(mode, num_modes).float()

    context = style @ world.style_to_context
    context = context + mode_hint * (mode_one_hot @ world.mode_to_context)
    context = context + noise_std * torch.randn_like(context)
    context = F.layer_norm(context, (context_dim,))

    target_offsets = torch.einsum("bs,bsd->bd", style, world.mode_offsets[mode])
    target = world.prototypes[mode] + target_offsets + noise_std * torch.randn(n, latent_dim, device=device)
    target = F.layer_norm(target, (latent_dim,))
    return SyntheticLatentSet(context=context, target=target, mode=mode)


def gaussian_mixture_nll(
    router_logits: torch.Tensor,
    candidates: torch.Tensor,
    target: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    latent_dim = target.shape[-1]
    sq_dist = (candidates - target.unsqueeze(1)).square().sum(dim=-1)
    log_prob = -0.5 * sq_dist / (sigma * sigma)
    log_prob = log_prob - 0.5 * latent_dim * math.log(2.0 * math.pi * sigma * sigma)
    log_mix = F.log_softmax(router_logits, dim=-1) + log_prob
    return -torch.logsumexp(log_mix, dim=-1).mean()


def usage_balance_loss(router_logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(router_logits, dim=-1)
    usage = probs.mean(dim=0)
    return (usage * (usage.clamp_min(1e-8).log() + math.log(usage.numel()))).sum()


def router_responsibility_loss(
    router_logits: torch.Tensor,
    candidates: torch.Tensor,
    target: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    sq_dist = (candidates.detach() - target.unsqueeze(1)).square().sum(dim=-1)
    log_prob = -0.5 * sq_dist / (sigma * sigma)
    responsibilities = F.softmax(F.log_softmax(router_logits.detach(), dim=-1) + log_prob, dim=-1)
    return -(responsibilities * F.log_softmax(router_logits, dim=-1)).sum(dim=-1).mean()


class SingleLatentHead(nn.Module):
    def __init__(self, context_dim: int, width: int, latent_dim: int) -> None:
        super().__init__()
        self.net = make_mlp(context_dim, width, latent_dim)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred = self.net(context).unsqueeze(1)
        logits = context.new_zeros(context.shape[0], 1)
        return logits, pred

    def loss(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        sigma: float,
        usage_weight: float,
        router_resp_weight: float,
    ) -> torch.Tensor:
        del sigma, usage_weight, router_resp_weight
        _, pred = self(context)
        return F.mse_loss(pred.squeeze(1), target)


class IndependentMixtureHead(nn.Module):
    def __init__(self, context_dim: int, width: int, latent_dim: int, num_candidates: int) -> None:
        super().__init__()
        self.num_candidates = num_candidates
        self.latent_dim = latent_dim
        self.trunk = make_mlp(context_dim, width, width)
        self.router = nn.Linear(width, num_candidates)
        self.candidates = nn.Linear(width, num_candidates * latent_dim)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(context)
        logits = self.router(hidden)
        candidates = self.candidates(hidden).view(context.shape[0], self.num_candidates, self.latent_dim)
        return logits, candidates

    def loss(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        sigma: float,
        usage_weight: float,
        router_resp_weight: float,
    ) -> torch.Tensor:
        logits, candidates = self(context)
        return (
            gaussian_mixture_nll(logits, candidates, target, sigma)
            + usage_weight * usage_balance_loss(logits)
            + router_resp_weight * router_responsibility_loss(logits, candidates, target, sigma)
        )


class CandidateOffsetMixtureHead(nn.Module):
    def __init__(self, context_dim: int, width: int, latent_dim: int, num_candidates: int) -> None:
        super().__init__()
        self.num_candidates = num_candidates
        self.latent_dim = latent_dim
        self.trunk = make_mlp(context_dim, width, width)
        self.router = nn.Linear(width, num_candidates)
        self.gates = nn.Linear(width, num_candidates)
        self.prototypes = nn.Parameter(torch.randn(num_candidates, latent_dim) * 0.02)
        self.expert_offsets = nn.ModuleList(
            make_mlp(width, max(width // 2, latent_dim), latent_dim) for _ in range(num_candidates)
        )

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(context)
        logits = self.router(hidden)
        gates = torch.sigmoid(self.gates(hidden)).unsqueeze(-1)
        offsets = torch.stack([expert(hidden) for expert in self.expert_offsets], dim=1)
        candidates = self.prototypes.unsqueeze(0) + gates * offsets
        return logits, candidates

    def loss(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        sigma: float,
        usage_weight: float,
        router_resp_weight: float,
    ) -> torch.Tensor:
        logits, candidates = self(context)
        return (
            gaussian_mixture_nll(logits, candidates, target, sigma)
            + usage_weight * usage_balance_loss(logits)
            + router_resp_weight * router_responsibility_loss(logits, candidates, target, sigma)
        )


class CandidateScoredOffsetMixtureHead(nn.Module):
    def __init__(self, context_dim: int, width: int, latent_dim: int, num_candidates: int) -> None:
        super().__init__()
        self.num_candidates = num_candidates
        self.latent_dim = latent_dim
        self.shared = make_mlp(context_dim, width, width)
        self.prototypes = nn.Parameter(torch.randn(num_candidates, latent_dim) * 0.02)
        self.expert_states = nn.ModuleList(make_mlp(width, width, width) for _ in range(num_candidates))
        self.expert_offsets = nn.ModuleList(nn.Linear(width, latent_dim) for _ in range(num_candidates))
        self.expert_scores = nn.ModuleList(nn.Linear(width, 1) for _ in range(num_candidates))
        self.expert_gates = nn.ModuleList(nn.Linear(width, 1) for _ in range(num_candidates))

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(context)
        expert_states = [expert(shared) for expert in self.expert_states]
        logits = torch.cat([score(state) for score, state in zip(self.expert_scores, expert_states)], dim=-1)
        offsets = torch.stack(
            [offset(state) for offset, state in zip(self.expert_offsets, expert_states)],
            dim=1,
        )
        gates = torch.sigmoid(
            torch.cat([gate(state) for gate, state in zip(self.expert_gates, expert_states)], dim=-1)
        ).unsqueeze(-1)
        candidates = self.prototypes.unsqueeze(0) + gates * offsets
        return logits, candidates

    def loss(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        sigma: float,
        usage_weight: float,
        router_resp_weight: float,
    ) -> torch.Tensor:
        logits, candidates = self(context)
        return (
            gaussian_mixture_nll(logits, candidates, target, sigma)
            + usage_weight * usage_balance_loss(logits)
            + router_resp_weight * router_responsibility_loss(logits, candidates, target, sigma)
        )


def sample_batch(dataset: SyntheticLatentSet, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, dataset.context.shape[0], (batch_size,), device=dataset.context.device)
    return dataset.context[idx], dataset.target[idx]


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataset: SyntheticLatentSet,
    sigma: float,
    usage_weight: float,
    num_modes: int,
) -> dict[str, object]:
    logits, candidates = model(dataset.context)
    mse_per_candidate = (candidates - dataset.target.unsqueeze(1)).square().mean(dim=-1)
    probs = F.softmax(logits, dim=-1)
    chosen = probs.argmax(dim=-1)
    best = mse_per_candidate.argmin(dim=-1)
    chosen_mse = mse_per_candidate.gather(1, chosen.unsqueeze(1)).mean().item()
    min_mse = mse_per_candidate.min(dim=-1).values.mean().item()
    entropy = (-(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)).mean().item()
    soft_usage = probs.mean(dim=0)
    hard_usage = torch.bincount(chosen, minlength=probs.shape[-1]).float() / chosen.numel()

    mode_candidate_mse = torch.empty(num_modes, probs.shape[-1], device=dataset.context.device)
    for mode_idx in range(num_modes):
        mask = dataset.mode == mode_idx
        mode_candidate_mse[mode_idx] = mse_per_candidate[mask].mean(dim=0)

    best_perm = None
    best_perm_mse = None
    if probs.shape[-1] >= num_modes:
        for perm in itertools.permutations(range(probs.shape[-1]), num_modes):
            score = sum(mode_candidate_mse[mode_idx, candidate_idx].item() for mode_idx, candidate_idx in enumerate(perm))
            score /= num_modes
            if best_perm_mse is None or score < best_perm_mse:
                best_perm_mse = score
                best_perm = perm

    nll = gaussian_mixture_nll(logits, candidates, dataset.target, sigma).item()
    balance = usage_balance_loss(logits).item() if probs.shape[-1] > 1 else 0.0
    return {
        "nll": nll,
        "loss_with_balance": nll + usage_weight * balance,
        "min_mse": min_mse,
        "chosen_mse": chosen_mse,
        "router_entropy": entropy,
        "soft_usage": [round(v, 4) for v in soft_usage.detach().cpu().tolist()],
        "hard_usage": [round(v, 4) for v in hard_usage.detach().cpu().tolist()],
        "mode_to_candidate_perm": list(best_perm) if best_perm is not None else [],
        "mode_perm_mse": best_perm_mse,
        "best_assignment_usage": [
            round(v, 4)
            for v in (torch.bincount(best, minlength=probs.shape[-1]).float() / best.numel()).cpu().tolist()
        ],
    }


def train_model(
    *,
    name: str,
    model: nn.Module,
    train_set: SyntheticLatentSet,
    val_set: SyntheticLatentSet,
    steps: int,
    batch_size: int,
    lr: float,
    sigma: float,
    usage_weight: float,
    router_resp_weight: float,
    num_modes: int,
) -> dict[str, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for _ in range(steps):
        context, target = sample_batch(train_set, batch_size)
        loss = model.loss(context, target, sigma, usage_weight, router_resp_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    metrics = evaluate_model(model, val_set, sigma, usage_weight, num_modes)
    metrics["name"] = name
    metrics["parameters"] = sum(param.numel() for param in model.parameters())
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast synthetic probe for single latent, mixture latent, and V4 candidate-offset heads."
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-size", type=int, default=8192)
    parser.add_argument("--val-size", type=int, default=2048)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--context-dim", type=int, default=48)
    parser.add_argument("--style-dim", type=int, default=12)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--mode-hint", type=float, default=0.35)
    parser.add_argument("--prototype-radius", type=float, default=3.0)
    parser.add_argument("--offset-scale", type=float, default=1.4)
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--sigma", type=float, default=0.55)
    parser.add_argument("--usage-weight", type=float, default=0.05)
    parser.add_argument("--router-resp-weight", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    world = build_synthetic_world(
        num_modes=args.k,
        context_dim=args.context_dim,
        style_dim=args.style_dim,
        latent_dim=args.latent_dim,
        prototype_radius=args.prototype_radius,
        offset_scale=args.offset_scale,
        device=device,
    )

    train_set = sample_synthetic_latents(
        world=world,
        n=args.train_size,
        mode_hint=args.mode_hint,
        noise_std=args.noise_std,
        device=device,
    )
    val_set = sample_synthetic_latents(
        world=world,
        n=args.val_size,
        mode_hint=args.mode_hint,
        noise_std=args.noise_std,
        device=device,
    )

    models: list[tuple[str, nn.Module]] = [
        ("single_latent", SingleLatentHead(args.context_dim, args.width, args.latent_dim).to(device)),
        (
            "independent_mixture",
            IndependentMixtureHead(args.context_dim, args.width, args.latent_dim, args.k).to(device),
        ),
        (
            "v4_candidate_offset",
            CandidateOffsetMixtureHead(args.context_dim, args.width, args.latent_dim, args.k).to(device),
        ),
        (
            "v4_candidate_scored_offset",
            CandidateScoredOffsetMixtureHead(args.context_dim, args.width, args.latent_dim, args.k).to(device),
        ),
    ]

    results = {
        "config": vars(args) | {"device": str(device)},
        "results": [
            train_model(
                name=name,
                model=model,
                train_set=train_set,
                val_set=val_set,
                steps=args.steps,
                batch_size=args.batch_size,
                lr=args.lr,
                sigma=args.sigma,
                usage_weight=args.usage_weight,
                router_resp_weight=args.router_resp_weight,
                num_modes=args.k,
            )
            for name, model in models
        ],
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
