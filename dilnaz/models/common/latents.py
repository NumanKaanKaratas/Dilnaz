import math

import torch
import torch.nn.functional as F


def split_factorized_latent(
    latents: torch.Tensor,
    semantic_latent_size: int,
    surface_latent_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected = int(semantic_latent_size) + int(surface_latent_size)
    if latents.shape[-1] != expected:
        raise ValueError(f"factorized latent last dimension must be {expected}, got {latents.shape[-1]}")
    semantic = latents[..., :semantic_latent_size]
    surface = latents[..., semantic_latent_size:]
    return semantic, surface


def compose_factorized_latent(semantic: torch.Tensor, surface: torch.Tensor) -> torch.Tensor:
    if semantic.shape[:-1] != surface.shape[:-1]:
        raise ValueError("semantic and surface latent prefixes must match")
    return torch.cat([semantic, surface.to(dtype=semantic.dtype, device=semantic.device)], dim=-1)


def semantic_unit_latents(latents: torch.Tensor) -> torch.Tensor:
    raw = latents.float()
    norm = raw.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(raw)
    fallback[..., 0] = 1.0
    return torch.where(norm > 1e-6, raw / norm.clamp_min(1e-6), fallback)


def normalize_semantic_latents(latents: torch.Tensor) -> torch.Tensor:
    scale = math.sqrt(latents.shape[-1])
    return semantic_unit_latents(latents).to(latents.dtype) * scale


def normalize_factorized_latents(
    latents: torch.Tensor,
    semantic_latent_size: int,
    surface_latent_size: int,
) -> torch.Tensor:
    semantic, surface = split_factorized_latent(latents, semantic_latent_size, surface_latent_size)
    return compose_factorized_latent(normalize_semantic_latents(semantic), surface.tanh())


def angular_noise_like(latents: torch.Tensor, min_cos: torch.Tensor, max_cos: torch.Tensor) -> torch.Tensor:
    unit = semantic_unit_latents(latents)
    noise = torch.randn_like(unit)
    noise = noise - (noise * unit).sum(dim=-1, keepdim=True) * unit
    noise = F.normalize(noise, dim=-1, eps=1e-6)
    cos = torch.empty_like(min_cos).uniform_(0.0, 1.0)
    cos = min_cos + cos * (max_cos - min_cos)
    sin = torch.sqrt((1.0 - cos.square()).clamp_min(0.0))
    return (cos.unsqueeze(-1) * unit + sin.unsqueeze(-1) * noise) * math.sqrt(latents.shape[-1])
