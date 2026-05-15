from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import ModelOutput


@dataclass
class NazOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    reconstruction_loss: Optional[torch.FloatTensor] = None
    mse_loss: Optional[torch.FloatTensor] = None
    mse_mean: Optional[torch.FloatTensor] = None
    mixture_nll: Optional[torch.FloatTensor] = None
    responsibility_loss: Optional[torch.FloatTensor] = None
    usage_balance_loss: Optional[torch.FloatTensor] = None
    moe_balance_loss: Optional[torch.FloatTensor] = None
    min_mse: Optional[torch.FloatTensor] = None
    chosen_mse: Optional[torch.FloatTensor] = None
    router_entropy: Optional[torch.FloatTensor] = None
    candidate_usage: Optional[torch.FloatTensor] = None
    moe_usage: Optional[torch.FloatTensor] = None
    cosine_loss: Optional[torch.FloatTensor] = None
    latent_cos: Optional[torch.FloatTensor] = None
    latent_predictions: Optional[torch.FloatTensor] = None
    predicted_latents: Optional[torch.FloatTensor] = None
    target_latents: Optional[torch.FloatTensor] = None
    num_targets: Optional[torch.LongTensor] = None


@dataclass
class NazGenerationOutput(ModelOutput):
    prompt_latents: Optional[torch.FloatTensor] = None
    generated_latents: Optional[torch.FloatTensor] = None


@dataclass
class NazGenerationStep(ModelOutput):
    latent: Optional[torch.FloatTensor] = None
    latent_cos_to_previous: Optional[torch.FloatTensor] = None
    should_stop: Optional[torch.Tensor] = None
    candidate_index: Optional[torch.LongTensor] = None


@dataclass
class NazDynamicsOutput(ModelOutput):
    candidate_latents: Optional[torch.FloatTensor] = None
    router_logits: Optional[torch.FloatTensor] = None
    selected_latents: Optional[torch.FloatTensor] = None
    selected_indices: Optional[torch.LongTensor] = None
