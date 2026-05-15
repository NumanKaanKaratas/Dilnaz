from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import ModelOutput


@dataclass
class DilOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    semantic: Optional[torch.FloatTensor] = None
    distill_loss: Optional[torch.FloatTensor] = None
    mean_geometry_loss: Optional[torch.FloatTensor] = None
    variance_loss: Optional[torch.FloatTensor] = None
