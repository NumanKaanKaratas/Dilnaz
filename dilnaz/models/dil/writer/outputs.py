from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import ModelOutput

from dilnaz.surface import PackedSurface


@dataclass
class DilWriterOutput(ModelOutput):
    token_logits: Optional[torch.FloatTensor] = None
    state_valid_logits: Optional[torch.FloatTensor] = None
    emit_logits: Optional[torch.FloatTensor] = None
    query_surface: Optional[PackedSurface] = None
