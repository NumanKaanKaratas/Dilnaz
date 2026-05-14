from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import ModelOutput

from dilnaz.surface import PackedSurface


@dataclass
class DilWriterOutput(ModelOutput):
    token_logits: Optional[torch.FloatTensor] = None
    query_surface: Optional[PackedSurface] = None


@dataclass
class DilWriterGeneration(ModelOutput):
    token_ids: Optional[torch.LongTensor] = None
    token_mask: Optional[torch.BoolTensor] = None
    lengths: Optional[torch.LongTensor] = None
    stopped: Optional[torch.BoolTensor] = None
