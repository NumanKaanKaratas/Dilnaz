from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class NazBackboneLayerCache:
    key: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None
    delta_state: Optional[torch.Tensor] = None
    conv_state: Optional[torch.Tensor] = None


@dataclass
class NazBackboneCache:
    layers: list[NazBackboneLayerCache]
    position: int = 0

    @classmethod
    def empty(cls, num_layers: int) -> "NazBackboneCache":
        return cls([NazBackboneLayerCache() for _ in range(num_layers)])
