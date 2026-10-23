from typing import Dict

import torch
from torch import nn

from .head import ValueHead
from .torso import ModelTorso


class Agent(nn.Module):
    """
    Full AlphaStar Agent (Torso -> Core -> Head).

    Current Implementation:
    - Torso: Active (Scalar only)
    - Core: Placeholder (Pass-through)
    - Head: Active (Value only)
    """

    def __init__(self, torso_dim: int = 256):
        super().__init__()
        self.torso = ModelTorso(output_dim=torso_dim)
        # TODO: Core (LSTM) and PolicyHead.
        self.value_head = ValueHead(input_dim=torso_dim)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 1. Torso: dict of observations -> [Batch, Torso_Dim]
        torso_out = self.torso(observations)
        # 2. Core (LSTM) - TODO: pass-through for now.
        core_out = torso_out
        # 3. Heads: value only for now.
        return self.value_head(core_out)
