import torch
from torch import nn


class ValueHead(nn.Module):
    """
    Estimates the value V(s) of the current state.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, torso_output: torch.Tensor) -> torch.Tensor:
        # torso_output: [Batch, Torso_Dim] -> [Batch, 1]
        return self.net(torso_output)
