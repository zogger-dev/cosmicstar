from typing import Dict

import torch
from torch import nn


class ScalarEncoder(nn.Module):
    """Encodes scalar features (minerals, vespene, food, etc.) into an embedding."""

    def __init__(self, output_dim: int = 64):
        super().__init__()
        # The scalar feature count is fixed by the dataset, so infer it on first call.
        self.net = nn.Sequential(
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModelTorso(nn.Module):
    """
    AlphaStar Torso (Encoder & Fusion).
    Fuses Spatial, Entity, and Scalar streams.

    Current Implementation:
    - Scalar Stream: Active
    - Entity Stream: Placeholder
    - Spatial Stream: Placeholder
    """

    def __init__(self, output_dim: int = 256, scalar_dim: int = 64):
        super().__init__()
        self.scalar_encoder = ScalarEncoder(output_dim=scalar_dim)
        # TODO: entity stream (Transformer) and spatial stream (ResNet).
        fusion_in = scalar_dim
        self.fusion = nn.Sequential(nn.Linear(fusion_in, output_dim), nn.ReLU())

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = []

        # 1. Scalar Stream
        if "scalars" in observations:
            embeddings.append(self.scalar_encoder(observations["scalars"]))

        # 2. Entity Stream (TODO)
        # 3. Spatial Stream (TODO)

        if not embeddings:
            raise ValueError("No valid input streams found in observations!")

        # Fusion: concatenate all embeddings -> [Batch, sum(embedding_dims)]
        fusion = torch.cat(embeddings, dim=-1)
        # Project to the fixed torso output size.
        return self.fusion(fusion)
