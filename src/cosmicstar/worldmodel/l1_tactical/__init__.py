"""Layer 1 (`L1`) tactical representations (`0.3125 s` `step=7` frames, squad pooler, chunk flow, energy & SAE)."""

from . import chunk_decoder, energy, frames, sae, squad_pooler
from .chunk_decoder import L1ChunkFlowDecoder
from .energy import EnergyModel, State, build_gram_state, evaluate_spatial_fields_grid, gram_layout
from .frames import ContinuousPotentialField, Frames, SpatioTemporalBlock, extract_entity_perception
from .sae import MultiLayerSAEStack, TopKLayerSAE
from .squad_pooler import (
    D_SQUAD_DEFAULT,
    HindsightOutcomePosterior,
    PerceiverSquadCompressor,
    extract_combat_losses,
    innovation_dim,
    pack_innovation,
    unpack_innovation,
)

__all__ = [
    "ContinuousPotentialField",
    "D_SQUAD_DEFAULT",
    "EnergyModel",
    "Frames",
    "HindsightOutcomePosterior",
    "L1ChunkFlowDecoder",
    "MultiLayerSAEStack",
    "PerceiverSquadCompressor",
    "SpatioTemporalBlock",
    "State",
    "TopKLayerSAE",
    "build_gram_state",
    "chunk_decoder",
    "energy",
    "evaluate_spatial_fields_grid",
    "extract_combat_losses",
    "extract_entity_perception",
    "frames",
    "gram_layout",
    "innovation_dim",
    "pack_innovation",
    "sae",
    "squad_pooler",
    "unpack_innovation",
]
