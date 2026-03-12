import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Any

class ScalarEncoder(nn.Module):
    """Encodes scalar features (minerals, vespene, food, etc.) into an embedding."""
    output_dim: int = 64

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.output_dim)(x)
        x = nn.relu(x)
        return x

class ModelTorso(nn.Module):
    """
    AlphaStar Torso (Encoder & Fusion).
    Fuses Spatial, Entity, and Scalar streams.
    
    Current Implementation:
    - Scalar Stream: Active
    - Entity Stream: Placeholder
    - Spatial Stream: Placeholder
    """
    output_dim: int = 256
    
    @nn.compact
    def __call__(self, observations: Dict[str, jnp.ndarray]):
        embeddings = []
        
        # 1. Scalar Stream
        if 'scalars' in observations:
            scalar_input = observations['scalars']
            # Ensure shape is [Batch, D]
            scalar_emb = ScalarEncoder(output_dim=64)(scalar_input)
            embeddings.append(scalar_emb)
            
        # 2. Entity Stream (TODO)
        # if 'entities' in observations:
        #   entity_emb = Transformer(...)(observations['entities'])
        #   embeddings.append(entity_emb)
        
        # 3. Spatial Stream (TODO)
        # if 'spatial' in observations:
        #   spatial_emb = ResNet(...)(observations['spatial'])
        #   embeddings.append(spatial_emb)
        
        if not embeddings:
            raise ValueError("No valid input streams found in observations!")
            
        # Fusion: Concatenate all embeddings
        # shape: [Batch, sum(embedding_dims)]
        fusion = jnp.concatenate(embeddings, axis=-1)
        
        # Project to fixed Torso output size
        fusion = nn.Dense(features=self.output_dim)(fusion)
        fusion = nn.relu(fusion)
        
        return fusion
