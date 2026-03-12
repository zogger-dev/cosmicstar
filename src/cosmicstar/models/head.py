import jax
import jax.numpy as jnp
from flax import linen as nn

class ValueHead(nn.Module):
    """
    Estimates the value V(s) of the current state.
    """
    
    @nn.compact
    def __call__(self, torso_output):
        # torso_output shape: [Batch, Torso_Dim]
        
        # Simple MLP for value estimation
        x = nn.Dense(features=128)(torso_output)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        
        # Output shape: [Batch, 1]
        return x
