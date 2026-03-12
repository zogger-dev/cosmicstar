from flax import linen as nn
from .head import ValueHead

class Agent(nn.Module):
    """
    Full AlphaStar Agent (Torso -> Core -> Head).
    
    Current Implementation:
    - Torso: Active (Scalar only)
    - Core: Placeholder (Pass-through)
    - Head: Active (Value only)
    """
    
    @nn.compact
    def __call__(self, observations):
        # 1. Torso
        # Input: Dict of observations
        # Output: [Batch, Torso_Dim]
        torso_out = ModelTorso()(observations)
        
        # 2. Core (LSTM) - TODO
        # For now, just pass through the torso output as the "state representation"
        core_out = torso_out
        
        # 3. Heads
        # Value Head
        value = ValueHead()(core_out)
        
        # Policy Head (TODO)
        # policy_logits = PolicyHead()(core_out)
        
        return value
