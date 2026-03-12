from pysc2.agents import base_agent
from pysc2.lib import actions

class CosmicAgent(base_agent.BaseAgent):
    """A basic agent for StarCraft II."""
    
    def step(self, obs):
        # super(CosmicAgent, self).step(obs) # Disable super call as it expects TimeStep
        self.steps += 1
        self.reward += 0 # simplified tracking
        
        # TODO: Implement agent logic here
        
        # Determine valid action
        # For verification, try to build a marine or scv if possible, or just Move Camera
        # For now, just return no-op
        return actions.FUNCTIONS.no_op()
