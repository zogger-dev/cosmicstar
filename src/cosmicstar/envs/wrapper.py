import numpy as np
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import units

class CosmicEnvWrapper:
    """Wraps sc2_env.SC2Env to provide structured tensors and shaped rewards."""

    def __init__(self, env):
        self._env = env
        self._last_score = 0
        self._last_minerals = 0
        self._last_vespene = 0
        self._last_food_used = 0
        
        # Define max constants for tensor shapes
        self.max_units = 512
        self.n_actions = len(actions.FUNCTIONS)

    def reset(self):
        """Resets the environment and returns the initial observation."""
        timesteps = self._env.reset()
        self._reset_state()
        return self._process_step(timesteps[0])

    def step(self, action):
        """Steps the environment with the given action."""
        # TODO: act_to_pysc2 converter needed here. 
        # For now, we assume action is already a pysc2 FunctionCall or we pass a no-op as placeholder
        # In full implementation, 'action' would be a tensor dictionary.
        
        # Placeholder: If action is None, send No-Op
        if action is None:
           pysc2_action = actions.FUNCTIONS.no_op()
        else:
           pysc2_action = action
           
        timesteps = self._env.step([pysc2_action])
        return self._process_step(timesteps[0])

    def _reset_state(self):
        self._last_score = 0
        self._last_minerals = 0
        self._last_vespene = 0
        self._last_food_used = 0

    def _process_step(self, timestep):
        """Converts pysc2 timestep to (obs, reward, done, info)."""
        obs = self._extract_observation(timestep)
        reward = self._compute_reward(timestep)
        done = timestep.last()
        return obs, reward, done, {}

    def _extract_observation(self, timestep):
        """Converts raw timestep observation to structured tensors."""
        
        # 1. Scalar Features
        # [PlayerID, Minerals, Vespene, FoodUsed, FoodCap, ArmyCount, ...]
        player = timestep.observation.player
        scalar_obs = np.array([
            float(player.player_id),
            float(player.minerals),
            float(player.vespene),
            float(player.food_used),
            float(player.food_cap),
            float(player.army_count),
            float(player.warp_gate_count),
            float(player.larva_count),
            float(timestep.observation.game_loop[0]), # game_loop is usually single-element array
        ], dtype=np.float32)

        # 2. Entity Features (Units)
        # We rely on 'feature_units' being enabled in AgentInterfaceFormat
        raw_units = timestep.observation.feature_units
        unit_features = np.zeros((self.max_units, 21), dtype=np.float32) # Updated dim to 21
        
        # Basic feature extraction loop
        for i, u in enumerate(raw_units):
            if i >= self.max_units: break
            # Example features: [unit_type, owner, health, shield, x, y, ...]
            unit_features[i] = [
                u.unit_type,
                u.alliance, # owner -> alliance (or keep owner if preferred, but alliance is 1=Self, 4=Enemy)
                u.health,
                u.shield,
                u.energy,
                u.x, u.y,
                0, # is_flying placeholder
                u.is_selected,
                u.is_powered,
                u.cloak,
                u.order_id_0,
                u.order_progress_0,
                u.buff_id_0,
                u.buff_id_1,
                u.addon_unit_type,
                0, 0, 0, 0, 0 # Padding
            ]
            
        # 3. Action Mask
        # Create binary mask size [Total_Functions]
        action_mask = np.zeros(self.n_actions, dtype=np.float32)
        if timestep.observation.available_actions.any():
            action_mask[timestep.observation.available_actions] = 1.0

        return {
            "scalar": scalar_obs,
            "units": unit_features,
            "mask": action_mask
        }

    def _compute_reward(self, timestep):
        """Computes shaped reward."""
        reward = timestep.reward # Base win/loss
        
        # Heuristic: small reward for resource collection
        current_minerals = timestep.observation.player.minerals
        current_vespene = timestep.observation.player.vespene
        current_food = timestep.observation.player.food_used

        minerals_diff = max(0, current_minerals - self._last_minerals) # Be careful with spending
        # Actually, standard rewards track 'score_cumulative', but let's use gathered delta if possible
        # Or better: use score details if available. 
        # For this pass, we'll just stick to the plan's shape:
        
        # Simplified shaping:
        # +0.001 per mineral collected is hard to track purely from 'current', 
        # because spending lowers current.
        # Better to track 'score' if available or accept noise.
        # Let's use 'score_cumulative.total_value_units' etc if present, otherwise ignore for V1.
        
        # Fallback: Just keep existing sparse reward for now + very tiny step penalty to encourage speed?
        # Or follow plan: " +0.01 per supply created"
        if current_food > self._last_food_used:
             reward += 0.01 * (current_food - self._last_food_used)
        
        self._last_minerals = current_minerals # This logic is flawed for gathering vs spending, need 'collection_rate' or score.
        self._last_vespene = current_vespene
        self._last_food_used = current_food
        
        return reward
