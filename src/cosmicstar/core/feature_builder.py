import numpy as np


class FeatureBuilder:
    """Extracts structured observations and actions from SC2 protocol data."""

    def __init__(self):
        pass

    def extract_state(self, obs, game_loop: int) -> dict:
        """
        Extracts observations from the raw SC2 observation object.

        Args:
            obs: The observation object returned by controller.observe()
            game_loop: Current game loop

        Returns:
            Dictionary containing processed state
        """
        observation = obs.observation

        # Scalar features
        scalars = np.array(
            [
                observation.player_common.minerals,
                observation.player_common.vespene,
                observation.player_common.food_used,
                observation.player_common.food_cap,
                # Normalize game loop slightly to keep it in reasonable range if needed,
                # but raw is fine for now or handled by downstream validation
                game_loop,
            ],
            dtype=np.float32,
        )

        # Unit Counts (Simple Feature)
        unit_count = len(observation.raw_data.units)

        # We can add more complex extraction here (spatial features, entity lists)
        # For now, sticking to what data_processor.py was doing but structured.

        return {
            "frame": game_loop,
            "scalars": scalars,
            "unit_count": unit_count,
            # In the future, 'screen' and 'minimap' would be extracted here
        }

    def extract_actions(self, obs) -> np.ndarray:
        """
        Extracts actions from the observation.

        Args:
            obs: The observation object returned by controller.observe()

        Returns:
            Numpy array of action IDs
        """
        actions = []
        if obs.actions:
            for action in obs.actions:
                if action.HasField("action_feature_layer"):
                    if action.action_feature_layer.HasField("unit_command"):
                        actions.append(
                            action.action_feature_layer.unit_command.ability_id
                        )

        return np.array(actions, dtype=np.int32)
