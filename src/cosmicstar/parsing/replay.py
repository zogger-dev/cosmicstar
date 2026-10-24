import os
import sys
from typing import Generator, Dict, Any

from pysc2 import run_configs
from pysc2.lib import remote_controller
from s2clientprotocol import sc2api_pb2 as sc_pb

import cosmicstar.envs.wine_run_config  # noqa: F401  registers the Wine/Proton run config on Linux

from cosmicstar.core.config import ReplayConfig
from cosmicstar.core.feature_builder import FeatureBuilder


class ReplayParser:
    """
    Manages the SC2 process to parse replays.
    """

    def __init__(self):
        self.feature_builder = FeatureBuilder()

    def parse(self, config: ReplayConfig) -> Generator[Dict[str, Any], None, None]:
        """
        Generator that yields parsed frames from the replay.
        """
        run_config = run_configs.get()

        # Validate paths
        if not os.path.exists(config.replay_path):
            raise FileNotFoundError(f"Replay not found: {config.replay_path}")

        print(f"Starting Replay Parsing for: {os.path.basename(config.replay_path)}")

        # Start SC2 Process
        # Note: We let run_config manages the process lifecycle,
        # but we need to arguably manage ports if running in parallel.
        # pysc2 'start' method automatically finds ports if not specified,
        # but for parallel we might need to pass 'network_settings' if supported.
        # For now, we rely on pysc2's default behavior which usually binds to available ports
        # but we wrap in a try/finally to ensure cleanup.

        interface = sc_pb.InterfaceOptions(
            raw=True,
            score=True,
            feature_layer=sc_pb.SpatialCameraSetup(width=config.interface.camera_width),
        )

        # Apply resolutions
        interface.feature_layer.resolution.x = config.interface.screen_resolution[0]
        interface.feature_layer.resolution.y = config.interface.screen_resolution[1]
        interface.feature_layer.minimap_resolution.x = (
            config.interface.minimap_resolution[0]
        )
        interface.feature_layer.minimap_resolution.y = (
            config.interface.minimap_resolution[1]
        )

        try:
            with run_config.start(want_rgb=config.interface.render) as controller:
                replay_data = run_config.replay_data(config.replay_path)
                info = controller.replay_info(replay_data)

                # Verify Map
                map_data = None
                if info.local_map_path and os.path.exists(info.local_map_path):
                    map_data = run_config.map_data(info.local_map_path)
                elif config.map_data_path and os.path.exists(config.map_data_path):
                    map_data = run_config.map_data(config.map_data_path)

                # Determine Player ID to observe
                player_id = config.observed_player_id
                if player_id is None:
                    # Default: Observe the first player or the one with a result
                    if len(info.player_info) > 0:
                        player_id = info.player_info[0].player_info.player_id
                    else:
                        player_id = 1

                print(
                    f"Parsed Info - Map: {info.map_name}, Duration: {info.game_duration_loops}, Observing: {player_id}"
                )

                # Start Replay
                controller.start_replay(
                    sc_pb.RequestStartReplay(
                        replay_data=replay_data,
                        map_data=map_data,
                        options=interface,
                        disable_fog=config.disable_fog,
                        observed_player_id=player_id,
                    )
                )

                # Main Loop
                while True:
                    controller.step(config.step_mul)
                    obs = controller.observe()

                    if controller.status != remote_controller.Status.in_replay:
                        break

                    # Extract Data
                    state = self.feature_builder.extract_state(
                        obs, obs.observation.game_loop
                    )
                    actions = self.feature_builder.extract_actions(obs)

                    # Combine into a sample result
                    yield {
                        "state": state,
                        "actions": actions,
                        "info": {
                            "replay_name": os.path.basename(config.replay_path),
                            "player_id": player_id,
                        },
                    }

                    if obs.player_result:
                        break

        except Exception as e:
            # Re-raise or log
            print(f"Error parsing replay {config.replay_path}: {e}")
            raise e
