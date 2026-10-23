from dataclasses import dataclass, field
from typing import Optional, Tuple

@dataclass
class InterfaceConfig:
    """Configuration for the SC2 Interface."""
    screen_resolution: Tuple[int, int] = (84, 84)
    minimap_resolution: Tuple[int, int] = (64, 64)
    camera_width: int = 24
    render: bool = False  # Whether to render the game (slower, for debug)

@dataclass
class ReplayConfig:
    """Configuration for running a replay."""
    replay_path: str
    map_data_path: Optional[str] = None  # Path to the local map file if needed
    step_mul: int = 8  # Game loops per step
    interface: InterfaceConfig = field(default_factory=InterfaceConfig)
    disable_fog: bool = False
    observed_player_id: Optional[int] = None  # If None, will try to detect
