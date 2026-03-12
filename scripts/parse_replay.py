import sys
import os
from absl import flags, app

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from cosmicstar.core.config import ReplayConfig, InterfaceConfig
from cosmicstar.parsing.replay import ReplayParser

FLAGS = flags.FLAGS
flags.DEFINE_string("replay_path", r"C:\Users\prade\sc2_data\my_replays\2025-12-13 - (T)ejade,(P)Clue VS (Z)zogger,(Z)Shampagne.SC2Replay", "Path to replay file")
flags.DEFINE_integer("step_mul", 100, "Game loops per step")
flags.DEFINE_boolean("render", False, "Whether to render the game")

def main(argv):
    del argv  # Unused
    
    replay_path = FLAGS.replay_path
    if not os.path.exists(replay_path):
        print(f"Replay file not found: {replay_path}")
        return

    # Create Configuration
    config = ReplayConfig(
        replay_path=replay_path,
        step_mul=FLAGS.step_mul,
        interface=InterfaceConfig(render=FLAGS.render)
    )
    
    # Initialize Parser
    parser = ReplayParser()
    
    print(f"Loading replay: {replay_path}")
    
    try:
        # Iterate through the generator
        for sample in parser.parse(config):
            game_loop = sample['state']['frame']
            
            # Print debug info every 1000 frames or for the first few
            if game_loop < 500 or game_loop % 1000 == 0:
                print(f"Frame {game_loop}")
                print(f"  Minerals: {sample['state']['scalars'][0]}")
                print(f"  Army: {sample['state']['scalars'][2]}")
                print(f"  Units: {sample['state']['unit_count']}")
                if len(sample['actions']) > 0:
                    print(f"  Actions: {sample['actions']}")

        print("Replay parsing finished.")

    except Exception as e:
        print(f"Parser Error: {e}")

if __name__ == "__main__":
    app.run(main)
