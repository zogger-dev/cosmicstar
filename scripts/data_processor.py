import sys
import os
import torch
from absl import flags, app

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from cosmicstar.core.config import ReplayConfig, InterfaceConfig
from cosmicstar.parsing.replay import ReplayParser

# Output directory for processed data
DATA_OUTPUT_DIR = r"C:\Users\prade\sc2_data\processed"
if not os.path.exists(DATA_OUTPUT_DIR):
    os.makedirs(DATA_OUTPUT_DIR)

FLAGS = flags.FLAGS
flags.DEFINE_string("replay_path", r"C:\Users\prade\sc2_data\my_replays\CosmicStar_Gen_2025-12-15-02-56-06.SC2Replay", "Path to replay file")

def main(argv):
    del argv
    
    replay_path = FLAGS.replay_path
    if not os.path.exists(replay_path):
        print(f"Replay not found: {replay_path}")
        return
        
    replay_name = os.path.basename(replay_path).replace(".SC2Replay", "")
    print(f"Processing: {replay_name}")
    
    # Configure
    config = ReplayConfig(
        replay_path=replay_path,
        step_mul=8,  # Standard APM step
        interface=InterfaceConfig() # Default 84x84
    )
    
    parser = ReplayParser()
    samples = []
    
    try:
        step_count = 0
        for data in parser.parse(config):
            step_count += 1
            
            # Logic from original script: Save every 100 steps (was 800 frames = 100 steps * 8)
            # OR if there are actions.
            
            has_actions = len(data['actions']) > 0
            
            if has_actions or step_count % 100 == 0:
                # Reconstruct the sample format expected by dataset.py
                # Note: 'scalars' in FeatureBuilder includes game_loop at index 4, 
                # but original script separated frame.
                # data['state']['scalars'] has 5 elements.
                
                # Original scalars: [minerals, vespene, food_used, food_cap]
                # New scalars: [minerals, vespene, food_used, food_cap, game_loop]
                # dataset.py likely expects specific size. Let's inspect dataset.py later if needed.
                # For now, slice off the game_loop to match original expectations if strictly needed, 
                # or keep it. The original script had 4 scalars.
                
                original_scalars = data['state']['scalars'][:4]
                
                sample = {
                    "frame": data['state']['frame'],
                    "scalars": original_scalars,
                    "unit_count": data['state']['unit_count'],
                    "action_ids": data['actions']
                }
                samples.append(sample)
            
            if step_count % 100 == 0:
                 print(f"Step {step_count} | Extracted {len(samples)} samples", end='\r')

        print(f"\nReplay processed. Total Samples: {len(samples)}")
        
        # Save
        output_file = os.path.join(DATA_OUTPUT_DIR, f"{replay_name}.pt")
        torch.save(samples, output_file)
        print(f"Saved to: {output_file}")

    except Exception as e:
        print(f"Error processing replay: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    app.run(main)
