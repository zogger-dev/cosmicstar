import sys
import os
from absl import flags
from pysc2.env import sc2_env
from pysc2.lib import features

# Add parent directory to path to import agents/env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cosmicstar.agents.base_agent import CosmicAgent
from cosmicstar.envs.wrapper import CosmicEnvWrapper

FLAGS = flags.FLAGS
flags.DEFINE_bool("render", True, "Whether to render with pygame.")

def main(unused_argv):
    agent = CosmicAgent()
    try:
        with sc2_env.SC2Env(
            map_name="Simple64",
            players=[sc2_env.Agent(sc2_env.Race.terran),
                     sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64),
                use_feature_units=True),
            step_mul=8,
            game_steps_per_episode=0,
            visualize=True,
            save_replay_episodes=1,
            replay_dir="C:/Users/prade/sc2_data/my_replays",
            replay_prefix="CosmicStar_Gen") as env:
            
            # Wrap the environment
            env = CosmicEnvWrapper(env)
            
            # Reset Wrapper (which resets env internally)
            obs, reward, done, info = env.reset()
            # Pass dictionary observation to agent
            agent.reset()
            
            while True:
                # Agent acts on structured observation (obs is now a dict)
                # TODO: Update agent to return valid pysc2 actions or tensor actions
                # For now, we rely on agent receiving raw obs, but we need to pass structured.
                # Let's pass 'obs' to agent.step()
                step_actions = [agent.step(obs)]
                
                # Wrapper step
                if step_actions[0] is None:
                     # Handle no-op if agent returns None
                     from pysc2.lib import actions
                     step_actions = [actions.FUNCTIONS.no_op()]

                obs, reward, done, info = env.step(step_actions[0])
                
                if done:
                    break
                
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    from absl import app
    app.run(main)
