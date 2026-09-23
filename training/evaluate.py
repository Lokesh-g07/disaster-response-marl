"""Evaluation script for trained MAPPO models."""

import argparse
import os
import sys
import time
from pathlib import Path
import yaml
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation.envs.disaster_env import DisasterEnv
from simulation.scenarios.loader import load_scenario
from rl.mappo import MAPPO

def evaluate(model_path, scenario_path, num_episodes=5, delay=0.1, render=True):
    # Load default config for MAPPO init
    config_path = Path("training/configs/default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    scenario = load_scenario(scenario_path)
    env = DisasterEnv(scenario, render_mode="ansi" if render else None)
    
    num_agents = len(env.possible_agents)
    obs_shape = env.observation_space(env.possible_agents[0]).shape
    action_dim = env.action_space(env.possible_agents[0]).n
    global_state_shape = (env.height, env.width, 5)
    
    device = "cpu"
    
    mappo = MAPPO(
        obs_shape=(4, 5, 5),
        global_state_shape=global_state_shape,
        action_dim=action_dim,
        device=device
    )
    
    mappo.load(model_path)
    print(f"Loaded model weights from {model_path}")
    
    all_rewards = []
    all_rescues = []
    
    for episode in range(1, num_episodes + 1):
        obs_dict, infos = env.reset()
        episode_rewards = {agent: 0.0 for agent in env.possible_agents}
        
        if render:
            print(f"\n--- Episode {episode} ---")
            print(env.render())
            time.sleep(delay)
            
        step = 0
        while True:
            # Active agents only
            active_agents = env.agents
            if not active_agents:
                break
                
            obs_list = [obs_dict[agent] for agent in active_agents]
            obs_batch = np.stack(obs_list)
            
            # During evaluation, we only need local observations
            # We can use the actor directly
            with torch.no_grad():
                obs_tensor = torch.tensor(obs_batch, dtype=torch.float32).to(device)
                action_dists = mappo.actor(obs_tensor)
                actions = action_dists.sample().cpu().numpy()
                
            action_dict = {agent: int(actions[i]) for i, agent in enumerate(active_agents)}
            
            next_obs_dict, rewards_dict, terminations, truncations, next_infos = env.step(action_dict)
            
            for agent, r in rewards_dict.items():
                episode_rewards[agent] += r
                
            if render:
                print(f"Step {step + 1}: Rewards = { {k: round(v, 2) for k, v in rewards_dict.items()} }")
                print(env.render())
                time.sleep(delay)
                
            obs_dict = next_obs_dict
            step += 1
            
            if all([terminations.get(a, True) or truncations.get(a, True) for a in active_agents]):
                break
                
        total_reward = sum(episode_rewards.values())
        rescued = env.survivors_rescued_total
        print(f"Episode {episode} | Total Reward: {total_reward:.2f} | Rescued: {rescued}/{env.initial_survivor_count}")
        all_rewards.append(total_reward)
        all_rescues.append(rescued)
        
    print("\n=== Evaluation Summary ===")
    print(f"Average Reward: {np.mean(all_rewards):.2f} +/- {np.std(all_rewards):.2f}")
    print(f"Average Rescued: {np.mean(all_rescues):.2f} +/- {np.std(all_rescues):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained MAPPO")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--scenario", type=str, default="simulation/scenarios/examples/fire_small.json", help="Path to scenario JSON")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between steps")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    
    args = parser.parse_args()
    evaluate(args.model, args.scenario, args.episodes, args.delay, not args.no_render)
