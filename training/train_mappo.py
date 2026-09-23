"""Training script for MAPPO on CrisisRL environments."""

import os
import sys
from pathlib import Path
import yaml
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation.envs.disaster_env import DisasterEnv
from simulation.scenarios.loader import load_scenario
from rl.mappo import MAPPO, RolloutBuffer

def main():
    # Load config
    config_path = Path("training/configs/default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    env_cfg = config["env"]
    train_cfg = config["training"]
    log_cfg = config["logging"]
    
    # Setup environment
    scenario = load_scenario(env_cfg["scenario"])
    # override max steps if specified in config
    if "max_steps" in env_cfg:
        scenario["hazard"]["max_steps"] = env_cfg["max_steps"]
        
    env = DisasterEnv(scenario)
    
    num_agents = len(env.possible_agents)
    obs_shape = env.observation_space(env.possible_agents[0]).shape
    # For actor input, we expect (4, 5, 5) assuming channels first if we didn't permute,
    # but the env returns (5, 5, 4). The MAPPO class handles permuting (batch, H, W, C) -> (batch, C, H, W).
    # So we can pass the env obs_shape directly or just (4, 5, 5) depending on convention.
    # The actor code checks for obs.dim() == 4 and obs.shape[-1] == 4, and permutes.
    
    action_dim = env.action_space(env.possible_agents[0]).n
    global_state_shape = (env.height, env.width, 5)
    
    device = train_cfg["device"] if torch.cuda.is_available() and train_cfg["device"] == "cuda" else "cpu"
    print(f"Using device: {device}")
    
    mappo = MAPPO(
        obs_shape=(4, 5, 5), # We know we will permute to this
        global_state_shape=global_state_shape,
        action_dim=action_dim,
        actor_lr=train_cfg["actor_lr"],
        critic_lr=train_cfg["critic_lr"],
        clip_param=train_cfg["clip_param"],
        ppo_epoch=train_cfg["ppo_epoch"],
        num_mini_batch=train_cfg["num_mini_batch"],
        entropy_coef=train_cfg["entropy_coef"],
        value_loss_coef=train_cfg["value_loss_coef"],
        max_grad_norm=train_cfg["max_grad_norm"],
        device=device
    )
    
    buffer = RolloutBuffer(
        num_steps=train_cfg["num_steps"],
        num_agents=num_agents,
        obs_shape=obs_shape, # Buffer stores raw env obs (5, 5, 4)
        global_state_shape=global_state_shape,
        device=device
    )
    
    os.makedirs(log_cfg["save_dir"], exist_ok=True)
    
    # Training loop
    for episode in range(1, train_cfg["num_episodes"] + 1):
        obs_dict, infos = env.reset()
        
        episode_rewards = {agent: 0.0 for agent in env.possible_agents}
        
        for step in range(train_cfg["num_steps"]):
            # Format inputs
            obs_list = [obs_dict[agent] for agent in env.possible_agents]
            obs_batch = np.stack(obs_list)
            
            # Global state is same for all agents in CTDE
            global_state = env.state()
            global_states_batch = np.stack([global_state for _ in range(num_agents)])
            
            # Get actions
            actions, log_probs, values = mappo.get_actions(obs_batch, global_states_batch)
            
            # Step environment
            action_dict = {agent: int(actions[i]) for i, agent in enumerate(env.possible_agents)}
            next_obs_dict, rewards_dict, terminations, truncations, next_infos = env.step(action_dict)
            
            # Extract rewards and dones
            rewards = np.array([rewards_dict.get(a, 0.0) for a in env.possible_agents])
            # If agent terminated earlier, rewards might not have it, so default 0.0
            
            dones = np.array([terminations.get(a, True) or truncations.get(a, True) for a in env.possible_agents])
            
            # Insert into buffer
            buffer.insert(obs_batch, global_states_batch, actions, log_probs, rewards, values, dones)
            
            # Update episode rewards
            for agent, r in rewards_dict.items():
                episode_rewards[agent] += r
                
            obs_dict = next_obs_dict
            
            if all(dones):
                break
                
        # Compute GAE and update
        # Get next value for bootstrapping
        next_obs_list = [obs_dict.get(a, np.zeros(obs_shape)) for a in env.possible_agents]
        # In reality, if episode is done, next_value doesn't matter much due to next_done=1
        global_state = env.state()
        global_states_batch = np.stack([global_state for _ in range(num_agents)])
        next_values = mappo.get_values(global_states_batch)
        
        buffer.compute_returns_and_advantages(
            next_value=torch.tensor(next_values, dtype=torch.float32).to(device),
            next_done=torch.tensor(dones, dtype=torch.float32).to(device),
            gamma=train_cfg["gamma"],
            gae_lambda=train_cfg["gae_lambda"]
        )
        
        actor_loss, critic_loss, entropy_loss = mappo.update(buffer)
        buffer.clear()
        
        # Logging
        if episode % 10 == 0:
            avg_reward = sum(episode_rewards.values()) / num_agents
            print(f"Episode: {episode}/{train_cfg['num_episodes']} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Actor Loss: {actor_loss:.4f} | "
                  f"Critic Loss: {critic_loss:.4f} | "
                  f"Entropy: {entropy_loss:.4f}")
                  
        if episode % log_cfg["save_interval"] == 0:
            save_path = Path(log_cfg["save_dir"]) / f"mappo_ep_{episode}.pt"
            mappo.save(str(save_path))
            print(f"Saved model to {save_path}")

if __name__ == "__main__":
    main()
