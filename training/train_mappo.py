"""Training script for MAPPO on CrisisRL environments.

v2: Updated for dual-branch actor with coordination vector observations.
    The training loop now extracts both "visual" and "vector" from the
    Dict observations returned by DisasterEnv v2.
"""

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
    config_path = Path("training/configs/smoke.yaml")
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
    action_dim = env.action_space(env.possible_agents[0]).n
    global_state_shape = (env.height, env.width, 5)
    visual_obs_shape = env.obs_shape  # (5, 5, 4)
    coord_vector_dim = env.coord_vector_dim

    device = train_cfg["device"] if torch.cuda.is_available() and train_cfg["device"] == "cuda" else "cpu"
    print(f"Using device: {device}")
    print(f"Agents: {env.possible_agents}")
    print(f"Visual obs: {visual_obs_shape} | Coord vector dim: {coord_vector_dim}")

    mappo = MAPPO(
        obs_shape=(4, 5, 5),  # channels-first convention
        global_state_shape=global_state_shape,
        action_dim=action_dim,
        coord_vector_dim=coord_vector_dim,
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
        obs_shape=visual_obs_shape,  # Buffer stores raw env visual obs (5, 5, 4)
        global_state_shape=global_state_shape,
        coord_vector_dim=coord_vector_dim,
        device=device
    )

    os.makedirs(log_cfg["save_dir"], exist_ok=True)

    # Training loop
    for episode in range(1, train_cfg["num_episodes"] + 1):
        obs_dict, infos = env.reset()

        episode_rewards = {agent: 0.0 for agent in env.possible_agents}

        for step in range(train_cfg["num_steps"]):
            # Extract visual and vector observations
            visual_list = [obs_dict[agent]["visual"] for agent in env.possible_agents]
            vector_list = [obs_dict[agent]["vector"] for agent in env.possible_agents]
            obs_batch = np.stack(visual_list)
            coord_batch = np.stack(vector_list)

            # Global state is same for all agents in CTDE
            global_state = env.state()
            global_states_batch = np.stack([global_state for _ in range(num_agents)])

            # Get actions
            actions, log_probs, values = mappo.get_actions(obs_batch, coord_batch, global_states_batch)

            # Step environment
            action_dict = {agent: int(actions[i]) for i, agent in enumerate(env.possible_agents)}
            next_obs_dict, rewards_dict, terminations, truncations, next_infos = env.step(action_dict)

            # Extract rewards and dones
            rewards = np.array([rewards_dict.get(a, 0.0) for a in env.possible_agents])
            dones = np.array([terminations.get(a, True) or truncations.get(a, True) for a in env.possible_agents])

            # Insert into buffer
            buffer.insert(obs_batch, coord_batch, global_states_batch, actions, log_probs, rewards, values, dones)

            # Update episode rewards
            for agent, r in rewards_dict.items():
                episode_rewards[agent] += r

            obs_dict = next_obs_dict

            if all(dones):
                break

        # Compute GAE and update
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
