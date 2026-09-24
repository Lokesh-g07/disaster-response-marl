"""
Training script for the Shared DQN baseline.
"""

import os
import argparse
import yaml
import numpy as np
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation.envs.disaster_env import DisasterEnv
from simulation.scenarios.loader import load_scenario
from rl.dqn.shared_dqn import SharedDQN
from evaluation.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Train Shared DQN Baseline")
    parser.add_config = parser.add_argument(
        "--config", type=str, default="training/configs/dqn.yaml",
        help="Path to configuration yaml"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def set_seeds(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    args = parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seeds(args.seed)

    env_cfg = config["env"]
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]
    log_cfg = config["logging"]

    scenario = load_scenario(env_cfg["scenario"])
    env = DisasterEnv(scenario)
    
    # Initialize DQN
    agent_id = env.possible_agents[0]
    action_dim = env.action_space(agent_id).n
    obs_space = env.observation_space(agent_id)
    obs_shape = obs_space.spaces["visual"].shape
    chw_obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
    coord_vector_dim = obs_space.spaces["vector"].shape[0]

    device = train_cfg.get("device", "cpu")

    dqn = SharedDQN(
        obs_shape=chw_obs_shape,
        coord_vector_dim=coord_vector_dim,
        action_dim=action_dim,
        lr=train_cfg["learning_rate"],
        gamma=train_cfg["gamma"],
        buffer_capacity=train_cfg["buffer_capacity"],
        batch_size=train_cfg["batch_size"],
        target_update_freq=train_cfg["target_update_frequency"],
        epsilon_start=train_cfg["epsilon_start"],
        epsilon_end=train_cfg["epsilon_end"],
        epsilon_decay=train_cfg["epsilon_decay"],
        device=device
    )

    os.makedirs(log_cfg["save_dir"], exist_ok=True)
    eval_interval = eval_cfg.get("eval_interval", 50)
    eval_episodes = eval_cfg.get("eval_episodes", 5)
    log_interval  = log_cfg.get("log_interval", 10)
    save_interval = log_cfg.get("save_interval", 100)

    learning_starts = train_cfg.get("learning_starts", 1000)
    train_frequency = train_cfg.get("train_frequency", 1)

    global_step = 0

    print(f"Using device: {device}")
    print(f"Agents: {env.possible_agents}")
    print(f"Visual obs: {obs_shape} | Coord vector dim: {coord_vector_dim}")

    for episode in range(1, train_cfg["num_episodes"] + 1):
        obs_dict, _ = env.reset(seed=args.seed + episode)
        episode_reward = 0.0
        episode_losses = []

        done_flag = False
        while not done_flag and len(env.agents) > 0:
            actions = {}
            for agent_id in env.agents:
                if agent_id in obs_dict:
                    actions[agent_id] = dqn.get_action(obs_dict[agent_id], explore=True)

            next_obs_dict, rewards_dict, terminations, truncations, _ = env.step(actions)

            done_flag = all(terminations.get(a, True) or truncations.get(a, True) for a in env.possible_agents)

            # Insert transitions into shared buffer
            for agent_id in env.possible_agents:
                if agent_id in obs_dict:
                    # Treat early termination properly
                    r = rewards_dict.get(agent_id, 0.0)
                    d = 1.0 if (terminations.get(agent_id, False) or truncations.get(agent_id, False)) else 0.0
                    
                    visual = obs_dict[agent_id]["visual"]
                    vector = obs_dict[agent_id]["vector"]
                    
                    next_raw = next_obs_dict.get(agent_id)
                    if next_raw is not None:
                        next_visual = next_raw["visual"]
                        next_vector = next_raw["vector"]
                    else:
                        next_visual = np.zeros_like(visual)
                        next_vector = np.zeros_like(vector)

                    dqn.buffer.push(visual, vector, actions[agent_id], r, next_visual, next_vector, d)
                    episode_reward += r

            obs_dict = next_obs_dict
            global_step += 1

            if global_step > learning_starts and global_step % train_frequency == 0:
                loss = dqn.update()
                if loss is not None:
                    episode_losses.append(loss)

        avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0.0

        if episode % log_interval == 0:
            print(f"Episode: {episode}/{train_cfg['num_episodes']} | "
                  f"Avg Total Reward: {episode_reward:.2f} | "
                  f"Avg Loss: {avg_loss:.4f} | Epsilon: {dqn.epsilon:.3f}")

        if episode % save_interval == 0:
            save_path = Path(log_cfg["save_dir"]) / f"shared_dqn_ep_{episode}.pt"
            dqn.save(str(save_path))
            print(f"Saved model to {save_path}")

        if episode % eval_interval == 0:
            from rl.dqn.shared_dqn import SharedDQNPolicy
            evaluator = Evaluator(env_cfg["scenario"])
            policy = SharedDQNPolicy(dqn)
            
            # Save explore epsilon and set to greedy
            current_eps = dqn.epsilon
            
            summary = evaluator.evaluate(policy, num_episodes=eval_episodes, base_seed=args.seed)
            print(f"--- Eval Episode {episode} ---")
            print(f"Mean Rescued: {summary.mean_survivors_rescued:.2f}")
            print(f"Mean Reward: {summary.mean_total_reward:.2f}")
            print(f"--------------------------")
            
            # Restore explore epsilon
            dqn.epsilon = current_eps

if __name__ == "__main__":
    main()
