import os
import sys
import time
import json
import csv
from pathlib import Path
import yaml
import numpy as np
import random
import torch
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation.envs.disaster_env import DisasterEnv
from simulation.scenarios.loader import load_scenario
from rl.mappo import MAPPO, RolloutBuffer

def evaluate(mappo, eval_scenarios, eval_seeds, max_agents):
    results = []
    
    for scenario_path in eval_scenarios:
        scenario_data = load_scenario(scenario_path)
        for seed in eval_seeds:
            env = DisasterEnv(scenario_data, max_agents=max_agents)
            obs_dict, _ = env.reset(seed=seed)
            done = False
            total_reward = 0.0
            
            # For action distribution tracking
            action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            
            while not done:
                active_agents = env.agents
                if not active_agents:
                    break
                    
                visual_list = [obs_dict[agent]["visual"] for agent in active_agents]
                vector_list = [obs_dict[agent]["vector"] for agent in active_agents]
                
                with torch.no_grad():
                    visual_tensor = torch.tensor(np.stack(visual_list), dtype=torch.float32).to(mappo.device)
                    vector_tensor = torch.tensor(np.stack(vector_list), dtype=torch.float32).to(mappo.device)
                    action_dists = mappo.actor(visual_tensor, vector_tensor)
                    actions = action_dists.logits.argmax(dim=-1).cpu().numpy()
                    
                action_dict = {agent: int(actions[i]) for i, agent in enumerate(active_agents)}
                
                for a in action_dict.values():
                    action_counts[a] += 1
                
                next_obs, rewards, terminations, truncations, infos = env.step(action_dict)
                total_reward += sum(rewards.values())
                obs_dict = next_obs
                done = all([terminations.get(a, True) or truncations.get(a, True) for a in active_agents])
                
            results.append({
                "scenario": os.path.basename(scenario_path),
                "seed": seed,
                "reward": total_reward,
                "rescued": env.survivors_rescued_total,
                "casualties": env.casualties_total,
                "action_dist": action_counts
            })
            
    return results

def main():
    config_path = Path("training/configs/multi.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    env_cfg = config["env"]
    train_cfg = config["training"]
    log_cfg = config["logging"]
    eval_cfg = config.get("evaluation", {})

    train_scenarios = env_cfg["train_scenarios"]
    train_seeds = env_cfg["train_seeds"]
    eval_seeds = env_cfg.get("eval_seeds", [])
    max_agents = env_cfg.get("max_agents", 2)

    os.makedirs(log_cfg["save_dir"], exist_ok=True)

    # Initialize environment once to get dimensions (assume all match max_agents)
    init_scenario = load_scenario(train_scenarios[0])
    dummy_env = DisasterEnv(init_scenario, max_agents=max_agents)
    
    action_dim = dummy_env.action_space(dummy_env.possible_agents[0]).n
    # Note: Using a fixed global state shape based on maximum size (e.g., 20x20)
    # The CNN critic handles variable size via AdaptiveAvgPool, but the rollout buffer needs fixed dims.
    # Actually, if we use a buffer with max shape, we can pad. Let's just fix the buffer to a large enough grid (e.g. 15x15 or 20x20).
    # Since we are doing Phase A (fire_small 10x10 and flood_small 12x12), max is 12x12.
    MAX_H, MAX_W = 12, 12
    global_state_shape = (MAX_H, MAX_W, 5)
    visual_obs_shape = dummy_env.obs_shape  # (5, 5, 4)
    coord_vector_dim = dummy_env.coord_vector_dim

    device = train_cfg["device"] if torch.cuda.is_available() and train_cfg["device"] == "cuda" else "cpu"
    
    print(f"Using device: {device}")
    print(f"Train Scenarios: {train_scenarios}")
    print(f"Train Seeds: {train_seeds}")

    mappo = MAPPO(
        obs_shape=visual_obs_shape,
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
        num_agents=max_agents,
        obs_shape=visual_obs_shape,
        global_state_shape=global_state_shape,
        coord_vector_dim=coord_vector_dim,
        device=device
    )

    csv_file = open(os.path.join(log_cfg["save_dir"], "training_log.csv"), "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "scenario", "seed", "reward", "rescued", "casualties", "actor_loss", "critic_loss", "entropy", "wall_time", "total_steps"])
    
    total_env_steps = 0
    start_time = time.time()

    for episode in range(1, train_cfg["num_episodes"] + 1):
        # Sample scenario and seed
        scenario_path = random.choice(train_scenarios)
        scenario_data = load_scenario(scenario_path)
        seed = random.choice(train_seeds)

        env = DisasterEnv(scenario_data, max_agents=max_agents)
        obs_dict, infos = env.reset(seed=seed)

        episode_rewards = {agent: 0.0 for agent in env.possible_agents}

        for step in range(train_cfg["num_steps"]):
            # Extract visual and vector observations
            # We must handle cases where some agents are inactive, but MAPPO expects max_agents.
            # In Phase A, both scenarios have 2 agents.
            visual_list = [obs_dict[agent]["visual"] for agent in env.possible_agents]
            vector_list = [obs_dict[agent]["vector"] for agent in env.possible_agents]
            obs_batch = np.stack(visual_list)
            coord_batch = np.stack(vector_list)

            # Global state padding to MAX_H x MAX_W
            gs = env.state()
            gs_padded = np.zeros(global_state_shape, dtype=np.float32)
            h, w = gs.shape[0], gs.shape[1]
            gs_padded[:h, :w, :] = gs
            global_states_batch = np.stack([gs_padded for _ in range(max_agents)])

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

            for agent, r in rewards_dict.items():
                episode_rewards[agent] += r

            obs_dict = next_obs_dict
            total_env_steps += len(env.possible_agents)

            if all(dones):
                break

        # Compute GAE and update
        gs = env.state()
        gs_padded = np.zeros(global_state_shape, dtype=np.float32)
        h, w = gs.shape[0], gs.shape[1]
        gs_padded[:h, :w, :] = gs
        global_states_batch = np.stack([gs_padded for _ in range(max_agents)])
        
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
        avg_reward = sum(episode_rewards.values())
        elapsed_time = time.time() - start_time
        
        csv_writer.writerow([
            episode, os.path.basename(scenario_path), seed, avg_reward, 
            env.survivors_rescued_total, env.casualties_total, 
            actor_loss, critic_loss, entropy_loss, elapsed_time, total_env_steps
        ])
        csv_file.flush()

        if episode % 10 == 0:
            print(f"Ep {episode}/{train_cfg['num_episodes']} | "
                  f"Scen: {os.path.basename(scenario_path)} | Seed: {seed} | "
                  f"Reward: {avg_reward:.2f} | Rescued: {env.survivors_rescued_total} | "
                  f"Entropy: {entropy_loss:.4f}")

        # Evaluation
        if episode % eval_cfg.get("eval_interval", 50) == 0:
            print("\n--- Evaluation ---")
            eval_results = evaluate(mappo, train_scenarios, eval_seeds, max_agents)
            for res in eval_results:
                print(f"Eval {res['scenario']} (Seed {res['seed']}): Reward={res['reward']:.2f}, Rescued={res['rescued']}, Actions={res['action_dist']}")
            print("------------------\n")

        # Save checkpoint
        if episode % log_cfg["save_interval"] == 0:
            save_path = os.path.join(log_cfg["save_dir"], f"mappo_multi_ep_{episode}.pt")
            mappo.save(save_path)
            print(f"Saved model to {save_path}")

    csv_file.close()

if __name__ == "__main__":
    main()
