"""
Training script for Independent PPO on CrisisRL environments.

Independent PPO vs MAPPO
------------------------
MAPPO:
  - ONE shared actor (all agents share weights)
  - ONE centralized critic (sees full global state during training)
  - Cooperative training signal

Independent PPO (this script):
  - SEPARATE actor per agent (no parameter sharing)
  - SEPARATE local critic per agent (local 5x5x4 obs only)
  - Each agent trains independently on its own reward/trajectory

Usage:
    python training/train_ppo.py
    python training/train_ppo.py --config training/configs/ppo.yaml
    python training/train_ppo.py --episodes 50   # quick dev run
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation.envs.disaster_env import DisasterEnv
from simulation.scenarios.loader import load_scenario
from rl.ppo import IndependentPPOPolicy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="training/configs/ppo.yaml")
    p.add_argument("--episodes", type=int, default=None,
                   help="Override num_episodes in config (useful for quick dev runs)")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_cfg   = cfg["env"]
    train_cfg = cfg["training"]
    log_cfg   = cfg["logging"]
    eval_cfg  = cfg.get("evaluation", {})

    if args.episodes is not None:
        train_cfg["num_episodes"] = args.episodes

    # Device
    device = (
        train_cfg["device"]
        if (train_cfg["device"] == "cuda" and torch.cuda.is_available())
        else "cpu"
    )
    print(f"[PPO Training] Device: {device}")

    # Environment
    scenario = load_scenario(env_cfg["scenario"])
    scenario["hazard"]["max_steps"] = env_cfg.get("max_steps", 50)
    env = DisasterEnv(scenario)

    num_agents  = len(env.possible_agents)
    obs_shape   = env.observation_space(env.possible_agents[0]).shape   # (5,5,4)
    action_dim  = env.action_space(env.possible_agents[0]).n            # 5

    print(f"[PPO Training] Agents: {env.possible_agents}")
    print(f"[PPO Training] Obs shape: {obs_shape}  |  Action dim: {action_dim}")
    print(f"[PPO Training] Episodes: {train_cfg['num_episodes']}")

    # Build one IndependentPPO per agent
    policy = IndependentPPOPolicy(
        agent_ids=env.possible_agents,
        action_dim=action_dim,
        device=device,
        num_steps=train_cfg["num_steps"],
        actor_lr=train_cfg["actor_lr"],
        critic_lr=train_cfg["critic_lr"],
        clip_param=train_cfg["clip_param"],
        ppo_epoch=train_cfg["ppo_epoch"],
        num_mini_batch=train_cfg["num_mini_batch"],
        entropy_coef=train_cfg["entropy_coef"],
        value_loss_coef=train_cfg["value_loss_coef"],
        max_grad_norm=train_cfg["max_grad_norm"],
        gamma=train_cfg["gamma"],
        gae_lambda=train_cfg["gae_lambda"],
    )

    os.makedirs(log_cfg["save_dir"], exist_ok=True)
    eval_interval = eval_cfg.get("eval_interval", 50)
    eval_episodes = eval_cfg.get("eval_episodes", 5)
    log_interval  = log_cfg.get("log_interval", 10)
    save_interval = log_cfg.get("save_interval", 100)

    # Training loop
    for episode in range(1, train_cfg["num_episodes"] + 1):
        obs_dict, _ = env.reset(seed=args.seed)
        episode_rewards = {aid: 0.0 for aid in env.possible_agents}
        last_obs = dict(obs_dict)

        policy.train()
        policy.clear_all_buffers()

        # Rollout collection
        for step in range(train_cfg["num_steps"]):
            if not env.agents:
                break

            # Sample actions (stochastic, for training)
            action_dict, log_probs, values = policy.get_actions_train(obs_dict, env.agents)

            # Fill missing agents (terminated early) with STAY
            for aid in env.possible_agents:
                if aid not in action_dict:
                    action_dict[aid] = 0

            next_obs_dict, rewards_dict, terminations, truncations, _ = env.step(action_dict)

            done_flag = all(terminations.get(a, True) or truncations.get(a, True)
                            for a in env.possible_agents)

            # Insert into each agent's buffer
            for aid in env.possible_agents:
                if aid in obs_dict:
                    ppo_agent = policy.agents[aid]
                    if ppo_agent.buffer.ptr < ppo_agent.buffer.num_steps:
                        ppo_agent.buffer.insert(
                            obs=obs_dict[aid],
                            action=action_dict.get(aid, 0),
                            log_prob=log_probs.get(aid, 0.0),
                            reward=rewards_dict.get(aid, 0.0),
                            value=values.get(aid, 0.0),
                            done=float(terminations.get(aid, False) or truncations.get(aid, False)),
                        )

            for aid, r in rewards_dict.items():
                episode_rewards[aid] += r

            last_obs = {aid: next_obs_dict.get(aid, obs_dict.get(aid, np.zeros(obs_shape)))
                        for aid in env.possible_agents}
            obs_dict = next_obs_dict

            if done_flag:
                break

        # GAE computation and PPO update
        policy.compute_all_gae(
            next_obs_dict=last_obs,
            next_done=float(not bool(env.agents)),  # 1.0 if episode ended
            gamma=train_cfg["gamma"],
            gae_lambda=train_cfg["gae_lambda"],
        )
        losses = policy.update_all()

        # Logging
        if episode % log_interval == 0:
            avg_reward = sum(episode_rewards.values()) / max(1, num_agents)
            mean_actor_loss  = sum(l[0] for l in losses.values()) / len(losses)
            mean_critic_loss = sum(l[1] for l in losses.values()) / len(losses)
            mean_entropy     = sum(l[2] for l in losses.values()) / len(losses)
            print(
                f"Ep {episode:5d}/{train_cfg['num_episodes']} | "
                f"AvgR: {avg_reward:7.3f} | "
                f"ActorL: {mean_actor_loss:.4f} | "
                f"CriticL: {mean_critic_loss:.4f} | "
                f"Entropy: {mean_entropy:.4f}"
            )

        # Checkpoint
        if episode % save_interval == 0:
            ckpt_dir = Path(log_cfg["save_dir"]) / f"ep_{episode}"
            policy.save(str(ckpt_dir))
            print(f"  [Saved] {ckpt_dir}")

        # Quick evaluation
        if episode % eval_interval == 0:
            policy.eval()
            eval_rewards = []
            eval_rescued = []
            for _ in range(eval_episodes):
                eval_obs, _ = env.reset()
                total_r = 0.0
                while env.agents:
                    eval_acts = policy(eval_obs, env)
                    eval_obs, eval_rw, _, _, _ = env.step(eval_acts)
                    total_r += sum(eval_rw.values())
                eval_rewards.append(total_r)
                eval_rescued.append(env.survivors_rescued_total)
            print(
                f"  [Eval ep {episode}] "
                f"MeanR={sum(eval_rewards)/len(eval_rewards):.2f} | "
                f"MeanRescued={sum(eval_rescued)/len(eval_rescued):.2f}/{env.initial_survivor_count}"
            )
            policy.train()

    # Final checkpoint
    final_dir = Path(log_cfg["save_dir"]) / "final"
    policy.save(str(final_dir))
    print(f"\n[PPO Training] Completed. Final checkpoint: {final_dir}")


if __name__ == "__main__":
    main()
