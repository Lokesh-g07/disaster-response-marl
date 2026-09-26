import os
from typing import Dict, Any
from pathlib import Path

from rl.mappo import MAPPO
from rl.dqn.shared_dqn import SharedDQN, SharedDQNPolicy
from rl.ppo.independent_ppo import IndependentPPO, IndependentPPOPolicy
from baselines.rule_based import GreedyNearestPolicy, GreedyLargestZonePolicy
from evaluation.evaluator import make_mappo_greedy_policy
from simulation.envs.disaster_env import DisasterEnv

_policies: Dict[str, Any] = {}

def get_available_policies() -> list:
    return [
        {
            "id": "mappo",
            "name": "MAPPO (Dual-Branch)",
            "algorithm": "MAPPO",
            "checkpoint_available": os.path.exists("checkpoints/mappo_ep_350.pt"),
            "observation_type": "Dict(Visual, Vector)",
            "action_space": "Discrete(5)"
        },
        {
            "id": "shared_dqn",
            "name": "Shared DQN",
            "algorithm": "DQN",
            "checkpoint_available": os.path.exists("checkpoints/shared_dqn_ep_350.pt"),
            "observation_type": "Dict(Visual, Vector)",
            "action_space": "Discrete(5)"
        },
        {
            "id": "ppo",
            "name": "Independent PPO",
            "algorithm": "PPO",
            "checkpoint_available": os.path.exists("checkpoints/ppo"),
            "observation_type": "Visual(5,5,4)",
            "action_space": "Discrete(5)"
        },
        {
            "id": "greedy_nearest",
            "name": "Greedy-Nearest",
            "algorithm": "Rule-Based",
            "checkpoint_available": True,
            "observation_type": "Privileged Global State",
            "action_space": "Discrete(5)"
        },
        {
            "id": "greedy_largest_zone",
            "name": "Greedy-Largest-Zone",
            "algorithm": "Rule-Based",
            "checkpoint_available": True,
            "observation_type": "Privileged Global State",
            "action_space": "Discrete(5)"
        }
    ]

def get_policy(policy_id: str, env: DisasterEnv) -> Any:
    if policy_id in _policies:
        return _policies[policy_id]
        
    if policy_id == "greedy_nearest":
        policy = GreedyNearestPolicy()
        _policies[policy_id] = policy
        return policy
        
    if policy_id == "greedy_largest_zone":
        policy = GreedyLargestZonePolicy()
        _policies[policy_id] = policy
        return policy
        
    action_dim = env.action_space(env.possible_agents[0]).n
    gs = (env.height, env.width, 5)
    cvd = env.coord_vector_dim
    
    if policy_id == "mappo":
        ckpt = "checkpoints/mappo_ep_350.pt"
        if not os.path.exists(ckpt):
            raise ValueError(f"MAPPO checkpoint missing at {ckpt}")
        mappo = MAPPO(obs_shape=(4, 5, 5), global_state_shape=gs, action_dim=action_dim, coord_vector_dim=cvd, device="cpu")
        mappo.load(ckpt)
        mappo.actor.eval()
        policy = make_mappo_greedy_policy(mappo)
        _policies[policy_id] = policy
        return policy
        
    if policy_id == "shared_dqn":
        ckpt = "checkpoints/shared_dqn_ep_350.pt"
        if not os.path.exists(ckpt):
            raise ValueError(f"Shared DQN checkpoint missing at {ckpt}")
        dqn = SharedDQN(obs_shape=(4, 5, 5), coord_vector_dim=cvd, action_dim=action_dim, device="cpu")
        dqn.load(ckpt)
        policy = SharedDQNPolicy(dqn)
        _policies[policy_id] = policy
        return policy
        
    if policy_id == "ppo":
        ckpt_dir = "checkpoints/ppo"
        if not os.path.exists(ckpt_dir):
            raise ValueError(f"PPO checkpoint directory missing at {ckpt_dir}")
        ppo = IndependentPPO(agent_ids=env.possible_agents, action_dim=action_dim, device="cpu")
        ppo.load(ckpt_dir)
        for actor in ppo.actors.values():
            actor.eval()
        policy = IndependentPPOPolicy(ppo, deterministic=True)
        _policies[policy_id] = policy
        return policy
        
    raise ValueError(f"Unknown policy: {policy_id}")
