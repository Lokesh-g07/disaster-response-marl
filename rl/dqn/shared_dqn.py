"""
Shared DQN Baseline for CrisisRL.

Architecture
------------
Parameter-Shared DQN: All agents use the SAME Q-network.
The network is a dual-branch fusion network taking the upgraded Dict observation:
- visual: (5,5,4) -> CNN
- vector: (coord_dim) -> MLP
The features are concatenated and fed into an MLP to output Q-values for 5 actions.

Information Fairness
--------------------
DQN receives the same Dict observation as MAPPO's decentralized actor.
It does NOT receive the global state `(H,W,5)` (which MAPPO uses for its centralized critic).
It does NOT use privileged oracle-level zone assignments (like Greedy baselines).

Multi-Agent Handling
--------------------
All agents produce transitions `(obs, action, reward, next_obs, done)`.
These transitions are pushed into a SINGLE, shared experience replay buffer.
A single shared DQN network is optimized from minibatch samples from this buffer.
"""

import os
import random
import copy
from typing import Dict, List, Tuple, Any, Optional
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DQNNetwork(nn.Module):
    """
    Dual-branch Q-Network for Shared DQN.
    Takes Dict observation {"visual": (5,5,4), "vector": (coord_dim,)}
    and outputs Q-values for discrete actions.
    """
    def __init__(
        self,
        obs_shape: Tuple[int, ...] = (4, 5, 5),
        coord_vector_dim: int = 7,
        action_dim: int = 5,
        hidden_dim: int = 64,
        vec_hidden_dim: int = 32,
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.coord_vector_dim = coord_vector_dim
        channels = obs_shape[0]

        # --- Branch 1: CNN for local visual grid ---
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            nn.Flatten()
        )
        cnn_out_dim = 32 * obs_shape[1] * obs_shape[2]

        self.visual_compress = nn.Sequential(
            layer_init(nn.Linear(cnn_out_dim, hidden_dim)),
            nn.ReLU(),
        )

        # --- Branch 2: MLP for coordination vector ---
        self.vec_mlp = nn.Sequential(
            layer_init(nn.Linear(coord_vector_dim, vec_hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(vec_hidden_dim, vec_hidden_dim)),
            nn.ReLU(),
        )

        # --- Fusion: concat visual + vec -> Q-values ---
        fusion_in = hidden_dim + vec_hidden_dim
        self.fusion_mlp = nn.Sequential(
            layer_init(nn.Linear(fusion_in, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)  # small std for output
        )

    def forward(self, visual: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        # Permute HWC to CHW if needed
        if visual.shape[-1] == self.obs_shape[0]:
            visual = visual.permute(0, 3, 1, 2)
        
        vis_feat = self.cnn(visual)
        vis_feat = self.visual_compress(vis_feat)

        vec_feat = self.vec_mlp(vector)

        fused = torch.cat([vis_feat, vec_feat], dim=-1)
        q_values = self.fusion_mlp(fused)
        return q_values


class ReplayBuffer:
    def __init__(self, capacity: int, device: str):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, visual, vector, action, reward, next_visual, next_vector, done):
        self.buffer.append((visual, vector, action, reward, next_visual, next_vector, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        visuals, vectors, actions, rewards, next_visuals, next_vectors, dones = zip(*batch)

        visuals = torch.tensor(np.array(visuals), dtype=torch.float32, device=self.device)
        vectors = torch.tensor(np.array(vectors), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_visuals = torch.tensor(np.array(next_visuals), dtype=torch.float32, device=self.device)
        next_vectors = torch.tensor(np.array(next_vectors), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        return visuals, vectors, actions, rewards, next_visuals, next_vectors, dones

    def __len__(self):
        return len(self.buffer)


class SharedDQN:
    """
    Parameter-Shared DQN. Manages network, target network, buffer, and updates.
    """
    def __init__(
        self,
        obs_shape: Tuple[int, ...] = (4, 5, 5),
        coord_vector_dim: int = 7,
        action_dim: int = 5,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        device: str = "cpu"
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device

        self.q_network = DQNNetwork(obs_shape, coord_vector_dim, action_dim).to(device)
        self.target_network = DQNNetwork(obs_shape, coord_vector_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity, device)

        self.steps_done = 0

    def get_action(self, obs_dict: Dict[str, np.ndarray], explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        visual = torch.tensor(obs_dict["visual"], dtype=torch.float32, device=self.device).unsqueeze(0)
        vector = torch.tensor(obs_dict["vector"], dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.q_network(visual, vector)
            return q_values.argmax(dim=1).item()

    def update(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        visuals, vectors, actions, rewards, next_visuals, next_vectors, dones = self.buffer.sample(self.batch_size)

        # Q(s, a)
        q_values = self.q_network(visuals, vectors).gather(1, actions)

        # max_a Q'(s', a)
        with torch.no_grad():
            max_next_q = self.target_network(next_visuals, next_vectors).max(1)[0].unsqueeze(1)
            expected_q_values = rewards + (self.gamma * max_next_q * (1.0 - dones))

        # Huber Loss
        loss = F.smooth_l1_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.steps_done += 1

        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save(self, filepath: str):
        torch.save({
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps_done": self.steps_done,
        }, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.steps_done = checkpoint["steps_done"]


class SharedDQNPolicy:
    """
    Wrapper mapping the DQN instance to the Evaluator interface.
    """
    def __init__(self, dqn: SharedDQN):
        self.dqn = dqn
        self.dqn.q_network.eval()
        self.dqn.epsilon = 0.0  # Greedy for evaluation

    def __call__(self, obs_dict: Dict[str, Any], env) -> Dict[str, int]:
        actions = {}
        for agent_id in env.agents:
            if agent_id in obs_dict:
                actions[agent_id] = self.dqn.get_action(obs_dict[agent_id], explore=False)
        return actions

    def reset(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "shared_dqn"

