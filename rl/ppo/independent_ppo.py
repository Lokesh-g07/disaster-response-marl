"""
Independent PPO agent for CrisisRL.

Architecture contrast with MAPPO
----------------------------------
MAPPO (Centralized Training, Decentralized Execution):
  - ONE shared ActorNetwork for all agents (parameter sharing)
  - ONE CriticNetwork that observes the full global state (H, W, 5)
  - Training: centralized critic + shared actor
  - Execution: each agent uses the shared actor with its local obs

Independent PPO (this file):
  - SEPARATE ActorNetwork PER AGENT (no parameter sharing)
  - SEPARATE local CriticNetwork PER AGENT
  - The local critic uses only the same local 5x5x4 observation the actor sees
  - NO global state is given to the critic -- this is the key difference
  - Training: fully independent per-agent
  - Execution: each agent uses its own actor independently

Information fairness
--------------------
Both MAPPO actor and PPO actor receive:
  obs: np.ndarray of shape (5, 5, 4) -- local egocentric 5x5 window
    Channel 0: Fire
    Channel 1: Survivor
    Channel 2: Wall / Out-of-bounds
    Channel 3: Exit

MAPPO critic additionally receives:
  global_state: np.ndarray of shape (H, W, 5) -- full grid truth

PPO critic receives:
  local obs ONLY -- same (5, 5, 4) window as the actor

This means MAPPO has a training information advantage (centralized critic).
Both policies are EQUALLY informed at inference time (only local obs used for actions).

Task / Zone context
--------------------
The local observation (5x5x4) does NOT encode explicit task/zone assignments.
Any zone context that rule-based baselines use via privileged env.grid access
is NOT available to PPO -- this is intentional and preserves information fairness.
PPO must learn survivor-seeking behavior implicitly from reward signal alone,
just like MAPPO does.

The conceptual pipeline:
  agent -> local obs -> PPO actor -> action -> movement -> rescue

Task/zone context is NOT injected into the observation because:
  1. MAPPO does not receive it either.
  2. Adding it would break the information-equivalence between PPO and MAPPO.
  3. The rule-based baselines are labeled as privileged-state heuristics
     precisely because they CAN use this information.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Layer initialisation helper (same as used in MAPPO actor/critic)
# ---------------------------------------------------------------------------

def _layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Orthogonal weight init with constant bias -- matches MAPPO convention."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# ---------------------------------------------------------------------------
# PPO Actor network (identical architecture to MAPPO ActorNetwork)
# ---------------------------------------------------------------------------

class PPOActor(nn.Module):
    """
    Per-agent PPO actor network.

    Takes local egocentric observation of shape (5, 5, 4) and outputs
    a Categorical action distribution over 5 actions
    (STAY, UP, DOWN, LEFT, RIGHT).

    Architecture is intentionally identical to MAPPO's ActorNetwork so that
    any performance difference is attributable to the training setup
    (independent vs cooperative) rather than network capacity.

    Input:  obs (batch, 5, 5, 4) -- HWC, auto-permuted to (batch, 4, 5, 5)
    Output: torch.distributions.Categorical
    """

    OBS_CHANNELS = 4
    OBS_H = 5
    OBS_W = 5

    def __init__(self, action_dim: int = 5, hidden_dim: int = 64) -> None:
        super().__init__()
        # CNN feature extractor -- 4 input channels, same layout as MAPPO ActorNetwork
        self.cnn = nn.Sequential(
            _layer_init(nn.Conv2d(self.OBS_CHANNELS, 16, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            _layer_init(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        cnn_out = 32 * self.OBS_H * self.OBS_W  # 800

        self.mlp = nn.Sequential(
            _layer_init(nn.Linear(cnn_out, hidden_dim)),
            nn.ReLU(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )
        self.action_head = _layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)

    def forward(self, obs: torch.Tensor) -> Categorical:
        """
        Args:
            obs: Tensor of shape (batch, 5, 5, 4) or (batch, 4, 5, 5).
        Returns:
            Categorical distribution over action_dim actions.
        """
        if obs.dim() == 4 and obs.shape[-1] == self.OBS_CHANNELS:
            obs = obs.permute(0, 3, 1, 2)
        x = self.cnn(obs)
        x = self.mlp(x)
        return Categorical(logits=self.action_head(x))


# ---------------------------------------------------------------------------
# PPO local Critic network
# ---------------------------------------------------------------------------

class PPOCritic(nn.Module):
    """
    Per-agent LOCAL critic network.

    CRITICAL DISTINCTION from MAPPO:
      - MAPPO critic uses full global state (H, W, 5)
      - PPO critic uses only the same local 5x5x4 obs the actor sees

    This means Independent PPO has NO privileged global-state access during
    training, unlike MAPPO's centralized critic.

    Input:  obs (batch, 5, 5, 4) -- same as actor input
    Output: scalar value estimate V(local_obs)
    """

    OBS_CHANNELS = 4
    OBS_H = 5
    OBS_W = 5

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            _layer_init(nn.Conv2d(self.OBS_CHANNELS, 16, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            _layer_init(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        cnn_out = 32 * self.OBS_H * self.OBS_W  # 800

        self.mlp = nn.Sequential(
            _layer_init(nn.Linear(cnn_out, hidden_dim)),
            nn.ReLU(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )
        self.value_head = _layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: Tensor of shape (batch, 5, 5, 4) or (batch, 4, 5, 5).
        Returns:
            Value tensor of shape (batch, 1).
        """
        if obs.dim() == 4 and obs.shape[-1] == self.OBS_CHANNELS:
            obs = obs.permute(0, 3, 1, 2)
        x = self.cnn(obs)
        x = self.mlp(x)
        return self.value_head(x)


# ---------------------------------------------------------------------------
# Per-agent rollout buffer (local obs only -- no global state)
# ---------------------------------------------------------------------------

class PPORolloutBuffer:
    """
    Per-agent rollout buffer for Independent PPO.

    Unlike the MAPPO RolloutBuffer which stores multi-agent batches and
    global_states, this buffer is for ONE agent and uses only local obs.

    This makes each agent's training completely independent.

    Args:
        num_steps: Rollout horizon (steps per update).
        obs_shape: Local observation shape, e.g. (5, 5, 4).
        device: Torch device string.
    """

    def __init__(
        self,
        num_steps: int,
        obs_shape: Tuple[int, ...] = (5, 5, 4),
        device: str = "cpu",
    ) -> None:
        self.num_steps = num_steps
        self.obs_shape = obs_shape
        self.device = device

        self.obs        = torch.zeros((num_steps,) + obs_shape, dtype=torch.float32, device=device)
        self.actions    = torch.zeros(num_steps, dtype=torch.int64, device=device)
        self.log_probs  = torch.zeros(num_steps, dtype=torch.float32, device=device)
        self.rewards    = torch.zeros(num_steps, dtype=torch.float32, device=device)
        self.values     = torch.zeros(num_steps, dtype=torch.float32, device=device)
        self.dones      = torch.zeros(num_steps, dtype=torch.float32, device=device)
        self.returns    = torch.zeros(num_steps, dtype=torch.float32, device=device)
        self.advantages = torch.zeros(num_steps, dtype=torch.float32, device=device)

        self.ptr = 0  # write pointer

    def insert(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: float,
    ) -> None:
        """Insert a single step transition for one agent."""
        if self.ptr >= self.num_steps:
            raise IndexError("PPORolloutBuffer is full; call clear() before next rollout.")
        self.obs[self.ptr]       = torch.tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[self.ptr]   = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr]   = reward
        self.values[self.ptr]    = value
        self.dones[self.ptr]     = done
        self.ptr += 1

    def compute_returns_and_advantages(
        self,
        next_value: float,
        next_done: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """
        Compute Generalized Advantage Estimation (GAE) for one agent.

        Identical GAE formula to MAPPO's RolloutBuffer.compute_returns_and_advantages,
        but operates on a single agent's trajectory.

        Args:
            next_value: Bootstrap value for the step after the last stored step.
            next_done: 1.0 if episode ended at the step after the last stored step.
            gamma: Discount factor.
            gae_lambda: GAE smoothing coefficient.
        """
        last_gae = 0.0
        for step in reversed(range(self.ptr)):
            if step == self.ptr - 1:
                non_terminal = 1.0 - next_done
                nv = next_value
            else:
                non_terminal = 1.0 - self.dones[step + 1].item()
                nv = self.values[step + 1].item()

            delta = self.rewards[step].item() + gamma * nv * non_terminal - self.values[step].item()
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            self.advantages[step] = last_gae

        self.returns = self.advantages + self.values

    def get_minibatches(self, num_mini_batch: int):
        """
        Yield shuffled minibatches of (obs, actions, log_probs, returns, advantages, values).

        Only yields from the filled portion (0 .. ptr) of the buffer.
        """
        size = self.ptr
        mini_batch_size = max(1, size // num_mini_batch)
        indices = np.random.permutation(size)

        obs_flat      = self.obs[:size]
        actions_flat  = self.actions[:size]
        lp_flat       = self.log_probs[:size]
        ret_flat      = self.returns[:size]
        adv_flat      = self.advantages[:size]
        val_flat      = self.values[:size]

        for i in range(num_mini_batch):
            mb_idx = indices[i * mini_batch_size: (i + 1) * mini_batch_size]
            if len(mb_idx) == 0:
                continue
            yield (
                obs_flat[mb_idx],
                actions_flat[mb_idx],
                lp_flat[mb_idx],
                ret_flat[mb_idx],
                adv_flat[mb_idx],
                val_flat[mb_idx],
            )

    def clear(self) -> None:
        """Reset write pointer (does not zero tensors for speed)."""
        self.ptr = 0


# ---------------------------------------------------------------------------
# Independent PPO agent (one per rescue agent)
# ---------------------------------------------------------------------------

class IndependentPPO:
    """
    Single-agent Independent PPO.

    One IndependentPPO instance is created PER rescue agent.
    Agents do NOT share parameters and do NOT access each other's
    observations, values, or gradients.

    Default hyperparameters
    -----------------------
    actor_lr      : 3e-4  (matches MAPPO default)
    critic_lr     : 1e-3  (matches MAPPO default)
    clip_param    : 0.2   (matches MAPPO default)
    ppo_epoch     : 10    (matches MAPPO default)
    num_mini_batch: 4     (PPO default; MAPPO uses 1)
    entropy_coef  : 0.01  (matches MAPPO default)
    value_loss_coef: 0.5  (matches MAPPO default)
    max_grad_norm : 0.5   (matches MAPPO default)
    gamma         : 0.99
    gae_lambda    : 0.95
    num_steps     : 64    (rollout horizon per update)

    Args:
        agent_id: String identifier, e.g. "rescue_0".
        action_dim: Number of discrete actions (5 for CrisisRL).
        **kwargs: Override any default hyperparameter.
    """

    def __init__(
        self,
        agent_id: str,
        action_dim: int = 5,
        num_steps: int = 64,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        clip_param: float = 0.2,
        ppo_epoch: int = 10,
        num_mini_batch: int = 4,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
    ) -> None:
        self.agent_id = agent_id
        self.action_dim = action_dim
        self.num_steps = num_steps
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        # Separate actor and critic -- no parameter sharing across agents
        self.actor  = PPOActor(action_dim=action_dim).to(device)
        self.critic = PPOCritic().to(device)

        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.buffer = PPORolloutBuffer(num_steps=num_steps, device=device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        """
        Sample an action from the agent's policy.

        Args:
            obs: Local observation array of shape (5, 5, 4).

        Returns:
            (action, log_prob, value) as Python scalars.
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist  = self.actor(obs_t)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(obs_t).squeeze()
        return int(action.item()), float(log_prob.item()), float(value.item())

    @torch.no_grad()
    def get_value(self, obs: np.ndarray) -> float:
        """Bootstrap value for a local observation."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.critic(obs_t).squeeze().item())

    @torch.no_grad()
    def get_deterministic_action(self, obs: np.ndarray) -> int:
        """Greedy (argmax) action for evaluation."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.actor(obs_t).logits.argmax(dim=-1).item())

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def update(self) -> Tuple[float, float, float]:
        """
        Perform PPO update using the current buffer.

        Returns:
            (mean_actor_loss, mean_critic_loss, mean_entropy) over all epochs.
        """
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        update_count = 0

        for _ in range(self.ppo_epoch):
            for batch in self.buffer.get_minibatches(self.num_mini_batch):
                obs_b, acts_b, old_lp_b, ret_b, adv_b, old_val_b = batch

                # Normalize advantages per minibatch -- guard against single-element batch
                if adv_b.numel() > 1:
                    adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
                else:
                    adv_b = adv_b - adv_b.mean()

                # Actor forward
                dist = self.actor(obs_b)
                new_lp = dist.log_prob(acts_b)
                entropy = dist.entropy().mean()

                # Critic forward (local obs only)
                new_val = self.critic(obs_b).squeeze(-1)

                # Clipped surrogate actor loss
                ratio = torch.exp(new_lp - old_lp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean()

                # Clipped value loss (matches MAPPO convention)
                val_clipped = old_val_b + (new_val - old_val_b).clamp(-self.clip_param, self.clip_param)
                vl1 = (new_val  - ret_b).pow(2)
                vl2 = (val_clipped - ret_b).pow(2)
                critic_loss = 0.5 * torch.max(vl1, vl2).mean()

                # Combined loss
                loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                total_actor_loss  += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy     += entropy.item()
                update_count      += 1

        n = max(1, update_count)
        return total_actor_loss / n, total_critic_loss / n, total_entropy / n

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save actor, critic, and optimizer states."""
        torch.save({
            "agent_id": self.agent_id,
            "actor_state_dict":  self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict":  self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        """Load actor, critic, and optimizer states."""
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer_state_dict"])


# ---------------------------------------------------------------------------
# Multi-agent wrapper: manages one IndependentPPO per env agent
# ---------------------------------------------------------------------------

class IndependentPPOPolicy:
    """
    Manages a collection of independent PPO agents, one per env rescue agent.

    This is NOT a centralized controller -- it is simply a convenience wrapper
    that matches the PolicyFn interface (obs_dict, env) -> action_dict so that
    it can be passed directly to Evaluator.evaluate() alongside MAPPO and the
    rule-based baselines.

    At inference time, each agent uses its OWN actor independently.

    Args:
        agent_ids: List of agent identifiers from env.possible_agents.
        action_dim: Number of discrete actions.
        device: Torch device.
        **ppo_kwargs: Forwarded to each IndependentPPO constructor.
    """

    def __init__(
        self,
        agent_ids: List[str],
        action_dim: int = 5,
        device: str = "cpu",
        **ppo_kwargs,
    ) -> None:
        self.agents: Dict[str, IndependentPPO] = {
            aid: IndependentPPO(
                agent_id=aid,
                action_dim=action_dim,
                device=device,
                **ppo_kwargs,
            )
            for aid in agent_ids
        }
        self.device = device

    # ------------------------------------------------------------------
    # PolicyFn interface -- called by Evaluator
    # ------------------------------------------------------------------

    def __call__(
        self,
        obs_dict: Dict[str, np.ndarray],
        env,
    ) -> Dict[str, int]:
        """
        Deterministic (greedy) inference for evaluation.
        Each agent independently selects the argmax action.
        """
        actions = {}
        for agent_id in env.agents:
            if agent_id in obs_dict and agent_id in self.agents:
                obs = obs_dict[agent_id]
                actions[agent_id] = self.agents[agent_id].get_deterministic_action(obs)
        return actions

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def get_actions_train(
        self,
        obs_dict: Dict[str, np.ndarray],
        active_agents: List[str],
    ) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, float]]:
        """
        Sample stochastic actions for training step.

        Returns:
            actions: agent_id -> action int
            log_probs: agent_id -> log prob float
            values: agent_id -> value float
        """
        actions, log_probs, values = {}, {}, {}
        for aid in active_agents:
            if aid in self.agents and aid in obs_dict:
                a, lp, v = self.agents[aid].get_action(obs_dict[aid])
                actions[aid], log_probs[aid], values[aid] = a, lp, v
        return actions, log_probs, values

    def get_values_for(
        self,
        obs_dict: Dict[str, np.ndarray],
        agent_ids: List[str],
    ) -> Dict[str, float]:
        """Bootstrap values for specified agents."""
        return {
            aid: self.agents[aid].get_value(obs_dict[aid])
            for aid in agent_ids
            if aid in self.agents and aid in obs_dict
        }

    def update_all(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Run PPO update for every agent. Returns per-agent loss tuple.
        """
        return {aid: ppo.update() for aid, ppo in self.agents.items()}

    def compute_all_gae(
        self,
        next_obs_dict: Dict[str, np.ndarray],
        next_done: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute returns/advantages for all agent buffers."""
        for aid, ppo in self.agents.items():
            nv = ppo.get_value(next_obs_dict.get(aid, np.zeros((5, 5, 4))))
            ppo.buffer.compute_returns_and_advantages(nv, next_done, gamma, gae_lambda)

    def clear_all_buffers(self) -> None:
        for ppo in self.agents.values():
            ppo.buffer.clear()

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Save each agent's weights to <directory>/ppo_<agent_id>.pt."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        for aid, ppo in self.agents.items():
            ppo.save(str(Path(directory) / f"ppo_{aid}.pt"))

    def load(self, directory: str) -> None:
        """Load each agent's weights from <directory>/ppo_<agent_id>.pt."""
        for aid, ppo in self.agents.items():
            ppo.load(str(Path(directory) / f"ppo_{aid}.pt"))

    def eval(self) -> None:
        """Set all networks to eval mode."""
        for ppo in self.agents.values():
            ppo.actor.eval()
            ppo.critic.eval()

    def train(self) -> None:
        """Set all networks to train mode."""
        for ppo in self.agents.values():
            ppo.actor.train()
            ppo.critic.train()
