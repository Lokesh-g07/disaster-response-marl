"""Rollout buffer for multi-agent PPO experience collection and GAE calculation."""

import torch
import numpy as np

class RolloutBuffer:
    def __init__(self, num_steps, num_agents, obs_shape, global_state_shape, device="cpu"):
        self.num_steps = num_steps
        self.num_agents = num_agents
        self.obs_shape = obs_shape
        self.global_state_shape = global_state_shape
        self.device = device
        
        self.obs = torch.zeros((num_steps, num_agents) + obs_shape, dtype=torch.float32).to(device)
        self.global_states = torch.zeros((num_steps, num_agents) + global_state_shape, dtype=torch.float32).to(device)
        self.actions = torch.zeros((num_steps, num_agents), dtype=torch.int64).to(device)
        self.log_probs = torch.zeros((num_steps, num_agents), dtype=torch.float32).to(device)
        self.rewards = torch.zeros((num_steps, num_agents), dtype=torch.float32).to(device)
        self.values = torch.zeros((num_steps, num_agents), dtype=torch.float32).to(device)
        self.dones = torch.zeros((num_steps, num_agents), dtype=torch.float32).to(device)
        
        self.returns = torch.zeros((num_steps, num_agents), dtype=torch.float32).to(device)
        self.advantages = torch.zeros((num_steps, num_agents), dtype=torch.float32).to(device)
        
        self.step = 0

    def insert(self, obs, global_states, actions, log_probs, rewards, values, dones):
        """Insert a single transition into the buffer."""
        if self.step >= self.num_steps:
            raise IndexError("RolloutBuffer is full")
            
        self.obs[self.step] = torch.tensor(obs, dtype=torch.float32).to(self.device)
        self.global_states[self.step] = torch.tensor(global_states, dtype=torch.float32).to(self.device)
        self.actions[self.step] = torch.tensor(actions, dtype=torch.int64).to(self.device)
        self.log_probs[self.step] = torch.tensor(log_probs, dtype=torch.float32).to(self.device)
        self.rewards[self.step] = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        self.values[self.step] = torch.tensor(values, dtype=torch.float32).to(self.device)
        self.dones[self.step] = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        self.step += 1

    def compute_returns_and_advantages(self, next_value, next_done, gamma=0.99, gae_lambda=0.95):
        """Compute Generalized Advantage Estimation (GAE)."""
        last_gae_lam = 0
        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_non_terminal = 1.0 - next_done
                next_val = next_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_val = self.values[step + 1]
            
            delta = self.rewards[step] + gamma * next_val * next_non_terminal - self.values[step]
            self.advantages[step] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            
        self.returns = self.advantages + self.values

    def get_generator(self, num_mini_batch):
        """Generate minibatches of experience for training."""
        batch_size = self.num_steps * self.num_agents
        mini_batch_size = batch_size // num_mini_batch
        
        indices = np.random.permutation(batch_size)
        
        # Flatten the buffer dimensions to (num_steps * num_agents, ...)
        obs_flat = self.obs.view(-1, *self.obs_shape)
        global_states_flat = self.global_states.view(-1, *self.global_state_shape)
        actions_flat = self.actions.view(-1)
        log_probs_flat = self.log_probs.view(-1)
        returns_flat = self.returns.view(-1)
        advantages_flat = self.advantages.view(-1)
        values_flat = self.values.view(-1)
        
        for i in range(num_mini_batch):
            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size
            mb_indices = indices[start:end]
            
            yield (
                obs_flat[mb_indices],
                global_states_flat[mb_indices],
                actions_flat[mb_indices],
                log_probs_flat[mb_indices],
                returns_flat[mb_indices],
                advantages_flat[mb_indices],
                values_flat[mb_indices],
            )

    def clear(self):
        """Reset buffer state."""
        self.step = 0
