"""MAPPO Algorithm Implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
from .actor import ActorNetwork
from .critic import CriticNetwork

class MAPPO:
    def __init__(self,
                 obs_shape,
                 global_state_shape,
                 action_dim,
                 actor_lr=3e-4,
                 critic_lr=1e-3,
                 clip_param=0.2,
                 ppo_epoch=10,
                 num_mini_batch=1,
                 entropy_coef=0.01,
                 value_loss_coef=0.5,
                 max_grad_norm=0.5,
                 device="cpu"):
        
        self.device = device
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        
        self.actor = ActorNetwork(obs_shape, action_dim).to(self.device)
        self.critic = CriticNetwork(global_channels=global_state_shape[-1]).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
    def get_actions(self, obs, global_states):
        """Get actions and values for a set of observations."""
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            global_states_tensor = torch.tensor(global_states, dtype=torch.float32).to(self.device)
            
            action_dists = self.actor(obs_tensor)
            actions = action_dists.sample()
            log_probs = action_dists.log_prob(actions)
            
            values = self.critic(global_states_tensor)
            
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.squeeze(-1).cpu().numpy()
        
    def get_values(self, global_states):
        """Get value estimates for a set of global states."""
        with torch.no_grad():
            global_states_tensor = torch.tensor(global_states, dtype=torch.float32).to(self.device)
            values = self.critic(global_states_tensor)
        return values.squeeze(-1).cpu().numpy()
        
    def update(self, buffer):
        """Perform PPO update on actor and critic networks."""
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy_loss = 0
        
        for e in range(self.ppo_epoch):
            data_generator = buffer.get_generator(self.num_mini_batch)
            
            for sample in data_generator:
                obs_batch, global_states_batch, actions_batch, \
                old_log_probs_batch, returns_batch, advantages_batch, old_values_batch = sample
                
                # Normalize advantages
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
                
                # Evaluate actions
                action_dists = self.actor(obs_batch)
                log_probs = action_dists.log_prob(actions_batch)
                entropy = action_dists.entropy().mean()
                
                values = self.critic(global_states_batch).squeeze(-1)
                
                # Actor loss (Clipped surrogate objective)
                ratio = torch.exp(log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss (MSE with optional clipping)
                value_pred_clipped = old_values_batch + (values - old_values_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - returns_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
                critic_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                
                # Total loss
                loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy
                
                # Backpropagate
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy_loss += entropy.item()
                
        num_updates = self.ppo_epoch * self.num_mini_batch
        return (total_actor_loss / num_updates, 
                total_critic_loss / num_updates, 
                total_entropy_loss / num_updates)
    
    def save(self, path):
        """Save model weights."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
        
    def load(self, path):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
