"""Actor network for MAPPO (Decentralized execution)."""

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorNetwork(nn.Module):
    """
    Decentralized Actor Network.
    Takes local egocentric observation (5x5x4) and outputs action probabilities.
    """
    def __init__(self, obs_shape=(4, 5, 5), action_dim=5, hidden_dim=64):
        super(ActorNetwork, self).__init__()
        
        self.obs_shape = obs_shape
        channels = obs_shape[0]
        
        # Convolutional feature extractor for the local grid
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate CNN output size (32 * 5 * 5 = 800)
        cnn_out_dim = 32 * obs_shape[1] * obs_shape[2]
        
        # MLP policy head
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(cnn_out_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )
        
        self.action_head = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)

    def forward(self, obs):
        """
        Forward pass to get categorical action distribution.
        obs shape: (batch_size, 5, 5, 4) or (batch_size, 4, 5, 5) depending on convention.
        We will assume input is (batch_size, H, W, C) and permute to (batch_size, C, H, W)
        """
        # If input is (batch_size, 5, 5, 4)
        if obs.dim() == 4 and obs.shape[-1] == self.obs_shape[0]:
            obs = obs.permute(0, 3, 1, 2)
            
        x = self.cnn(obs)
        x = self.mlp(x)
        logits = self.action_head(x)
        
        return Categorical(logits=logits)
