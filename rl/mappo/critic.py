"""Critic network for MAPPO (Centralized training)."""

import torch
import torch.nn as nn
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CriticNetwork(nn.Module):
    """
    Centralized Critic Network.
    Takes the full global state (H, W, 5) and outputs state value V(s).
    """
    def __init__(self, global_channels=5, hidden_dim=64):
        super(CriticNetwork, self).__init__()
        
        # CNN to process arbitrary size global state grid
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(global_channels, 32, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Convert variable spatial dims to fixed 4x4
            nn.Flatten()
        )
        
        # Output of AdaptiveAvgPool2d(4,4) * 64 channels = 1024
        cnn_out_dim = 64 * 4 * 4
        
        # MLP value head
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(cnn_out_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )
        
        self.value_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward(self, global_state):
        """
        Forward pass to estimate V(s).
        global_state shape: (batch_size, H, W, C)
        """
        # Ensure it's (batch_size, C, H, W)
        if global_state.dim() == 4 and global_state.shape[-1] == 5:
            global_state = global_state.permute(0, 3, 1, 2)
            
        x = self.cnn(global_state)
        x = self.mlp(x)
        value = self.value_head(x)
        
        return value
