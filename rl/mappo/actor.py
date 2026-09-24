"""Actor network for MAPPO (Decentralized execution).

Architecture v2 -- Dual-Branch Fusion
--------------------------------------
The actor accepts a Dict observation with two components:

  "visual":  (batch, 5, 5, 4)  -- local egocentric grid
  "vector":  (batch, coord_dim) -- coordination vector

Processing pipeline:

  visual (5,5,4) --> CNN --> visual_features (64)
                                                \
                                                 --> concat --> fusion_mlp --> action_head --> 5 actions
                                                /
  vector (coord_dim) --> vec_mlp --> vec_features (32)

This fusion architecture lets the actor combine:
  - Spatial patterns from the local grid (fire avoidance, nearby survivors)
  - Coordination signals (own position, teammate locations, nearest target)

CTDE Compliance
---------------
The actor receives ONLY the Dict observation at execution time.
It NEVER receives the global state -- that is for the centralized critic only.

Backward Compatibility
-----------------------
Old checkpoints saved with the v1 ActorNetwork (single CNN, obs_shape=(4,5,5))
are NOT compatible with this architecture.  They must be retrained.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from typing import Dict, Optional, Tuple, Union


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorNetwork(nn.Module):
    """
    Decentralized Actor Network with dual-branch observation fusion.

    Takes Dict observation {"visual": (5,5,4), "vector": (coord_dim,)}
    and outputs Categorical action distribution over 5 actions.

    Args:
        obs_shape: Visual observation shape in CHW format, default (4, 5, 5).
        coord_vector_dim: Dimension of the coordination vector.
        action_dim: Number of discrete actions (5).
        hidden_dim: Hidden layer width (64).
        vec_hidden_dim: Vector branch hidden width (32).
    """
    def __init__(
        self,
        obs_shape: Tuple[int, ...] = (4, 5, 5),
        coord_vector_dim: int = 7,
        action_dim: int = 5,
        hidden_dim: int = 64,
        vec_hidden_dim: int = 32,
    ):
        super(ActorNetwork, self).__init__()

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
        cnn_out_dim = 32 * obs_shape[1] * obs_shape[2]  # 32 * 5 * 5 = 800

        # Compress CNN features
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

        # --- Fusion: concat visual_features + vec_features -> policy head ---
        fusion_in = hidden_dim + vec_hidden_dim  # 64 + 32 = 96
        self.fusion_mlp = nn.Sequential(
            layer_init(nn.Linear(fusion_in, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )

        self.action_head = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)

    def forward(
        self,
        obs: Union[Dict[str, torch.Tensor], torch.Tensor],
        coord_vector: Optional[torch.Tensor] = None,
    ) -> Categorical:
        """
        Forward pass to get categorical action distribution.

        Accepts either:
          1. obs = {"visual": (batch,5,5,4), "vector": (batch, coord_dim)}
          2. obs = (batch,5,5,4) visual tensor, coord_vector = (batch, coord_dim)

        The first form is used when observations come directly from the env.
        The second form is used during training when visual and vector are
        stored separately in the buffer.
        """
        if isinstance(obs, dict):
            visual = obs["visual"]
            vec = obs["vector"]
        else:
            visual = obs
            vec = coord_vector

        # Ensure visual is (batch, C, H, W)
        if visual.dim() == 4 and visual.shape[-1] == self.obs_shape[0]:
            visual = visual.permute(0, 3, 1, 2)

        # CNN branch
        cnn_out = self.cnn(visual)
        visual_features = self.visual_compress(cnn_out)

        # Vector branch
        vec_features = self.vec_mlp(vec)

        # Fusion
        fused = torch.cat([visual_features, vec_features], dim=-1)
        x = self.fusion_mlp(fused)
        logits = self.action_head(x)

        return Categorical(logits=logits)
