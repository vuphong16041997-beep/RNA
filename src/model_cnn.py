"""
CNN-based model for RNA secondary structure prediction.

Architecture: ResNet-style 2D CNN operating on L x L pair feature maps.

The model takes pairwise features (L, L, C) as input and predicts a
contact probability matrix (L, L) where each entry represents the
probability that base i pairs with base j.

Inspired by:
- SPOT-RNA (Singh et al., 2019)
- E2Efold (Chen et al., 2020)
- CNNFold (2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock2D(nn.Module):
    """Residual block with two conv2d layers and batch normalization."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class RNAContactCNN(nn.Module):
    """ResNet-style CNN for RNA contact prediction.

    Input: (batch, 9, L, L) pairwise features
    Output: (batch, L, L) contact probability matrix

    Architecture:
        1. Initial projection to hidden channels
        2. Stack of residual blocks with dilated convolutions
        3. Symmetrization layer
        4. Final 1x1 convolution to single channel
        5. Sigmoid activation
    """

    def __init__(self, in_channels: int = 9, hidden_channels: int = 64,
                 num_blocks: int = 8, dropout: float = 0.1):
        super().__init__()

        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )

        # Residual blocks with increasing dilation
        blocks = []
        for i in range(num_blocks):
            dilation = 2 ** (i % 4)  # 1, 2, 4, 8, 1, 2, 4, 8
            blocks.append(ResBlock2D(hidden_channels, dilation=dilation))
        self.res_blocks = nn.Sequential(*blocks)

        self.dropout = nn.Dropout2d(dropout)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, C, L, L) pairwise features
            mask: (batch, L, L) valid position mask

        Returns:
            (batch, L, L) contact probabilities
        """
        # Project input
        h = self.input_proj(x)

        # Residual blocks
        h = self.res_blocks(h)
        h = self.dropout(h)

        # Output
        logits = self.output_proj(h).squeeze(1)  # (batch, L, L)

        # Symmetrize: average upper and lower triangle
        logits = (logits + logits.transpose(-1, -2)) / 2

        # Apply mask
        if mask is not None:
            logits = logits * mask

        probs = torch.sigmoid(logits)

        return probs


class RNAContactCNNLarge(nn.Module):
    """Larger CNN with attention mechanism for longer sequences.

    Adds a self-attention layer between ResNet blocks for capturing
    long-range dependencies.
    """

    def __init__(self, in_channels: int = 9, hidden_channels: int = 96,
                 num_blocks: int = 12, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )

        # First block of ResNets
        blocks1 = []
        for i in range(num_blocks // 2):
            dilation = 2 ** (i % 4)
            blocks1.append(ResBlock2D(hidden_channels, dilation=dilation))
        self.res_blocks1 = nn.Sequential(*blocks1)

        # Row-wise and column-wise attention (axial attention)
        self.row_attn = nn.MultiheadAttention(
            hidden_channels, num_heads, dropout=dropout, batch_first=True
        )
        self.col_attn = nn.MultiheadAttention(
            hidden_channels, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_channels)

        # Second block of ResNets
        blocks2 = []
        for i in range(num_blocks // 2):
            dilation = 2 ** (i % 4)
            blocks2.append(ResBlock2D(hidden_channels, dilation=dilation))
        self.res_blocks2 = nn.Sequential(*blocks2)

        self.dropout = nn.Dropout2d(dropout)

        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
        )

    def forward(self, x, mask=None):
        B, C, H, W = x.shape

        h = self.input_proj(x)
        h = self.res_blocks1(h)

        # Axial attention: apply attention along rows and columns
        # Row attention
        h_perm = h.permute(0, 2, 3, 1)  # (B, H, W, C)
        h_flat = h_perm.reshape(B * H, W, -1)  # (B*H, W, C)
        h_attn, _ = self.row_attn(h_flat, h_flat, h_flat)
        h_attn = h_attn.reshape(B, H, W, -1)

        # Column attention
        h_perm2 = h.permute(0, 3, 2, 1)  # (B, W, H, C)
        h_flat2 = h_perm2.reshape(B * W, H, -1)
        h_attn2, _ = self.col_attn(h_flat2, h_flat2, h_flat2)
        h_attn2 = h_attn2.reshape(B, W, H, -1).permute(0, 2, 1, 3)

        # Combine
        h_combined = self.attn_norm((h_attn + h_attn2) / 2)
        h = h + h_combined.permute(0, 3, 1, 2)  # Residual

        h = self.res_blocks2(h)
        h = self.dropout(h)

        logits = self.output_proj(h).squeeze(1)
        logits = (logits + logits.transpose(-1, -2)) / 2

        if mask is not None:
            logits = logits * mask

        return torch.sigmoid(logits)
