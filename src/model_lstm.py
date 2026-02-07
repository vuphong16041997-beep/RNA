"""
LSTM/BiLSTM model for RNA secondary structure prediction.

Category 2: Pure deep learning, end-to-end folding (RNN-based).

Architecture:
    Sequence → BiLSTM encoder → pairwise outer product → contact matrix

BiLSTMs capture long-range sequential dependencies, which is important
because base pairing in RNA can occur between positions far apart in
the sequence. The bidirectional nature ensures both upstream and
downstream context is available at each position.

References:
- SPOT-RNA used BiLSTMs as part of its ensemble
- General RNN approaches for RNA structure surveyed in DL reviews
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNAContactLSTM(nn.Module):
    """BiLSTM-based model for RNA contact prediction.

    Pipeline:
        1. Embed one-hot sequence (L, 4) → (L, hidden_dim)
        2. BiLSTM encodes sequence context → (L, 2*hidden_dim)
        3. Outer product creates pairwise features → (L, L, 4*hidden_dim)
        4. 2D conv layers refine the contact map → (L, L, 1)
        5. Symmetrize + sigmoid → contact probabilities

    This combines sequential modeling (LSTM) with spatial modeling (CNN)
    for the final contact map refinement.
    """

    def __init__(self, in_channels: int = 9, hidden_dim: int = 64,
                 num_lstm_layers: int = 2, num_conv_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        # Sequence-level embedding (from one-hot, 4 features per position)
        self.seq_embed = nn.Linear(4, hidden_dim)

        # BiLSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
        )

        # Pairwise feature dimension: 2*hidden_dim (concat of i and j) * 2 (bidir)
        pair_dim = hidden_dim * 4

        # 2D CNN refinement of the contact map
        conv_layers = [
            nn.Conv2d(pair_dim + 1, hidden_dim, kernel_size=1),  # +1 for rel pos
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(num_conv_layers):
            conv_layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
            ])
        conv_layers.append(nn.Conv2d(hidden_dim, 1, kernel_size=1))
        self.conv_refine = nn.Sequential(*conv_layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, 9, L, L) pairwise features (CNN-style input)
               We extract the diagonal for sequence features.
            mask: (batch, L, L) valid position mask
        """
        B, C, H, W = x.shape

        # Extract per-position features from diagonal of pairwise input
        # x[:, :4, i, i] gives one-hot of position i
        seq_features = torch.diagonal(x[:, :4, :, :], dim1=2, dim2=3)  # (B, 4, L)
        seq_features = seq_features.permute(0, 2, 1)  # (B, L, 4)

        # Embed and encode with BiLSTM
        h = self.seq_embed(seq_features)  # (B, L, hidden_dim)
        h = self.dropout(h)
        lstm_out, _ = self.lstm(h)  # (B, L, 2*hidden_dim)

        # Create pairwise features via outer product
        # h_i concat h_j for all pairs
        L = lstm_out.shape[1]
        h_i = lstm_out.unsqueeze(2).expand(-1, -1, L, -1)  # (B, L, L, 2H)
        h_j = lstm_out.unsqueeze(1).expand(-1, L, -1, -1)  # (B, L, L, 2H)
        pair_feat = torch.cat([h_i, h_j], dim=-1)  # (B, L, L, 4H)
        pair_feat = pair_feat.permute(0, 3, 1, 2)  # (B, 4H, L, L)

        # Add relative position channel
        pos = torch.arange(L, dtype=torch.float32, device=x.device)
        rel_pos = torch.abs(pos.unsqueeze(0) - pos.unsqueeze(1)) / max(L, 1)
        rel_pos = rel_pos.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
        pair_feat = torch.cat([pair_feat, rel_pos], dim=1)

        # Pad to original size if needed
        if L < H:
            pad = H - L
            pair_feat = F.pad(pair_feat, (0, pad, 0, pad))

        # 2D CNN refinement
        logits = self.conv_refine(pair_feat[:, :, :H, :W]).squeeze(1)

        # Symmetrize
        logits = (logits + logits.transpose(-1, -2)) / 2

        if mask is not None:
            logits = logits * mask

        return torch.sigmoid(logits)


class RNAContactBiLSTMAttention(nn.Module):
    """BiLSTM with self-attention for RNA contact prediction.

    Adds a multi-head self-attention layer after the BiLSTM to capture
    very long-range dependencies that LSTMs may struggle with.
    This bridges Categories 2 and 3 (RNN + Attention).
    """

    def __init__(self, in_channels: int = 9, hidden_dim: int = 64,
                 num_lstm_layers: int = 2, num_heads: int = 4,
                 num_conv_layers: int = 4, dropout: float = 0.1):
        super().__init__()

        self.seq_embed = nn.Linear(4, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
        )

        # Project BiLSTM output to attention dimension
        self.lstm_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Pairwise features: hidden_dim * 2 (concat i,j after attention) + 1 (rel pos)
        pair_dim = hidden_dim * 2 + 1

        conv_layers = [
            nn.Conv2d(pair_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(num_conv_layers):
            conv_layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
            ])
        conv_layers.append(nn.Conv2d(hidden_dim, 1, kernel_size=1))
        self.conv_refine = nn.Sequential(*conv_layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, C, H, W = x.shape

        seq_features = torch.diagonal(x[:, :4, :, :], dim1=2, dim2=3)
        seq_features = seq_features.permute(0, 2, 1)

        h = self.seq_embed(seq_features)
        h = self.dropout(h)
        lstm_out, _ = self.lstm(h)

        # Project and apply self-attention
        h = self.lstm_proj(lstm_out)
        h_attn, _ = self.self_attn(h, h, h)
        h = self.attn_norm(h + h_attn)  # Residual + norm

        # Pairwise outer product
        L = h.shape[1]
        h_i = h.unsqueeze(2).expand(-1, -1, L, -1)
        h_j = h.unsqueeze(1).expand(-1, L, -1, -1)
        pair_feat = torch.cat([h_i, h_j], dim=-1)
        pair_feat = pair_feat.permute(0, 3, 1, 2)

        # Relative position
        pos = torch.arange(L, dtype=torch.float32, device=x.device)
        rel_pos = torch.abs(pos.unsqueeze(0) - pos.unsqueeze(1)) / max(L, 1)
        rel_pos = rel_pos.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
        pair_feat = torch.cat([pair_feat, rel_pos], dim=1)

        if L < H:
            pad = H - L
            pair_feat = F.pad(pair_feat, (0, pad, 0, pad))

        logits = self.conv_refine(pair_feat[:, :, :H, :W]).squeeze(1)
        logits = (logits + logits.transpose(-1, -2)) / 2

        if mask is not None:
            logits = logits * mask

        return torch.sigmoid(logits)
