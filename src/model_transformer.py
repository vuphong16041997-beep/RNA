"""
Transformer-based model for RNA secondary structure prediction.

Category 3: Transformer and attention-based models.

Architecture inspired by:
- E2Efold (Transformer encoder for RNA contact prediction)
- RNAformer (MSA-aware transformer)
- Axial attention approaches for 2D prediction

The Transformer is particularly well-suited for RNA because:
1. Self-attention naturally models all-pairs relationships
2. No information bottleneck (unlike RNNs)
3. Positional encodings can capture sequence distance
4. Multi-head attention learns different pairing patterns simultaneously
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position."""

    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PairwisePositionalEncoding(nn.Module):
    """2D positional encoding for pair representation.

    Encodes both absolute positions and relative distance between
    positions i and j in the pair matrix.
    """

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        self.rel_pos_embed = nn.Embedding(2 * max_len + 1, d_model)
        self.max_len = max_len

    def forward(self, L: int, device):
        """Returns: (L, L, d_model) relative position encoding."""
        pos = torch.arange(L, device=device)
        rel_pos = pos.unsqueeze(0) - pos.unsqueeze(1)  # (L, L)
        rel_pos = rel_pos.clamp(-self.max_len, self.max_len) + self.max_len
        return self.rel_pos_embed(rel_pos)


class AxialAttentionBlock(nn.Module):
    """Axial attention: apply attention along rows and columns separately.

    This is more memory-efficient than full 2D attention (O(L^2) vs O(L^4))
    while still capturing long-range 2D dependencies in the contact map.

    Used in protein structure prediction (AlphaFold) and adapted here for RNA.
    """

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.row_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.col_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: (batch, L, L, d_model)"""
        B, L, _, D = x.shape

        # Row-wise attention
        x_rows = x.reshape(B * L, L, D)
        attn_out, _ = self.row_attn(x_rows, x_rows, x_rows)
        attn_out = attn_out.reshape(B, L, L, D)
        x = self.norm1(x + attn_out)

        # Column-wise attention
        x_cols = x.permute(0, 2, 1, 3).reshape(B * L, L, D)
        attn_out, _ = self.col_attn(x_cols, x_cols, x_cols)
        attn_out = attn_out.reshape(B, L, L, D).permute(0, 2, 1, 3)
        x = self.norm2(x + attn_out)

        # Feed-forward
        x = self.norm3(x + self.ffn(x))

        return x


class RNATransformer(nn.Module):
    """Full Transformer model for RNA secondary structure prediction.

    Pipeline:
        1. Embed sequence with 1D Transformer encoder
        2. Create 2D pair representation via outer product
        3. Refine with axial attention blocks (2D Transformer)
        4. Predict contact matrix

    This architecture processes information at two levels:
    - Sequence level (1D): captures local motifs, base identity context
    - Pair level (2D): captures interaction patterns between positions
    """

    def __init__(self, in_channels: int = 9, d_model: int = 64,
                 num_heads: int = 4, num_1d_layers: int = 4,
                 num_2d_layers: int = 4, dropout: float = 0.1,
                 max_length: int = 500):
        super().__init__()

        # 1D: Sequence Transformer
        self.seq_embed = nn.Linear(4, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_length, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.seq_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_1d_layers
        )

        # 1D → 2D: Outer product projection
        self.outer_proj = nn.Linear(d_model * 2, d_model)

        # 2D positional encoding
        self.pair_pos_enc = PairwisePositionalEncoding(d_model, max_len=max_length)

        # 2D: Axial Attention blocks
        self.axial_blocks = nn.ModuleList([
            AxialAttentionBlock(d_model, num_heads, dropout)
            for _ in range(num_2d_layers)
        ])

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, 9, L_pad, L_pad) pairwise features
            mask: (batch, L_pad, L_pad) valid position mask
        """
        B, C, H, W = x.shape

        # Extract per-position one-hot from diagonal
        seq_features = torch.diagonal(x[:, :4, :, :], dim1=2, dim2=3)  # (B, 4, L)
        seq_features = seq_features.permute(0, 2, 1)  # (B, L, 4)

        # Find actual sequence length from mask
        if mask is not None:
            lengths = mask[:, :, 0].sum(dim=1).long()  # (B,)
            max_L = lengths.max().item()
        else:
            max_L = H
            lengths = torch.full((B,), H, dtype=torch.long, device=x.device)

        # Truncate to actual max length for efficiency
        seq_features = seq_features[:, :max_L, :]

        # 1D: Embed and encode sequence
        h = self.seq_embed(seq_features)  # (B, L, d_model)
        h = self.pos_enc(h)
        h = self.seq_transformer(h)  # (B, L, d_model)

        # Create 2D pair representation
        L = h.shape[1]
        h_i = h.unsqueeze(2).expand(-1, -1, L, -1)  # (B, L, L, D)
        h_j = h.unsqueeze(1).expand(-1, L, -1, -1)  # (B, L, L, D)
        pair_repr = torch.cat([h_i, h_j], dim=-1)  # (B, L, L, 2D)
        pair_repr = self.outer_proj(pair_repr)  # (B, L, L, D)

        # Add 2D positional encoding
        pair_repr = pair_repr + self.pair_pos_enc(L, x.device)

        # 2D: Axial attention refinement
        for block in self.axial_blocks:
            pair_repr = block(pair_repr)

        # Predict contact matrix
        logits = self.output_head(pair_repr).squeeze(-1)  # (B, L, L)

        # Symmetrize
        logits = (logits + logits.transpose(-1, -2)) / 2

        # Pad back to original size
        if L < H:
            padded_logits = torch.zeros(B, H, W, device=x.device)
            padded_logits[:, :L, :L] = logits
            logits = padded_logits

        if mask is not None:
            logits = logits * mask

        return torch.sigmoid(logits)


class RNATransformerLight(nn.Module):
    """Lightweight Transformer for faster training on smaller datasets.

    Uses only 1D Transformer + simple 2D convolutions instead of
    full axial attention. Good balance of speed and accuracy.
    """

    def __init__(self, in_channels: int = 9, d_model: int = 64,
                 num_heads: int = 4, num_layers: int = 4,
                 num_conv_layers: int = 4, dropout: float = 0.1,
                 max_length: int = 500):
        super().__init__()

        self.seq_embed = nn.Linear(4, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_length, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 2D refinement with convolutions
        pair_input_dim = d_model * 2 + 1  # concat + rel_pos
        conv_layers = [
            nn.Conv2d(pair_input_dim, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
        ]
        for i in range(num_conv_layers):
            dilation = 2 ** (i % 3)
            conv_layers.extend([
                nn.Conv2d(d_model, d_model, kernel_size=3,
                          padding=dilation, dilation=dilation),
                nn.BatchNorm2d(d_model),
                nn.ReLU(),
            ])
        conv_layers.append(nn.Conv2d(d_model, 1, kernel_size=1))
        self.conv2d = nn.Sequential(*conv_layers)

    def forward(self, x, mask=None):
        B, C, H, W = x.shape

        # Extract and encode sequence
        seq_features = torch.diagonal(x[:, :4, :, :], dim1=2, dim2=3)
        seq_features = seq_features.permute(0, 2, 1)

        h = self.seq_embed(seq_features)
        h = self.pos_enc(h)
        h = self.transformer(h)

        # Outer product to 2D
        L = h.shape[1]
        h_i = h.unsqueeze(2).expand(-1, -1, L, -1)
        h_j = h.unsqueeze(1).expand(-1, L, -1, -1)
        pair_feat = torch.cat([h_i, h_j], dim=-1).permute(0, 3, 1, 2)

        # Relative position
        pos = torch.arange(L, dtype=torch.float32, device=x.device)
        rel_pos = torch.abs(pos.unsqueeze(0) - pos.unsqueeze(1)) / max(L, 1)
        rel_pos = rel_pos.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
        pair_feat = torch.cat([pair_feat, rel_pos], dim=1)

        # Pad if needed
        if L < H:
            pair_feat = F.pad(pair_feat, (0, H - L, 0, H - L))

        logits = self.conv2d(pair_feat[:, :, :H, :W]).squeeze(1)
        logits = (logits + logits.transpose(-1, -2)) / 2

        if mask is not None:
            logits = logits * mask

        return torch.sigmoid(logits)
