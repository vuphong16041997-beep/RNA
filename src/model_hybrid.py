"""
Thermodynamic-aware hybrid model for RNA secondary structure prediction.

Category 1: Classical ML augmenting thermodynamics.

This model incorporates RNA folding physics as inductive biases:
1. Canonical base-pair scoring (AU, GC, GU Watson-Crick pairs)
2. Minimum loop size constraint (hairpin loops need >= 3 unpaired bases)
3. Stacking energy approximation (adjacent base pairs stabilize)
4. Learned corrections on top of thermodynamic priors

Inspired by:
- CONTRAfold/MXfold2: learn parameters combined with Turner energies
- CNN models that produce pseudo-free energies for DP algorithms

The key insight is: pure DL models must learn basic physics from scratch,
but hybrid models get it for free, letting the network focus on complex
patterns that physics alone can't capture (non-canonical pairs, tertiary
interactions, context-dependent effects).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Thermodynamic prior computation
# ---------------------------------------------------------------------------

# Canonical base pair scores (rough approximation of Turner energies)
# Real Turner parameters have hundreds of values; this is a simplified version
CANONICAL_PAIRS = {
    (0, 3): 2.0,  # A-U: moderate stability
    (3, 0): 2.0,  # U-A
    (1, 2): 3.0,  # C-G: strong stability (3 hydrogen bonds)
    (2, 1): 3.0,  # G-C
    (2, 3): 1.0,  # G-U: wobble pair, weaker
    (3, 2): 1.0,  # U-G
}

BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "U": 3}


def compute_thermodynamic_prior(sequence: str) -> np.ndarray:
    """Compute a physics-based prior for base pairing.

    Returns an (L, L) matrix where higher values indicate more
    thermodynamically favorable base pairs.

    Incorporates:
    1. Canonical pair compatibility (AU, GC, GU)
    2. Minimum loop size (>= 3 unpaired bases in hairpin)
    3. Approximate stacking bonus (adjacent pairs score higher)
    """
    L = len(sequence)
    seq_idx = [BASE_TO_IDX.get(b, 0) for b in sequence]
    prior = np.zeros((L, L), dtype=np.float32)

    for i in range(L):
        for j in range(i + 4, L):  # Minimum loop size of 3
            pair_key = (seq_idx[i], seq_idx[j])
            if pair_key in CANONICAL_PAIRS:
                score = CANONICAL_PAIRS[pair_key]

                # Distance penalty: very long-range pairs are less likely
                dist = j - i
                dist_factor = 1.0 / (1.0 + 0.01 * dist)

                prior[i, j] = score * dist_factor
                prior[j, i] = prior[i, j]

    # Add stacking bonus: if (i,j) and (i+1, j-1) are both canonical,
    # both get a bonus (stacked pairs are more stable)
    stacking = np.zeros_like(prior)
    for i in range(L - 1):
        for j in range(i + 5, L):
            if prior[i, j] > 0 and prior[i + 1, j - 1] > 0:
                stacking[i, j] += 1.0
                stacking[j, i] += 1.0
                stacking[i + 1, j - 1] += 1.0
                stacking[j - 1, i + 1] += 1.0

    prior = prior + stacking

    # Normalize to [0, 1]
    if prior.max() > 0:
        prior = prior / prior.max()

    return prior


def compute_thermodynamic_prior_batch(sequences: list, max_length: int) -> torch.Tensor:
    """Compute thermodynamic priors for a batch of sequences."""
    B = len(sequences)
    priors = torch.zeros(B, 1, max_length, max_length)

    for b, seq in enumerate(sequences):
        L = len(seq)
        prior = compute_thermodynamic_prior(seq)
        priors[b, 0, :L, :L] = torch.from_numpy(prior)

    return priors


# ---------------------------------------------------------------------------
# Hybrid model
# ---------------------------------------------------------------------------

class RNAHybridModel(nn.Module):
    """Thermodynamic-aware CNN for RNA structure prediction.

    The model receives both:
    1. Standard pairwise features (one-hot, relative position) — 9 channels
    2. Thermodynamic prior (canonical pair scores, stacking) — 1 channel

    The thermodynamic prior gives the network a strong initialization
    for where base pairs should be, and the learned CNN layers refine
    this based on patterns in the training data.

    Architecture:
        [seq features (9ch)] + [thermo prior (1ch)] → ResNet CNN → contact map
    """

    def __init__(self, in_channels: int = 10, hidden_channels: int = 64,
                 num_blocks: int = 8, dropout: float = 0.1):
        super().__init__()

        # Input: 9 (standard) + 1 (thermodynamic prior) = 10 channels
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )

        # Residual blocks with dilated convolutions
        blocks = []
        for i in range(num_blocks):
            dilation = 2 ** (i % 4)
            blocks.append(ResBlock2D(hidden_channels, dilation=dilation))
        self.res_blocks = nn.Sequential(*blocks)

        self.dropout = nn.Dropout2d(dropout)

        # Output: learned correction + thermodynamic base
        self.correction_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
        )

        # Learnable weight for combining prior and correction
        self.prior_weight = nn.Parameter(torch.tensor(1.0))
        self.correction_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, mask=None, thermo_prior=None):
        """
        Args:
            x: (batch, 9, L, L) pairwise features
            mask: (batch, L, L) valid position mask
            thermo_prior: (batch, 1, L, L) thermodynamic prior
                          If None, uses zeros (pure DL mode).
        """
        B, C, H, W = x.shape

        if thermo_prior is not None:
            x = torch.cat([x, thermo_prior], dim=1)
        else:
            zeros = torch.zeros(B, 1, H, W, device=x.device)
            x = torch.cat([x, zeros], dim=1)

        h = self.input_proj(x)
        h = self.res_blocks(h)
        h = self.dropout(h)

        correction = self.correction_head(h).squeeze(1)  # (B, L, L)

        # Combine thermodynamic prior with learned correction
        if thermo_prior is not None:
            prior = thermo_prior.squeeze(1)
            logits = (self.prior_weight * prior +
                      self.correction_weight * correction)
        else:
            logits = correction

        # Symmetrize
        logits = (logits + logits.transpose(-1, -2)) / 2

        if mask is not None:
            logits = logits * mask

        return torch.sigmoid(logits)


class ResBlock2D(nn.Module):
    """Residual block (same as in model_cnn.py)."""

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


# ---------------------------------------------------------------------------
# Constrained Nussinov-style dynamic programming decoder
# ---------------------------------------------------------------------------

def nussinov_decode(probs: np.ndarray, min_loop_size: int = 3,
                    threshold: float = 0.1) -> dict:
    """Nussinov-style DP to find optimal non-crossing structure.

    Instead of maximizing base pairs, maximizes the sum of predicted
    probabilities, subject to structural constraints:
    1. No crossing pairs (pseudoknot-free)
    2. Minimum loop size
    3. Each base pairs with at most one other

    This is Category 1 thinking: use DP algorithms with learned scores.

    Args:
        probs: (L, L) predicted pairing probabilities
        min_loop_size: minimum number of unpaired bases in hairpin
        threshold: minimum probability to consider a pair

    Returns:
        pair_map dict
    """
    L = probs.shape[0]

    # Symmetrize
    probs = (probs + probs.T) / 2

    # DP table: dp[i][j] = max total probability for subsequence i..j
    dp = np.zeros((L, L), dtype=np.float32)
    traceback = [[None] * L for _ in range(L)]

    # Fill DP table
    for span in range(min_loop_size + 1, L):
        for i in range(L - span):
            j = i + span

            # Option 1: i is unpaired
            best = dp[i + 1][j]
            traceback[i][j] = ("skip", i + 1, j)

            # Option 2: i pairs with some k (i+min_loop+1 <= k <= j)
            for k in range(i + min_loop_size + 1, j + 1):
                if probs[i][k] > threshold:
                    score = probs[i][k]
                    if k > i + 1:
                        score += dp[i + 1][k - 1]
                    if k < j:
                        score += dp[k + 1][j]

                    if score > best:
                        best = score
                        traceback[i][j] = ("pair", i, k)

            dp[i][j] = best

    # Traceback to recover pairs
    pair_map = {}

    def trace(i, j):
        if i >= j or traceback[i][j] is None:
            return
        action = traceback[i][j]
        if action[0] == "skip":
            trace(action[1], action[2])
        elif action[0] == "pair":
            a, b = action[1], action[2]
            pair_map[a] = b
            pair_map[b] = a
            if b > a + 1:
                trace(a + 1, b - 1)
            if b < j:
                trace(b + 1, j)

    trace(0, L - 1)
    return pair_map
