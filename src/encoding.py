"""
Encoding utilities for RNA sequences and structures.

RNA secondary structure prediction is formulated as:
- Input: RNA sequence (string of A, C, G, U)
- Output: Contact/pairing matrix (L x L binary matrix)

The contact matrix M[i,j] = 1 if base i pairs with base j, 0 otherwise.
This formulation allows using 2D convolutions (CNN) or graph-based (GNN) approaches.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional

# Try importing torch_geometric for GNN support
try:
    from torch_geometric.data import Data as PyGData
    from torch_geometric.data import Batch as PyGBatch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


# ---------------------------------------------------------------------------
# Nucleotide encoding
# ---------------------------------------------------------------------------

BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "U": 3}
IDX_TO_BASE = {v: k for k, v in BASE_TO_IDX.items()}
NUM_BASES = 4


def encode_sequence(sequence: str) -> np.ndarray:
    """One-hot encode an RNA sequence.

    Args:
        sequence: RNA sequence string (A, C, G, U)

    Returns:
        (L, 4) numpy array with one-hot encoding
    """
    L = len(sequence)
    encoding = np.zeros((L, NUM_BASES), dtype=np.float32)
    for i, base in enumerate(sequence):
        if base in BASE_TO_IDX:
            encoding[i, BASE_TO_IDX[base]] = 1.0
        else:
            # Unknown base: uniform distribution
            encoding[i, :] = 0.25
    return encoding


def encode_pair_features(sequence: str) -> np.ndarray:
    """Create pairwise feature matrix from sequence.

    For each pair (i, j), encode the pair type (AA, AC, AG, AU, ...).
    This gives a (L, L, 16) feature tensor.
    Also includes positional encoding via relative distance.

    Returns:
        (L, L, 17) numpy array
    """
    L = len(sequence)
    seq_idx = np.array([BASE_TO_IDX.get(b, 0) for b in sequence])

    # Pair type encoding: 4*base_i + base_j gives 16 categories
    pair_type = np.zeros((L, L, 16), dtype=np.float32)
    for i in range(L):
        for j in range(L):
            pair_idx = seq_idx[i] * 4 + seq_idx[j]
            pair_type[i, j, pair_idx] = 1.0

    # Relative position encoding (normalized)
    pos = np.arange(L, dtype=np.float32)
    rel_pos = np.abs(pos[:, None] - pos[None, :]) / L
    rel_pos = rel_pos[:, :, np.newaxis]

    return np.concatenate([pair_type, rel_pos], axis=-1)


def encode_structure_as_matrix(sample) -> np.ndarray:
    """Convert structure to L x L contact matrix (ground truth).

    M[i,j] = 1 if base i pairs with base j.
    The matrix is symmetric.
    """
    L = sample.length
    matrix = np.zeros((L, L), dtype=np.float32)

    for i, j in sample.pair_map.items():
        matrix[i, j] = 1.0
        matrix[j, i] = 1.0

    return matrix


def structure_to_dotbracket(pair_map: dict, length: int) -> str:
    """Convert pair map back to dot-bracket notation."""
    result = ["."] * length
    for i, j in pair_map.items():
        if i < j:
            result[i] = "("
            result[j] = ")"
    return "".join(result)


def contact_matrix_to_pairs(matrix: np.ndarray, threshold: float = 0.5) -> dict:
    """Convert predicted contact matrix to pair map.

    Args:
        matrix: (L, L) predicted contact probabilities
        threshold: probability threshold for considering a pair

    Returns:
        pair_map dict
    """
    L = matrix.shape[0]
    # Symmetrize
    matrix = (matrix + matrix.T) / 2

    # Greedy pairing: take highest-confidence pairs first
    # Each base can pair with at most one other base
    pair_map = {}
    paired = set()

    # Get all candidate pairs with prob > threshold, sorted by confidence
    candidates = []
    for i in range(L):
        for j in range(i + 4, L):  # Minimum loop size of 3
            prob = matrix[i, j]
            if prob > threshold:
                candidates.append((prob, i, j))

    candidates.sort(reverse=True)

    for prob, i, j in candidates:
        if i not in paired and j not in paired:
            pair_map[i] = j
            pair_map[j] = i
            paired.add(i)
            paired.add(j)

    return pair_map


# ---------------------------------------------------------------------------
# PyTorch Datasets
# ---------------------------------------------------------------------------

class RNAContactDataset(Dataset):
    """Dataset for CNN-based models: predicts L x L contact matrix."""

    def __init__(self, samples: list, max_length: int = 200):
        self.samples = [s for s in samples if s.length <= max_length]
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        L = sample.length

        # Encode sequence: (L, 4)
        seq_enc = encode_sequence(sample.sequence)

        # Create outer product feature: (L, L, 8)
        # Concatenate one-hot[i] and one-hot[j] for each pair
        seq_i = np.repeat(seq_enc[:, np.newaxis, :], L, axis=1)  # (L, L, 4)
        seq_j = np.repeat(seq_enc[np.newaxis, :, :], L, axis=0)  # (L, L, 4)
        pair_feat = np.concatenate([seq_i, seq_j], axis=-1)  # (L, L, 8)

        # Add relative position
        pos = np.arange(L, dtype=np.float32)
        rel_pos = np.abs(pos[:, None] - pos[None, :]) / max(L, 1)
        pair_feat = np.concatenate([pair_feat, rel_pos[:, :, np.newaxis]], axis=-1)

        # Pad to max_length
        padded_feat = np.zeros((self.max_length, self.max_length, 9), dtype=np.float32)
        padded_feat[:L, :L, :] = pair_feat

        # Ground truth contact matrix
        target = encode_structure_as_matrix(sample)
        padded_target = np.zeros((self.max_length, self.max_length), dtype=np.float32)
        padded_target[:L, :L] = target

        # Mask for valid positions
        mask = np.zeros((self.max_length, self.max_length), dtype=np.float32)
        mask[:L, :L] = 1.0

        # Transpose features to (C, H, W) for conv2d
        padded_feat = padded_feat.transpose(2, 0, 1)

        return {
            "features": torch.from_numpy(padded_feat),
            "target": torch.from_numpy(padded_target),
            "mask": torch.from_numpy(mask),
            "length": L,
            "name": sample.name,
        }


class RNAGraphDataset(Dataset):
    """Dataset for GNN-based models.

    Each RNA sequence is represented as a graph:
    - Nodes: nucleotides (with one-hot + positional features)
    - Edges: backbone connections (i, i+1) + potential pairs within window
    - Labels: edge labels (1 = paired, 0 = not paired)
    """

    def __init__(self, samples: list, max_length: int = 200,
                 window_size: int = 0):
        """
        Args:
            samples: List of RNASample objects
            max_length: Filter sequences longer than this
            window_size: If > 0, only consider pairs within this distance.
                         If 0, consider all pairs (fully connected).
        """
        self.samples = [s for s in samples if s.length <= max_length]
        self.window_size = window_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if not HAS_PYG:
            raise ImportError("torch_geometric is required for GNN models. "
                              "Install with: pip install torch-geometric")

        sample = self.samples[idx]
        L = sample.length

        # Node features: one-hot encoding + normalized position
        seq_enc = encode_sequence(sample.sequence)  # (L, 4)
        pos_enc = np.arange(L, dtype=np.float32)[:, np.newaxis] / max(L, 1)
        node_features = np.concatenate([seq_enc, pos_enc], axis=1)  # (L, 5)

        # Build edges
        src, dst = [], []
        edge_labels = []

        # Backbone edges (i <-> i+1)
        for i in range(L - 1):
            src.extend([i, i + 1])
            dst.extend([i + 1, i])
            is_pair_fwd = 1.0 if sample.pair_map.get(i) == i + 1 else 0.0
            is_pair_bwd = 1.0 if sample.pair_map.get(i + 1) == i else 0.0
            edge_labels.extend([is_pair_fwd, is_pair_bwd])

        # Candidate pair edges
        for i in range(L):
            start_j = i + 4  # Minimum loop size
            end_j = min(L, i + self.window_size) if self.window_size > 0 else L
            for j in range(start_j, end_j):
                is_paired = 1.0 if sample.pair_map.get(i) == j else 0.0
                src.extend([i, j])
                dst.extend([j, i])
                edge_labels.extend([is_paired, is_paired])

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_labels = torch.tensor(edge_labels, dtype=torch.float32)
        x = torch.from_numpy(node_features)

        # Build full contact matrix as target (for evaluation)
        target = encode_structure_as_matrix(sample)

        data = PyGData(
            x=x,
            edge_index=edge_index,
            edge_labels=edge_labels,
            y=torch.from_numpy(target),
            seq_len=L,
            name=sample.name,
        )

        return data


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------

def get_cnn_dataloaders(samples: list,
                        batch_size: int = 8,
                        max_length: int = 200,
                        val_split: float = 0.1,
                        test_split: float = 0.1,
                        seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders for CNN model."""
    np.random.seed(seed)
    n = len(samples)
    indices = np.random.permutation(n)

    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_test - n_val

    train_samples = [samples[i] for i in indices[:n_train]]
    val_samples = [samples[i] for i in indices[n_train:n_train + n_val]]
    test_samples = [samples[i] for i in indices[n_train + n_val:]]

    train_ds = RNAContactDataset(train_samples, max_length)
    val_ds = RNAContactDataset(val_samples, max_length)
    test_ds = RNAContactDataset(test_samples, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_gnn_dataloaders(samples: list,
                        batch_size: int = 8,
                        max_length: int = 200,
                        window_size: int = 0,
                        val_split: float = 0.1,
                        test_split: float = 0.1,
                        seed: int = 42):
    """Create train/val/test DataLoaders for GNN model."""
    if not HAS_PYG:
        raise ImportError("torch_geometric required. Install with: pip install torch-geometric")

    from torch_geometric.loader import DataLoader as PyGLoader

    np.random.seed(seed)
    n = len(samples)
    indices = np.random.permutation(n)

    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_test - n_val

    train_samples = [samples[i] for i in indices[:n_train]]
    val_samples = [samples[i] for i in indices[n_train:n_train + n_val]]
    test_samples = [samples[i] for i in indices[n_train + n_val:]]

    train_ds = RNAGraphDataset(train_samples, max_length, window_size)
    val_ds = RNAGraphDataset(val_samples, max_length, window_size)
    test_ds = RNAGraphDataset(test_samples, max_length, window_size)

    train_loader = PyGLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = PyGLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = PyGLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
