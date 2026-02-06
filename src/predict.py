"""
Prediction module for RNA secondary structure.

Given a trained model and an RNA sequence, predicts the secondary
structure in dot-bracket notation and as a contact matrix.
"""

import numpy as np
import torch
from typing import Dict, Optional

from .encoding import (
    encode_sequence, contact_matrix_to_pairs, structure_to_dotbracket
)
from .dataset import RNASample


def predict_cnn(model, sequence: str, device: str = "cpu",
                max_length: int = 200, threshold: float = 0.5) -> Dict:
    """Predict secondary structure using CNN model.

    Args:
        model: trained RNAContactCNN model
        sequence: RNA sequence string
        device: torch device
        max_length: maximum sequence length model was trained on
        threshold: probability threshold for base pairs

    Returns:
        Dict with 'structure', 'pair_map', 'contact_matrix', 'probabilities'
    """
    sequence = sequence.upper().replace("T", "U")
    L = len(sequence)

    if L > max_length:
        raise ValueError(f"Sequence length {L} exceeds max_length {max_length}. "
                         f"Use a model trained with larger max_length.")

    model.eval()
    model = model.to(device)

    # Encode sequence
    seq_enc = encode_sequence(sequence)

    # Create pairwise features (same as in RNAContactDataset)
    seq_i = np.repeat(seq_enc[:, np.newaxis, :], L, axis=1)
    seq_j = np.repeat(seq_enc[np.newaxis, :, :], L, axis=0)
    pair_feat = np.concatenate([seq_i, seq_j], axis=-1)

    pos = np.arange(L, dtype=np.float32)
    rel_pos = np.abs(pos[:, None] - pos[None, :]) / max(L, 1)
    pair_feat = np.concatenate([pair_feat, rel_pos[:, :, np.newaxis]], axis=-1)

    # Pad
    padded_feat = np.zeros((max_length, max_length, 9), dtype=np.float32)
    padded_feat[:L, :L, :] = pair_feat

    mask = np.zeros((max_length, max_length), dtype=np.float32)
    mask[:L, :L] = 1.0

    # To tensor
    feat_tensor = torch.from_numpy(padded_feat.transpose(2, 0, 1)).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(feat_tensor, mask_tensor)

    # Extract predictions for actual sequence
    probs = pred[0, :L, :L].cpu().numpy()

    # Convert to structure
    pair_map = contact_matrix_to_pairs(probs, threshold=threshold)
    structure = structure_to_dotbracket(pair_map, L)

    return {
        "sequence": sequence,
        "structure": structure,
        "pair_map": pair_map,
        "contact_matrix": (probs > threshold).astype(np.float32),
        "probabilities": probs,
    }


def predict_gnn(model, sequence: str, device: str = "cpu",
                window_size: int = 0, threshold: float = 0.5) -> Dict:
    """Predict secondary structure using GNN model.

    Args:
        model: trained GNN model
        sequence: RNA sequence string
        device: torch device
        window_size: same window_size used during training
        threshold: probability threshold for base pairs
    """
    try:
        from torch_geometric.data import Data as PyGData
    except ImportError:
        raise ImportError("torch_geometric required for GNN prediction")

    from .model_gnn import gnn_predictions_to_contact_matrix

    sequence = sequence.upper().replace("T", "U")
    L = len(sequence)

    model.eval()
    model = model.to(device)

    # Build graph for this sequence
    seq_enc = encode_sequence(sequence)
    pos_enc = np.arange(L, dtype=np.float32)[:, np.newaxis] / max(L, 1)
    node_features = np.concatenate([seq_enc, pos_enc], axis=1)

    # Build edges (backbone + candidate pairs)
    src, dst = [], []
    for i in range(L - 1):
        src.extend([i, i + 1])
        dst.extend([i + 1, i])

    for i in range(L):
        start_j = i + 4
        end_j = min(L, i + window_size) if window_size > 0 else L
        for j in range(start_j, end_j):
            src.extend([i, j])
            dst.extend([j, i])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x = torch.from_numpy(node_features)

    data = PyGData(x=x, edge_index=edge_index, seq_len=L)
    data = data.to(device)

    with torch.no_grad():
        edge_logits = model(data)

    # Convert to contact matrix
    contact_matrix = gnn_predictions_to_contact_matrix(data, edge_logits, threshold)

    pair_map = contact_matrix_to_pairs(contact_matrix, threshold=threshold)
    structure = structure_to_dotbracket(pair_map, L)

    return {
        "sequence": sequence,
        "structure": structure,
        "pair_map": pair_map,
        "contact_matrix": (contact_matrix > threshold).astype(np.float32),
        "probabilities": contact_matrix,
    }


def visualize_prediction(result: Dict, max_width: int = 80):
    """Print prediction results in a readable format."""
    seq = result["sequence"]
    struct = result["structure"]
    n_pairs = len([k for k, v in result["pair_map"].items() if k < v])

    print("=" * max_width)
    print("RNA Secondary Structure Prediction")
    print("=" * max_width)
    print(f"Length: {len(seq)} nucleotides")
    print(f"Predicted base pairs: {n_pairs}")
    print()

    # Print sequence and structure in chunks
    chunk_size = max_width
    for start in range(0, len(seq), chunk_size):
        end = min(start + chunk_size, len(seq))
        # Position ruler
        ruler = ""
        for pos in range(start, end):
            if pos % 10 == 0:
                ruler += str(pos % 100 // 10)
            elif pos % 5 == 0:
                ruler += "+"
            else:
                ruler += " "
        print(f"  {ruler}")
        print(f"  {seq[start:end]}")
        print(f"  {struct[start:end]}")
        print()

    # List base pairs
    print("Base pairs:")
    pairs = sorted([(k, v) for k, v in result["pair_map"].items() if k < v])
    for i, (a, b) in enumerate(pairs):
        pair_type = f"{seq[a]}-{seq[b]}"
        canonical = pair_type in ("A-U", "U-A", "G-C", "C-G", "G-U", "U-G")
        marker = "" if canonical else " (non-canonical)"
        print(f"  {a:3d}-{b:3d}  {pair_type}{marker}")

    print("=" * max_width)
