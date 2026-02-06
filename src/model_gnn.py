"""
GNN-based model for RNA secondary structure prediction.

Architecture: Message-passing GNN that operates on RNA sequence graphs.

Graph representation:
- Nodes: nucleotides with features (one-hot encoding + position)
- Edges: backbone connections + candidate pairing edges
- Task: predict which edges are actual base pairs

This approach is inspired by recent work:
- "Exploring the Efficiency of Deep Graph Neural Networks for RNA" (2024)
- BPfold (2025)

GNNs are natural for RNA structure because:
1. RNA structure IS a graph (bases = nodes, pairs = edges)
2. Message passing captures local and global structural patterns
3. Variable-length sequences are handled naturally (no padding needed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import (
        GCNConv, GATConv, GINConv, TransformerConv,
        global_mean_pool, BatchNorm
    )
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


def _check_pyg():
    if not HAS_PYG:
        raise ImportError(
            "torch_geometric is required for GNN models.\n"
            "Install with: pip install torch-geometric\n"
            "See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
        )


# ---------------------------------------------------------------------------
# GCN-based model
# ---------------------------------------------------------------------------

class RNAStructureGCN(nn.Module):
    """Graph Convolutional Network for RNA base pair prediction.

    Simple but effective: uses GCN message passing layers to learn
    node representations, then predicts edge labels (paired or not).
    """

    def __init__(self, node_features: int = 5, hidden_dim: int = 64,
                 num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        _check_pyg()

        self.input_proj = nn.Linear(node_features, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(BatchNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Edge classifier: takes concatenated node features of both endpoints
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Initial projection
        x = F.relu(self.input_proj(x))

        # Message passing
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new  # Residual connection

        # Edge prediction
        src, dst = edge_index
        edge_features = torch.cat([x[src], x[dst]], dim=-1)
        edge_logits = self.edge_classifier(edge_features).squeeze(-1)

        return edge_logits


# ---------------------------------------------------------------------------
# GAT-based model (Graph Attention Network)
# ---------------------------------------------------------------------------

class RNAStructureGAT(nn.Module):
    """Graph Attention Network for RNA structure prediction.

    GAT uses attention mechanisms to weigh neighbor contributions,
    which is particularly useful for RNA where not all neighbors
    are equally important for determining structure.
    """

    def __init__(self, node_features: int = 5, hidden_dim: int = 64,
                 num_layers: int = 6, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        _check_pyg()

        self.input_proj = nn.Linear(node_features, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATConv(
                hidden_dim, hidden_dim // num_heads,
                heads=num_heads, dropout=dropout, concat=True
            ))
            self.norms.append(BatchNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Edge classifier with attention-enhanced features
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.input_proj(x))

        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new

        src, dst = edge_index
        edge_features = torch.cat([x[src], x[dst]], dim=-1)
        edge_logits = self.edge_classifier(edge_features).squeeze(-1)

        return edge_logits


# ---------------------------------------------------------------------------
# Transformer-based GNN
# ---------------------------------------------------------------------------

class RNAStructureTransformerGNN(nn.Module):
    """Transformer-style GNN for RNA structure prediction.

    Uses TransformerConv which applies multi-head attention in
    the message passing step - combining the benefits of
    Transformers and GNNs.
    """

    def __init__(self, node_features: int = 5, hidden_dim: int = 64,
                 num_layers: int = 6, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        _check_pyg()

        self.input_proj = nn.Linear(node_features, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(TransformerConv(
                hidden_dim, hidden_dim // num_heads,
                heads=num_heads, dropout=dropout, concat=True
            ))
            self.norms.append(BatchNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.input_proj(x))

        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new

        src, dst = edge_index
        edge_features = torch.cat([x[src], x[dst]], dim=-1)
        edge_logits = self.edge_classifier(edge_features).squeeze(-1)

        return edge_logits


# ---------------------------------------------------------------------------
# Helper to reconstruct contact matrix from GNN predictions
# ---------------------------------------------------------------------------

def gnn_predictions_to_contact_matrix(data, edge_logits, threshold=0.5):
    """Convert GNN edge predictions back to L x L contact matrix.

    Args:
        data: PyG Data object with edge_index and seq_len
        edge_logits: predicted edge logits from GNN
        threshold: probability threshold

    Returns:
        (L, L) numpy contact matrix
    """
    import numpy as np

    L = data.seq_len if isinstance(data.seq_len, int) else data.seq_len.item()
    probs = torch.sigmoid(edge_logits).detach().cpu().numpy()

    contact_matrix = np.zeros((L, L), dtype=np.float32)
    edge_index = data.edge_index.cpu().numpy()

    for k in range(edge_index.shape[1]):
        i, j = edge_index[0, k], edge_index[1, k]
        if i < L and j < L:
            contact_matrix[i, j] = max(contact_matrix[i, j], probs[k])

    # Symmetrize
    contact_matrix = (contact_matrix + contact_matrix.T) / 2

    return contact_matrix
