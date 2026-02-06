# RNA Secondary Structure Prediction with Deep Learning & GNNs

Predict RNA secondary structure from sequence using CNN and Graph Neural Network models.

## Problem

Given an RNA sequence (e.g., `ACGCGAGAUUUGGGUGGGUAUGUCAGCUGCGGGUGUGGUG`), predict which bases pair with each other to form the secondary structure (stems, loops, bulges).

**Formulation:** The problem is cast as predicting an L x L **contact matrix** where entry (i, j) = 1 if base i pairs with base j.

## Models

| Model | Architecture | Strengths |
|-------|-------------|-----------|
| `cnn` | ResNet-style 2D CNN with dilated convolutions | Fast training, good for short-medium sequences |
| `cnn-large` | CNN + axial attention | Better long-range dependencies |
| `gnn-gcn` | Graph Convolutional Network | Natural graph representation of RNA |
| `gnn-gat` | Graph Attention Network | Attention-weighted message passing |
| `gnn-transformer` | Transformer-style GNN | Best of transformers + GNNs |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# For GNN models, also install:
pip install torch-geometric
```

### 2. Train on synthetic data (quick test)

```bash
python main.py train --model cnn --data synthetic --epochs 10
```

### 3. Train on real RNA data (curated sample)

```bash
python main.py train --model cnn --data sample --epochs 50
```

### 4. Predict your sequence

```bash
python main.py predict --model cnn \
    --sequence ACGCGAGAUUUGGGUGGGUAUGUCAGCUGCGGGUGUGGUG
```

### 5. Try GNN models

```bash
python main.py train --model gnn-gcn --data synthetic --epochs 20
python main.py predict --model gnn-gcn \
    --sequence ACGCGAGAUUUGGGUGGGUAUGUCAGCUGCGGGUGUGGUG
```

## Datasets

### Included

- **Synthetic**: Auto-generated RNA sequences with simple stem-loop structures (for debugging/testing)
- **Sample**: Curated set of well-characterized RNA structures (tRNA, rRNA, ribozymes)

### For serious training (download externally)

| Dataset | Size | Description | URL |
|---------|------|-------------|-----|
| **bpRNA-1m** | ~100k sequences | Standard benchmark | https://bprna.cgrb.oregonstate.edu/ |
| **ArchiveII** | ~4k sequences | High-quality curated set | https://rna.urmc.rochester.edu/pub/archiveII.tar.gz |
| **RNAcentral** | 45M+ sequences | Largest RNA database | https://rnacentral.org/ |
| **PDB (RNA)** | ~100s | Experimentally determined 3D structures | https://www.rcsb.org/ |

Place downloaded `.bpseq`, `.ct`, or `.dbn` files in the `data/` directory, then train with:

```bash
python main.py train --model cnn --data directory --data-dir data/bprna
```

## Project Structure

```
RNA/
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
├── src/
│   ├── dataset.py          # Data loading, parsing (.bpseq, .ct, .dbn, .sta)
│   ├── encoding.py         # Sequence encoding, contact matrices, DataLoaders
│   ├── model_cnn.py        # CNN architectures (ResNet, ResNet+Attention)
│   ├── model_gnn.py        # GNN architectures (GCN, GAT, TransformerGNN)
│   ├── train.py            # Training loops, metrics (precision/recall/F1)
│   ├── predict.py          # Inference and structure reconstruction
│   └── visualize.py        # Contact maps, arc diagrams, training plots
├── data/                   # Dataset storage
├── models/                 # Saved model checkpoints
└── notebooks/              # Jupyter notebooks for exploration
```

## How It Works

### CNN Approach
1. Encode the RNA sequence as pairwise features: for each (i, j) position, concatenate one-hot encodings of bases i and j plus relative distance
2. This creates an (L, L, 9) feature tensor (like an image)
3. Apply ResNet-style 2D convolutions with dilated kernels to capture patterns at multiple scales
4. Output a symmetric (L, L) probability matrix
5. Greedily extract base pairs (each base pairs with at most one other)

### GNN Approach
1. Represent the RNA as a graph: nucleotides = nodes, backbone + candidate pairs = edges
2. Node features: one-hot encoding + positional encoding
3. Apply message-passing layers (GCN/GAT/Transformer) to learn representations
4. Classify each edge as "paired" or "not paired"
5. Convert edge predictions back to a contact matrix

## Key References

- [SPOT-RNA](https://www.nature.com/articles/s41467-019-13395-9) - CNN ensemble for RNA structure (2019)
- [BPfold](https://www.nature.com/articles/s41467-025-60048-1) - DL + base pair motif energy (2025)
- [RhoFold+](https://www.nature.com/articles/s41592-024-02487-0) - RNA language model for 3D structure (2024)
- [GNN for RNA](https://www.biorxiv.org/content/10.1101/2024.10.11.617338v1) - GNN architecture exploration (2024)
- [Arnie](https://github.com/WaymentSteeleLab/arnie) - RNA structure prediction wrapper library
