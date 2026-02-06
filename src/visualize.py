"""
Visualization utilities for RNA secondary structure prediction.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


def plot_contact_matrix(probs: np.ndarray, target: np.ndarray = None,
                        sequence: str = None, save_path: str = None,
                        title: str = "Contact Matrix"):
    """Plot predicted contact matrix, optionally with ground truth overlay.

    Args:
        probs: (L, L) predicted probabilities
        target: (L, L) ground truth (optional)
        sequence: RNA sequence for axis labels
        save_path: path to save figure
        title: plot title
    """
    fig, axes = plt.subplots(1, 2 if target is not None else 1,
                             figsize=(12 if target is not None else 6, 5))

    if target is None:
        axes = [axes]

    # Predicted
    im = axes[0].imshow(probs, cmap="Blues", vmin=0, vmax=1,
                        aspect="equal", origin="upper")
    axes[0].set_title("Predicted")
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Position")
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    # Ground truth
    if target is not None:
        axes[1].imshow(target, cmap="Reds", vmin=0, vmax=1,
                       aspect="equal", origin="upper")
        axes[1].set_title("Ground Truth")
        axes[1].set_xlabel("Position")
        axes[1].set_ylabel("Position")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.close(fig)


def plot_training_history(history: dict, save_path: str = None):
    """Plot training loss and validation F1 over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()

    ax2.plot(epochs, history["val_f1"], label="Val F1", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Validation F1 Score")
    ax2.legend()

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training history to {save_path}")

    plt.close(fig)


def plot_arc_diagram(sequence: str, structure: str, save_path: str = None):
    """Plot an arc diagram showing base pairs.

    Each base pair is shown as an arc connecting the two paired positions.
    """
    from .dataset import RNASample
    sample = RNASample("plot", sequence, structure)

    L = len(sequence)
    fig, ax = plt.subplots(figsize=(max(10, L * 0.15), 4))

    # Draw sequence
    for i, base in enumerate(sequence):
        color = {"A": "#e74c3c", "U": "#3498db",
                 "G": "#2ecc71", "C": "#f39c12"}.get(base, "gray")
        ax.text(i, -0.3, base, ha="center", va="top",
                fontsize=max(4, min(8, 400 // L)),
                fontweight="bold", color=color)

    # Draw arcs for base pairs
    pairs = [(k, v) for k, v in sample.pair_map.items() if k < v]
    max_dist = max([b - a for a, b in pairs], default=1)

    for a, b in pairs:
        center = (a + b) / 2
        radius = (b - a) / 2
        height = radius * 0.6

        # Color by pair type
        pair_type = f"{sequence[a]}{sequence[b]}"
        if pair_type in ("AU", "UA"):
            color = "#e74c3c"
        elif pair_type in ("GC", "CG"):
            color = "#2ecc71"
        elif pair_type in ("GU", "UG"):
            color = "#9b59b6"
        else:
            color = "#95a5a6"

        arc = matplotlib.patches.Arc(
            (center, 0), b - a, height * 2,
            angle=0, theta1=0, theta2=180,
            color=color, linewidth=1.5, alpha=0.7
        )
        ax.add_patch(arc)

    ax.set_xlim(-1, L)
    ax.set_ylim(-1, max_dist * 0.4 + 1)
    ax.set_aspect("auto")
    ax.axis("off")
    ax.set_title(f"RNA Structure Arc Diagram ({L} nt, {len(pairs)} pairs)")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved arc diagram to {save_path}")

    plt.close(fig)
