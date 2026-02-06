"""
Training pipeline for RNA secondary structure prediction models.

Supports both CNN and GNN model architectures.

Metrics:
- Precision: fraction of predicted pairs that are correct
- Recall (Sensitivity): fraction of true pairs that are predicted
- F1 score: harmonic mean of precision and recall
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Optional
from pathlib import Path


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred: np.ndarray, target: np.ndarray,
                    mask: Optional[np.ndarray] = None,
                    threshold: float = 0.5) -> Dict[str, float]:
    """Compute precision, recall, F1 for contact prediction.

    Args:
        pred: predicted contact probabilities (L, L)
        target: ground truth contact matrix (L, L)
        mask: valid position mask (L, L)
        threshold: probability threshold for positive prediction
    """
    if mask is not None:
        pred = pred * mask
        target = target * mask

    pred_binary = (pred > threshold).astype(np.float32)

    # Only count upper triangle to avoid double-counting
    L = pred.shape[0]
    triu_mask = np.triu(np.ones((L, L), dtype=np.float32), k=1)
    if mask is not None:
        triu_mask = triu_mask * mask

    pred_pairs = pred_binary * triu_mask
    true_pairs = target * triu_mask

    tp = np.sum(pred_pairs * true_pairs)
    fp = np.sum(pred_pairs * (1 - true_pairs))
    fn = np.sum((1 - pred_pairs) * true_pairs)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


# ---------------------------------------------------------------------------
# CNN Training
# ---------------------------------------------------------------------------

def train_cnn(model, train_loader, val_loader, device,
              epochs: int = 50, lr: float = 1e-3,
              save_dir: str = "models",
              patience: int = 10) -> Dict:
    """Train CNN model for contact prediction.

    Uses binary cross-entropy loss with class weighting to handle
    the severe class imbalance (most positions are unpaired).
    """
    os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5,
                                  factor=0.5)

    best_f1 = 0.0
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_losses = []

        for batch in train_loader:
            features = batch["features"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)

            pred = model(features, mask)

            # Weighted BCE: upweight positive pairs (paired bases are rare)
            pos_weight = (mask.sum() - target.sum()) / (target.sum() + 1e-8)
            pos_weight = torch.clamp(pos_weight, min=1.0, max=50.0)
            loss_fn = nn.BCELoss(reduction="none")
            loss = loss_fn(pred, target)
            # Apply weight to positive samples
            weight = torch.where(target > 0.5, pos_weight, torch.ones_like(target))
            loss = (loss * weight * mask).sum() / (mask.sum() + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # --- Validation ---
        model.eval()
        val_losses = []
        all_metrics = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                target = batch["target"].to(device)
                mask = batch["mask"].to(device)

                pred = model(features, mask)

                loss = nn.BCELoss(reduction="none")(pred, target)
                loss = (loss * mask).sum() / (mask.sum() + 1e-8)
                val_losses.append(loss.item())

                # Compute metrics per sample
                for i in range(pred.shape[0]):
                    L = batch["length"][i].item()
                    p = pred[i, :L, :L].cpu().numpy()
                    t = target[i, :L, :L].cpu().numpy()
                    metrics = compute_metrics(p, t)
                    all_metrics.append(metrics)

        avg_val_loss = np.mean(val_losses)
        avg_f1 = np.mean([m["f1"] for m in all_metrics]) if all_metrics else 0

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_f1"].append(avg_f1)

        scheduler.step(avg_f1)

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val F1: {avg_f1:.4f}")

        # Save best model
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1,
            }, os.path.join(save_dir, "best_cnn_model.pt"))
            print(f"  -> Saved best model (F1: {best_f1:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break

    return history


# ---------------------------------------------------------------------------
# GNN Training
# ---------------------------------------------------------------------------

def train_gnn(model, train_loader, val_loader, device,
              epochs: int = 50, lr: float = 1e-3,
              save_dir: str = "models",
              patience: int = 10) -> Dict:
    """Train GNN model for edge (base pair) prediction."""
    os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5,
                                  factor=0.5)

    best_f1 = 0.0
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_losses = []

        for batch in train_loader:
            batch = batch.to(device)
            edge_logits = model(batch)

            # Weighted BCE for edge classification
            edge_labels = batch.edge_labels
            pos_count = edge_labels.sum()
            neg_count = (edge_labels.numel() - pos_count)
            pos_weight = torch.clamp(neg_count / (pos_count + 1e-8),
                                     min=1.0, max=50.0)

            loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=pos_weight.unsqueeze(0)
            )
            loss = loss_fn(edge_logits, edge_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # --- Validation ---
        model.eval()
        val_losses = []
        all_f1s = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                edge_logits = model(batch)
                edge_labels = batch.edge_labels

                loss = nn.BCEWithLogitsLoss()(edge_logits, edge_labels)
                val_losses.append(loss.item())

                # Compute edge-level metrics
                probs = torch.sigmoid(edge_logits).cpu().numpy()
                labels = edge_labels.cpu().numpy()
                pred_binary = (probs > 0.5).astype(np.float32)

                tp = np.sum(pred_binary * labels)
                fp = np.sum(pred_binary * (1 - labels))
                fn = np.sum((1 - pred_binary) * labels)

                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                all_f1s.append(f1)

        avg_val_loss = np.mean(val_losses)
        avg_f1 = np.mean(all_f1s) if all_f1s else 0

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_f1"].append(avg_f1)

        scheduler.step(avg_f1)

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val F1: {avg_f1:.4f}")

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_f1": best_f1,
            }, os.path.join(save_dir, "best_gnn_model.pt"))
            print(f"  -> Saved best model (F1: {best_f1:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break

    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_cnn(model, test_loader, device) -> Dict:
    """Evaluate CNN model on test set."""
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)

            pred = model(features, mask)

            for i in range(pred.shape[0]):
                L = batch["length"][i].item()
                p = pred[i, :L, :L].cpu().numpy()
                t = target[i, :L, :L].cpu().numpy()
                metrics = compute_metrics(p, t)
                metrics["name"] = batch["name"][i]
                metrics["length"] = L
                all_metrics.append(metrics)

    # Aggregate
    avg_precision = np.mean([m["precision"] for m in all_metrics])
    avg_recall = np.mean([m["recall"] for m in all_metrics])
    avg_f1 = np.mean([m["f1"] for m in all_metrics])

    print(f"\nTest Results ({len(all_metrics)} samples):")
    print(f"  Precision: {avg_precision:.4f}")
    print(f"  Recall:    {avg_recall:.4f}")
    print(f"  F1 Score:  {avg_f1:.4f}")

    return {
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
        "per_sample": all_metrics,
    }
