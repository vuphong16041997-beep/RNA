#!/usr/bin/env python3
"""
Benchmark and compare all model architectures on the same dataset.

This script:
1. Loads a dataset (synthetic, sample, or your bpRNA directory)
2. Trains all CNN-style models on the same train/val/test split
3. Evaluates each model and compares F1 scores
4. Predicts your specific sequence with each model
5. Generates a comparison report

Usage:
    # Quick comparison with synthetic data
    python benchmark.py --data synthetic --epochs 10

    # Compare on your bpRNA data
    python benchmark.py --data directory --data-dir data/bpRNA_data --epochs 30

    # Predict your sequence with all trained models
    python benchmark.py --predict-only --sequence ACGCGAGAUUUGGGUGGGUAUGUCAGCUGCGGGUGUGGUG
"""

import argparse
import os
import sys
import json
import time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import prepare_dataset
from src.encoding import get_cnn_dataloaders
from src.model_cnn import RNAContactCNN, RNAContactCNNLarge
from src.model_lstm import RNAContactLSTM, RNAContactBiLSTMAttention
from src.model_transformer import RNATransformer, RNATransformerLight
from src.model_hybrid import RNAHybridModel, nussinov_decode
from src.train import train_cnn, evaluate_cnn, compute_metrics
from src.predict import predict_cnn, visualize_prediction
from src.encoding import structure_to_dotbracket
from src.visualize import plot_contact_matrix


# Model registry: name -> (class, kwargs_override)
MODEL_REGISTRY = {
    "cnn": (RNAContactCNN, {"in_channels": 9}),
    "cnn-large": (RNAContactCNNLarge, {"in_channels": 9}),
    "lstm": (RNAContactLSTM, {"in_channels": 9}),
    "lstm-attn": (RNAContactBiLSTMAttention, {"in_channels": 9}),
    "transformer-light": (RNATransformerLight, {"in_channels": 9}),
    "hybrid": (RNAHybridModel, {"in_channels": 10}),
}

# Category labels
CATEGORIES = {
    "cnn": "Cat.2 Pure DL (CNN)",
    "cnn-large": "Cat.2 Pure DL (CNN+Attn)",
    "lstm": "Cat.2 Pure DL (BiLSTM)",
    "lstm-attn": "Cat.2+3 (BiLSTM+Attn)",
    "transformer-light": "Cat.3 Transformer",
    "hybrid": "Cat.1 Thermo+DL",
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(name, hidden_dim, num_layers, max_length, dropout):
    """Build a model by name."""
    cls, overrides = MODEL_REGISTRY[name]
    in_ch = overrides.get("in_channels", 9)

    if name == "cnn":
        return cls(in_channels=in_ch, hidden_channels=hidden_dim,
                   num_blocks=num_layers, dropout=dropout)
    elif name == "cnn-large":
        return cls(in_channels=in_ch, hidden_channels=hidden_dim,
                   num_blocks=num_layers, dropout=dropout)
    elif name == "lstm":
        return cls(in_channels=in_ch, hidden_dim=hidden_dim,
                   num_lstm_layers=max(2, num_layers // 4),
                   num_conv_layers=max(2, num_layers // 2),
                   dropout=dropout)
    elif name == "lstm-attn":
        return cls(in_channels=in_ch, hidden_dim=hidden_dim,
                   num_lstm_layers=max(2, num_layers // 4),
                   num_conv_layers=max(2, num_layers // 2),
                   dropout=dropout)
    elif name == "transformer-light":
        return cls(in_channels=in_ch, d_model=hidden_dim, num_heads=4,
                   num_layers=max(2, num_layers // 2),
                   num_conv_layers=max(2, num_layers // 2),
                   dropout=dropout, max_length=max_length)
    elif name == "hybrid":
        return cls(in_channels=in_ch, hidden_channels=hidden_dim,
                   num_blocks=num_layers, dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {name}")


def run_benchmark(args):
    """Train and evaluate all models."""
    device = get_device()
    print(f"Device: {device}")
    print("=" * 70)

    # Load data
    samples = prepare_dataset(
        mode=args.data, data_dir=args.data_dir,
        n_synthetic=args.n_synthetic, max_length=args.max_length,
    )
    print(f"Total samples: {len(samples)}")
    print(f"Length range: {min(s.length for s in samples)}-{max(s.length for s in samples)}")
    print(f"Avg pairs per sample: {np.mean([len(s.pair_map)//2 for s in samples]):.1f}")
    print("=" * 70)

    # Create shared train/val/test split
    train_loader, val_loader, test_loader = get_cnn_dataloaders(
        samples, batch_size=args.batch_size, max_length=args.max_length,
        val_split=0.1, test_split=0.1, seed=42,
    )
    print(f"Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)} | "
          f"Test: {len(test_loader.dataset)}")
    print("=" * 70)

    results = {}
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    models_to_run = args.models.split(",") if args.models else list(MODEL_REGISTRY.keys())

    for model_name in models_to_run:
        if model_name not in MODEL_REGISTRY:
            print(f"Skipping unknown model: {model_name}")
            continue

        print(f"\n{'='*70}")
        print(f"Training: {model_name} [{CATEGORIES.get(model_name, '')}]")
        print(f"{'='*70}")

        model = build_model(model_name, args.hidden_dim, args.num_layers,
                            args.max_length, args.dropout)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")

        # Train
        t0 = time.time()
        history = train_cnn(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr,
            save_dir=save_dir, patience=args.patience,
        )
        train_time = time.time() - t0

        # Load best checkpoint and evaluate
        ckpt_path = os.path.join(save_dir, "best_cnn_model.pt")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])

            eval_result = evaluate_cnn(model, test_loader, device)

            results[model_name] = {
                "category": CATEGORIES.get(model_name, ""),
                "parameters": n_params,
                "train_time_s": round(train_time, 1),
                "best_val_f1": round(max(history["val_f1"]), 4),
                "test_precision": round(eval_result["avg_precision"], 4),
                "test_recall": round(eval_result["avg_recall"], 4),
                "test_f1": round(eval_result["avg_f1"], 4),
                "epochs_trained": len(history["train_loss"]),
            }

            # Save model with unique name for later prediction
            model_save_path = os.path.join(save_dir, f"bench_{model_name}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "best_f1": eval_result["avg_f1"],
            }, model_save_path)

    # Print comparison table
    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS")
    print("=" * 90)
    print(f"{'Model':<20} {'Category':<25} {'Params':>10} {'Train(s)':>10} "
          f"{'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-" * 90)

    sorted_results = sorted(results.items(), key=lambda x: x[1]["test_f1"], reverse=True)
    for name, r in sorted_results:
        print(f"{name:<20} {r['category']:<25} {r['parameters']:>10,} "
              f"{r['train_time_s']:>10.1f} "
              f"{r['test_precision']:>8.4f} {r['test_recall']:>8.4f} "
              f"{r['test_f1']:>8.4f}")

    print("=" * 90)

    if sorted_results:
        best_name = sorted_results[0][0]
        best_f1 = sorted_results[0][1]["test_f1"]
        print(f"\nBest model: {best_name} (F1: {best_f1:.4f})")

    # Save results
    results_path = os.path.join(save_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results


def predict_with_all_models(args):
    """Predict a sequence with all trained benchmark models."""
    device = get_device()
    sequence = args.sequence.upper().replace("T", "U")
    save_dir = args.save_dir

    print(f"Sequence: {sequence}")
    print(f"Length:   {len(sequence)}")
    print("=" * 70)

    # Find all benchmark model checkpoints
    predictions = {}
    for model_name in MODEL_REGISTRY:
        ckpt_path = os.path.join(save_dir, f"bench_{model_name}.pt")
        if not os.path.exists(ckpt_path):
            continue

        model = build_model(model_name, args.hidden_dim, args.num_layers,
                            args.max_length, dropout=0)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Greedy decoding
        result = predict_cnn(model, sequence, device=str(device),
                             max_length=args.max_length, threshold=0.5)

        # Nussinov DP decoding
        nuss_pairs = nussinov_decode(result["probabilities"], threshold=0.1)
        nuss_struct = structure_to_dotbracket(nuss_pairs, len(sequence))

        predictions[model_name] = {
            "greedy": result["structure"],
            "nussinov": nuss_struct,
            "n_pairs_greedy": len([k for k, v in result["pair_map"].items() if k < v]),
            "n_pairs_nussinov": len([k for k, v in nuss_pairs.items() if k < v]),
        }

        # Save contact map
        plot_contact_matrix(
            result["probabilities"], sequence=sequence,
            save_path=os.path.join(save_dir, f"contact_{model_name}.png"),
            title=f"{model_name} [{CATEGORIES.get(model_name, '')}]",
        )

    # Print comparison
    print(f"\n{'Model':<20} {'Category':<25} {'Pairs':>6} {'Structure (greedy)'}")
    print("-" * 100)
    for name, pred in predictions.items():
        print(f"{name:<20} {CATEGORIES.get(name, ''):<25} "
              f"{pred['n_pairs_greedy']:>6} {pred['greedy']}")

    print(f"\n{'Model':<20} {'Category':<25} {'Pairs':>6} {'Structure (Nussinov DP)'}")
    print("-" * 100)
    for name, pred in predictions.items():
        print(f"{name:<20} {CATEGORIES.get(name, ''):<25} "
              f"{pred['n_pairs_nussinov']:>6} {pred['nussinov']}")

    # Consensus: positions where majority of models agree
    L = len(sequence)
    greedy_structs = [pred["greedy"] for pred in predictions.values()]
    if greedy_structs:
        print(f"\nSequence:  {sequence}")
        consensus = []
        for i in range(L):
            chars = [s[i] for s in greedy_structs if i < len(s)]
            paired = sum(1 for c in chars if c != ".")
            if paired > len(chars) / 2:
                consensus.append("*")
            else:
                consensus.append(".")
        print(f"Consensus: {''.join(consensus)}  (* = majority predict paired)")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark all RNA structure prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--data", type=str, default="synthetic",
                        choices=["synthetic", "sample", "directory"])
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--save-dir", type=str, default="models/benchmark")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=200)
    parser.add_argument("--n-synthetic", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--models", type=str, default="",
                        help="Comma-separated list of models (default: all)")

    parser.add_argument("--sequence", type=str,
                        default="ACGCGAGAUUUGGGUGGGUAUGUCAGCUGCGGGUGUGGUG",
                        help="RNA sequence to predict after training")
    parser.add_argument("--predict-only", action="store_true",
                        help="Skip training, just predict with existing models")

    args = parser.parse_args()

    if args.predict_only:
        predict_with_all_models(args)
    else:
        run_benchmark(args)
        print("\n\nPredicting your sequence with all trained models...\n")
        predict_with_all_models(args)


if __name__ == "__main__":
    main()
