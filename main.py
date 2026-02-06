#!/usr/bin/env python3
"""
RNA Secondary Structure Prediction using Deep Learning and GNNs.

Usage:
    # Step 1: Train with synthetic data (quick test)
    python main.py train --model cnn --data synthetic --epochs 10

    # Step 2: Train with curated real RNA data
    python main.py train --model cnn --data sample --epochs 50

    # Step 3: Train GNN model
    python main.py train --model gnn-gcn --data sample --epochs 50

    # Step 4: Predict structure for your sequence
    python main.py predict --model cnn --sequence ACGCGAGAUUUGGGUGGGUAUGUCAGCUGCGGGUGUGGUG

    # Step 5: Predict with GNN
    python main.py predict --model gnn-gcn --sequence ACGCGAGAUUUGGGUGGGUAUGUCAGCUGCGGGUGUGGUG
"""

import argparse
import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import prepare_dataset
from src.encoding import get_cnn_dataloaders
from src.model_cnn import RNAContactCNN, RNAContactCNNLarge
from src.train import train_cnn, train_gnn, evaluate_cnn
from src.predict import predict_cnn, predict_gnn, visualize_prediction
from src.visualize import plot_training_history, plot_contact_matrix, plot_arc_diagram


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cmd_train(args):
    """Train a model."""
    device = get_device()
    print(f"Using device: {device}")

    # Load data
    samples = prepare_dataset(
        mode=args.data,
        data_dir=args.data_dir,
        n_synthetic=args.n_synthetic,
        max_length=args.max_length,
    )
    print(f"Dataset: {len(samples)} samples")

    if len(samples) < 10:
        print("WARNING: Very few samples. Results will be unreliable.")
        print("For real training, use bpRNA-1m or ArchiveII datasets.")

    if args.model == "cnn":
        model = RNAContactCNN(
            in_channels=9,
            hidden_channels=args.hidden_dim,
            num_blocks=args.num_layers,
            dropout=args.dropout,
        )
        print(f"Model: RNAContactCNN ({sum(p.numel() for p in model.parameters()):,} parameters)")

        train_loader, val_loader, test_loader = get_cnn_dataloaders(
            samples, batch_size=args.batch_size, max_length=args.max_length,
        )
        print(f"Train: {len(train_loader.dataset)}, "
              f"Val: {len(val_loader.dataset)}, "
              f"Test: {len(test_loader.dataset)}")

        history = train_cnn(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr,
            save_dir=args.save_dir, patience=args.patience,
        )

        # Plot training history
        plot_training_history(history, save_path=os.path.join(args.save_dir, "training_history.png"))

        # Evaluate on test set
        checkpoint = torch.load(os.path.join(args.save_dir, "best_cnn_model.pt"),
                                map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        evaluate_cnn(model, test_loader, device)

    elif args.model == "cnn-large":
        model = RNAContactCNNLarge(
            in_channels=9,
            hidden_channels=args.hidden_dim,
            num_blocks=args.num_layers,
            dropout=args.dropout,
        )
        print(f"Model: RNAContactCNNLarge ({sum(p.numel() for p in model.parameters()):,} parameters)")

        train_loader, val_loader, test_loader = get_cnn_dataloaders(
            samples, batch_size=args.batch_size, max_length=args.max_length,
        )

        history = train_cnn(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr,
            save_dir=args.save_dir, patience=args.patience,
        )
        plot_training_history(history, save_path=os.path.join(args.save_dir, "training_history.png"))

    elif args.model.startswith("gnn"):
        try:
            from src.encoding import get_gnn_dataloaders
        except ImportError:
            print("ERROR: torch_geometric is required for GNN models.")
            print("Install: pip install torch-geometric")
            sys.exit(1)

        gnn_type = args.model.split("-")[1] if "-" in args.model else "gcn"

        if gnn_type == "gcn":
            from src.model_gnn import RNAStructureGCN
            model = RNAStructureGCN(
                node_features=5, hidden_dim=args.hidden_dim,
                num_layers=args.num_layers, dropout=args.dropout,
            )
        elif gnn_type == "gat":
            from src.model_gnn import RNAStructureGAT
            model = RNAStructureGAT(
                node_features=5, hidden_dim=args.hidden_dim,
                num_layers=args.num_layers, dropout=args.dropout,
            )
        elif gnn_type == "transformer":
            from src.model_gnn import RNAStructureTransformerGNN
            model = RNAStructureTransformerGNN(
                node_features=5, hidden_dim=args.hidden_dim,
                num_layers=args.num_layers, dropout=args.dropout,
            )
        else:
            print(f"Unknown GNN type: {gnn_type}")
            print("Available: gcn, gat, transformer")
            sys.exit(1)

        print(f"Model: {type(model).__name__} ({sum(p.numel() for p in model.parameters()):,} parameters)")

        train_loader, val_loader, test_loader = get_gnn_dataloaders(
            samples, batch_size=args.batch_size, max_length=args.max_length,
            window_size=args.window_size,
        )

        history = train_gnn(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr,
            save_dir=args.save_dir, patience=args.patience,
        )
        plot_training_history(history, save_path=os.path.join(args.save_dir, "training_history.png"))

    print("\nTraining complete!")


def cmd_predict(args):
    """Predict structure for a sequence."""
    device = get_device()
    sequence = args.sequence.upper().replace("T", "U")
    print(f"Sequence: {sequence}")
    print(f"Length: {len(sequence)}")

    if args.model == "cnn" or args.model == "cnn-large":
        model_path = os.path.join(args.save_dir, "best_cnn_model.pt")
        if not os.path.exists(model_path):
            print(f"ERROR: No trained model found at {model_path}")
            print("Train a model first: python main.py train --model cnn --data synthetic")
            sys.exit(1)

        if args.model == "cnn":
            model = RNAContactCNN(
                in_channels=9, hidden_channels=args.hidden_dim,
                num_blocks=args.num_layers, dropout=0,
            )
        else:
            model = RNAContactCNNLarge(
                in_channels=9, hidden_channels=args.hidden_dim,
                num_blocks=args.num_layers, dropout=0,
            )

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        result = predict_cnn(model, sequence, device=str(device),
                             max_length=args.max_length,
                             threshold=args.threshold)

    elif args.model.startswith("gnn"):
        model_path = os.path.join(args.save_dir, "best_gnn_model.pt")
        if not os.path.exists(model_path):
            print(f"ERROR: No trained model found at {model_path}")
            print("Train first: python main.py train --model gnn-gcn --data synthetic")
            sys.exit(1)

        gnn_type = args.model.split("-")[1] if "-" in args.model else "gcn"

        if gnn_type == "gcn":
            from src.model_gnn import RNAStructureGCN
            model = RNAStructureGCN(node_features=5, hidden_dim=args.hidden_dim,
                                    num_layers=args.num_layers, dropout=0)
        elif gnn_type == "gat":
            from src.model_gnn import RNAStructureGAT
            model = RNAStructureGAT(node_features=5, hidden_dim=args.hidden_dim,
                                    num_layers=args.num_layers, dropout=0)
        elif gnn_type == "transformer":
            from src.model_gnn import RNAStructureTransformerGNN
            model = RNAStructureTransformerGNN(node_features=5, hidden_dim=args.hidden_dim,
                                              num_layers=args.num_layers, dropout=0)
        else:
            print(f"Unknown GNN type: {gnn_type}")
            sys.exit(1)

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        result = predict_gnn(model, sequence, device=str(device),
                             window_size=args.window_size,
                             threshold=args.threshold)
    else:
        print(f"Unknown model: {args.model}")
        sys.exit(1)

    # Display results
    visualize_prediction(result)

    # Save visualizations
    plot_contact_matrix(
        result["probabilities"],
        sequence=sequence,
        save_path=os.path.join(args.save_dir, "prediction_contact_map.png"),
        title=f"Contact Matrix ({args.model})",
    )

    plot_arc_diagram(
        result["sequence"],
        result["structure"],
        save_path=os.path.join(args.save_dir, "prediction_arc_diagram.png"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="RNA Secondary Structure Prediction with Deep Learning & GNNs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- Train ---
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--model", type=str, default="cnn",
                              choices=["cnn", "cnn-large", "gnn-gcn", "gnn-gat", "gnn-transformer"],
                              help="Model architecture")
    train_parser.add_argument("--data", type=str, default="synthetic",
                              choices=["synthetic", "sample", "directory"],
                              help="Dataset mode")
    train_parser.add_argument("--data-dir", type=str, default="data",
                              help="Directory for data files")
    train_parser.add_argument("--save-dir", type=str, default="models",
                              help="Directory to save models")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--hidden-dim", type=int, default=64)
    train_parser.add_argument("--num-layers", type=int, default=8)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--max-length", type=int, default=200)
    train_parser.add_argument("--n-synthetic", type=int, default=5000)
    train_parser.add_argument("--window-size", type=int, default=0,
                              help="GNN: max distance for candidate pairs (0=all)")
    train_parser.add_argument("--patience", type=int, default=10,
                              help="Early stopping patience")

    # --- Predict ---
    pred_parser = subparsers.add_parser("predict", help="Predict structure")
    pred_parser.add_argument("--model", type=str, default="cnn",
                             choices=["cnn", "cnn-large", "gnn-gcn", "gnn-gat", "gnn-transformer"])
    pred_parser.add_argument("--sequence", type=str, required=True,
                             help="RNA sequence to predict")
    pred_parser.add_argument("--save-dir", type=str, default="models")
    pred_parser.add_argument("--hidden-dim", type=int, default=64)
    pred_parser.add_argument("--num-layers", type=int, default=8)
    pred_parser.add_argument("--max-length", type=int, default=200)
    pred_parser.add_argument("--window-size", type=int, default=0)
    pred_parser.add_argument("--threshold", type=float, default=0.5,
                             help="Probability threshold for base pairs")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
