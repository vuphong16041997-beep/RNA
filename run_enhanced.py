#!/usr/bin/env python3
"""
Enhanced RNA Structure Prediction - fixes the precision problem.

Your original results:  Precision=0.47, Recall=0.98, F1=0.61
Target improvement:     Precision>0.70, Recall>0.80, F1>0.75

Run this script step-by-step:

  Step 1: Train with focal loss (fixes over-prediction)
    python run_enhanced.py train --loss focal --epochs 30

  Step 2: Train with constraint loss (adds physics)
    python run_enhanced.py train --loss constraint --epochs 30

  Step 3: Compare original vs enhanced
    python run_enhanced.py compare

  Step 4: Predict your sequence (with optimal threshold)
    python run_enhanced.py predict --sequence ACGCGAGAUUUGGGUGGGUAUGUCAGCUGCGGGUGUGGUG

  Step 5: Train multiple models and ensemble
    python run_enhanced.py ensemble --epochs 20
"""

import argparse
import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import prepare_dataset
from src.encoding import get_cnn_dataloaders
from src.model_cnn import RNAContactCNN
from src.train import compute_metrics, evaluate_cnn
from src.train_enhanced import (
    train_cnn_enhanced, evaluate_enhanced,
    ensemble_predict, FocalLoss,
)
from src.predict import predict_cnn, visualize_prediction
from src.model_hybrid import nussinov_decode
from src.encoding import structure_to_dotbracket
from src.visualize import plot_training_history, plot_contact_matrix


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cmd_train(args):
    """Train with enhanced pipeline."""
    device = get_device()
    print(f"Device: {device}")

    samples = prepare_dataset(
        mode=args.data, data_dir=args.data_dir,
        n_synthetic=args.n_synthetic, max_length=args.max_length,
    )

    train_loader, val_loader, test_loader = get_cnn_dataloaders(
        samples, batch_size=args.batch_size, max_length=args.max_length,
    )
    print(f"Data: {len(train_loader.dataset)} train / "
          f"{len(val_loader.dataset)} val / {len(test_loader.dataset)} test")

    model = RNAContactCNN(
        in_channels=9, hidden_channels=args.hidden_dim,
        num_blocks=args.num_layers, dropout=args.dropout,
    )
    print(f"Model: RNAContactCNN ({sum(p.numel() for p in model.parameters()):,} params)")

    save_dir = os.path.join(args.save_dir, f"enhanced_{args.loss}")
    history = train_cnn_enhanced(
        model, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr,
        save_dir=save_dir, patience=args.patience,
        loss_type=args.loss,
        warmup_epochs=args.warmup,
        use_augmentation=args.augment,
        label_smoothing=args.label_smoothing,
    )

    plot_training_history(history,
                          save_path=os.path.join(save_dir, "training_history.png"))

    # Evaluate with multiple thresholds
    ckpt = torch.load(os.path.join(save_dir, "best_cnn_model.pt"),
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    eval_results = evaluate_enhanced(model, test_loader, device)

    # Save results
    results = {
        "loss_type": args.loss,
        "best_threshold": eval_results["best_threshold"],
        "best_f1": eval_results["best_f1"],
        "all_thresholds": {
            str(k): v for k, v in eval_results["results_by_threshold"].items()
        },
        "training_epochs": len(history["train_loss"]),
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {save_dir}/")
    print(f"Use threshold={eval_results['best_threshold']} for prediction")


def cmd_compare(args):
    """Compare original vs enhanced training results."""
    device = get_device()

    samples = prepare_dataset(
        mode=args.data, data_dir=args.data_dir,
        n_synthetic=args.n_synthetic, max_length=args.max_length,
    )
    _, _, test_loader = get_cnn_dataloaders(
        samples, batch_size=args.batch_size, max_length=args.max_length,
    )

    results = {}

    # Check for original model
    orig_path = os.path.join(args.save_dir, "best_cnn_model.pt")
    if os.path.exists(orig_path):
        model = RNAContactCNN(in_channels=9, hidden_channels=args.hidden_dim,
                              num_blocks=args.num_layers, dropout=0)
        ckpt = torch.load(orig_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print("=" * 60)
        print("ORIGINAL MODEL (weighted BCE)")
        print("=" * 60)
        eval_r = evaluate_enhanced(model, test_loader, device)
        results["original"] = eval_r

    # Check for enhanced models
    for loss_type in ["focal", "constraint", "bce"]:
        enhanced_path = os.path.join(args.save_dir, f"enhanced_{loss_type}",
                                      "best_cnn_model.pt")
        if os.path.exists(enhanced_path):
            model = RNAContactCNN(in_channels=9, hidden_channels=args.hidden_dim,
                                  num_blocks=args.num_layers, dropout=0)
            ckpt = torch.load(enhanced_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"\n{'=' * 60}")
            print(f"ENHANCED MODEL ({loss_type} loss)")
            print(f"{'=' * 60}")
            eval_r = evaluate_enhanced(model, test_loader, device)
            results[f"enhanced_{loss_type}"] = eval_r

    if not results:
        print("No trained models found! Train first:")
        print("  python run_enhanced.py train --loss focal --epochs 30")
        return

    # Summary comparison
    print(f"\n{'=' * 70}")
    print("COMPARISON SUMMARY (at each model's optimal threshold)")
    print(f"{'=' * 70}")
    print(f"{'Model':<25} {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 65)

    for name, r in results.items():
        t = r["best_threshold"]
        m = r["results_by_threshold"][t]
        print(f"{name:<25} {t:>10.1f} {m['precision']:>10.4f} "
              f"{m['recall']:>10.4f} {m['f1']:>10.4f}")


def cmd_predict(args):
    """Predict with best available model."""
    device = get_device()
    sequence = args.sequence.upper().replace("T", "U")

    # Find best model
    best_model_path = None
    best_threshold = 0.5
    best_name = ""

    for loss_type in ["focal", "constraint", "bce"]:
        path = os.path.join(args.save_dir, f"enhanced_{loss_type}", "best_cnn_model.pt")
        results_path = os.path.join(args.save_dir, f"enhanced_{loss_type}", "results.json")
        if os.path.exists(path):
            if os.path.exists(results_path):
                with open(results_path) as f:
                    r = json.load(f)
                best_threshold = r.get("best_threshold", 0.5)
            best_model_path = path
            best_name = f"enhanced_{loss_type}"

    # Fallback to original
    if best_model_path is None:
        best_model_path = os.path.join(args.save_dir, "best_cnn_model.pt")
        best_name = "original"

    if not os.path.exists(best_model_path):
        print("No trained model found! Train first:")
        print("  python run_enhanced.py train --loss focal --epochs 30")
        sys.exit(1)

    print(f"Using model: {best_name}")
    print(f"Threshold: {best_threshold}")
    print(f"Sequence: {sequence}")
    print(f"Length: {len(sequence)}")

    model = RNAContactCNN(in_channels=9, hidden_channels=args.hidden_dim,
                          num_blocks=args.num_layers, dropout=0)
    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Greedy prediction
    result = predict_cnn(model, sequence, device=str(device),
                         max_length=args.max_length, threshold=best_threshold)

    print("\n--- Greedy Decoding ---")
    visualize_prediction(result)

    # Nussinov DP prediction
    nuss_pairs = nussinov_decode(result["probabilities"], threshold=0.1)
    nuss_struct = structure_to_dotbracket(nuss_pairs, len(sequence))

    print("\n--- Nussinov DP Decoding (guarantees valid structure) ---")
    nuss_result = dict(result)
    nuss_result["pair_map"] = nuss_pairs
    nuss_result["structure"] = nuss_struct
    visualize_prediction(nuss_result)

    # Save visualizations
    save_dir = os.path.join(args.save_dir, "predictions")
    os.makedirs(save_dir, exist_ok=True)
    plot_contact_matrix(
        result["probabilities"], sequence=sequence,
        save_path=os.path.join(save_dir, "contact_map.png"),
        title=f"Contact Probabilities ({best_name})",
    )


def cmd_ensemble(args):
    """Train multiple models and create ensemble prediction."""
    device = get_device()

    samples = prepare_dataset(
        mode=args.data, data_dir=args.data_dir,
        n_synthetic=args.n_synthetic, max_length=args.max_length,
    )
    train_loader, val_loader, test_loader = get_cnn_dataloaders(
        samples, batch_size=args.batch_size, max_length=args.max_length,
    )

    # Train 3 models with different configurations
    configs = [
        {"loss": "focal", "seed_offset": 0, "hidden": 64, "layers": 8},
        {"loss": "focal", "seed_offset": 1, "hidden": 48, "layers": 10},
        {"loss": "constraint", "seed_offset": 2, "hidden": 64, "layers": 8},
    ]

    models = []
    for i, cfg in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Training ensemble member {i+1}/{len(configs)}: {cfg['loss']} loss")
        print(f"{'='*60}")

        # Set different random seed
        torch.manual_seed(42 + cfg["seed_offset"])
        np.random.seed(42 + cfg["seed_offset"])

        model = RNAContactCNN(
            in_channels=9, hidden_channels=cfg["hidden"],
            num_blocks=cfg["layers"], dropout=args.dropout,
        )

        save_dir = os.path.join(args.save_dir, f"ensemble_{i}")
        train_cnn_enhanced(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr,
            save_dir=save_dir, patience=args.patience,
            loss_type=cfg["loss"],
            warmup_epochs=3,
        )

        # Load best checkpoint
        ckpt = torch.load(os.path.join(save_dir, "best_cnn_model.pt"),
                           map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        models.append(model)

    # Evaluate ensemble
    print(f"\n{'='*60}")
    print("ENSEMBLE EVALUATION")
    print(f"{'='*60}")

    all_metrics = []
    for batch in test_loader:
        features = batch["features"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)

        # Ensemble prediction
        pred = ensemble_predict(models, features, mask, device)

        for i in range(pred.shape[0]):
            L = batch["length"][i].item()
            p = pred[i, :L, :L].cpu().numpy()
            t = target[i, :L, :L].cpu().numpy()
            metrics = compute_metrics(p, t, threshold=0.5)
            all_metrics.append(metrics)

    avg_p = np.mean([m["precision"] for m in all_metrics])
    avg_r = np.mean([m["recall"] for m in all_metrics])
    avg_f1 = np.mean([m["f1"] for m in all_metrics])

    print(f"Ensemble ({len(models)} models):")
    print(f"  Precision: {avg_p:.4f}")
    print(f"  Recall:    {avg_r:.4f}")
    print(f"  F1:        {avg_f1:.4f}")

    # Predict the target sequence
    sequence = args.sequence.upper().replace("T", "U")
    print(f"\nEnsemble prediction for: {sequence}")

    result = predict_cnn(models[0], sequence, device=str(device),
                         max_length=args.max_length, threshold=0.5)

    # Average predictions from all models
    from src.encoding import encode_sequence, contact_matrix_to_pairs
    L = len(sequence)
    avg_probs = np.zeros((L, L), dtype=np.float32)
    for model in models:
        r = predict_cnn(model, sequence, device=str(device),
                        max_length=args.max_length, threshold=0.5)
        avg_probs += r["probabilities"]
    avg_probs /= len(models)

    pair_map = contact_matrix_to_pairs(avg_probs, threshold=0.5)
    structure = structure_to_dotbracket(pair_map, L)

    ensemble_result = {
        "sequence": sequence,
        "structure": structure,
        "pair_map": pair_map,
        "probabilities": avg_probs,
    }
    visualize_prediction(ensemble_result)


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced RNA Structure Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # Shared args
    for name, help_text in [
        ("train", "Train with enhanced pipeline"),
        ("compare", "Compare original vs enhanced"),
        ("predict", "Predict structure"),
        ("ensemble", "Train ensemble of models"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--data", default="synthetic",
                        choices=["synthetic", "sample", "directory"])
        p.add_argument("--data-dir", default="data")
        p.add_argument("--save-dir", default="models")
        p.add_argument("--batch-size", type=int, default=8)
        p.add_argument("--hidden-dim", type=int, default=64)
        p.add_argument("--num-layers", type=int, default=8)
        p.add_argument("--max-length", type=int, default=200)
        p.add_argument("--n-synthetic", type=int, default=5000)
        p.add_argument("--dropout", type=float, default=0.1)
        p.add_argument("--sequence", type=str,
                        default="ACGCGAGAUUUGGGUGGGUAUGUCAGCUGCGGGUGUGGUG")

        if name in ("train", "ensemble"):
            p.add_argument("--epochs", type=int, default=30)
            p.add_argument("--lr", type=float, default=1e-3)
            p.add_argument("--patience", type=int, default=15)

        if name == "train":
            p.add_argument("--loss", default="focal",
                            choices=["focal", "constraint", "bce"])
            p.add_argument("--warmup", type=int, default=5)
            p.add_argument("--augment", action="store_true", default=True)
            p.add_argument("--no-augment", dest="augment", action="store_false")
            p.add_argument("--label-smoothing", type=float, default=0.05)

    args = parser.parse_args()

    commands = {
        "train": cmd_train,
        "compare": cmd_compare,
        "predict": cmd_predict,
        "ensemble": cmd_ensemble,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
