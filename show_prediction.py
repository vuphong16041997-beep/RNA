#!/usr/bin/env python3
"""
Generate visual RNA secondary structure prediction for your sequence.

Produces:
  1. Terminal output: dot-bracket structure
  2. results/contact_map.png: heatmap of base pair probabilities
  3. results/arc_diagram.png: arc diagram showing base pairs
  4. results/prediction_summary.txt: text summary

Usage:
    python show_prediction.py
    python show_prediction.py --sequence GGCCAUUCAAGGCC
    python show_prediction.py --sequence ACGCGAGAUUUGGGUGGGUAUGUCAGCUGCGGGUGUGGUG
"""

import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model_cnn import RNAContactCNN
from src.predict import predict_cnn, visualize_prediction
from src.model_hybrid import nussinov_decode, compute_thermodynamic_prior
from src.encoding import structure_to_dotbracket, contact_matrix_to_pairs
from src.visualize import plot_contact_matrix, plot_arc_diagram


def find_model():
    """Find the best available trained model."""
    candidates = [
        "models/enhanced_focal/best_cnn_model.pt",
        "models/enhanced_constraint/best_cnn_model.pt",
        "models/best_cnn_model.pt",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def detect_model_config(ckpt_path):
    """Detect model hidden_dim and num_blocks from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    # Detect hidden_channels from input_proj weight shape
    hidden = state["input_proj.0.weight"].shape[0]

    # Count residual blocks
    num_blocks = 0
    while f"res_blocks.{num_blocks}.conv1.weight" in state:
        num_blocks += 1

    return hidden, num_blocks


def main():
    parser = argparse.ArgumentParser(description="Show RNA structure prediction")
    parser.add_argument("--sequence", type=str,
                        default="ACGCGAGAUUUGGGUGGGUAUGUCAGCUGCGGGUGUGGUG")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=200)
    args = parser.parse_args()

    seq = args.sequence.upper().replace("T", "U")
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Find model
    model_path = args.model_path or find_model()
    if model_path is None:
        print("ERROR: No trained model found!")
        print("Train first:  python main.py train --model cnn --data synthetic --epochs 10")
        sys.exit(1)

    # Detect model architecture from checkpoint
    hidden, num_blocks = detect_model_config(model_path)
    print(f"Model: {model_path} (hidden={hidden}, blocks={num_blocks})")

    model = RNAContactCNN(in_channels=9, hidden_channels=hidden,
                          num_blocks=num_blocks, dropout=0)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print(f"Sequence: {seq}")
    print(f"Length:   {len(seq)} nt")

    # ===== METHOD 1: ML Prediction (Greedy) =====
    result = predict_cnn(model, seq, device="cpu",
                         max_length=args.max_length, threshold=args.threshold)

    print("\n" + "=" * 70)
    print("METHOD 1: Neural Network + Greedy Decoding")
    print("=" * 70)
    visualize_prediction(result)

    # ===== METHOD 2: ML + Nussinov DP =====
    nuss_pairs = nussinov_decode(result["probabilities"], threshold=0.1)
    nuss_struct = structure_to_dotbracket(nuss_pairs, len(seq))

    print("\n" + "=" * 70)
    print("METHOD 2: Neural Network + Nussinov DP (valid structure)")
    print("=" * 70)
    nuss_result = dict(result)
    nuss_result["pair_map"] = nuss_pairs
    nuss_result["structure"] = nuss_struct
    visualize_prediction(nuss_result)

    # ===== METHOD 3: Pure Thermodynamics =====
    thermo_prior = compute_thermodynamic_prior(seq)
    thermo_pairs = nussinov_decode(thermo_prior, threshold=0.1)
    thermo_struct = structure_to_dotbracket(thermo_pairs, len(seq))

    print("\n" + "=" * 70)
    print("METHOD 3: Thermodynamic Only (no ML, pure physics)")
    print("=" * 70)
    thermo_result = {
        "sequence": seq, "structure": thermo_struct,
        "pair_map": thermo_pairs, "probabilities": thermo_prior,
    }
    visualize_prediction(thermo_result)

    # ===== GENERATE IMAGES =====
    print("\n" + "=" * 70)
    print("SAVING VISUAL OUTPUTS")
    print("=" * 70)

    # 1. Contact probability heatmap
    plot_contact_matrix(
        result["probabilities"],
        title=f"Base Pair Probability Map (CNN model)\n{seq[:30]}...",
        save_path=os.path.join(out_dir, "contact_map.png"),
    )

    # 2. Arc diagram - ML prediction
    plot_arc_diagram(
        nuss_result["sequence"], nuss_result["structure"],
        save_path=os.path.join(out_dir, "arc_diagram_ml.png"),
    )

    # 3. Arc diagram - thermodynamic prediction
    plot_arc_diagram(
        thermo_result["sequence"], thermo_result["structure"],
        save_path=os.path.join(out_dir, "arc_diagram_thermo.png"),
    )

    # 4. Save text summary
    summary_path = os.path.join(out_dir, "prediction_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"RNA Secondary Structure Prediction\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Sequence:  {seq}\n")
        f.write(f"Length:    {len(seq)} nt\n\n")

        f.write(f"Method 1 - Neural Network (Greedy):\n")
        f.write(f"  Structure: {result['structure']}\n")
        n1 = len([k for k, v in result['pair_map'].items() if k < v])
        f.write(f"  Pairs:     {n1}\n\n")

        f.write(f"Method 2 - Neural Network + Nussinov DP:\n")
        f.write(f"  Structure: {nuss_struct}\n")
        n2 = len([k for k, v in nuss_pairs.items() if k < v])
        f.write(f"  Pairs:     {n2}\n\n")

        f.write(f"Method 3 - Thermodynamic Only:\n")
        f.write(f"  Structure: {thermo_struct}\n")
        n3 = len([k for k, v in thermo_pairs.items() if k < v])
        f.write(f"  Pairs:     {n3}\n\n")

        f.write(f"Base pairs (Method 2 - recommended):\n")
        for a, b in sorted([(k, v) for k, v in nuss_pairs.items() if k < v]):
            f.write(f"  {a:3d}-{b:3d}  {seq[a]}-{seq[b]}\n")

    print(f"\nFiles saved to {out_dir}/:")
    for fname in sorted(os.listdir(out_dir)):
        fpath = os.path.join(out_dir, fname)
        size = os.path.getsize(fpath)
        print(f"  {fname:<35} ({size:,} bytes)")

    print(f"\n>>> OPEN THESE FILES TO SEE THE VISUAL STRUCTURE:")
    print(f"    {os.path.abspath(os.path.join(out_dir, 'arc_diagram_ml.png'))}")
    print(f"    {os.path.abspath(os.path.join(out_dir, 'contact_map.png'))}")
    print(f"    {os.path.abspath(os.path.join(out_dir, 'prediction_summary.txt'))}")


if __name__ == "__main__":
    main()
