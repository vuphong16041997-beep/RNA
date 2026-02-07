#!/usr/bin/env python3
"""
Dataset setup helper for RNA secondary structure prediction.

This script helps you download and organize datasets for training.
Datasets are stored locally in data/ (gitignored - NOT uploaded to GitHub).

Usage:
    # See what datasets are available
    python setup_data.py info

    # Download bpRNA-1m from the web (if available)
    python setup_data.py download-bprna

    # Verify your local bpRNA data after manual download
    python setup_data.py verify --data-dir data/bpRNA_data

    # Show statistics about your dataset
    python setup_data.py stats --data-dir data/bpRNA_data
"""

import os
import sys
import argparse
import tarfile
import zipfile
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import (
    load_dataset_from_dir, parse_bpseq, parse_bprna_st,
    parse_ct_file, parse_dbn_file, generate_synthetic_dataset,
)


def cmd_info(args):
    """Show dataset information and download instructions."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              RNA DATASET SETUP GUIDE                               ║
╚══════════════════════════════════════════════════════════════════════╝

IMPORTANT: Datasets go in the data/ folder (gitignored, NOT on GitHub).
           Only your CODE goes on GitHub, not the data.

┌──────────────────────────────────────────────────────────────────────┐
│  OPTION 1: Synthetic Data (built-in, no download needed)            │
│                                                                      │
│  python main.py train --model cnn --data synthetic --epochs 10       │
│                                                                      │
│  ✓ 5000 auto-generated RNA sequences with stem-loop structures       │
│  ✓ Good for testing that your code works                             │
│  ✗ Not real RNA - won't generalize to real sequences                 │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  OPTION 2: bpRNA-1m (recommended, manual download)                  │
│                                                                      │
│  1. Go to: https://bprna.cgrb.oregonstate.edu/download.php           │
│  2. Download bpRNA-1m (or bpRNA-1m(90) for non-redundant set)       │
│  3. Extract into data/bpRNA_data/                                    │
│                                                                      │
│  Your folder should look like:                                       │
│  data/                                                               │
│  └── bpRNA_data/                                                     │
│      ├── bpRNA_RFAM_1.st                                             │
│      ├── bpRNA_RFAM_2.st                                             │
│      ├── bpRNA_RFAM_3.st                                             │
│      └── ... (~100,000 .st files)                                    │
│                                                                      │
│  Then run:                                                           │
│  python setup_data.py verify --data-dir data/bpRNA_data              │
│  python run_enhanced.py train --loss focal --data directory \\        │
│      --data-dir data/bpRNA_data --epochs 50                          │
│                                                                      │
│  ✓ ~100k real RNA sequences with validated structures                │
│  ✓ Standard benchmark used by all papers                             │
│  ✓ Multiple RNA families (tRNA, rRNA, miRNA, etc.)                   │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  OPTION 3: ArchiveII (smaller, high quality)                        │
│                                                                      │
│  1. Download: https://rna.urmc.rochester.edu/pub/archiveII.tar.gz    │
│  2. Extract:                                                         │
│     mkdir -p data/archiveII                                          │
│     tar -xzf archiveII.tar.gz -C data/archiveII/                    │
│                                                                      │
│  Then run:                                                           │
│  python run_enhanced.py train --loss focal --data directory \\        │
│      --data-dir data/archiveII --epochs 50                           │
│                                                                      │
│  ✓ ~4000 high-quality sequences                                     │
│  ✓ .ct format (connectivity tables)                                  │
│  ✓ Good for initial real-data training                               │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  OPTION 4: bpRNA from HuggingFace (easiest download)                │
│                                                                      │
│  pip install datasets                                                │
│  python setup_data.py download-hf                                    │
│                                                                      │
│  Downloads from: huggingface.co/datasets/multimolecule/bprna         │
│  Converts to .dbn format in data/bprna_hf/                           │
└──────────────────────────────────────────────────────────────────────┘

WHY NOT PUT DATA ON GITHUB?
  - bpRNA-1m is ~2 GB when extracted
  - GitHub has a 100 MB file limit
  - Datasets are meant to be downloaded separately
  - Your code should work with ANY data in data/
  - Other researchers download datasets to their own machines
""")


def cmd_verify(args):
    """Verify dataset directory and show file counts."""
    data_dir = args.data_dir

    if not os.path.exists(data_dir):
        print(f"ERROR: Directory not found: {data_dir}")
        print(f"Create it and add your data files there.")
        sys.exit(1)

    # Count files by extension
    extensions = Counter()
    total_files = 0
    for filepath in Path(data_dir).rglob("*"):
        if filepath.is_file():
            extensions[filepath.suffix.lower()] += 1
            total_files += 1

    print(f"Dataset directory: {data_dir}")
    print(f"Total files: {total_files}")
    print(f"\nFile types found:")
    for ext, count in extensions.most_common():
        supported = ext in (".st", ".sta", ".bpseq", ".ct", ".dbn")
        marker = "  ✓ supported" if supported else ""
        print(f"  {ext:>8}: {count:>6} files{marker}")

    supported_count = sum(
        count for ext, count in extensions.items()
        if ext in (".st", ".sta", ".bpseq", ".ct", ".dbn")
    )
    print(f"\nSupported data files: {supported_count}")

    if supported_count == 0:
        print("\nWARNING: No supported files found!")
        print("Expected file formats: .st, .sta, .bpseq, .ct, .dbn")
        print("Run: python setup_data.py info  for download instructions")
        return

    # Try loading a sample
    print(f"\nLoading sample (first 100 files, max length 600)...")
    samples = load_dataset_from_dir(data_dir, max_samples=100, max_length=600)
    print(f"Successfully loaded: {len(samples)} samples")

    if samples:
        lengths = [s.length for s in samples]
        pairs = [len(s.pair_map) // 2 for s in samples]
        print(f"\nSample statistics (first {len(samples)}):")
        print(f"  Length range: {min(lengths)} - {max(lengths)} nt")
        print(f"  Average length: {sum(lengths)/len(lengths):.0f} nt")
        print(f"  Average pairs: {sum(pairs)/len(pairs):.1f}")
        print(f"\nFirst 5 samples:")
        for s in samples[:5]:
            print(f"  {s.name}: {s.length} nt, {len(s.pair_map)//2} pairs")
            print(f"    Seq: {s.sequence[:50]}{'...' if s.length > 50 else ''}")
            print(f"    Str: {s.structure[:50]}{'...' if s.length > 50 else ''}")

    print(f"\n{'='*60}")
    print("Your data is ready! Now train with:")
    print(f"  python run_enhanced.py train --loss focal --data directory \\")
    print(f"      --data-dir {data_dir} --epochs 50")


def cmd_stats(args):
    """Show detailed statistics about a dataset."""
    data_dir = args.data_dir
    max_len = args.max_length

    print(f"Loading all data from {data_dir} (max_length={max_len})...")
    samples = load_dataset_from_dir(data_dir, max_length=max_len)
    print(f"Loaded {len(samples)} samples\n")

    if not samples:
        print("No samples loaded. Check your data directory.")
        return

    lengths = [s.length for s in samples]
    pairs = [len(s.pair_map) // 2 for s in samples]
    pair_fractions = [len(s.pair_map) / (2 * s.length) for s in samples]

    # Base composition
    base_counts = Counter()
    for s in samples:
        base_counts.update(s.sequence)

    total_bases = sum(base_counts.values())

    print(f"{'='*50}")
    print(f"DATASET STATISTICS")
    print(f"{'='*50}")
    print(f"Total samples:     {len(samples):,}")
    print(f"Total bases:       {total_bases:,}")
    print(f"\nLength distribution:")
    print(f"  Min:    {min(lengths)}")
    print(f"  Max:    {max(lengths)}")
    print(f"  Mean:   {sum(lengths)/len(lengths):.1f}")
    print(f"  Median: {sorted(lengths)[len(lengths)//2]}")

    print(f"\nBase pairs per sequence:")
    print(f"  Min:    {min(pairs)}")
    print(f"  Max:    {max(pairs)}")
    print(f"  Mean:   {sum(pairs)/len(pairs):.1f}")
    print(f"  Avg fraction paired: {sum(pair_fractions)/len(pair_fractions):.2%}")

    print(f"\nBase composition:")
    for base in "ACGU":
        count = base_counts.get(base, 0)
        print(f"  {base}: {count:>10,}  ({count/total_bases:.1%})")

    # Length histogram (text-based)
    print(f"\nLength distribution:")
    bins = [0, 50, 100, 150, 200, 300, 400, 500, 600]
    for i in range(len(bins) - 1):
        count = sum(1 for l in lengths if bins[i] <= l < bins[i+1])
        bar = "█" * (count * 40 // len(samples))
        print(f"  {bins[i]:>4}-{bins[i+1]:<4}: {count:>6} {bar}")
    overflow = sum(1 for l in lengths if l >= bins[-1])
    if overflow:
        print(f"  {bins[-1]:>4}+   : {overflow:>6}")


def cmd_download_hf(args):
    """Download bpRNA from HuggingFace and convert to .dbn format."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install the datasets library first:")
        print("  pip install datasets")
        sys.exit(1)

    out_dir = os.path.join(args.data_dir, "bprna_hf")
    os.makedirs(out_dir, exist_ok=True)

    print("Downloading bpRNA from HuggingFace...")
    print("Source: huggingface.co/datasets/multimolecule/bprna")

    ds = load_dataset("multimolecule/bprna", split="train")
    print(f"Downloaded {len(ds)} samples")

    # Convert to .dbn format
    dbn_path = os.path.join(out_dir, "bprna_full.dbn")
    count = 0
    with open(dbn_path, "w") as f:
        for item in ds:
            seq = item.get("sequence", "")
            # HuggingFace bpRNA uses different column names
            struct = item.get("secondary_structure", item.get("ss", ""))
            name = item.get("id", item.get("name", f"bprna_{count}"))

            if seq and struct and len(seq) == len(struct):
                f.write(f">{name}\n{seq}\n{struct}\n")
                count += 1

    print(f"Saved {count} samples to {dbn_path}")
    print(f"\nTrain with:")
    print(f"  python run_enhanced.py train --loss focal --data directory \\")
    print(f"      --data-dir {out_dir} --epochs 50")


def main():
    parser = argparse.ArgumentParser(
        description="RNA Dataset Setup Helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info", help="Show dataset information and instructions")

    verify_p = sub.add_parser("verify", help="Verify dataset directory")
    verify_p.add_argument("--data-dir", required=True)

    stats_p = sub.add_parser("stats", help="Dataset statistics")
    stats_p.add_argument("--data-dir", required=True)
    stats_p.add_argument("--max-length", type=int, default=600)

    hf_p = sub.add_parser("download-hf", help="Download from HuggingFace")
    hf_p.add_argument("--data-dir", default="data")

    args = parser.parse_args()

    commands = {
        "info": cmd_info,
        "verify": cmd_verify,
        "stats": cmd_stats,
        "download-hf": cmd_download_hf,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        cmd_info(args)


if __name__ == "__main__":
    main()
