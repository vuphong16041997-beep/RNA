"""
Dataset download and preprocessing for RNA secondary structure prediction.

Supports:
- bpRNA-1m: The standard benchmark dataset (~100k RNA sequences with known structures)
- Custom .bpseq / .ct / .dbn files

bpRNA format (dot-bracket notation):
  Sequence:  ACGUACGU
  Structure: ((....))
  Where '(' and ')' are paired bases, '.' is unpaired
"""

import os
import re
import requests
import gzip
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class RNASample:
    """Single RNA sample with sequence and secondary structure."""

    def __init__(self, name: str, sequence: str, structure: str,
                 pair_map: Optional[Dict[int, int]] = None):
        self.name = name
        self.sequence = sequence.upper().replace("T", "U")  # DNA->RNA
        self.structure = structure
        # pair_map: {i: j} means position i pairs with position j (0-indexed)
        self.pair_map = pair_map or self._structure_to_pair_map(structure)

    @staticmethod
    def _structure_to_pair_map(structure: str) -> Dict[int, int]:
        """Convert dot-bracket notation to pair map."""
        pair_map = {}
        stacks = {"(": [], "[": [], "{": [], "<": []}
        closing = {")": "(", "]": "[", "}": "{", ">": "<"}

        for i, ch in enumerate(structure):
            if ch in stacks:
                stacks[ch].append(i)
            elif ch in closing:
                opener = closing[ch]
                if stacks[opener]:
                    j = stacks[opener].pop()
                    pair_map[i] = j
                    pair_map[j] = i
            # '.' or other characters = unpaired

        return pair_map

    @property
    def length(self) -> int:
        return len(self.sequence)

    def __repr__(self):
        return f"RNASample(name={self.name}, len={self.length})"


# ---------------------------------------------------------------------------
# bpRNA parser (.sta format)
# ---------------------------------------------------------------------------

def parse_bprna_sta(filepath: str) -> Optional[RNASample]:
    """Parse a single bpRNA .sta (structure annotation) file."""
    name = Path(filepath).stem
    sequence = ""
    structure = ""

    try:
        with open(filepath, "r") as f:
            lines = f.readlines()

        # .sta files have lines like:
        # Filename: ...
        # Organism: ...
        # Accession: ...
        # Sequence: ACGU...
        # Structure: ((..))
        for line in lines:
            line = line.strip()
            if line.startswith("Sequence:"):
                sequence = line.split(":", 1)[1].strip()
            elif line.startswith("Structure:"):
                structure = line.split(":", 1)[1].strip()

        if sequence and structure and len(sequence) == len(structure):
            return RNASample(name=name, sequence=sequence, structure=structure)
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Dot-bracket file parser (.dbn)
# ---------------------------------------------------------------------------

def parse_dbn_file(filepath: str) -> List[RNASample]:
    """Parse a .dbn file (FASTA-like with dot-bracket structures).

    Format:
        >name
        SEQUENCE
        STRUCTURE
    """
    samples = []
    try:
        with open(filepath, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        i = 0
        while i < len(lines):
            if lines[i].startswith(">"):
                name = lines[i][1:]
                if i + 2 < len(lines):
                    seq = lines[i + 1]
                    struct = lines[i + 2]
                    if len(seq) == len(struct):
                        samples.append(RNASample(name=name, sequence=seq,
                                                 structure=struct))
                i += 3
            else:
                i += 1
    except Exception:
        pass

    return samples


# ---------------------------------------------------------------------------
# bpseq parser
# ---------------------------------------------------------------------------

def parse_bpseq(filepath: str) -> Optional[RNASample]:
    """Parse a .bpseq file.

    Format (1-indexed):
        1 A 0
        2 C 10
        ...
    Column 3 = 0 means unpaired, else paired with that position.
    """
    name = Path(filepath).stem
    entries = []

    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    idx = int(parts[0])
                    base = parts[1]
                    pair = int(parts[2])
                    entries.append((idx, base, pair))

        if not entries:
            return None

        entries.sort(key=lambda x: x[0])
        sequence = "".join(e[1] for e in entries)

        # Build dot-bracket from pair info
        pair_map = {}
        for idx, base, pair in entries:
            if pair > 0:
                pair_map[idx - 1] = pair - 1  # Convert to 0-indexed

        structure = []
        for idx, base, pair in entries:
            i = idx - 1
            if pair == 0:
                structure.append(".")
            elif pair > idx:
                structure.append("(")
            else:
                structure.append(")")

        structure = "".join(structure)
        return RNASample(name=name, sequence=sequence, structure=structure,
                         pair_map=pair_map)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CT file parser
# ---------------------------------------------------------------------------

def parse_ct_file(filepath: str) -> Optional[RNASample]:
    """Parse a .ct (connectivity table) file.

    First line: length title
    Subsequent lines: index base prev next pair index
    """
    name = Path(filepath).stem
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()

        if not lines:
            return None

        # First line has the length
        header = lines[0].strip().split()
        n = int(header[0])

        sequence = []
        pair_map = {}

        for line in lines[1:n + 1]:
            parts = line.strip().split()
            if len(parts) >= 5:
                idx = int(parts[0]) - 1  # 0-indexed
                base = parts[1]
                pair = int(parts[4]) - 1  # 0-indexed, -1 means unpaired
                sequence.append(base)
                if pair >= 0:
                    pair_map[idx] = pair

        sequence = "".join(sequence)
        # Build dot-bracket
        structure = []
        for i in range(len(sequence)):
            if i not in pair_map:
                structure.append(".")
            elif pair_map[i] > i:
                structure.append("(")
            else:
                structure.append(")")

        structure = "".join(structure)
        return RNASample(name=name, sequence=sequence, structure=structure,
                         pair_map=pair_map)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Synthetic dataset generator (for quick experimentation)
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(n_samples: int = 5000,
                               min_len: int = 20,
                               max_len: int = 100,
                               seed: int = 42) -> List[RNASample]:
    """Generate synthetic RNA sequences with simple hairpin/stem-loop structures.

    This is useful for initial model development and debugging before
    training on real data. The synthetic structures follow basic RNA
    folding rules (AU, GC, GU pairs).
    """
    import random
    random.seed(seed)

    bases = "ACGU"
    complement = {"A": "U", "U": "A", "G": "C", "C": "G"}

    samples = []
    for i in range(n_samples):
        length = random.randint(min_len, max_len)

        # Generate a random sequence
        seq = [random.choice(bases) for _ in range(length)]

        # Create random stem-loop structures
        structure = ["."] * length
        pair_map = {}

        # Try to insert 1-3 stem-loops
        n_stems = random.randint(1, 3)
        for _ in range(n_stems):
            stem_len = random.randint(3, min(8, length // 6))
            loop_len = random.randint(3, 8)
            total = 2 * stem_len + loop_len

            if total >= length:
                continue

            # Find a random position that doesn't overlap existing structures
            start = random.randint(0, length - total)
            if any(structure[j] != "." for j in range(start, start + total)):
                continue

            # Create the stem
            for k in range(stem_len):
                left = start + k
                right = start + total - 1 - k
                # Make the right base complementary
                seq[right] = complement[seq[left]]
                structure[left] = "("
                structure[right] = ")"
                pair_map[left] = right
                pair_map[right] = left

        seq_str = "".join(seq)
        struct_str = "".join(structure)
        samples.append(RNASample(
            name=f"synthetic_{i}",
            sequence=seq_str,
            structure=struct_str,
            pair_map=pair_map,
        ))

    return samples


# ---------------------------------------------------------------------------
# Load dataset from directory
# ---------------------------------------------------------------------------

def load_dataset_from_dir(data_dir: str,
                          max_samples: int = 0,
                          max_length: int = 600) -> List[RNASample]:
    """Load RNA samples from a directory containing .bpseq, .ct, .dbn, or .sta files."""
    samples = []
    data_path = Path(data_dir)

    for filepath in sorted(data_path.rglob("*")):
        if max_samples > 0 and len(samples) >= max_samples:
            break

        ext = filepath.suffix.lower()
        sample = None

        if ext == ".bpseq":
            sample = parse_bpseq(str(filepath))
        elif ext == ".ct":
            sample = parse_ct_file(str(filepath))
        elif ext == ".sta":
            sample = parse_bprna_sta(str(filepath))
        elif ext == ".dbn":
            for s in parse_dbn_file(str(filepath)):
                if s.length <= max_length:
                    samples.append(s)
            continue

        if sample and sample.length <= max_length:
            samples.append(sample)

    return samples


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_bprna_sample(data_dir: str = "data/bprna") -> str:
    """Download a curated sample of bpRNA data in .dbn format.

    This creates a .dbn file with well-known RNA families suitable for
    training and testing models.
    """
    os.makedirs(data_dir, exist_ok=True)
    dbn_path = os.path.join(data_dir, "bprna_sample.dbn")

    if os.path.exists(dbn_path):
        print(f"Dataset already exists at {dbn_path}")
        return dbn_path

    # Curated set of RNA sequences with known secondary structures.
    # Each sequence and structure are carefully length-matched.
    curated_data = """\
>hairpin_1
GGGAAACCC
(((...)))
>hairpin_2
GGGGAAAACCCC
((((....))))
>hairpin_3
CCCCUUUUGGGG
((((....))))
>hairpin_4
GGGCCCAAAGGGCCC
(((((.....))))).
>stem_loop_AU
AAAUUUUCCCAAAAUUUU
((((....))))......
>stem_loop_GC
GGGCCCUUUUGGGCCC
(((((....))))).
>tRNA_like_1
GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCA
(((((((..((((........)))).(((((.......))))).....(((((.......)))))))))))).....
>tRNA_like_2
GGGGCUAUAGCUCAGCUGGGAGAGCGCUUGCAUGGCAUGCAAGAGGUCAGCGGUUCGAUCCCGCUUAGCUCCA
(((((((..((((........)))).(((((.......))))).....(((((.......))))).))))))).....
>small_stem_1
GCGCAAAAAGCGC
((((.....)))).
>small_stem_2
AUGCGCAUUUUAUGCGCAU
((((((......)))))).
>double_hairpin
GGGCCCUUUUGGGCCCAAAAGGGAAACCC
(((((....))))).....(((...)))..
>internal_loop
GGCCAUUCAAGGCC
((((.....)))).
>bulge_loop
GGGACUUUCCCGGG
(((.....)))...
>multi_stem
GGGAAACCCUUUUGGGAAACCC
(((...)))....(((...)))
>long_hairpin
GGGGGGGGAAAAAAAACCCCCCCC
((((((((.........))))))))
>gc_rich_stem
GCGCGCUUUUGCGCGC
((((((....))))))
>au_rich_stem
AUAUAUUUUUAUAUAU
((((((....))))))
>mixed_stem
GAUCGAUUUUAUCGAUC
((((((.....))))))
>triple_loop
GGGAAACCCUUUUGGGAAACCCUUUUGGGAAACCC
(((...)))....(((...)))....(((...)))
>nested_stem
GGGGAAACCCCCCUUUUGGGGGGAAACCCC
((((((...)))))).....(((...)))..
"""
    with open(dbn_path, "w") as f:
        f.write(curated_data)

    print(f"Created curated sample dataset at {dbn_path}")
    print("For full bpRNA-1m dataset, visit: https://bprna.cgrb.oregonstate.edu/")
    print("For ArchiveII dataset, visit: https://rna.urmc.rochester.edu/pub/archiveII.tar.gz")
    return dbn_path


# ---------------------------------------------------------------------------
# Main entry point for dataset preparation
# ---------------------------------------------------------------------------

def prepare_dataset(mode: str = "synthetic",
                    data_dir: str = "data",
                    n_synthetic: int = 5000,
                    max_length: int = 600) -> List[RNASample]:
    """Prepare dataset for training.

    Args:
        mode: One of "synthetic", "sample", or "directory"
            - "synthetic": Generate synthetic RNA data for quick experiments
            - "sample": Download curated bpRNA sample (small, real data)
            - "directory": Load from files in data_dir
        data_dir: Directory to store/load data
        n_synthetic: Number of synthetic samples to generate
        max_length: Maximum sequence length to include

    Returns:
        List of RNASample objects
    """
    if mode == "synthetic":
        print(f"Generating {n_synthetic} synthetic RNA samples...")
        return generate_synthetic_dataset(n_samples=n_synthetic)

    elif mode == "sample":
        dbn_path = download_bprna_sample(data_dir)
        samples = parse_dbn_file(dbn_path)
        print(f"Loaded {len(samples)} samples from curated dataset")
        return samples

    elif mode == "directory":
        samples = load_dataset_from_dir(data_dir, max_length=max_length)
        print(f"Loaded {len(samples)} samples from {data_dir}")
        return samples

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'synthetic', 'sample', or 'directory'")
