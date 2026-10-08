#!/usr/bin/env python3
"""
score_at_uz.py - rescore a benchmark score.json at a given utilization.

Reads RESULTS_DIR/score.json written by score.py, recalculates the GPU class
score (median per-GPU cost per million tokens) at utilization UZ, prints it on
stdout and records it in score.json as custom_uz_score.

Usage:
  score_at_uz.py RESULTS_DIR UZ

  UZ  utilization, a fraction greater than 0 and at most 1
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

def _fraction(text):
    """Parse a number greater than 0 and at most 1."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text}")
    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError(f"must be greater than 0 and at most 1, got {text}")
    return value

def score_at_uz(results_dir, uz):
    """Rescore score.json at utilization uz and record it as custom_uz_score."""
    path = results_dir / "score.json"
    if not path.is_file():
        sys.exit(f"ERROR: no {path}; run score.py first")
    data = json.loads(path.read_text())
    # Uz only divides the fixed term, which is stored at the score's own Uz.
    score = statistics.median(g["energy_cost_per_mtok"] + g["fixed_cost_per_mtok"] * data["uz"] / uz
                              for g in data["gpus"])
    data["custom_uz_score"] = {"uz": uz, "cost_per_mtok": score}
    path.write_text(json.dumps(data, indent=2) + "\n")
    return score

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("uz", type=_fraction, help="utilization, greater than 0 and at most 1")
    args = parser.parse_args()
    return score_at_uz(args.results_dir, args.uz)

if __name__ == "__main__":
    print(main())
