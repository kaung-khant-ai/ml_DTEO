"""
prepare_splits.py

Goal
----
Create balanced train/val/test splits from the unified manifest produced by scan_videos.py.
Stratification is done across one or more categorical columns (default: label + source),
so each split keeps a similar distribution of real vs ai_generated per dataset/source.

Why this matters
---------------
- Prevents data leakage and skewed evaluation.
- Keeps your splits balanced even when datasets differ in size.
- Reproducible: random seed controls determinism.

Input
-----
data/manifests/manifest.csv with columns at least:
  clip_id, source, subset, label, subtype, duration_s, fps, width, height, sha256, rel_path, notes

Output
------
data/manifests/manifest_splits.csv  (same columns as input, but 'subset' is overwritten with train/val/test)

Examples
--------
# Default fractions: val=0.1 test=0.1 (thus train=0.8), stratify by label+source
python scripts/prepare_splits.py

# Custom fractions and seed
python scripts/prepare_splits.py --val 0.15 --test 0.15 --seed 1337

# Use only 'label' for stratification (ignore 'source')
python scripts/prepare_splits.py --strata label

# Hold out a source entirely for OOD testing (e.g., celebdf only in test)
python scripts/prepare_splits.py --holdout-sources celebdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

import pandas as pd
import numpy as np

DEFAULT_IN = Path("data/manifests/manifest.csv")
DEFAULT_OUT = Path("data/manifests/manifest_splits.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create balanced train/val/test splits from a unified manifest.")
    p.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN,
                   help=f"Input manifest CSV (default: {DEFAULT_IN})")
    p.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT,
                   help=f"Output manifest CSV (default: {DEFAULT_OUT})")

    p.add_argument("--val", dest="frac_val", type=float, default=0.10,
                   help="Validation fraction (default: 0.10)")
    p.add_argument("--test", dest="frac_test", type=float, default=0.10,
                   help="Test fraction (default: 0.10)")
    p.add_argument("--seed", dest="seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")

    # Which columns to stratify on — comma-separated, e.g., "label,source" or "label"
    p.add_argument("--strata", dest="strata", type=str, default="label,source",
                   help='Comma-separated columns to stratify on (default: "label,source")')

    # Deduplicate by sha256 first (recommended if you scanned overlapping folders)
    p.add_argument("--dedupe", action="store_true",
                   help="If set, drop duplicate rows by sha256 before splitting.")

    # (Optional) put some sources *entirely* into test for out-of-distribution evaluation
    p.add_argument("--holdout-sources", dest="holdout_sources", type=str, default="",
                   help="Comma-separated list of sources to reserve for TEST only (e.g., 'celebdf,dfdc').")

    args = p.parse_args()

    if args.frac_val < 0 or args.frac_test < 0 or (args.frac_val + args.frac_test) >= 1:
        p.error("Fractions must be >=0 and val+test < 1 (train gets the remainder).")

    return args


def summarize(df: pd.DataFrame, title: str, by: List[str] = ["subset", "label", "source"]) -> None:
    """
    Print a small pivot-like summary so you can quickly verify balance.
    """
    print(f"\n=== {title} ===")
    if not set(by).issubset(df.columns):
        print(f"[warn] cannot summarize by {by}, columns missing; printing overall counts only.")
        print(df["subset"].value_counts(dropna=False))
        return

    # Count by grouping keys
    counts = df.groupby(by).size().rename("count").reset_index()
    # Make a nicer table: subset x (label, source)
    try:
        table = counts.pivot_table(index=by[0], columns=by[1:], values="count", fill_value=0, aggfunc="sum")
        print(table)
    except Exception:
        print(counts.sort_values(by))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    in_path: Path = args.in_path
    out_path: Path = args.out_path

    if not in_path.exists():
        print(f"[err] Input manifest not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(in_path)
    required_cols = {"clip_id", "source", "label", "sha256", "rel_path"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[err] Input manifest missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    # 1) (Optional) deduplicate by sha256 — keeps the first occurrence
    if args.dedupe:
        before = len(df)
        df = df.sort_values("sha256").drop_duplicates("sha256", keep="first")
        print(f"[dedupe] dropped {before - len(df)} duplicates (by sha256)")

    # 2) Parse stratification columns
    strata_cols = [c.strip() for c in args.strata.split(",") if c.strip()]
    for c in strata_cols:
        if c not in df.columns:
            print(f"[err] stratification column '{c}' not in manifest columns.", file=sys.stderr)
            sys.exit(1)

    # 3) Optional: move holdout sources entirely to TEST split (OOD evaluation)
    holdout = [s.strip() for s in args.holdout_sources.split(",") if s.strip()]
    df["subset"] = "train"  # initialize; will be overwritten below
    if holdout:
        mask_hold = df["source"].isin(holdout)
        df.loc[mask_hold, "subset"] = "test"
        print(f"[holdout] placed {mask_hold.sum()} rows from sources {holdout} into TEST only")

    # 4) We will stratify *only* the non-holdout rows
    df_work = df.loc[df["subset"] != "test"].copy()

    # Shuffle for reproducibility before group-wise allocation
    df_work = df_work.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # 5) Allocate per-group counts
    #    For each stratum (e.g., label+source= (ai_generated, celebdf)), divide into train/val/test
    val_frac = args.frac_val
    test_frac = args.frac_test
    df_work["subset"] = ""  # reset working subset

    for keys, group in df_work.groupby(strata_cols):
        n = len(group)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        # Ensure at least 1 in train if the group is non-empty
        n_train = max(0, n - n_test - n_val)

        # Edge cases: tiny groups
        if n > 0 and n_train == 0:
            # Prefer reducing val/test by one if possible to guarantee at least 1 train
            if n_val > 0:
                n_val -= 1
                n_train += 1
            elif n_test > 0:
                n_test -= 1
                n_train += 1

        # Index slicing is safe because we shuffled earlier
        idx = group.index
        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]

        df_work.loc[test_idx, "subset"] = "test"
        df_work.loc[val_idx, "subset"] = "val"
        df_work.loc[train_idx, "subset"] = "train"

    # 6) Merge back with holdout rows (already marked as 'test')
    out = pd.concat([df_work, df.loc[df["subset"] == "test"]], ignore_index=True)

    # 7) Save and print summaries
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[done] wrote splits → {out_path.as_posix()}")

    # Helpful summaries — check that balance looks reasonable
    summarize(out, "Counts by subset/label/source", by=["subset", "label", "source"])
    print("\nSubset sizes:\n", out["subset"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
