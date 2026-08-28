"""Loaders for features.csv (TabNet + baselines) and sequences.npy (LSTM).

M1's exports:
  data/features.csv   — engineered features + DEFAULT_LABEL column
  data/sequences.npy  — shape (n, 48, n_feats_seq), row-aligned with features.csv

Labels come from features.csv[DEFAULT_LABEL]; sequences.npy has no label file.
Row order in sequences.npy is assumed to match row order in features.csv.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple
from config import CFG


def _load_labels_from_features() -> np.ndarray:
    df = pd.read_csv(CFG.features_csv, usecols=[CFG.label_column])
    return df[CFG.label_column].to_numpy().astype(np.int64)


def load_tabular() -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, feature_names) for TabNet + baselines.

    Auto-drops any non-numeric columns (IDs, dates) alongside the label.
    """
    df = pd.read_csv(CFG.features_csv)
    if CFG.label_column not in df.columns:
        raise KeyError(f"{CFG.label_column!r} not in features.csv columns: "
                       f"{list(df.columns)[:10]}...")
    y = df[CFG.label_column].to_numpy().astype(np.int64)

    # Drop label + any non-numeric columns (IDs, dates, strings)
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    drop_cols = list(set([CFG.label_column] + non_numeric))
    dropped_extras = [c for c in drop_cols if c != CFG.label_column]
    if dropped_extras:
        print(f"[data] auto-dropped non-numeric columns: {dropped_extras}")

    X = df.drop(columns=drop_cols).to_numpy().astype(np.float32)
    feat_names = [c for c in df.columns if c not in drop_cols]
    return X, y, feat_names


def load_sequences() -> Tuple[np.ndarray, np.ndarray]:
    """Return (sequences, y) with shapes (n, 48, n_feats_seq) and (n,).

    Labels are pulled from features.csv[DEFAULT_LABEL]. Row order must match.
    """
    seq = np.load(CFG.sequences_npy).astype(np.float32)
    y = _load_labels_from_features()
    if len(seq) != len(y):
        raise ValueError(
            f"Row-count mismatch between sequences.npy ({len(seq)}) and "
            f"features.csv ({len(y)}). M1 needs to confirm row order alignment."
        )
    return seq, y


if __name__ == "__main__":
    X, y, names = load_tabular()
    print(f"tabular: X={X.shape}, y={y.shape}, positives={y.sum()}/{len(y)} "
          f"({y.mean():.1%})")
    print(f"first 5 features: {names[:5]}")
    seq, y2 = load_sequences()
    print(f"sequences: seq={seq.shape}, y={y2.shape}")
    assert (y == y2).all(), "label vectors disagree between tabular and sequence loaders"
    print("✓ label vectors agree")
