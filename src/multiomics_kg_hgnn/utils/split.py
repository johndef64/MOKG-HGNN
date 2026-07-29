"""Stratified train/val/test split generator.

Vendored from the MOGNN-TF baseline so the MOKG-HGNN framework has no
dependency on the benchmark package. The logic is byte-for-byte equivalent to
the original (same stratification, same random_state handling, same output file
names), so the indices it produces are identical to the baseline's.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_stratified_shuffle_indices(
    out_dir: str,
    labels_csv: str,
    test_size: float = 0.2,
    val_size: float = 0.25,
    train_fraction: float = 1.0,
    label_column: str = 'icluster_cluster_assignment',
    random_state: int = 42,
    logger=None,
):
    if logger is None:
        logger = logging.getLogger("split_utils")

    labels_csv = Path(labels_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"SPLIT SEED: {random_state}")

    df = pd.read_csv(labels_csv)
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in CSV.")
    y = df[label_column].to_numpy()

    indices = np.arange(len(df))
    # Test split
    train_idx, test_idx, y_train, _ = train_test_split(
        indices, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # Validation split
    train_idx, val_idx, _, _ = train_test_split(
        train_idx, y_train, test_size=val_size, stratify=y_train, random_state=random_state
    )
    # Optional: reduce the training set while keeping stratification
    train_fraction_indices = None
    if train_fraction < 1.0:
        if not (0.0 < train_fraction < 1.0):
            raise ValueError("train_fraction must be in (0, 1].")
        y_train_full = y[train_idx]
        train_fraction_indices, _, _, _ = train_test_split(
            train_idx, y_train_full,
            test_size=(1.0 - train_fraction),
            stratify=y_train_full,
            random_state=random_state,
        )
        train_fraction_indices = np.asarray(train_fraction_indices, dtype=int)
        logger.info(
            f"Train fraction applied: {train_fraction:.3f}. "
            f"Train set reduced {len(train_idx)} -> {len(train_fraction_indices)} samples."
        )

    # Save indices to disk
    prefix = out_dir / "common_trimmed_shuffle_index"
    logger.info(f"Saving train/val/test indices to {out_dir}")
    logger.info(f"Train indices: {len(train_idx)} samples")
    np.savetxt(str(prefix) + "_train.tsv", train_idx, fmt='%d')

    if train_fraction_indices is not None and train_fraction < 1.0:
        pct = int(round(train_fraction * 100))
        np.savetxt(str(prefix) + f"_train_fraction_{pct}.tsv", train_fraction_indices, fmt="%d")
        logger.info(f"Train fraction indices saved successfully with {len(train_fraction_indices)} samples.")

    np.savetxt(str(prefix) + "_val.tsv", val_idx, fmt='%d')
    logger.info(f"Validation indices saved successfully with {len(val_idx)} samples.")
    np.savetxt(str(prefix) + "_test.tsv", test_idx, fmt='%d')
    logger.info(f"Test indices saved successfully with {len(test_idx)} samples.")
    logger.info("Indices saved successfully.")

    # Log stratified distribution of the splits
    logger.info("Stratified distribution of splits:")

    def print_distribution(name, idx):
        unique, counts = np.unique(y[idx], return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        logger.info(f"{name} size={len(idx)} distribution: {dist}")

    print_distribution("Train", train_idx)
    if train_fraction_indices is not None and train_fraction < 1.0:
        print_distribution("Train (after fraction reduction)", train_fraction_indices)
    print_distribution("Validation", val_idx)
    print_distribution("Test", test_idx)
    logger.info("Stratified splits generated successfully.")
