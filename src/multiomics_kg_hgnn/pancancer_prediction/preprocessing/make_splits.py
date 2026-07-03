"""Generate the stratified train/val/test split indices for one or more seeds.

Standalone step (in MOGNN-TF these were produced inside the experiment runner).
The hetero pipeline needs the splits BEFORE feature selection, so this exposes
them as an explicit preprocessing step. Reuses MOGNN-TF's split generator so the
indices are identical to the baseline (same stratification, same file names).

    conda run -n gnn python -m multiomics_kg_hgnn.pancancer_prediction.preprocessing.make_splits \
        --seeds 42 43 44 45 46 47 48 49 50 51 52
"""

import os

from multiomics_gnn.pancancer_prediction.utils.split import generate_stratified_shuffle_indices

LABELS = "data/training/molecular_subtype.csv"
SPLITS_ROOT = "data/training/splits"


def make_splits(seeds, labels_csv=LABELS, splits_root=SPLITS_ROOT,
                test_size=0.2, val_size=0.25):
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(
            f"{labels_csv} not found. Run the data preprocessing (data-wrap) first.")
    for seed in seeds:
        out_dir = os.path.join(splits_root, f"splits_seed_{seed}")
        base = os.path.join(out_dir, "common_trimmed_shuffle_index_train.tsv")
        if os.path.exists(base):
            print(f"[splits] seed {seed}: already present, skipping ({out_dir})")
            continue
        print(f"[splits] seed {seed}: generating -> {out_dir}")
        generate_stratified_shuffle_indices(
            labels_csv=labels_csv, out_dir=out_dir,
            test_size=test_size, val_size=val_size, random_state=seed)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate stratified train/val/test splits.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="Seeds to generate (e.g. --seeds 42 43 44).")
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--splits-root", default=SPLITS_ROOT)
    args = ap.parse_args()
    make_splits(args.seeds, labels_csv=args.labels, splits_root=args.splits_root)
