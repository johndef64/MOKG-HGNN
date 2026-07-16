"""Assemble train/val/test HeteroData datasets + loaders for the new model.

This is the HeteroData analog of the data-prep section of MOGNN-TF's
experiment_runner (load omics -> scale on train -> build graph features ->
wrap per split) plus builder.build_data_loader. It reuses the shared topology
template and injects per-patient features via patient_features.build_features.

CLI (quick sanity build):
    conda run -n gnn python -m multiomics_kg_hgnn.pancancer_prediction.preprocessing.make_datasets
"""

import os

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler

from multiomics_kg_hgnn.pancancer_prediction.datasets.HeteroOmicDataset import HeteroOmicsDataset
from multiomics_kg_hgnn.pancancer_prediction.preprocessing import patient_features as pf

# repo-root-relative defaults (cwd = repo root)
HETERO_DIR = "data/prior_knowledge/hetero"
TEMPLATE = os.path.join(HETERO_DIR, "hetero_graph_template.pt")
EXPRESSION = "data/training/expression_data_pancan.tsv"
CNV = "data/training/cnv_data_pancan.tsv"
# full 743-miRNA matrix (from feature_selection.export_full_mirna), so the loader
# can serve whichever miRNAs the (variance-selected) template asks for.
MIRNA = "data/training/mirna_data_full.tsv"
LABELS = "data/training/molecular_subtype.csv"


def make_datasets(
    split_dir,
    template_path=TEMPLATE,
    hetero_dir=HETERO_DIR,
    expression_path=EXPRESSION,
    cnv_path=CNV,
    mirna_path=MIRNA,
    labels_csv=LABELS,
    use_cnv=True,
    use_mirna=True,
    scaler="standard",
):
    """Return (train_ds, val_ds, test_ds, num_classes, classes).

    ``classes`` are the ORIGINAL iCluster label values present in the data (e.g.
    1..28 with 24/LAML absent), in the same order as the contiguous 0..C-1 encoding
    used for training. It maps each model index back to its real subtype so that
    per-class reports carry the true iCluster name (not a renumbered C{idx+1}).
    """
    template = torch.load(template_path, weights_only=False)
    samples, y, classes = pf.load_labels(labels_csv)
    train_idx, val_idx, test_idx = pf.load_split(split_dir)

    # features: scaler fitted on TRAIN rows only, applied to all patients
    feats = pf.build_features(
        hetero_dir=hetero_dir, expression_path=expression_path, labels_csv=labels_csv,
        train_idx=train_idx, cnv_path=cnv_path, mirna_path=mirna_path,
        use_cnv=use_cnv, use_mirna=use_mirna, scaler=scaler,
    )

    make = lambda idx: HeteroOmicsDataset(template, feats, y, idx)
    print(f"[datasets] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
          f"| classes={len(classes)}")
    return make(train_idx), make(val_idx), make(test_idx), len(classes), classes


def build_loaders(train_ds, val_ds, test_ds, batch_size=16, num_workers=0,
                  weighted_sampling=False, gamma=1.0):
    """DataLoaders over lists of HeteroData (PyG collate offsets node indices)."""
    sampler, shuffle = None, True
    if weighted_sampling:
        w = train_ds.get_weight_pancan(gamma=gamma)
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        shuffle = False
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                              sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", default="data/training/splits/splits_seed_42")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-cnv", action="store_true")
    ap.add_argument("--no-mirna", action="store_true")
    args = ap.parse_args()

    tr, va, te, ncls, _ = make_datasets(
        args.split_dir, use_cnv=not args.no_cnv, use_mirna=not args.no_mirna)
    train_loader, _, _ = build_loaders(tr, va, te, batch_size=args.batch_size)
    batch = next(iter(train_loader))
    print("\n=== first training batch (HeteroData) ===")
    print(batch)
    print("y:", tuple(batch.y.shape), "| num graphs:", batch.num_graphs)
