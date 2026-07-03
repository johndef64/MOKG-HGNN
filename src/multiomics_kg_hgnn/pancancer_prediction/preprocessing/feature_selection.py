"""Unified variance-based feature selection for the heterogeneous model.

Methodology (matches MOGNN-TF and satisfies the coherence requirement):
- ONE criterion for every molecular level: variance.
- ONE moment: all levels selected together, downstream, from the FULL sets
  (~15.9k genes, 743 miRNA, all TFs) -- not miRNA-first-then-genes.
- Computed on the TRAIN split only (no leakage), exactly like the old runner's
  ``variance_FS`` (genes) and ``fileter_tf_per_variance`` (TFs).

MOGNN-TF selects genes, TFs and miRNAs all by variance
(preprocess_pancan_traning.py: high_variance_expression_gene / variance_FS /
fileter_tf_per_variance; config feature_selection_method: variance), so applying
variance to all three here keeps the comparison controlled.

Outputs (into ``out_dir``) feed build_hetero_graph.py:
    selected_genes.csv   (col 'symbol')   -> --gene-list
    selected_tf.csv      (col 'TF')        -> --tf-list
    selected_mirna.txt   (one per line)    -> --mirna-list

The full miRNA matrix (743) is materialized once from data/omics/mirna.zip,
since data/training only ships the pre-cut top-100 panel.
"""

import os
import zipfile

import numpy as np
import pandas as pd

from multiomics_kg_hgnn.pancancer_prediction.preprocessing import patient_features as pf

# repo-root-relative defaults
OMICS_MIRNA_ZIP = "data/omics/mirna.zip"
FULL_MIRNA_TSV = "data/training/mirna_data_full.tsv"
EXPRESSION = "data/training/expression_data_pancan.tsv"
LABELS = "data/training/molecular_subtype.csv"
TF_VOCAB = "data/training/tf_nodes_all_in_vocab.csv"  # full TF vocabulary (col 'TF')


def export_full_mirna(mirna_zip=OMICS_MIRNA_ZIP, out_tsv=FULL_MIRNA_TSV):
    """Materialize the full pancan miRNA matrix (743) from data/omics/mirna.zip,
    sample-indexed, so downstream code can rank/select over ALL miRNAs."""
    if os.path.exists(out_tsv):
        print(f"[FS] full miRNA already present: {out_tsv}")
        return out_tsv
    with zipfile.ZipFile(mirna_zip) as z:
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        df = pd.read_csv(z.open(name))
    sample_col = "sample_id" if "sample_id" in df.columns else "sample"
    df = df.rename(columns={sample_col: "sample"}).set_index("sample")
    df.to_csv(out_tsv, sep="\t")
    print(f"[FS] exported full miRNA {df.shape} -> {out_tsv}")
    return out_tsv


def _train_variance(df, train_idx, samples, cols=None):
    """Variance per column computed on TRAIN rows only (leakage-free)."""
    aligned = df.reindex(samples)
    if cols is not None:
        aligned = aligned.reindex(columns=cols)
    train_rows = aligned.iloc[train_idx]
    return train_rows.var(axis=0, ddof=1, numeric_only=True).sort_values(ascending=False)


def select_features(
    split_dir,
    out_dir,
    top_genes=700,
    top_tf=200,
    top_mirna=100,
    expression_path=EXPRESSION,
    labels_csv=LABELS,
    tf_vocab_path=TF_VOCAB,
    mirna_full_path=None,
):
    """Write the selected gene / TF / miRNA panels for a given split."""
    os.makedirs(out_dir, exist_ok=True)
    samples, _, _ = pf.load_labels(labels_csv)
    train_idx, _, _ = pf.load_split(split_dir)
    mirna_full_path = mirna_full_path or export_full_mirna()

    expr = pf.read_omic(expression_path)

    # genes: top-N by train variance over the full expression matrix
    gene_var = _train_variance(expr, train_idx, samples)
    genes = gene_var.head(top_genes).index.tolist()
    pd.DataFrame({"symbol": genes}).to_csv(os.path.join(out_dir, "selected_genes.csv"), index=False)

    # TFs: top-K by train variance, restricted to the TF vocabulary (TFs are genes)
    tf_vocab = pd.read_csv(tf_vocab_path)["TF"].astype(str).str.strip().tolist()
    tf_present = [t for t in tf_vocab if t in expr.columns]
    tf_var = _train_variance(expr, train_idx, samples, cols=tf_present)
    tfs = tf_var.head(top_tf).index.tolist()
    pd.DataFrame({"TF": tfs}).to_csv(os.path.join(out_dir, "selected_tf.csv"), index=False)

    # miRNAs: top-M by train variance over the FULL 743 panel
    mirna = pf.read_omic(mirna_full_path)
    mirna_var = _train_variance(mirna, train_idx, samples)
    mirnas = mirna_var.head(top_mirna).index.tolist()
    with open(os.path.join(out_dir, "selected_mirna.txt"), "w") as fh:
        fh.write("\n".join(mirnas) + "\n")

    print(f"[FS] split={os.path.basename(split_dir)} | "
          f"genes {len(genes)}/{expr.shape[1]} | "
          f"TF {len(tfs)}/{len(tf_present)} | miRNA {len(mirnas)}/{mirna.shape[1]}")
    print(f"[FS] written -> {out_dir}/selected_genes.csv, selected_tf.csv, selected_mirna.txt")
    return {"genes": genes, "tf": tfs, "mirna": mirnas}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Unified variance FS (gene/TF/miRNA) on the train split.")
    ap.add_argument("--split-dir", default="data/training/splits/splits_seed_42")
    ap.add_argument("--out-dir", default=None, help="Default: data/training/feature_selection/<split>")
    ap.add_argument("--top-genes", type=int, default=700)
    ap.add_argument("--top-tf", type=int, default=200)
    ap.add_argument("--top-mirna", type=int, default=100)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(
        "data/training/feature_selection", os.path.basename(args.split_dir))
    select_features(
        split_dir=args.split_dir, out_dir=out_dir,
        top_genes=args.top_genes, top_tf=args.top_tf, top_mirna=args.top_mirna)
