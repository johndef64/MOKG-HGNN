"""Per-patient feature injection for the heterogeneous backbone.

This is the HeteroData analog of MOGNN-TF's ``down_unified_data_with_TF``: it
takes the raw omics matrices and produces, for every patient, the node feature
tensors aligned to the shared topology template
(``data/prior_knowledge/hetero/hetero_graph_template.pt``).

Difference from the old homogeneous pipeline: there, gene/miRNA/TF were collapsed
into one node-index space and features were zero-padded into shared channels.
Here each node type keeps its OWN feature matrix and channel count, which is what
the type-specific projection of a HeteroGNN expects:

    gene   -> [N_patients, n_gene, C_mol]   (expression [, CNV])
    TF     -> [N_patients, n_tf,   C_mol]   (expression [, CNV]; TFs are genes)
    miRNA  -> [N_patients, n_mirna, 1]      (miRNA expression)

Alignment is by identifier, not by row position: omics columns are reindexed to
the node-vocabulary order stored in ``node_gene.csv`` / ``node_TF.csv`` /
``node_miRNA.csv``. Nodes present in the template but absent from the omics are
zero-filled, and the coverage is reported (relevant e.g. when the template's
miRNA vocabulary is larger than the miRNA panel actually measured).

The pathway / GO_term / disease scales carry NO per-patient signal: they stay
featureless in the template and the model gives them learned embeddings.
"""

import os

import numpy as np
import pandas as pd

_SCALERS = {"standard": "StandardScaler", "minmax": "MinMaxScaler", "robust": "RobustScaler"}


# ---------------------------------------------------------------------------
# raw omics + labels
# ---------------------------------------------------------------------------
def read_omic(path, sep="\t"):
    """Read an omics matrix and return it indexed by TCGA ``sample``.

    Handles both layouts seen in data/training/:
      - expression/CNV: an unnamed row-index col, a ``sample`` column, then features
      - miRNA: ``sample`` already as the first column (becomes the index)
    Feature columns only are returned (``sample`` moved to the index).
    """
    df = pd.read_csv(path, sep=sep, index_col=0)
    if "sample" in df.columns:
        df = df.set_index("sample")
    else:
        df.index.name = "sample"  # miRNA layout: sample was col 0
    # drop any non-feature columns (the label ships inside the expression file)
    df = df.drop(columns=[c for c in ("icluster_cluster_assignment", "sample_id") if c in df.columns])
    return df


def load_labels(labels_csv, label_column="icluster_cluster_assignment"):
    """Return (samples, y_encoded, classes). ``samples`` is the canonical order
    the positional train/val/test splits index into (same as split.py)."""
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(labels_csv, index_col=0)
    samples = df["sample"].astype(str).to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(df[label_column].to_numpy())
    return samples, y.astype(np.int64), le.classes_


def load_split(split_dir, prefix="common_trimmed_shuffle_index"):
    """Load positional train/val/test indices from a splits_seed_* directory."""
    def _rd(name):
        return np.loadtxt(os.path.join(split_dir, f"{prefix}_{name}.tsv"), dtype=int)
    return _rd("train"), _rd("val"), _rd("test")


# ---------------------------------------------------------------------------
# alignment + scaling
# ---------------------------------------------------------------------------
def _align_samples(df, samples):
    """Reindex omics rows to the canonical sample order; zero-fill missing."""
    aligned = df.reindex(samples)
    missing = int(aligned.isna().all(axis=1).sum())
    if missing:
        print(f"[features] {missing}/{len(samples)} samples missing in an omic -> zero-filled")
    return aligned.fillna(0.0)


def _scale(df, train_idx, kind):
    """Fit the scaler on TRAIN rows only, transform the whole matrix (no leakage;
    matches the old runner)."""
    if kind is None or str(kind).lower() in {"none", ""}:
        return df
    import importlib
    cls = getattr(importlib.import_module("sklearn.preprocessing"), _SCALERS[str(kind).lower()])
    scaler = cls()
    scaler.fit(df.iloc[train_idx])
    return pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)


def _stack_to_nodes(channels, node_ids, samples):
    """Build a [N_patients, n_nodes, C] tensor by reindexing each channel's
    columns to the node vocabulary order. Nodes absent from a channel -> 0.

    channels: list of (name, DataFrame indexed by sample, columns = feature ids)
    node_ids: ordered list of node identifiers (the type's vocabulary)
    """
    mats, covered = [], None
    for name, df in channels:
        sub = df.reindex(columns=node_ids)  # [N_samples, n_nodes], NaN where absent
        present = sub.notna().any(axis=0).to_numpy()
        covered = present if covered is None else (covered | present)
        mats.append(sub.reindex(samples).fillna(0.0).to_numpy(dtype=np.float32))
    x = np.stack(mats, axis=-1)  # [N_patients, n_nodes, C]
    cov = 100.0 * covered.mean() if covered is not None else 0.0
    return x, cov


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def build_features(
    hetero_dir,
    expression_path,
    labels_csv,
    train_idx,
    cnv_path=None,
    mirna_path=None,
    use_cnv=True,
    use_mirna=True,
    scaler="standard",
):
    """Assemble per-type per-patient feature tensors aligned to the template vocab.

    Returns a dict:
        {"gene": [N, n_gene, C], "TF": [N, n_tf, C], "miRNA": [N, n_mirna, 1]}
    (miRNA present only if use_mirna). C = 1 (expr) or 2 (expr+cnv).
    """
    samples, _, _ = load_labels(labels_csv)

    gene_ids = pd.read_csv(os.path.join(hetero_dir, "node_gene.csv"))["symbol"].astype(str).tolist()
    tf_ids = pd.read_csv(os.path.join(hetero_dir, "node_TF.csv"))["TF"].astype(str).tolist()

    expr = _scale(_align_samples(read_omic(expression_path), samples), train_idx, scaler)
    channels = [("expr", expr)]
    if use_cnv and cnv_path:
        cnv = _scale(_align_samples(read_omic(cnv_path), samples), train_idx, scaler)
        channels.append(("cnv", cnv))

    feats = {}
    feats["gene"], cov_g = _stack_to_nodes(channels, gene_ids, samples)
    feats["TF"], cov_t = _stack_to_nodes(channels, tf_ids, samples)
    print(f"[features] gene: {feats['gene'].shape} (coverage {cov_g:.1f}%) | "
          f"TF: {feats['TF'].shape} (coverage {cov_t:.1f}%)")

    if use_mirna and mirna_path:
        mirna_ids = pd.read_csv(os.path.join(hetero_dir, "node_miRNA.csv"))["miRNA"].astype(str).tolist()
        mirna = _scale(_align_samples(read_omic(mirna_path), samples), train_idx, scaler)
        feats["miRNA"], cov_m = _stack_to_nodes([("mirna", mirna)], mirna_ids, samples)
        print(f"[features] miRNA: {feats['miRNA'].shape} (coverage {cov_m:.1f}%)")
        if cov_m < 50:
            print(f"[features] NOTE: only {cov_m:.1f}% of template miRNA nodes are measured. "
                  f"For a clean run rebuild the template with a miRNA vocabulary matching "
                  f"your panel (build_hetero_graph.py), or accept the zero-featured nodes.")

    return feats
