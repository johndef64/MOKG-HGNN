#!/usr/bin/env python
"""
TODO 6 (Level A) — Biological interpretation of the TF layer.

Post-hoc, model-free ranking of the transcription factors (TFs) that enter the
MOGNN-TF graph, by how well each one discriminates the iCluster molecular
subtypes. This is the "minimal, zero-risk" version of the TF interpretation
analysis discussed in docs/TODO_v2_comune_CiBM_JBHI.md: it reuses the same
univariate F-score (one-vs-rest f_classif) machinery already used for gene
feature selection, but applied to the TF nodes only and broken down per
subtype.

What it does, end to end:
  1. Loads the pan-cancer expression matrix and the iCluster labels, exactly as
     the experiment runner does (same file, same label column).
  2. Restricts to the training split (seed 42 by default) so no test/val
     information leaks into the ranking.
  3. Reproduces the *exact* set of TFs that the model puts on the graph, via the
     same top-variance selection (fileter_tf_per_variance) used at training time
     (default num_tf = 200).
  4. For every iCluster subtype, computes a one-vs-rest ANOVA F-score for each
     selected TF on the training expression. Higher F = the TF separates that
     subtype from the rest more strongly.
  5. Emits three artefacts under results/pancan/tf_interpretation/:
        - tf_fscore_per_subtype.csv  : long table (TF x subtype, F + p + rank)
        - tf_global_ranking.csv      : TFs ranked by aggregate discriminativeness
        - tf_top_per_subtype.md      : human-readable top-K TFs per subtype
     plus a run log printed to stdout.

This is interpretation of *which TFs carry subtype signal*, not a causal claim.
Frame it in the paper as correlative, post-hoc evidence.

Usage:
    python scripts/analysis/tf_fscore_ranking.py
    python scripts/analysis/tf_fscore_ranking.py --num-tf 200 --split-seed 42 --top-k 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

# --- repo paths -------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = REPO_ROOT / "data" / "training"
EXPRESSION_PATH = TRAIN_DIR / "expression_data_pancan.tsv"
TF_VOCAB_PATH = TRAIN_DIR / "tf_nodes_all_in_vocab.csv"
SPLITS_DIR = TRAIN_DIR / "splits"
OUT_DIR = REPO_ROOT / "results" / "pancan" / "tf_interpretation"

LABEL_COL = "icluster_cluster_assignment"


def icluster_name(original_label: int) -> str:
    """Original iCluster labels are 1..28 with 24 absent; render as C<n>."""
    return f"C{int(original_label)}"


def load_training_expression(split_seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return (exp_train, y_compact_train, original_labels_present).

    Mirrors experiment_runner: read the same expression file, take the label
    column, drop non-numeric helper columns, slice to the training indices.
    Labels are remapped to a contiguous 0..C-1 space (the iCluster set has a
    hole at 24), and we keep the original-label vector to name the subtypes.
    """
    if not EXPRESSION_PATH.exists():
        sys.exit(f"[ERROR] expression file not found: {EXPRESSION_PATH}")

    exp = pd.read_csv(EXPRESSION_PATH, sep="\t", index_col=0, header=0)
    if LABEL_COL not in exp.columns:
        sys.exit(f"[ERROR] label column '{LABEL_COL}' missing from expression file")

    labels_raw = exp[LABEL_COL].to_numpy(dtype=np.int64)
    # keep only numeric gene columns (drops 'sample' and the label col)
    exp = exp.drop(columns=[LABEL_COL]).select_dtypes(include=[np.number]).astype(np.float32)

    seed_dir = SPLITS_DIR / f"splits_seed_{split_seed}"
    train_idx_f = seed_dir / "common_trimmed_shuffle_index_train.tsv"
    if not train_idx_f.exists():
        sys.exit(f"[ERROR] training split not found: {train_idx_f}")
    train_idx = np.loadtxt(train_idx_f, dtype=int, delimiter="\t")

    exp_train = exp.iloc[train_idx]
    labels_train_raw = labels_raw[train_idx]

    classes = np.sort(np.unique(labels_train_raw))
    mapping = {lab: i for i, lab in enumerate(classes)}
    y_compact = np.array([mapping[v] for v in labels_train_raw], dtype=int)

    return exp_train, y_compact, classes


def select_graph_tfs(exp_train: pd.DataFrame, num_tf: int) -> list[str]:
    """Reproduce fileter_tf_per_variance: top-`num_tf` TFs by training-set
    expression variance, restricted to the TFLink vocabulary. This is the exact
    set of TFs the model places on the graph."""
    if not TF_VOCAB_PATH.exists():
        sys.exit(f"[ERROR] TF vocabulary not found: {TF_VOCAB_PATH}")
    tf_vocab = pd.read_csv(TF_VOCAB_PATH)
    tf_list = tf_vocab["TF"].astype(str).str.strip().tolist()

    # only TFs that actually exist as columns in the expression matrix
    available = [tf for tf in tf_list if tf in exp_train.columns]
    missing = len(tf_list) - len(available)
    if missing:
        print(f"[warn] {missing} TFs in vocab are not expression columns; ignored.")

    tf_variance = exp_train[available].var().sort_values(ascending=False)
    top_tf = tf_variance.head(num_tf).index.tolist()
    return top_tf


def compute_fscore_table(
    exp_train: pd.DataFrame,
    y_compact: np.ndarray,
    classes_original: np.ndarray,
    top_tf: list[str],
) -> pd.DataFrame:
    """One-vs-rest ANOVA F-score per (TF, subtype). Returns a long dataframe."""
    X = exp_train[top_tf].to_numpy()
    rows = []
    n_classes = len(classes_original)
    for c in range(n_classes):
        y_bin = (y_compact == c).astype(int)
        F, p = f_classif(X, y_bin)
        F = np.nan_to_num(F, nan=0.0, posinf=np.finfo(float).max, neginf=0.0)
        order = np.argsort(F)[::-1]
        rank = np.empty_like(order)
        rank[order] = np.arange(1, len(order) + 1)
        for j, tf in enumerate(top_tf):
            rows.append(
                {
                    "subtype_compact": c,
                    "subtype_icluster": icluster_name(classes_original[c]),
                    "tf": tf,
                    "f_score": float(F[j]),
                    "p_value": float(p[j]),
                    "rank_in_subtype": int(rank[j]),
                }
            )
    return pd.DataFrame(rows)


def build_global_ranking(long_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-subtype F-scores into a single TF discriminativeness rank.

    We report, per TF:
      - max_f / best_subtype : the subtype it discriminates most strongly,
      - mean_f               : average discriminativeness across subtypes,
      - n_subtypes_top10     : in how many subtypes it lands in the top 10.
    Global ranking is by max_f (the strongest single-subtype signal), which is
    the most defensible summary for "this TF marks a subtype".
    """
    g = long_df.groupby("tf")
    summary = pd.DataFrame(
        {
            "max_f": g["f_score"].max(),
            "mean_f": g["f_score"].mean(),
            "n_subtypes_top10": long_df[long_df["rank_in_subtype"] <= 10]
            .groupby("tf")
            .size()
            .reindex(g.size().index)
            .fillna(0)
            .astype(int),
        }
    )
    # best subtype = argmax over subtypes per TF
    idx = long_df.loc[long_df.groupby("tf")["f_score"].idxmax()]
    best = idx.set_index("tf")["subtype_icluster"]
    summary["best_subtype"] = best
    summary = summary.sort_values("max_f", ascending=False).reset_index()
    summary.insert(0, "global_rank", np.arange(1, len(summary) + 1))
    return summary[
        ["global_rank", "tf", "best_subtype", "max_f", "mean_f", "n_subtypes_top10"]
    ]


def write_top_per_subtype_md(long_df: pd.DataFrame, top_k: int, out_path: Path) -> None:
    lines = [
        "# Top discriminative TFs per iCluster subtype (training split, F-score OvR)",
        "",
        f"Top {top_k} TFs by one-vs-rest ANOVA F-score, per subtype. "
        "Higher F = stronger separation of that subtype from the rest.",
        "",
    ]
    for sub, sdf in long_df.groupby("subtype_icluster", sort=True):
        top = sdf.sort_values("f_score", ascending=False).head(top_k)
        tf_str = ", ".join(
            f"{r.tf} (F={r.f_score:.1f})" for r in top.itertuples()
        )
        lines.append(f"- **{sub}**: {tf_str}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-tf", type=int, default=200, help="TFs on the graph (default 200).")
    ap.add_argument("--split-seed", type=int, default=42, help="Training split seed (default 42).")
    ap.add_argument("--top-k", type=int, default=10, help="Top-K TFs listed per subtype (default 10).")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading training expression (split seed {args.split_seed}) ...")
    exp_train, y_compact, classes_original = load_training_expression(args.split_seed)
    print(
        f"      train samples={exp_train.shape[0]} genes={exp_train.shape[1]} "
        f"subtypes={len(classes_original)} "
        f"(iCluster {classes_original.min()}..{classes_original.max()}, 24 absent)"
    )

    print(f"[2/4] Selecting top-{args.num_tf} TFs by training variance ...")
    top_tf = select_graph_tfs(exp_train, args.num_tf)
    print(f"      selected {len(top_tf)} TFs; first 10: {top_tf[:10]}")

    print("[3/4] Computing one-vs-rest F-scores per subtype ...")
    long_df = compute_fscore_table(exp_train, y_compact, classes_original, top_tf)
    global_df = build_global_ranking(long_df)

    print("[4/4] Writing artefacts ...")
    long_out = OUT_DIR / "tf_fscore_per_subtype.csv"
    glob_out = OUT_DIR / "tf_global_ranking.csv"
    md_out = OUT_DIR / "tf_top_per_subtype.md"
    long_df.to_csv(long_out, index=False)
    global_df.to_csv(glob_out, index=False)
    write_top_per_subtype_md(long_df, args.top_k, md_out)

    print(f"      {long_out.relative_to(REPO_ROOT)}")
    print(f"      {glob_out.relative_to(REPO_ROOT)}")
    print(f"      {md_out.relative_to(REPO_ROOT)}")

    print("\n=== Top 20 TFs by max one-vs-rest F-score (global) ===")
    with pd.option_context("display.max_rows", 20, "display.width", 120):
        print(global_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
