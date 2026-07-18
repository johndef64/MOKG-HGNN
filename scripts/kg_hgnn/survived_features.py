"""Extract the miRNA / TF that SURVIVE in the MOKG-HGNN graph at a given gene
level+seed, so the MOGNN-TF collapse can be restricted to the exact same panel.

A miRNA/TF node survives only if it has an edge to a selected gene (build_hetero_
graph.load_molecular_edges applies this). The heterogeneous templates carry only
counts, not names, so we reproduce the survival rule here by reusing the builder's
own functions — deterministic, and it does NOT stream the 4.6GB KG (only the
molecular edge files), so it runs in seconds.

Given a feature-selection dir (selected_genes/tf/mirna), writes:
    <out-dir>/survived_mirna.txt   (one miRNA per line)
    <out-dir>/survived_tf.txt      (one TF per line)

    conda run -n gnn python scripts/kg_hgnn/survived_features.py \
        --fs-dir data/training/feature_selection/collapse_g700_seed42 \
        --out-dir data/training/feature_selection/collapse_g700_seed42
"""

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PRIORS = REPO / "scripts" / "preprocessing" / "priors"
for p in (str(REPO), str(PRIORS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import build_hetero_graph as bh


def _read_list(path, col=None):
    if path.endswith(".txt"):
        return [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
    df = pd.read_csv(path)
    return df[col if col in df.columns else df.columns[0]].astype(str).str.strip().tolist()


def survived(fs_dir, metapath=False):
    """(survived_mirna, survived_tf) for the panel in fs_dir, via the builder's rule."""
    genes = _read_list(os.path.join(fs_dir, "selected_genes.csv"), "symbol")
    tf_keep = set(_read_list(os.path.join(fs_dir, "selected_tf.csv"), "TF"))
    mirna_keep = set(_read_list(os.path.join(fs_dir, "selected_mirna.txt")))

    # gene vocab exactly as the builder derives it (drops genes without an Entrez)
    sym2entrez = bh.load_symbol_entrez_map()
    gene_nodes = pd.DataFrame({"symbol": genes})
    tmp = os.path.join(fs_dir, "_gene_nodes_tmp.csv")
    gene_nodes.to_csv(tmp, index=False)
    try:
        _, sym2idx, _ = bh.load_gene_vocab(tmp, sym2entrez)
    finally:
        os.remove(tmp)

    _, vocabs = bh.load_molecular_edges(
        sym2idx, mirna_keep=mirna_keep, tf_keep=tf_keep, metapath=metapath)
    return vocabs["miRNA"], vocabs["TF"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs-dir", required=True,
                    help="Dir with selected_genes.csv / selected_tf.csv / selected_mirna.txt")
    ap.add_argument("--out-dir", default=None, help="Default: same as --fs-dir")
    ap.add_argument("--metapath", action="store_true",
                    help="Match a template built WITH metapath (survival then includes co-targets).")
    args = ap.parse_args()
    out_dir = args.out_dir or args.fs_dir
    os.makedirs(out_dir, exist_ok=True)

    mirna, tf = survived(args.fs_dir, metapath=args.metapath)
    with open(os.path.join(out_dir, "survived_mirna.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(mirna) + ("\n" if mirna else ""))
    with open(os.path.join(out_dir, "survived_tf.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(tf) + ("\n" if tf else ""))
    print(f"[survived] miRNA={len(mirna)} TF={len(tf)} -> {out_dir}/survived_{{mirna,tf}}.txt")


if __name__ == "__main__":
    main()
