"""Graph topology of the two architectures along the feature-scarcity grid.

For every (level, split) it measures how fragmented the graph that carries the message
passing is, on the graphs the models were actually trained on:
  molecular  : the supra-adjacency MOGNN-TF used (adj_used.npz saved by each existing
               with-shallow run; the no-shallow runs build the identical graph, since
               panel and construction are the same and model.parallel does not touch it)
  multi-scale: the MOKG-HGNN template of the same (level, split)
               (data/prior_knowledge/hetero/collapse_g<L>_seed<S>.pt), both whole and
               restricted to its molecular layer (gene, miRNA, TF and the edges among them)
Metrics on the MOLECULAR nodes (the only nodes carrying patient data): fraction with no
neighbour (self-loops excluded), and fraction lying in the largest connected component of
the graph they belong to. Also the size of the semantic scales.

Writes results/tables/graph_topology.csv (per run) and graph_topology_by_level.{csv,md}.

  python experiments/no_shallow_matched/graph_topology.py
"""

import csv
import glob
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

MOLECULAR = ("gene", "miRNA", "TF")
TEMPLATES = C.REPO / "data" / "prior_knowledge" / "hetero"


def _stats(adj, mol_mask):
    """adj: symmetric scipy sparse (self-loops allowed); mol_mask: bool per node."""
    a = adj.tocsr().copy()
    a.setdiag(0)
    a.eliminate_zeros()
    a = ((a + a.T) > 0).astype(np.int8)
    deg = np.asarray(a.sum(axis=1)).ravel()
    _, lab = connected_components(a, directed=False)
    lcc = np.bincount(lab).argmax()
    m = np.asarray(mol_mask)
    return {"isolated_frac": float((deg[m] == 0).mean()),
            "in_lcc_frac": float((lab[m] == lcc).mean()),
            "n_components": int(lab.max() + 1)}


def molecular_graph(level, seed):
    import yaml
    for uc in glob.glob(str(C.EXISTING_ROOT / f"mognntf_g{level}" / "run_*" / "used_config.yaml")):
        cfg = yaml.safe_load(open(uc))
        if int(cfg["project"]["split_seed"]) != seed:
            continue
        f = Path(uc).parent / "adj_used.npz"
        if not f.exists():
            return None
        adj = sp.load_npz(f)
        n = len(cfg["data"]["gene_list"]) + len(cfg["data"]["mirna_keep"]) + len(cfg["data"]["tf_list"])
        if adj.shape[0] != n:
            raise RuntimeError(f"g{level} s{seed}: adj has {adj.shape[0]} nodes, panel has {n}")
        return _stats(adj, np.ones(adj.shape[0], bool)) | {"n_nodes": n}
    return None


def multiscale_graph(level, seed):
    import torch
    f = TEMPLATES / f"collapse_g{level}_seed{seed}.pt"
    if not f.exists():
        return None
    t = torch.load(f, weights_only=False)
    types = list(t.node_types)
    off, o = {}, 0
    for nt in types:
        off[nt] = o
        o += int(t[nt].num_nodes)
    rows, cols, rows_m, cols_m = [], [], [], []
    for (s, _, d), ei in ((et, t[et].edge_index.numpy()) for et in t.edge_types):
        r, c = ei[0] + off[s], ei[1] + off[d]
        rows.append(r); cols.append(c)
        if s in MOLECULAR and d in MOLECULAR:
            rows_m.append(r); cols_m.append(c)
    mol = np.zeros(o, bool)
    for nt in MOLECULAR:
        mol[off[nt]:off[nt] + int(t[nt].num_nodes)] = True

    def adj(rs, cs):
        r = np.concatenate(rs) if rs else np.array([], int)
        c = np.concatenate(cs) if cs else np.array([], int)
        return sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(o, o))

    full = _stats(adj(rows, cols), mol)
    # molecular layer alone: drop semantic nodes entirely
    idx = np.where(mol)[0]
    mol_only = _stats(adj(rows_m, cols_m).tocsr()[idx][:, idx], np.ones(len(idx), bool))
    sem = {nt: int(t[nt].num_nodes) for nt in types if nt not in MOLECULAR}
    return {"full": full, "mol_layer": mol_only, "semantic_nodes": sem,
            "n_molecular": int(mol.sum())}


def main():
    rows = []
    for g in C.LEVELS:
        for s in C.SEEDS:
            mg, ms = molecular_graph(g, s), multiscale_graph(g, s)
            r = {"genes": g, "split_seed": s}
            if mg:
                r |= {"mol_n_nodes": mg["n_nodes"], "mol_isolated_frac": round(mg["isolated_frac"], 4),
                      "mol_in_lcc_frac": round(mg["in_lcc_frac"], 4), "mol_components": mg["n_components"]}
            if ms:
                r |= {"ms_n_molecular": ms["n_molecular"],
                      "ms_mollayer_isolated_frac": round(ms["mol_layer"]["isolated_frac"], 4),
                      "ms_full_isolated_frac": round(ms["full"]["isolated_frac"], 4),
                      "ms_full_in_lcc_frac": round(ms["full"]["in_lcc_frac"], 4),
                      "ms_full_components": ms["full"]["n_components"],
                      **{f"ms_{k}_nodes": v for k, v in ms["semantic_nodes"].items()}}
            rows.append(r)
            print(r, flush=True)
    C.TABLES.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(C.TABLES / "graph_topology.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

    cols = ["mol_isolated_frac", "mol_in_lcc_frac", "ms_mollayer_isolated_frac",
            "ms_full_isolated_frac", "ms_full_in_lcc_frac", "ms_pathway_nodes", "ms_GO_term_nodes",
            "ms_disease_nodes"]
    agg = []
    for g in C.LEVELS:
        rg = [r for r in rows if r["genes"] == g]
        a = {"genes": g}
        for c in cols:
            v = [r[c] for r in rg if c in r]
            a[c] = (f"{np.mean(v):.3f}" if "frac" in c else f"{np.mean(v):.0f}") if v else ""
            a[c + "_n"] = len(v)
        agg.append(a)
    with open(C.TABLES / "graph_topology_by_level.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0]))
        w.writeheader(); w.writerows(agg)
    lines = ["# graph_topology_by_level", "",
             "Means over the splits available (n). Fractions refer to MOLECULAR nodes (gene, miRNA, "
             "TF). mol_* = graph MOGNN-TF trained on (adj_used.npz); ms_mollayer_* = MOKG-HGNN "
             "template restricted to its molecular layer; ms_full_* = whole MOKG-HGNN template.", "",
             "| " + " | ".join(agg[0]) + " |", "|" + "---|" * len(agg[0])]
    lines += ["| " + " | ".join(str(a[k]) for k in agg[0]) + " |" for a in agg]
    (C.TABLES / "graph_topology_by_level.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
