"""Node vocabularies (node_gene/miRNA/TF.csv) for one (level, split), verified on its template.

The collapse templates data/prior_knowledge/hetero/collapse_g{L}_seed{S}.pt hold only node
counts and edges. Feature injection reads the node NAMES from <hetero_dir>/node_*.csv, and
the shared data/prior_knowledge/hetero/ holds the names of whatever graph was built last.
Here the names are regenerated per (level, split) from the panel (unified train-only FS +
the builder's survival rule, no KG streaming) into results/four_arms/hetero/g{L}_s{S}/, and
the molecular edges rebuilt from those names with the builder's own functions must equal
the template's edges: that proves the node order. If they differ, ensure() raises.

  python scripts/four_arms/hetero_dirs.py            # all 35, prints a line per pair
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402

NS = core.NS
PRIORS = core.REPO / "scripts" / "preprocessing" / "priors"
if str(PRIORS) not in sys.path:
    sys.path.insert(0, str(PRIORS))

MOLECULAR = ("gene", "miRNA", "TF")


def hetero_dir(level, seed):
    return core.HETERO / f"g{level}_s{seed}"


def _build(level, seed):
    import pandas as pd
    import torch
    import build_hetero_graph as bh

    genes_sel, tf, mirna = NS.get_panel(level, seed, NS.load_yaml_config(NS.ORIGINAL_CONFIG))
    sym2entrez = bh.load_symbol_entrez_map()
    genes = [g for g in genes_sel if g in sym2entrez]          # = bh.load_gene_vocab
    sym2idx = {g: i for i, g in enumerate(genes)}
    edges, vocabs = bh.load_molecular_edges(sym2idx, mirna_keep=set(mirna), tf_keep=set(tf))
    rebuilt = bh._assemble_heterodata(
        n_gene=len(genes), n_mirna=len(vocabs["miRNA"]), n_tf=len(vocabs["TF"]),
        n_pathway=0, n_go=0, n_disease=0, mol_edges=edges, gene_pathway=None, gene_go=None,
        go_go=None, gene_disease=None, with_disease=False, with_pathway=False, with_go=False,
        make_undirected=True)
    template = torch.load(core.template_path(level, seed), weights_only=False)

    problems = []
    names = {"gene": genes, "miRNA": vocabs["miRNA"], "TF": vocabs["TF"]}
    for nt in MOLECULAR:
        if template[nt].num_nodes != len(names[nt]):
            problems.append(f"{nt}: template {template[nt].num_nodes} nodes, rebuilt {len(names[nt])}")
    for et in rebuilt.edge_types:
        if et not in template.edge_types:
            problems.append(f"{et} missing in template")
        elif not torch.equal(template[et].edge_index, rebuilt[et].edge_index):
            problems.append(f"{et}: edge_index differs")
    mol_et = [et for et in template.edge_types if et[0] in MOLECULAR and et[2] in MOLECULAR]
    if sorted(mol_et) != sorted(rebuilt.edge_types):
        problems.append(f"molecular edge types differ: {sorted(mol_et)} vs {sorted(rebuilt.edge_types)}")
    if problems:
        raise RuntimeError(f"[hetero_dir] g{level} s{seed}: node order NOT verified: {problems}")

    d = hetero_dir(level, seed)
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"symbol": genes, "idx": range(len(genes))}).to_csv(d / "node_gene.csv", index=False)
    pd.DataFrame({"miRNA": vocabs["miRNA"], "idx": range(len(vocabs["miRNA"]))}).to_csv(
        d / "node_miRNA.csv", index=False)
    pd.DataFrame({"TF": vocabs["TF"], "idx": range(len(vocabs["TF"]))}).to_csv(d / "node_TF.csv", index=False)
    check = {"level": level, "split_seed": seed, "verified": True,
             "num_nodes": {nt: len(names[nt]) for nt in MOLECULAR},
             "genes_without_entrez": len(genes_sel) - len(genes),
             "edges_checked": {"__".join(et): int(rebuilt[et].edge_index.shape[1])
                               for et in rebuilt.edge_types},
             "template": str(core.template_path(level, seed).relative_to(core.REPO)).replace("\\", "/")}
    core.write_json(d / "check.json", check)
    return check


def ensure(level, seed):
    """Folder with verified node vocabularies for (level, seed); built on first use."""
    d = hetero_dir(level, seed)
    c = d / "check.json"
    if not (c.exists() and json.loads(c.read_text()).get("verified")):
        _build(level, seed)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=core.LEVELS)
    ap.add_argument("--seeds", type=int, nargs="+", default=core.SEEDS)
    ap.add_argument("--rebuild", action="store_true", help="ignore existing check.json")
    a = ap.parse_args()
    for g in a.levels:
        for s in a.seeds:
            if a.rebuild:
                c = _build(g, s)
            else:
                ensure(g, s)
                c = json.loads((hetero_dir(g, s) / "check.json").read_text())
            print(f"[hetero_dir] g{g} s{s}: verified | nodes {c['num_nodes']}", flush=True)


if __name__ == "__main__":
    main()
