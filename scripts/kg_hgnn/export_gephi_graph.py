#!/usr/bin/env python
"""Export the multi-scale heterogeneous template as Gephi node/edge tables.

Reads the node/edge CSVs written by build_hetero_graph.py (the surviving
connectivity after feature selection) and emits two CSVs importable in Gephi
(or gephi-lite online): a node table (Id, Label, Scale) coloured by biological
scale, and an edge table (Source, Target, Relation) with the seven canonical
relations only (reverse edges and optional metapaths are dropped).

Usage:
    python scripts/kg_hgnn/export_gephi_graph.py \
        --hetero-dir data/prior_knowledge/hetero \
        --names-dir  data/prior_knowledge/PKT \
        --out-dir    paper/gephi
"""
import argparse
import os
import csv

# node type -> (node CSV, id column). idx column is always "idx".
NODE_FILES = {
    "gene":     ("node_gene.csv",     "symbol"),
    "miRNA":    ("node_miRNA.csv",    "miRNA"),
    "TF":       ("node_TF.csv",       "TF"),
    "pathway":  ("node_pathway.csv",  "reactome_id"),
    "GO_term":  ("node_GO_term.csv",  "go_id"),
    "disease":  ("node_disease.csv",  "mondo_id"),
}
# canonical (forward) edge files -> (src type, dst type). rev_* / shares_target dropped.
EDGE_FILES = {
    "edge_gene_interacts_gene.csv":         ("gene", "gene"),
    "edge_miRNA_targets_gene.csv":          ("miRNA", "gene"),
    "edge_TF_regulates_gene.csv":           ("TF", "gene"),
    "edge_gene_member_of_pathway.csv":      ("gene", "pathway"),
    "edge_gene_annotated_with_GO_term.csv": ("gene", "GO_term"),
    "edge_GO_term_is_a_GO_term.csv":        ("GO_term", "GO_term"),
    "edge_gene_associated_with_disease.csv":("gene", "disease"),
}
SCALE = {  # pretty scale label for colouring in Gephi
    "gene": "gene", "miRNA": "miRNA", "TF": "TF",
    "pathway": "pathway", "GO_term": "GO", "disease": "disease",
}


def load_names(path):
    m = {}
    if not os.path.exists(path):
        return m
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m[row["id"]] = row.get("label", "")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hetero-dir", default="data/prior_knowledge/hetero")
    ap.add_argument("--names-dir", default="data/prior_knowledge/PKT")
    ap.add_argument("--out-dir", default="paper/gephi")
    args = ap.parse_args()

    pw_name = load_names(os.path.join(args.names_dir, "pathway_names.csv"))
    go_name = load_names(os.path.join(args.names_dir, "go_names.csv"))

    # ---- nodes: (type, idx) -> global Id; also collect rows ----
    id_of = {}            # (type, idx) -> "type:rawid"
    node_rows = []        # (Id, Label, Scale)
    for ntype, (fname, idcol) in NODE_FILES.items():
        path = os.path.join(args.hetero_dir, fname)
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                idx = int(row["idx"])
                raw = row[idcol]
                gid = f"{ntype}:{raw}"
                if ntype == "pathway":
                    label = pw_name.get(raw, raw)
                elif ntype == "GO_term":
                    label = go_name.get(raw, raw)
                else:
                    label = raw                # gene/TF/miRNA symbol, disease MONDO id
                id_of[(ntype, idx)] = gid
                node_rows.append((gid, label, SCALE[ntype]))

    # ---- edges: read positional idx columns (0 = src_idx, 2 = dst_idx) ----
    edge_rows = []        # (Source, Target, Relation)
    for fname, (stype, dtype) in EDGE_FILES.items():
        path = os.path.join(args.hetero_dir, fname)
        if not os.path.exists(path):
            print(f"[warn] missing {fname}, skipped")
            continue
        rel = fname[len("edge_"):-len(".csv")]
        with open(path, encoding="utf-8") as f:
            r = csv.reader(f)
            next(r)  # header
            for cols in r:
                if len(cols) < 3:
                    continue
                s = id_of.get((stype, int(cols[0])))
                d = id_of.get((dtype, int(cols[2])))
                if s and d:
                    edge_rows.append((s, d, rel))

    os.makedirs(args.out_dir, exist_ok=True)
    np = os.path.join(args.out_dir, "gephi_nodes.csv")
    ep = os.path.join(args.out_dir, "gephi_edges.csv")
    with open(np, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Label", "Scale"])
        w.writerows(node_rows)
    with open(ep, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Source", "Target", "Type", "Relation"])
        for s, d, rel in edge_rows:
            w.writerow([s, d, "Undirected", rel])

    print(f"[saved] {np}  ({len(node_rows)} nodes)")
    print(f"[saved] {ep}  ({len(edge_rows)} edges)")
    # scale breakdown
    from collections import Counter
    sc = Counter(r[2] for r in node_rows)
    print("nodes by scale:", dict(sc))


if __name__ == "__main__":
    main()
