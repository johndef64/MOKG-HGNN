"""Annotate OLD explainability CSVs with human-readable names + real subtype labels.

Note: explain.py now writes the `name` column and the real iCluster subtype labels
natively, so a FRESH run needs no post-processing. This script is only for OLD runs
produced before that (e.g. results/explanations/20260715_152808): it back-fills the
`name` column and (with --per-class-csv) remaps the renumbered C1..C27 -> real
iCluster, writing *_named.csv next to the originals.

Name tables come from data/prior_knowledge/PKT/{pathway,go}_names.csv (built once
from PKT/nodes.zip, so we never touch the 4.6GB JSON again).

Optionally remaps the rinumbered C1..C27 subtype labels back to the REAL iCluster
values (C24/LAML is absent -> a gap): pass --per-class-csv <a run's
per_class_metrics.csv> which carries the class_idx -> real class mapping.

    conda run -n gnn python scripts/kg_hgnn/annotate_explanations.py \
        --dir results/explanations/20260715_152808 \
        --per-class-csv results/best_model_full_hetero_sage/20260715_152808/per_class_metrics.csv
"""

import argparse
import csv
import os

PKT = "data/prior_knowledge/PKT"
PATHWAY_NAMES = os.path.join(PKT, "pathway_names.csv")
GO_NAMES = os.path.join(PKT, "go_names.csv")


def load_names(path):
    if not os.path.isfile(path):
        return {}
    return {r["id"]: r["label"] for r in csv.DictReader(open(path, encoding="utf-8"))}


def load_real_labels(per_class_csv):
    """encoded 'Cn' (n = idx+1) -> real iCluster label, from a per_class file."""
    if not per_class_csv or not os.path.isfile(per_class_csv):
        return {}
    idx2real, idx2tumor = {}, {}
    for r in csv.DictReader(open(per_class_csv, encoding="utf-8")):
        idx2real[int(r["class_idx"])] = r["class"]
        idx2tumor[int(r["class_idx"])] = r.get("tumor", "")
    # explain label "Cn" -> encoded idx n-1 -> real
    remap, tumor = {}, {}
    for idx, real in idx2real.items():
        remap[f"C{idx + 1}"] = real
        tumor[f"C{idx + 1}"] = idx2tumor.get(idx, "")
    return remap, tumor


def annotate(in_csv, names, remap, tumor, out_csv):
    rows = list(csv.DictReader(open(in_csv, encoding="utf-8")))
    if not rows:
        return 0
    fields = list(rows[0].keys())
    for extra in ("name", "subtype_real", "tumor"):
        if extra not in fields:
            fields.append(extra)
    for r in rows:
        r["name"] = names.get(r.get("id", ""), "")
        if remap:
            r["subtype_real"] = remap.get(r["subtype"], r["subtype"])
            r["tumor"] = tumor.get(r["subtype"], "")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="results/explanations/<run> dir")
    ap.add_argument("--per-class-csv", default=None,
                    help="a run's per_class_metrics.csv, to remap C1..C27 -> real iCluster")
    args = ap.parse_args()

    pnames = load_names(PATHWAY_NAMES)
    gnames = load_names(GO_NAMES)
    if not pnames or not gnames:
        raise SystemExit(f"name tables missing under {PKT}/ — regenerate with: "
                         f"python scripts/kg_hgnn/extract_pkt_names.py")
    remap, tumor = load_real_labels(args.per_class_csv) if args.per_class_csv else ({}, {})

    for base, names in [("top_pathway_by_subtype.csv", pnames),
                        ("top_GO_term_by_subtype.csv", gnames)]:
        src = os.path.join(args.dir, base)
        if not os.path.isfile(src):
            print(f"[skip] {src} not found"); continue
        out = src.replace(".csv", "_named.csv")
        n = annotate(src, names, remap, tumor, out)
        print(f"[saved] {out}  ({n} rows{' + real labels' if remap else ''})")


if __name__ == "__main__":
    main()
