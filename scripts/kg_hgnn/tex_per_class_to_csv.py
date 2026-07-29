"""Convert the MOGNN-TF per-class LaTeX table into a tidy CSV reference.

The paper table (paper/table_per_class_mognn-tf.tex) reports, for five compared
models, per-class precision/recall/F1 as "mean $\\pm$ s.d." over eleven seeds,
with the best value per metric in \\textbf{}. This turns it into a long-format CSV
so per_class_aggregate.py can join our runs against it.

Output columns:
    model, class, metric, mean, sd, is_best

    conda run -n gnn python scripts/kg_hgnn/tex_per_class_to_csv.py
"""

import argparse
import csv
import os
import re

# Column groups in the tabular, left to right (3 columns each: P, R, F1).
MODELS = ["Li24 (GAT)", "Li24 (GCN)", "MoGCN", "MOGNN-TF (GAT)", "MOGNN-TF (GCN)"]
METRICS = ["precision", "recall", "f1"]

# "\textbf{0.863} $\pm$ 0.162" / "0.445 $\pm$ 0.192"
CELL_RE = re.compile(
    r"^\s*(?P<bold>\\textbf\{)?\s*(?P<mean>[0-9.]+)\}?\s*\$\\pm\$\s*(?P<sd>[0-9.]+)\s*$")
ROW_RE = re.compile(r"^\s*(?P<cls>C\d+)\s*&")


def parse_cell(cell):
    """('\\textbf{0.863} $\\pm$ 0.162') -> (0.863, 0.162, True)."""
    m = CELL_RE.match(cell)
    if not m:
        raise ValueError(f"unparsable cell: {cell!r}")
    return float(m.group("mean")), float(m.group("sd")), bool(m.group("bold"))


def parse_table(tex_path):
    rows = []
    for line in open(tex_path, encoding="utf-8"):
        if not ROW_RE.match(line):
            continue
        line = line.strip().rstrip("\\").strip()
        cells = [c.strip() for c in line.split("&")]
        cls, cells = cells[0], cells[1:]
        if len(cells) != len(MODELS) * len(METRICS):
            raise ValueError(f"{cls}: expected {len(MODELS) * len(METRICS)} cells, got {len(cells)}")
        for mi, model in enumerate(MODELS):
            for qi, metric in enumerate(METRICS):
                mean, sd, best = parse_cell(cells[mi * len(METRICS) + qi])
                rows.append({"model": model, "class": cls, "metric": metric,
                             "mean": mean, "sd": sd, "is_best": int(best)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default="paper/table_per_class_mognn-tf.tex")
    ap.add_argument("--out", default="paper/table_per_class_mognn-tf.csv")
    args = ap.parse_args()

    rows = parse_table(args.tex)
    if not rows:
        raise SystemExit(f"no class rows parsed from {args.tex}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "class", "metric", "mean", "sd", "is_best"])
        w.writeheader()
        w.writerows(rows)

    classes = sorted({r["class"] for r in rows}, key=lambda c: int(c[1:]))
    print(f"[saved] {args.out}")
    print(f"  {len(rows)} rows | {len(MODELS)} models x {len(classes)} classes x {len(METRICS)} metrics")
    print(f"  classes: {classes[0]}..{classes[-1]} "
          f"(missing: {sorted(set(range(1, 29)) - {int(c[1:]) for c in classes})})")


if __name__ == "__main__":
    main()
