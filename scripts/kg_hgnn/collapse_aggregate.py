"""Aggregate the feature-collapse experiment: mean ± s.d. of test macro-F1 per
(model, gene-count), a comparison table, and the two degradation curves overlaid.

Reads:
  - MOKG-HGNN: results/feature_collapse/mokghgnn_g<N>/<ts>/metrics.json
  - MOGNN-TF : results/feature_collapse/collapse_mognntf_summary.csv (written by
               scripts/collapse_mognntf.py), or per-run metrics if present.

Writes into --results:
  - feature_collapse_table.csv   (model, genes, n, macro_f1 mean/sd, acc mean/sd)
  - feature_collapse_curve.png   (macro-F1 vs #genes, both models, ± s.d. band)

    conda run -n gnn python scripts/kg_hgnn/collapse_aggregate.py --results results/feature_collapse
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _collect_mokghgnn(root):
    """(genes -> list of test_macro_f1, list of test_accuracy) from metrics.json."""
    by_g = {}
    for mj in glob.glob(os.path.join(root, "mokghgnn_g*", "**", "metrics.json"), recursive=True):
        m = re.search(r"mokghgnn_g(\d+)", mj)
        if not m:
            continue
        g = int(m.group(1))
        d = json.load(open(mj))
        by_g.setdefault(g, {"f1": [], "acc": []})
        by_g[g]["f1"].append(float(d.get("test_macro_f1")))
        by_g[g]["acc"].append(float(d.get("test_accuracy")))
    return by_g


def _collect_mognntf(root):
    by_g = {}
    summ = os.path.join(root, "collapse_mognntf_summary.csv")
    if os.path.exists(summ):
        for r in csv.DictReader(open(summ)):
            g = int(r["genes"])
            by_g.setdefault(g, {"f1": [], "acc": []})
            if r.get("test_macro_f1") not in (None, "", "None"):
                by_g[g]["f1"].append(float(r["test_macro_f1"]))
            if r.get("test_accuracy") not in (None, "", "None"):
                by_g[g]["acc"].append(float(r["test_accuracy"]))
    return by_g


def _mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0), len(xs)


def _shrinkage(table_path, rows):
    """Human-readable reasons why writing `rows` would LOSE data already on disk:
    a (model, genes) point disappearing, or its seed count going down."""
    if not os.path.exists(table_path):
        return []
    try:
        old = list(csv.DictReader(open(table_path)))
    except Exception:
        return []
    new = {(r["model"], int(r["genes"])): int(r["n_seeds"]) for r in rows}
    lost = []
    for r in old:
        try:
            key, n_old = (r["model"], int(r["genes"])), int(r["n_seeds"])
        except (KeyError, ValueError):
            continue
        if key not in new:
            lost.append(f"{key[0]} genes={key[1]}: {n_old} seed(s) on disk, absent now")
        elif new[key] < n_old:
            lost.append(f"{key[0]} genes={key[1]}: {n_old} seed(s) on disk, only {new[key]} now")
    return lost


def _report_robustness(data, root):
    """The actual question: which model degrades more slowly as genes shrink?

    Reports, per model, the drop from the richest to the poorest gene count and
    the slope of macro-F1 vs log2(genes) — a per-halving degradation rate, which
    is comparable across models even if their absolute F1 differs.
    """
    curves, out = {}, []
    for model, by_g in data.items():
        pts = [(g, _mean_sd(v["f1"])[0]) for g, v in sorted(by_g.items())
               if _mean_sd(v["f1"])[0] is not None]
        if len(pts) >= 2:
            curves[model] = pts

    if not curves:
        return

    print(f"\n{'model':10} {'range':>16} {'F1 drop':>9} {'per halving':>12}")
    for model, pts in curves.items():
        lo_g, lo_f1 = pts[0]        # fewest genes
        hi_g, hi_f1 = pts[-1]       # most genes
        drop = hi_f1 - lo_f1
        # least-squares slope of F1 vs log2(genes): F1 lost per halving of genes
        xs = [math.log2(g) for g, _ in pts]
        ys = [f for _, f in pts]
        mx, my = st.mean(xs), st.mean(ys)
        den = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else float("nan")
        out.append({"model": model, "genes_max": hi_g, "genes_min": lo_g,
                    "f1_at_max": round(hi_f1, 4), "f1_at_min": round(lo_f1, 4),
                    "f1_drop": round(drop, 4), "f1_per_halving": round(slope, 4)})
        print(f"{model:10} {hi_g:>6}->{lo_g:<9} {drop:>+9.4f} {slope:>12.4f}")

    path = os.path.join(root, "feature_collapse_robustness.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"[saved] {path}")

    if len(out) == 2:
        a, b = out                      # smaller drop = more robust
        winner = min(out, key=lambda r: r["f1_drop"])
        gap = abs(a["f1_drop"] - b["f1_drop"])
        print(f"\n=> {winner['model']} degrades LESS: {gap:.4f} macro-F1 of difference "
              f"over the {a['genes_max']}->{a['genes_min']} gene range.")
        print("   (a smaller drop / flatter slope = the model leans less on raw gene features)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/feature_collapse")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite the table even if it would lose points/seeds already there.")
    args = ap.parse_args()

    data = {"MOKG-HGNN": _collect_mokghgnn(args.results),
            "MOGNN-TF": _collect_mognntf(args.results)}

    # --- table ---
    rows = []
    for model, by_g in data.items():
        for g in sorted(by_g, reverse=True):
            mf, sf, n = _mean_sd(by_g[g]["f1"])
            ma, sa, _ = _mean_sd(by_g[g]["acc"])
            if mf is None:
                continue
            rows.append({"model": model, "genes": g, "n_seeds": n,
                         "macro_f1_mean": round(mf, 4), "macro_f1_sd": round(sf, 4),
                         "accuracy_mean": round(ma, 4) if ma is not None else "",
                         "accuracy_sd": round(sa, 4) if sa is not None else ""})
    if not rows:
        print(f"No results found under {args.results}/ — run the experiment first.")
        return

    # Guard against clobbering a complete table with a partial one. The runs live
    # on the server; a local checkout often holds only some of them, and silently
    # overwriting a good table with fewer seeds destroys real results.
    table_path = os.path.join(args.results, "feature_collapse_table.csv")
    lost = _shrinkage(table_path, rows)
    if lost and not args.force:
        print(f"\n[REFUSED] {table_path} already reports MORE than what was just aggregated:")
        for msg in lost:
            print(f"    {msg}")
        print("  Left untouched — it likely came from a fuller run (e.g. on the server).")
        print("  Aggregate where all runs live, or pass --force to overwrite anyway.")
        return

    with open(table_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[saved] {table_path}")
    print(f"\n{'model':10} {'genes':>6} {'n':>3} {'macro-F1':>16}")
    for r in rows:
        print(f"{r['model']:10} {r['genes']:>6} {r['n_seeds']:>3}  "
              f"{r['macro_f1_mean']:.4f} ± {r['macro_f1_sd']:.4f}")

    _report_robustness(data, args.results)

    # --- curve ---
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"MOKG-HGNN": "#4C72B0", "MOGNN-TF": "#C44E52"}
    for model, by_g in data.items():
        gs = sorted(by_g)
        pts = [(g, *_mean_sd(by_g[g]["f1"])[:2]) for g in gs if _mean_sd(by_g[g]["f1"])[0] is not None]
        if not pts:
            continue
        xs = [p[0] for p in pts]; ms = [p[1] for p in pts]; ss = [p[2] for p in pts]
        ax.plot(xs, ms, "-o", color=colors.get(model, None), label=model, lw=2)
        ax.fill_between(xs, [m - s for m, s in zip(ms, ss)],
                        [m + s for m, s in zip(ms, ss)], color=colors.get(model), alpha=0.18)
    ax.set_xscale("log")
    ax.set_xlabel("number of selected genes (log scale)")
    ax.set_ylabel("test macro-F1")
    ax.set_title("Feature-collapse: does the graph save performance?")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    ax.invert_xaxis()   # left→right = fewer genes = harder
    out_png = os.path.join(args.results, "feature_collapse_curve.png")
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)
    print(f"[saved] {out_png}")


if __name__ == "__main__":
    main()
