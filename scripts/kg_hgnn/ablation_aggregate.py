"""Aggregate the ablation study: mean ± s.d. of test macro-F1 per variant, with
the delta vs the full baseline, plus a bar plot. Paper-style table.

Reads results/ablation/<variant>/<ts>/metrics.json.
Writes into --results:
  - ablation_table.csv   (variant, n, macro_f1 mean/sd, delta vs full, accuracy)
  - ablation_bars.png    (macro-F1 per variant, ± s.d., full highlighted)

    conda run -n gnn python scripts/kg_hgnn/ablation_aggregate.py --results results/ablation
"""

import argparse
import csv
import glob
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# display order (baseline first)
ORDER = ["full", "no_metapath", "no_disease", "no_pathway", "no_go",
         "readout_mol", "readout_pathway", "no_cnv", "no_mirna"]


def _collect(root):
    by_v = {}
    for mj in glob.glob(os.path.join(root, "*", "**", "metrics.json"), recursive=True):
        # variant = first path component under root
        rel = os.path.relpath(mj, root)
        variant = rel.split(os.sep)[0]
        d = json.load(open(mj))
        by_v.setdefault(variant, {"f1": [], "acc": []})
        by_v[variant]["f1"].append(float(d.get("test_macro_f1")))
        by_v[variant]["acc"].append(float(d.get("test_accuracy")))
    return by_v


def _ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0), len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/ablation")
    args = ap.parse_args()

    by_v = _collect(args.results)
    if not by_v:
        print(f"No results under {args.results}/ — run ablation.sh first.")
        return

    full_mean = _ms(by_v["full"]["f1"])[0] if "full" in by_v else None
    variants = [v for v in ORDER if v in by_v] + [v for v in by_v if v not in ORDER]

    rows = []
    for v in variants:
        mf, sf, n = _ms(by_v[v]["f1"])
        ma, sa, _ = _ms(by_v[v]["acc"])
        delta = (mf - full_mean) if (full_mean is not None and mf is not None) else None
        rows.append({
            "variant": v, "n_seeds": n,
            "macro_f1_mean": round(mf, 4) if mf is not None else "",
            "macro_f1_sd": round(sf, 4) if sf is not None else "",
            "delta_vs_full": round(delta, 4) if delta is not None else "",
            "accuracy_mean": round(ma, 4) if ma is not None else "",
        })

    table = os.path.join(args.results, "ablation_table.csv")
    with open(table, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[saved] {table}\n")
    print(f"{'variant':16} {'n':>2} {'macro-F1':>16} {'Δ vs full':>10}")
    for r in rows:
        d = r["delta_vs_full"]
        dtxt = f"{d:+.4f}" if isinstance(d, float) else ""
        print(f"{r['variant']:16} {r['n_seeds']:>2}  {r['macro_f1_mean']:.4f} "
              f"± {r['macro_f1_sd']:.4f}  {dtxt:>10}")

    # --- bar plot ---
    labels = [r["variant"] for r in rows]
    means = [r["macro_f1_mean"] for r in rows]
    sds = [r["macro_f1_sd"] for r in rows]
    colors = ["#55A868" if v == "full" else "#4C72B0" for v in labels]

    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(labels)), 5))
    ax.bar(range(len(labels)), means, yerr=sds, capsize=4, color=colors)
    if full_mean is not None:
        ax.axhline(full_mean, color="#55A868", ls="--", lw=1, alpha=0.7, label="full baseline")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("test macro-F1")
    ax.set_title("Ablation: one factor at a time (± s.d.)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = os.path.join(args.results, "ablation_bars.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
