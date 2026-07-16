"""Aggregate per-class metrics across all seeds of an experiment and compare them
against the published MOGNN-TF per-class table.

This is the real per-class experiment: a single run already writes its own
per_class_metrics.csv, so the value added here is (a) mean ± s.d. across the
seeds of a multi-run experiment and (b) the head-to-head against the baseline.

Reads   results/<experiment>/<ts>/per_class_metrics.csv  (all runs found)
        paper/table_per_class_mognn-tf.csv               (reference, see
                                                          tex_per_class_to_csv.py)
Writes into --results:
  - per_class_aggregate.csv   ours: class, n_seeds, support, P/R/F1 mean+sd
  - per_class_vs_mognntf.csv  joined: ours vs reference, delta, win flag
  - per_class_vs_mognntf.png  F1 per class, ours vs reference, sorted by delta

    conda run -n gnn python scripts/kg_hgnn/per_class_aggregate.py \
        --results results/best_model_full_hetero_sage
"""

import argparse
import csv
import glob
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REFERENCE = "paper/table_per_class_mognn-tf.csv"
MAPPING = "data/omics/TCGA_iCluster_C1-C28_mapping.csv"
BASELINE = "MOGNN-TF (GCN)"   # the tuned baseline we compare against
METRICS = ["precision", "recall", "f1"]


def load_labels(path=MAPPING):
    """{class: short label}. Mixed clusters have no dominant acronym (the 'tumor'
    column of a run is then empty/nan), so fall back to the iCluster_label."""
    if not os.path.isfile(path):
        return {}
    out = {}
    for row in csv.DictReader(open(path, encoding="utf-8")):
        acr = (row.get("dominant_tumor_acronym") or "").strip()
        lab = (row.get("iCluster_label") or "").strip()
        out[row["iCluster"]] = acr or lab
    return out


def _class_key(c):
    """C1, C2, ... C28 sort numerically, not lexicographically."""
    try:
        return int(str(c).lstrip("C"))
    except ValueError:
        return 10**6


def collect_runs(root, labels):
    """{class: {metric: [values across seeds]}} + support + the run dirs used."""
    by_cls, support, runs = {}, {}, []
    for path in sorted(glob.glob(os.path.join(root, "*", "per_class_metrics.csv"))):
        runs.append(os.path.dirname(path))
        for row in csv.DictReader(open(path, encoding="utf-8")):
            cls = row["class"]
            rec = by_cls.setdefault(cls, {m: [] for m in METRICS})
            for m in METRICS:
                rec[m].append(float(row[m]))
            tumor = (row.get("tumor") or "").strip()
            if tumor in ("", "nan"):          # mixed cluster: no dominant acronym
                tumor = labels.get(cls, "")
            # support/tumor are split-dependent constants; keep the last seen
            support[cls] = (row.get("support", ""), tumor)
    return by_cls, support, runs


def _ms(xs):
    if not xs:
        return None, None
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)


def load_reference(path, model):
    """{class: {metric: (mean, sd)}} for one model of the reference table."""
    if not os.path.isfile(path):
        return {}
    out = {}
    for row in csv.DictReader(open(path, encoding="utf-8")):
        if row["model"] != model:
            continue
        out.setdefault(row["class"], {})[row["metric"]] = (float(row["mean"]), float(row["sd"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="experiment folder holding the run subdirs, e.g. results/best_model_full_hetero_sage")
    ap.add_argument("--reference", default=REFERENCE)
    ap.add_argument("--baseline", default=BASELINE,
                    help=f"model column of the reference table (default: {BASELINE})")
    args = ap.parse_args()

    by_cls, support, runs = collect_runs(args.results, load_labels())
    if not by_cls:
        raise SystemExit(
            f"No per_class_metrics.csv under {args.results}/*/ — train the model first "
            f"(each run saves it automatically), or use eval_per_class.py for older runs.")

    classes = sorted(by_cls, key=_class_key)
    n_seeds = max(len(by_cls[c]["f1"]) for c in classes)
    print(f"[runs] {len(runs)} run(s) with per-class metrics under {args.results}/")
    for r in runs:
        print(f"       {os.path.basename(r)}")

    # --- our aggregate ------------------------------------------------------
    ours, rows = {}, []
    for cls in classes:
        rec = {}
        for m in METRICS:
            mean, sd = _ms(by_cls[cls][m])
            rec[m] = (mean, sd)
        ours[cls] = rec
        sup, tumor = support.get(cls, ("", ""))
        rows.append({
            "class": cls, "tumor": tumor, "support": sup,
            "n_seeds": len(by_cls[cls]["f1"]),
            **{f"{m}_{k}": round(v, 4)
               for m in METRICS
               for k, v in zip(("mean", "sd"), rec[m])},
        })

    agg_path = os.path.join(args.results, "per_class_aggregate.csv")
    with open(agg_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[saved] {agg_path}")

    macro = _ms([ours[c]["f1"][0] for c in classes])
    print(f"[check] mean of per-class F1 (= macro-F1 averaged over {n_seeds} seeds): {macro[0]:.4f}")

    # --- comparison vs the reference table ----------------------------------
    ref = load_reference(args.reference, args.baseline)
    if not ref:
        print(f"\n[warn] no reference at {args.reference} for model '{args.baseline}' — "
              f"run scripts/kg_hgnn/tex_per_class_to_csv.py first. Skipping comparison.")
        return

    cmp_rows = []
    for cls in classes:
        if cls not in ref:
            print(f"[warn] class {cls} absent from the reference table — skipped in comparison")
            continue
        r = {"class": cls, "tumor": support.get(cls, ("", ""))[1],
             "support": support.get(cls, ("", ""))[0], "n_seeds_ours": len(by_cls[cls]["f1"])}
        for m in METRICS:
            om, osd = ours[cls][m]
            rm, rsd = ref[cls][m]
            r[f"ours_{m}"] = round(om, 4)
            r[f"ours_{m}_sd"] = round(osd, 4)
            r[f"ref_{m}"] = round(rm, 4)
            r[f"ref_{m}_sd"] = round(rsd, 4)
            r[f"delta_{m}"] = round(om - rm, 4)
        # a win we can defend: our mean beats theirs by more than their own s.d.
        r["f1_win"] = int(r["delta_f1"] > 0)
        r["f1_win_beyond_sd"] = int(r["delta_f1"] > ref[cls]["f1"][1])
        cmp_rows.append(r)

    cmp_path = os.path.join(args.results, "per_class_vs_mognntf.csv")
    with open(cmp_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cmp_rows[0].keys()))
        w.writeheader(); w.writerows(cmp_rows)
    print(f"[saved] {cmp_path}")

    # --- console summary, hardest classes first ------------------------------
    wins = [r for r in cmp_rows if r["f1_win"]]
    strong = [r for r in cmp_rows if r["f1_win_beyond_sd"]]
    print(f"\nF1 vs {args.baseline} — we win on {len(wins)}/{len(cmp_rows)} classes "
          f"({len(strong)} beyond their s.d.)\n")
    print(f"{'class':5} {'subtype':22} {'sup':>4} {'ours F1':>15} {'ref F1':>15} {'delta':>8}")
    for r in sorted(cmp_rows, key=lambda x: x["delta_f1"], reverse=True):
        flag = " *" if r["f1_win_beyond_sd"] else ""
        print(f"{r['class']:5} {str(r['tumor'])[:22]:22} {str(r['support']):>4} "
              f"{r['ours_f1']:.3f} ± {r['ours_f1_sd']:.3f}  "
              f"{r['ref_f1']:.3f} ± {r['ref_f1_sd']:.3f}  {r['delta_f1']:+.3f}{flag}")

    print(f"\nours   = MOKG-HGNN, {n_seeds} seeds, {os.path.basename(args.results)}")
    print(f"ref    = {args.baseline}, 11 seeds (paper table)")
    print("*      = our mean exceeds theirs by more than their own s.d.")

    # --- plot ---------------------------------------------------------------
    ordered = sorted(cmp_rows, key=lambda x: x["delta_f1"], reverse=True)
    labels = [r["class"] for r in ordered]
    x = range(len(ordered))
    fig, ax = plt.subplots(figsize=(max(9, 0.42 * len(ordered)), 5.5))
    ax.bar([i - 0.2 for i in x], [r["ours_f1"] for r in ordered], width=0.4,
           yerr=[r["ours_f1_sd"] for r in ordered], capsize=2,
           color="#55A868", label=f"MOKG-HGNN ({n_seeds} seeds)")
    ax.bar([i + 0.2 for i in x], [r["ref_f1"] for r in ordered], width=0.4,
           yerr=[r["ref_f1_sd"] for r in ordered], capsize=2,
           color="#4C72B0", label=f"{args.baseline} (11 seeds)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel("test F1")
    ax.set_title("Per-class F1: MOKG-HGNN vs baseline (sorted by delta)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    plot = os.path.join(args.results, "per_class_vs_mognntf.png")
    fig.savefig(plot, dpi=140); plt.close(fig)
    print(f"\n[saved] {plot}")


if __name__ == "__main__":
    main()
