"""Formal statistical analysis of the ablation study (paper-style).

Complements ablation_aggregate.py (which does mean±sd + table + bar plot). Here we
test whether the per-variant differences are significant, using the standard
paper protocol for comparing methods across seeds:

  1. Friedman test  — omnibus: are the variants different at all (across seeds)?
  2. Wilcoxon signed-rank, each variant vs `full`, PAIRED by split seed.
  3. Holm-Bonferroni correction over the family of vs-full comparisons.

Pairing is by SPLIT seed (read from each run's config.json data.split_dir), so the
same seed's `full` and variant F1 are compared head-to-head — the correct paired
design. Test macro-F1 is the metric.

IMPORTANT (power): with only 5 seeds the paired Wilcoxon cannot reach p<0.05 (its
smallest attainable two-sided p at n=5 is 0.0625). Non-significant here means
"underpowered", not "no effect". Read effect sizes (mean delta) alongside p.

    conda run -n gnn python scripts/kg_hgnn/ablation_stats.py --results results/ablation_hetero_sage

    command line example (Windows):
    cd "d:/Testing/MOKG-HGNN" && python scripts/kg_hgnn/ablation_stats.py --results results/ablation_hetero_sage 2>&1
"""

import argparse
import glob
import json
import os

from scipy import stats

# same display order as ablation_aggregate.py (baseline first)
ORDER = ["full", "no_metapath", "no_disease", "no_pathway", "no_go",
         "readout_mol", "readout_pathway", "no_cnv", "no_mirna"]


def _seed_of(run_dir):
    """Split seed for a run, from its config.json (data.split_dir='..._seed_42')."""
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    return cfg["data"]["split_dir"].rstrip("/").split("_")[-1]


def collect(root):
    """{variant: {seed: test_macro_f1}} — one F1 per (variant, split seed)."""
    by_v = {}
    for mj in glob.glob(os.path.join(root, "*", "**", "metrics.json"), recursive=True):
        variant = os.path.relpath(mj, root).split(os.sep)[0]
        run_dir = os.path.dirname(mj)
        try:
            seed = _seed_of(run_dir)
        except (FileNotFoundError, KeyError):
            continue  # a run without config.json can't be paired
        f1 = float(json.load(open(mj))["test_macro_f1"])
        by_v.setdefault(variant, {})[seed] = f1
    return by_v


def holm(pvals):
    """Holm-Bonferroni: return adjusted p-values in the input order."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(idx):
        val = (m - rank) * pvals[i]
        running = max(running, val)          # enforce monotonicity
        adj[i] = min(running, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/ablation")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    by_v = collect(args.results)
    if "full" not in by_v:
        print(f"No 'full' baseline under {args.results}/ — cannot run vs-full tests.")
        return

    full = by_v["full"]
    variants = [v for v in ORDER if v in by_v] + [v for v in by_v if v not in ORDER]
    others = [v for v in variants if v != "full"]

    # seeds present in EVERY variant (so Friedman gets complete blocks)
    common = set(full)
    for v in others:
        common &= set(by_v[v])
    common = sorted(common)
    print(f"# Ablation statistics — {args.results}")
    print(f"# paired by split seed; common seeds across all variants: {common} (n={len(common)})\n")

    # --- 1) Friedman omnibus over all variants (incl. full) -----------------
    if len(common) >= 2:
        cols = [[by_v[v][s] for s in common] for v in variants]
        chi2, p_fried = stats.friedmanchisquare(*cols)
        print(f"Friedman (all {len(variants)} variants): chi2={chi2:.3f}  p={p_fried:.4f}")
    else:
        print("Friedman: not enough common seeds.")
    print()

    # --- 2) Wilcoxon each variant vs full, paired --------------------------
    rows, praw = [], []
    for v in others:
        seeds = sorted(set(full) & set(by_v[v]))
        a = [full[s] for s in seeds]           # full
        b = [by_v[v][s] for s in seeds]        # variant
        mean_delta = sum(x - y for x, y in zip(a, b)) / len(seeds)  # full - variant
        try:
            _, p = stats.wilcoxon(a, b)
        except ValueError:
            p = 1.0                            # all-zero differences -> no effect
        rows.append([v, len(seeds), mean_delta, p])
        praw.append(p)

    padj = holm(praw)

    print(f"{'variant':16} {'n':>2} {'delta_vs_full':>13} {'p_raw':>8} {'p_holm':>8}  signif")
    print("-" * 60)
    for (v, n, d, p), pa in zip(rows, padj):
        sig = "*" if pa < args.alpha else ""
        print(f"{v:16} {n:>2} {d:+13.4f} {p:8.4f} {pa:8.4f}  {sig}")

    print(f"\n(delta_vs_full = full - variant; positive => removing it HURTS. "
          f"alpha={args.alpha}, Holm-corrected.)")
    if len(common) < 6:
        print(f"[warning] only {len(common)} seeds: paired Wilcoxon is underpowered "
              f"(min attainable p at n=5 is 0.0625). Treat non-significant as "
              f"'insufficient data', and read the effect sizes.")


if __name__ == "__main__":
    main()
