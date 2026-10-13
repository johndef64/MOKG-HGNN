"""Analysis of the version-3 factorial experiment (plan §3): six arms, paired contrasts.

Criteria fixed BEFORE the runs (plan §3), applied to every contrast at every level on the
splits shared by the arms involved:
  material      |mean paired difference| >= 0.02 and the sign agrees in >= 80% of the splits
                (4 of 5); 0.02 = twice the run-to-run noise between replicates (~0.01)
  parity        |mean| < 0.01
  inconclusive  otherwise
Across all levels: sign test (two-sided binomial) on the pooled paired differences.

Writes results/four_arms/tables/{summary_metrics,by_level,paired_contrasts,
decomposition_700}.{csv,md} and contrasts.json.

  python scripts/four_arms/compare.py                  # requires all six arms complete
  python scripts/four_arms/compare.py --allow-partial  # whatever is available (preview)
"""

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "ns_compare", core.REPO / "scripts" / "no_shallow" / "compare.py")
nsc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nsc)          # slope_per_halving, crossings, fmt_cross

MATERIAL, PARITY, AGREE = 0.02, 0.01, 0.8

CONTRASTS = [
    ("S_MOL", "shallow contribution in MOGNN-TF (MOL+S - MOL)", {"mol_s": 1, "mol": -1}),
    ("S_KG", "shallow contribution in MOKG-HGNN (KG+S - KG)", {"kg_s": 1, "kg": -1}),
    ("I", "interaction S_KG - S_MOL", {"kg_s": 1, "kg": -1, "mol_s": -1, "mol": 1}),
    ("G_0", "graph vs graph without shallow (KG - MOL)", {"kg": 1, "mol": -1}),
    ("G_S", "graph vs graph with shallow (KG+S - MOL+S)", {"kg_s": 1, "mol_s": -1}),
    ("GAMMA_MOL", "molecular graph on top of features (MOL+S - SH-tf)", {"mol_s": 1, "sh_tf": -1}),
    ("GAMMA_KG", "knowledge graph on top of features (KG+S - SH-kg)", {"kg_s": 1, "sh_kg": -1}),
    ("A_MOL", "molecular graph alone vs features alone (MOL - SH-tf)", {"mol": 1, "sh_tf": -1}),
    ("A_KG", "knowledge graph alone vs features alone (KG - SH-kg)", {"kg": 1, "sh_kg": -1}),
]
# Added AFTER seeing the results (declared as post hoc in the report): the same shallow MLP
# under the two training recipes. Since G_S = GAMMA_KG - GAMMA_MOL + R_RECIPE exactly, it
# tells how much of the graph-vs-graph difference with shallow comes from the recipe.
POST_HOC = [
    ("R_RECIPE", "POST HOC: training recipe, same MLP (SH-kg - SH-tf)", {"sh_kg": 1, "sh_tf": -1}),
]


def paired(series, weights, level):
    arms = list(weights)
    seeds = sorted(set.intersection(*[{s for (g, s) in series[a] if g == level} for a in arms]))
    return seeds, [sum(w * series[a][(level, s)] for a, w in weights.items()) for s in seeds]


def verdict(diffs):
    n = len(diffs)
    if n == 0:
        return "missing"
    m = sum(diffs) / n
    agree = sum(1 for d in diffs if d * m > 0)
    if abs(m) >= MATERIAL and agree >= math.ceil(AGREE * n):
        return "material +" if m > 0 else "material -"
    if abs(m) < PARITY:
        return "parity"
    return "inconclusive"


def sign_test(diffs):
    pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    n = pos + neg
    if n == 0:
        return pos, neg, float("nan")
    k = min(pos, neg)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return pos, neg, min(1.0, 2 * p)


def _write(name, rows, title, note=""):
    core.TABLES.mkdir(parents=True, exist_ok=True)
    with open(core.TABLES / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cols = list(rows[0])
    md = [f"# {title}", "", note, "", "| " + " | ".join(cols) + " |",
          "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (core.TABLES / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def f4(x):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.4f}"


def fa(x):
    return "" if x is None else f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args()
    series = core.load_f1_series()

    missing = [(arm, g, s) for arm in core.ARMS for g in core.LEVELS for s in core.SEEDS
               if (g, s) not in series[arm]]
    if missing and not a.allow_partial:
        sys.exit(f"[compare] {len(missing)} runs missing, e.g. {missing[:5]}; "
                 f"use --allow-partial for a preview")

    means, by_level = {}, []
    for arm, label in core.ARMS.items():
        means[arm] = {}
        for g in core.LEVELS:
            v = [series[arm][(g, s)] for s in core.SEEDS if (g, s) in series[arm]]
            if v:
                m, sd = core.NS.mean_sd(v)
                means[arm][g] = m
                by_level.append({"arm": arm, "description": label, "genes": g, "n": len(v),
                                 "mean_f1": f"{m:.4f}", "sd": f"{sd:.4f}"})
    if not by_level:
        sys.exit("[compare] no data")

    summary = []
    for arm, label in core.ARMS.items():
        mm = means[arm]
        if not mm:
            continue
        pts = sorted(mm.items())
        summary.append({
            "arm": arm, "description": label, "levels": len(mm),
            "f1_at_700": f"{mm[700]:.4f}" if 700 in mm else "",
            "f1_at_20": f"{mm[20]:.4f}" if 20 in mm else "",
            "drop_700_to_20": f"{mm[700] - mm[20]:.4f}" if 700 in mm and 20 in mm else "",
            "loss_per_halving": f"{nsc.slope_per_halving(pts):.4f}" if len(pts) > 1 else "",
            "crossing_with_kg": nsc.fmt_cross(nsc.crossings(mm, means["kg"])) if arm != "kg" else "n/a",
        })

    contrasts, cjson = [], {}
    for key, label, w in CONTRASTS + POST_HOC:
        pooled = []
        cjson[key] = {"label": label, "levels": {}}
        for g in core.LEVELS:
            seeds, d = paired(series, w, g)
            pooled += d
            m, sd = core.NS.mean_sd(d) if d else (None, None)
            dz = (m / sd) if d and sd else None
            v = verdict(d)
            contrasts.append({"contrast": key, "genes": g, "n_splits": len(d), "mean_diff": f4(m),
                              "sd_diff": f"{sd:.4f}" if sd is not None else "",
                              "dz": f"{dz:+.2f}" if dz is not None else "",
                              "n_pos": sum(x > 0 for x in d), "n_neg": sum(x < 0 for x in d),
                              "verdict": v})
            cjson[key]["levels"][g] = {"splits": seeds, "diffs": d, "mean": m, "verdict": v}
        pos, neg, p = sign_test(pooled)
        cjson[key]["sign_test"] = {"n_pos": pos, "n_neg": neg, "p_two_sided": p}
        contrasts.append({"contrast": key, "genes": "all", "n_splits": len(pooled),
                          "mean_diff": f4(sum(pooled) / len(pooled)) if pooled else "",
                          "sd_diff": "", "dz": "", "n_pos": pos, "n_neg": neg,
                          "verdict": f"sign test p={p:.3g}" if pooled else "missing"})

    decomp = []
    for arch, (sh, graph, full) in {"MOGNN-TF": ("sh_tf", "mol", "mol_s"),
                                    "MOKG-HGNN": ("sh_kg", "kg", "kg_s")}.items():
        m = {k: means[k].get(700) for k in (sh, graph, full)}
        decomp.append({
            "architecture": arch, "features_only (SH)": fa(m[sh]), "graph_only": fa(m[graph]),
            "graph_plus_shallow": fa(m[full]),
            "graph_on_features": f4(m[full] - m[sh]) if m[full] is not None and m[sh] is not None else "",
            "shallow_on_graph": f4(m[full] - m[graph]) if m[full] is not None and m[graph] is not None else "",
        })

    partial = " PARTIAL PREVIEW: some runs are missing." if missing else ""
    note = ("Test macro-F1 over the 5 frozen splits, identical panel per level, model seed 2025."
            + partial)
    _write("by_level", by_level, "Six arms by level", note)
    _write("summary_metrics", summary, "Six arms: summary", note)
    _write("paired_contrasts", contrasts, "Paired contrasts (plan §3)",
           note + f" Material: |mean| >= {MATERIAL} with sign agreement in >= {AGREE:.0%} of "
           f"splits; parity: |mean| < {PARITY}; otherwise inconclusive.")
    _write("decomposition_700", decomp, "Decomposition at 700 genes (means)", note)
    core.write_json(core.TABLES / "contrasts.json", {"partial": bool(missing), "contrasts": cjson})
    print((core.TABLES / "summary_metrics.md").read_text(encoding="utf-8"))
    print((core.TABLES / "decomposition_700.md").read_text(encoding="utf-8"))
    print(f"[compare] tables -> {core.TABLES}" + (f" ({len(missing)} runs missing)" if missing else ""))


if __name__ == "__main__":
    main()
