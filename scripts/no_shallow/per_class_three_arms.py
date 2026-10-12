"""Per-class F1 of the three arms at the full panel (700 genes), paired by split.

The per-class figure of the paper compares MOKG-HGNN with the published MOGNN-TF (shallow
branch included, different seeds and panel). Here the three arms are compared on the
identical panel and the identical five splits:
  MS: results/feature_collapse/mokghgnn_g700/<ts>/per_class_metrics.csv
  WS: results/feature_collapse/mognntf_g700/run_*/confusion_matrix_cm.csv
  NS: results/no_shallow/runs/no_shallow_g700_seed<S>.json
Class names: older MOKG-HGNN runs may carry the renumbered labels (C24 instead of C25,
see the iCluster label fix); classes are aligned by position in the encoder order and
the test support of every class is checked to be identical across the three arms.

Writes results/no_shallow/tables/per_class_three_arms.{csv,md}.

  python scripts/no_shallow/per_class_three_arms.py [--level 700]
"""

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def ms_runs(level):
    out = {}
    for f in glob.glob(str(C.EXISTING_ROOT / f"mokghgnn_g{level}" / "*" / "per_class_metrics.csv")):
        run = Path(f).parent
        cfg = json.loads((run / "config.json").read_text())
        seed = int(cfg["data"]["split_dir"].rstrip("/\\").split("_")[-1])
        rows = list(csv.DictReader(open(f, newline="", encoding="utf-8")))
        rows.sort(key=lambda r: int(r["class_idx"]))
        out[seed] = [(float(r["f1"]), int(r["support"])) for r in rows]
    return out


def ws_runs(level):
    out = {}
    for seed, run in C.existing_mognntf_run_dirs(level).items():
        if not (run / "confusion_matrix_cm.csv").exists():
            continue
        f1, sup, _ = C.per_class_from_confusion(run)
        out[seed] = (list(f1), [(f1[k], sup[k]) for k in f1])
    return out


def load(g):
    """(names, seeds, per, support): per[arm][class] = list of per-split F1 (seed order),
    for the splits available in all three arms, after the support check."""
    ms, ws = ms_runs(g), ws_runs(g)
    ns = {s: d for (lv, s), d in C.read_new_runs().items() if lv == g}
    seeds = sorted(set(ms) & set(ws) & set(ns))
    if not seeds:
        sys.exit("[per-class] no split available for all three arms")
    names = ws[seeds[0]][0]
    per = {k: {n: [] for n in names} for k in ("MS", "WS", "NS")}
    for s in seeds:
        w_names, w = ws[s]
        m = ms[s]
        if w_names != names or len(m) != len(names):
            sys.exit(f"[per-class] class list mismatch on split {s}")
        for i, n in enumerate(names):
            nf, nsup = ns[s]["per_class_f1"][n], ns[s]["per_class_support"][n]
            if not (m[i][1] == w[i][1] == nsup):
                sys.exit(f"[per-class] support mismatch split {s} class {n}")
            per["MS"][n].append(m[i][0]); per["WS"][n].append(w[i][0]); per["NS"][n].append(nf)
    support = {n: ws[seeds[0]][1][i][1] for i, n in enumerate(names)}
    return names, seeds, per, support


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=700)
    args = ap.parse_args()
    g = args.level
    ms, ws = ms_runs(g), ws_runs(g)
    ns = {s: d for (lv, s), d in C.read_new_runs().items() if lv == g}
    seeds = sorted(set(ms) & set(ws) & set(ns))
    if not seeds:
        sys.exit("[per-class] no split available for all three arms")
    names = ws[seeds[0]][0]                     # real iCluster labels, encoder order
    per = {k: {n: [] for n in names} for k in ("MS", "WS", "NS")}
    for s in seeds:
        w_names, w = ws[s]
        m = ms[s]
        n_f1 = [ns[s]["per_class_f1"][n] for n in w_names]
        n_sup = [ns[s]["per_class_support"][n] for n in w_names]
        if w_names != names or len(m) != len(names):
            sys.exit(f"[per-class] class list mismatch on split {s}")
        for i, n in enumerate(names):
            if not (m[i][1] == w[i][1] == n_sup[i]):
                sys.exit(f"[per-class] support mismatch split {s} class {n}: "
                         f"MS {m[i][1]} WS {w[i][1]} NS {n_sup[i]}")
            per["MS"][n].append(m[i][0]); per["WS"][n].append(w[i][0]); per["NS"][n].append(n_f1[i])
    rows = []
    for n in names:
        mean = {k: sum(per[k][n]) / len(seeds) for k in per}
        rows.append({"class": n, "support_per_split": ws[seeds[0]][1][names.index(n)][1],
                     "MS_f1": f"{mean['MS']:.3f}", "WS_f1": f"{mean['WS']:.3f}", "NS_f1": f"{mean['NS']:.3f}",
                     "MS_minus_WS": f"{mean['MS'] - mean['WS']:+.3f}",
                     "MS_minus_NS": f"{mean['MS'] - mean['NS']:+.3f}",
                     "MS_beats_NS_splits": f"{sum(a > b for a, b in zip(per['MS'][n], per['NS'][n]))}/{len(seeds)}"})
    macro = {k: sum(sum(per[k][n]) for n in names) / (len(names) * len(seeds)) for k in per}
    C.TABLES.mkdir(parents=True, exist_ok=True)
    out = C.TABLES / "per_class_three_arms.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    note = (f"Level {g} genes, splits {seeds} (identical panel and splits for the three arms). "
            f"Mean of per-class F1 over the splits. Macro (mean over classes and splits): "
            f"MS {macro['MS']:.4f}, WS {macro['WS']:.4f}, NS {macro['NS']:.4f}.")
    lines = ["# per_class_three_arms", "", note, "", "| " + " | ".join(rows[0]) + " |",
             "|" + "---|" * len(rows[0])] + ["| " + " | ".join(str(r[k]) for k in rows[0]) + " |" for r in rows]
    (C.TABLES / "per_class_three_arms.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
