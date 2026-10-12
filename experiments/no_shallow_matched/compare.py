"""Comparison tables for the no-shallow matched experiment (ticket §5).

Series (all on the identical panel and the identical 5 frozen splits per level):
  MS = multi-scale MOKG-HGNN (HeteroSAGE), existing runs, read only
  WS = molecular MOGNN-TF WITH shallow branch, existing runs, read only
  NS = molecular MOGNN-TF WITHOUT shallow branch, new runs (results/runs/*.json)

Writes CSV + Markdown into results/tables/:
  comparison_by_panel        one row per level: mean +- s.d. of the three series,
                             Delta(NS - MS), Delta(NS - WS), accuracy (secondary)
  summary_metrics            one row per series: drop 700->20, loss per halving,
                             crossing with MS, s.d. at 700 and at 20
  paired_differences         one row per (level, split): paired differences
  paired_differences_summary one row per (level, contrast): mean, s.d., dz, sign
                             concordance across the 5 splits
  sensitivity_by_panel /     POST HOC (rule fixed after the results, 2026-10-12): NS
  sensitivity_summary /      without degenerate runs (constant predictors) and per-level
  sensitivity.json           medians, with the same §8 thresholds. Never replaces the above.

Definitions (identical to scripts/kg_hgnn/collapse_aggregate.py, which produced the
published 0.014 / 0.029):
  * s.d. is the sample s.d. (n - 1) across the 5 splits;
  * loss per halving = least-squares slope of the per-level MEAN macro-F1 on log2(genes)
    (positive = F1 lost each time the panel is halved);
  * drop 700->20 = mean F1 at 700 minus mean F1 at 20.
Crossing with MS: consecutive levels (700 -> 20) where the sign of the mean difference
changes; the log2-linear interpolation point is reported as a descriptive aid only.

Statistics: with 5 splits the paired Wilcoxon cannot go below p = 0.0625, so no tests
are computed; differences are reported as effect sizes (mean paired difference, its
s.d., dz = mean / s.d., and how many of the 5 splits agree in sign).

  python experiments/no_shallow_matched/compare.py
  python experiments/no_shallow_matched/compare.py --allow-partial   # draft tables
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

SERIES = {"MS": "Multi-scale (MOKG-HGNN, HeteroSAGE)",
          "WS": "Molecular with shallow (MOGNN-TF)",
          "NS": "Molecular without shallow (MOGNN-TF, no theta2)"}


def load_series(allow_partial=False, runs_dir=C.RUNS):
    ms = C.read_multiscale_existing()
    ws = C.read_with_shallow_existing()
    ns_raw = C.read_new_runs("no_shallow", runs_dir)
    grid = [(g, s) for g in C.LEVELS for s in C.SEEDS]

    # integrity of the new runs: right variant, full training, same panel/split as audited
    sanity = json.loads(C.SANITY_JSON.read_text()) if C.SANITY_JSON.exists() else {}
    panels = sanity.get("A2_panels_and_configs", {})
    splits = sanity.get("A1_splits", {})
    problems = []
    for (g, s), d in ns_raw.items():
        if d.get("model_parallel") is not False:
            problems.append(f"g{g} s{s}: model_parallel is not False")
        if d.get("smoke_epochs"):
            problems.append(f"g{g} s{s}: smoke run")
        p = panels.get(f"g{g}_seed{s}")
        if p and p["panel_hash"] != d["panel_hash"]:
            problems.append(f"g{g} s{s}: panel hash differs from the audited panel")
        sp = splits.get(str(s))
        if sp and sp["sha256"] != d["split_hash"]:
            problems.append(f"g{g} s{s}: split hash differs from the audited split")
        w = ws.get((g, s))
        if w and (w["n_mirna"] != d["n_mirna"] or w["n_tf"] != d["n_tf"]):
            problems.append(f"g{g} s{s}: panel counts differ from the with-shallow run")
    if problems:
        sys.exit("[compare] integrity problems:\n  " + "\n  ".join(problems))

    missing = {name: [k for k in grid if k not in src]
               for name, src in (("MS", ms), ("WS", ws), ("NS", ns_raw))}
    if any(missing.values()) and not allow_partial:
        sys.exit(f"[compare] incomplete series {{name: missing (level, split)}}: "
                 f"{ {k: v for k, v in missing.items() if v} }. Use --allow-partial for drafts.")
    ns = {k: {"f1": v["test_macro_f1"], "acc": v["test_accuracy"]} for k, v in ns_raw.items()}
    return {"MS": ms, "WS": ws, "NS": ns}, missing


def level_stats(src, g, key="f1"):
    vals = [src[(g, s)][key] for s in C.SEEDS if (g, s) in src]
    if not vals:
        return None, None, 0
    m, sd = C.mean_sd(vals)
    return m, sd, len(vals)


def slope_per_halving(points):
    """points: [(genes, mean_f1)] -> LS slope of F1 on log2(genes)."""
    xs = [math.log2(g) for g, _ in points]
    ys = [f for _, f in points]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else float("nan")


def crossings(mean_a, mean_b):
    """Intervals of consecutive levels (700 -> 20) where sign(mean_a - mean_b) changes.
    Returns [{"between": (L_hi, L_lo), "approx_genes": x}] (x by log2-linear interp.)."""
    out = []
    levels = [g for g in C.LEVELS if g in mean_a and g in mean_b]
    for hi, lo in zip(levels, levels[1:]):
        d1, d2 = mean_a[hi] - mean_b[hi], mean_a[lo] - mean_b[lo]
        if d1 == 0:
            out.append({"between": (hi, hi), "approx_genes": float(hi)})
        elif d1 * d2 < 0:
            t = d1 / (d1 - d2)
            x = 2 ** (math.log2(hi) + t * (math.log2(lo) - math.log2(hi)))
            out.append({"between": (hi, lo), "approx_genes": round(x, 1)})
    if levels and mean_a[levels[-1]] - mean_b[levels[-1]] == 0:
        out.append({"between": (levels[-1], levels[-1]), "approx_genes": float(levels[-1])})
    return out


def fmt_cross(cr):
    if not cr:
        return "none in 700-20"
    return "; ".join(f"{a}-{b} genes (~{c['approx_genes']:.0f})" if a != b else f"at {a} genes"
                     for c in cr for a, b in [c["between"]])


def _w_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def _w_md(path, rows, title, note=""):
    cols = list(rows[0].keys())
    lines = [f"# {title}", ""] + ([note, ""] if note else [])
    lines += ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pm(m, sd):
    return "" if m is None else f"{m:.4f} ± {sd:.4f}"


def build_tables(series):
    means = {k: {g: level_stats(v, g)[0] for g in C.LEVELS if level_stats(v, g)[2]}
             for k, v in series.items()}

    # --- comparison_by_panel --------------------------------------------------
    by_panel = []
    for g in C.LEVELS:
        row = {"genes": g}
        for k in ("MS", "WS", "NS"):
            m, sd, n = level_stats(series[k], g)
            row[f"{k}_macro_f1"] = _pm(m, sd)
            row[f"{k}_n"] = n
        for a, b in (("NS", "MS"), ("NS", "WS")):
            ma, mb = means[a].get(g), means[b].get(g)
            row[f"delta_{a}_minus_{b}"] = "" if ma is None or mb is None else f"{ma - mb:+.4f}"
        for k in ("MS", "WS", "NS"):
            m, sd, _ = level_stats(series[k], g, "acc")
            row[f"{k}_accuracy"] = _pm(m, sd)
        by_panel.append(row)

    # --- summary_metrics -------------------------------------------------------
    summary = []
    for k in ("MS", "WS", "NS"):
        pts = sorted(means[k].items())
        full = len(pts) == len(C.LEVELS)
        m700, sd700, _ = level_stats(series[k], 700)
        m20, sd20, _ = level_stats(series[k], 20)
        summary.append({
            "series": k, "description": SERIES[k],
            "f1_at_700": f"{m700:.4f}" if m700 is not None else "",
            "f1_at_20": f"{m20:.4f}" if m20 is not None else "",
            "drop_700_to_20": f"{m700 - m20:.4f}" if (m700 is not None and m20 is not None) else "",
            "loss_per_halving": f"{slope_per_halving(pts):.4f}" if full else "(incomplete)",
            "crossing_with_MS": ("n/a" if k == "MS" else
                                 fmt_cross(crossings(means[k], means["MS"])) if full else "(incomplete)"),
            "sd_at_700": f"{sd700:.4f}" if sd700 is not None else "",
            "sd_at_20": f"{sd20:.4f}" if sd20 is not None else "",
        })

    # --- paired differences ------------------------------------------------------
    paired, paired_sum = [], []
    contrasts = (("NS", "MS"), ("WS", "MS"), ("NS", "WS"))
    for g in C.LEVELS:
        for s in C.SEEDS:
            row = {"genes": g, "split_seed": s}
            for k in ("MS", "WS", "NS"):
                v = series[k].get((g, s))
                row[f"{k}_f1"] = f"{v['f1']:.4f}" if v else ""
            for a, b in contrasts:
                va, vb = series[a].get((g, s)), series[b].get((g, s))
                row[f"d_{a}_minus_{b}"] = f"{va['f1'] - vb['f1']:+.4f}" if va and vb else ""
            paired.append(row)
        for a, b in contrasts:
            ds = [series[a][(g, s)]["f1"] - series[b][(g, s)]["f1"] for s in C.SEEDS
                  if (g, s) in series[a] and (g, s) in series[b]]
            if not ds:
                continue
            m, sd = C.mean_sd(ds)
            pos, neg = sum(d > 0 for d in ds), sum(d < 0 for d in ds)
            paired_sum.append({
                "genes": g, "contrast": f"{a} - {b}", "n_splits": len(ds),
                "mean_diff": f"{m:+.4f}", "sd_diff": f"{sd:.4f}",
                "dz": f"{m / sd:+.2f}" if sd > 0 else "",
                "n_positive": pos, "n_negative": neg,
                "sign_concordant": f"{max(pos, neg)}/{len(ds)}"})
    return means, by_panel, summary, paired, paired_sum


def interpretation(series, means):
    """Reading per ticket §8, with criteria FIXED BEFORE seeing the results
    (written 2026-10-11, before any no-shallow run):
      * gap at 700: G_WS = WS - MS, G_NS = NS - MS. 'Si riduce molto' if the gap shrinks
        by >= 50 % AND by more than 0.02 macro-F1 (about two run-to-run s.d.); 'resta
        simile' if |G_NS - G_WS| <= 0.02; otherwise 'riduzione parziale'.
      * 'degrada piu' in fretta': NS loss per halving > WS loss per halving + 0.005.
      * crossing: interval(s) where NS crosses MS; 'sopra i 50 geni' if a crossing lies
        in an interval whose upper level is > 50 (i.e. 100-50 or above).
    """
    out = {}
    if all(700 in means[k] for k in ("MS", "WS", "NS")):
        g_ws = means["WS"][700] - means["MS"][700]
        g_ns = means["NS"][700] - means["MS"][700]
        change = g_ns - g_ws
        red = 1 - g_ns / g_ws if g_ws else float("nan")
        if abs(change) <= 0.02:
            verdict = "resta simile"
        elif g_ws > 0 and red >= 0.5 and -change > 0.02:
            verdict = "si riduce molto"
        elif change > 0:
            verdict = "aumenta"
        else:
            verdict = "riduzione parziale"
        out["gap_700"] = {"G_WS": g_ws, "G_NS": g_ns, "change": change,
                          "relative_reduction": red, "verdict": verdict}
    if all(len(means[k]) == len(C.LEVELS) for k in ("WS", "NS")):
        s_ws = slope_per_halving(sorted(means["WS"].items()))
        s_ns = slope_per_halving(sorted(means["NS"].items()))
        out["slope"] = {"WS": s_ws, "NS": s_ns,
                        "NS_degrades_faster": s_ns > s_ws + 0.005,
                        "NS_degrades_slower": s_ns < s_ws - 0.005}
    if len(means["NS"]) == len(C.LEVELS) and len(means["MS"]) == len(C.LEVELS):
        cr_ns = crossings(means["NS"], means["MS"])
        cr_ws = crossings(means["WS"], means["MS"])
        out["crossing"] = {"NS_vs_MS": cr_ns, "WS_vs_MS": cr_ws,
                           "NS_crossing_above_50": any(c["between"][0] > 50 and c["between"][1] >= 50
                                                       for c in cr_ns)}
    return out


def degenerate_runs(runs_dir=C.RUNS):
    """POST HOC rule, defined on 2026-10-12 AFTER seeing the results: a no-shallow run is
    degenerate when its test F1 is > 0 for at most one class (a constant predictor).
    Two such runs appeared (20 genes, splits 44 and 45)."""
    return sorted(k for k, d in C.read_new_runs("no_shallow", runs_dir).items()
                  if sum(v > 0 for v in d["per_class_f1"].values()) <= 1)


def _median(vals):
    v = sorted(vals)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def sensitivity(series, means):
    """POST HOC sensitivity analysis (not part of the criteria fixed before the runs).
    Same §8 thresholds (interpretation()), applied to (a) NS without the degenerate runs,
    (b) per-level MEDIANS of all three series instead of means. The primary analysis
    (all 35 runs, means) is unchanged and stays the reference."""
    degen = degenerate_runs()
    ns_ok = {k: v for k, v in series["NS"].items() if k not in degen}
    means_excl = {**means, "NS": {g: level_stats(ns_ok, g)[0] for g in C.LEVELS
                                  if level_stats(ns_ok, g)[2]}}
    medians = {k: {g: _median([v[(g, s)]["f1"] for s in C.SEEDS if (g, s) in v])
                   for g in C.LEVELS if any((g, s) in v for s in C.SEEDS)}
               for k, v in series.items()}
    rows = []
    for g in C.LEVELS:
        m_all, sd_all, n_all = level_stats(series["NS"], g)
        m_ok, sd_ok, n_ok = level_stats(ns_ok, g)
        rows.append({"genes": g,
                     "NS_all_mean": _pm(m_all, sd_all), "NS_all_n": n_all,
                     "NS_excl_degenerate_mean": _pm(m_ok, sd_ok) if n_ok > 1 else
                     (f"{m_ok:.4f}" if n_ok else ""), "NS_excl_n": n_ok,
                     "MS_median": f"{medians['MS'][g]:.4f}", "WS_median": f"{medians['WS'][g]:.4f}",
                     "NS_median": f"{medians['NS'][g]:.4f}",
                     "delta_NS_excl_minus_MS": f"{means_excl['NS'][g] - means['MS'][g]:+.4f}",
                     "delta_NS_median_minus_MS_median": f"{medians['NS'][g] - medians['MS'][g]:+.4f}"})
    summary = []
    for name, m in (("NS excl. degenerate (means)", means_excl), ("medians", medians)):
        for k in ("MS", "WS", "NS") if name == "medians" else ("NS",):
            pts = sorted(m[k].items())
            summary.append({"analysis": name, "series": k,
                            "f1_at_700": f"{m[k][700]:.4f}", "f1_at_20": f"{m[k][20]:.4f}",
                            "drop_700_to_20": f"{m[k][700] - m[k][20]:.4f}",
                            "loss_per_halving": f"{slope_per_halving(pts):.4f}",
                            "crossing_with_MS": "n/a" if k == "MS" else fmt_cross(crossings(m[k], m["MS"]))})
    out = {"rule": "post hoc (2026-10-12): degenerate = test F1 > 0 for at most one class",
           "degenerate_runs": [{"genes": g, "split_seed": s} for g, s in degen],
           "excl_degenerate_means": interpretation(None, means_excl),
           "medians": interpretation(None, medians)}
    return rows, summary, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    series, missing = load_series(args.allow_partial)
    means, by_panel, summary, paired, paired_sum = build_tables(series)
    C.TABLES.mkdir(parents=True, exist_ok=True)
    note = ("Test macro-F1, mean ± s.d. over the 5 frozen splits; identical panel per level. "
            "MS = MOKG-HGNN, WS = MOGNN-TF with shallow, NS = MOGNN-TF without shallow.")
    if any(missing.values()):
        note += f" DRAFT: incomplete series {({k: len(v) for k, v in missing.items() if v})} missing runs."
    for name, rows in (("comparison_by_panel", by_panel), ("summary_metrics", summary),
                       ("paired_differences", paired),
                       ("paired_differences_summary", paired_sum)):
        if not rows:
            continue
        _w_csv(C.TABLES / f"{name}.csv", rows)
        _w_md(C.TABLES / f"{name}.md", rows, name, note)
        print(f"[saved] {C.TABLES / name}.csv / .md")
    interp = interpretation(series, means)
    (C.TABLES / "interpretation.json").write_text(json.dumps(interp, indent=2, default=str))
    print(json.dumps(interp, indent=2, default=str))

    if not any(missing.values()):
        s_rows, s_sum, s_out = sensitivity(series, means)
        s_note = ("POST HOC sensitivity analysis (rule defined after seeing the results): NS "
                  "without the degenerate runs (constant predictors), and per-level medians. "
                  "The primary analysis is comparison_by_panel / summary_metrics.")
        for name, rows in (("sensitivity_by_panel", s_rows), ("sensitivity_summary", s_sum)):
            _w_csv(C.TABLES / f"{name}.csv", rows)
            _w_md(C.TABLES / f"{name}.md", rows, name, s_note)
            print(f"[saved] {C.TABLES / name}.csv / .md")
        (C.TABLES / "sensitivity.json").write_text(json.dumps(s_out, indent=2, default=str))
        print(json.dumps(s_out, indent=2, default=str))


if __name__ == "__main__":
    main()
