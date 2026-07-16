"""Aggregate multi-seed runs into a summary JSON, per experiment folder.

Run with NO arguments and it scans results/ recursively, finds EVERY folder holding
more than one run (a run = a dir with metrics.json), and writes a summary.json into
each. That covers best_model_*/ , ablation_*/<variant>/ , feature_collapse/* , etc.

    conda run -n gnn python scripts/kg_hgnn/summarize_runs.py            # all of results/
    python scripts/kg_hgnn/summarize_runs.py --results results/best_model_full_hetero_sage
    python scripts/kg_hgnn/summarize_runs.py --root results/ablation_hetero_sage
    python scripts/kg_hgnn/summarize_runs.py --metric test_accuracy --min-runs 1

For every numeric metric in metrics.json it computes mean, sd, sem, median, min/max
and a 95% CI; identifies the best run (by --metric, default test_macro_f1); and
aggregates the per-class table across the runs (mean/sd of f1/precision/recall).

Runs are the multi-seed protocol: SPLIT seed varies (42..46), model seed fixed. The
95% CI uses the t-distribution (small n): mean ± t(0.975, n-1) * sem. With a single
run, sd/CI are null rather than 0 — one run has no spread.
"""
import argparse
import csv
import glob
import json
import os
import statistics as st

try:
    from scipy import stats as sstats
except ImportError:              # CI still works, just wider (normal approx)
    sstats = None


def _run_dirs(root):
    return sorted(d for d in glob.glob(os.path.join(root, "*"))
                  if os.path.isfile(os.path.join(d, "metrics.json")))


def _meta(run):
    """Seed / backbone / graph info for a run, best-effort."""
    out = {"run": os.path.basename(run)}
    try:
        cfg = json.load(open(os.path.join(run, "config.json")))
        out["split_seed"] = cfg["data"]["split_dir"].rstrip("/\\").split("_")[-1]
        out["model_seed"] = cfg["project"].get("seed")
        out["backbone"] = cfg["model"].get("backbone")
        out["readout_types"] = cfg["model"].get("readout_types")
    except (FileNotFoundError, KeyError):
        pass
    gi = os.path.join(run, "graph_info.json")
    if os.path.isfile(gi):
        try:
            g = json.load(open(gi))
            out["has_metapath"] = g.get("has_metapath")
            out["node_types"] = g.get("node_types")
        except json.JSONDecodeError:
            pass
    return out


def _stats(vals):
    """mean/sd/sem/median/min/max + 95% CI. sd & CI are null when n < 2."""
    n = len(vals)
    mean = st.mean(vals)
    if n < 2:
        return {"n": n, "mean": mean, "sd": None, "sem": None, "median": mean,
                "min": mean, "max": mean, "ci95_low": None, "ci95_high": None}
    sd = st.stdev(vals)
    sem = sd / (n ** 0.5)
    tcrit = sstats.t.ppf(0.975, n - 1) if sstats else 1.96
    return {"n": n, "mean": mean, "sd": sd, "sem": sem, "median": st.median(vals),
            "min": min(vals), "max": max(vals),
            "ci95_low": mean - tcrit * sem, "ci95_high": mean + tcrit * sem}


def _per_class(runs):
    """Mean/sd of per-class f1/precision/recall across runs, keyed by class."""
    acc = {}
    for run in runs:
        f = os.path.join(run, "per_class_metrics.csv")
        if not os.path.isfile(f) or os.path.getsize(f) == 0:
            continue
        for r in csv.DictReader(open(f)):
            e = acc.setdefault(r["class"], {"class_idx": int(r["class_idx"]),
                                            "tumor": r.get("tumor", ""),
                                            "support": int(r["support"]),
                                            "f1": [], "precision": [], "recall": []})
            for k in ("f1", "precision", "recall"):
                e[k].append(float(r[k]))
    out = []
    for cls, e in sorted(acc.items(), key=lambda kv: kv[1]["class_idx"]):
        row = {"class": cls, "class_idx": e["class_idx"], "tumor": e["tumor"],
               "support": e["support"]}
        for k in ("f1", "precision", "recall"):
            row[f"{k}_mean"] = st.mean(e[k])
            row[f"{k}_sd"] = st.stdev(e[k]) if len(e[k]) > 1 else None
        out.append(row)
    return out


def find_experiment_dirs(root, min_runs=2):
    """Every folder under `root` that directly holds >= min_runs run dirs.

    A run dir is any sub-dir with a metrics.json. Walks the whole tree, so nested
    layouts (ablation_x/<variant>/<ts>/, feature_collapse/<grid>/<ts>/) are found
    as well as flat ones (best_model_x/<ts>/).
    """
    found = []
    for dirpath, dirnames, _ in os.walk(root):
        if len(_run_dirs(dirpath)) >= min_runs:
            found.append(dirpath)
            dirnames[:] = []      # don't descend into the run dirs themselves
    return sorted(found)


def summarize(results_dir, metric="test_macro_f1", out=None, quiet=False):
    """Write <results_dir>/summary.json. Returns the summary dict (None if no runs)."""
    args = argparse.Namespace(results=results_dir, metric=metric, out=out)

    runs = _run_dirs(args.results)
    if not runs:
        if not quiet:
            print(f"No runs with metrics.json under {args.results}/")
        return None

    # collect every numeric metric across runs
    per_run, values = [], {}
    for run in runs:
        m = json.load(open(os.path.join(run, "metrics.json")))
        rec = _meta(run)
        rec["metrics"] = {k: float(v) for k, v in m.items()
                          if isinstance(v, (int, float))}
        per_run.append(rec)
        for k, v in rec["metrics"].items():
            values.setdefault(k, []).append(v)

    if args.metric not in values:
        if not quiet:
            print(f"[skip] {args.results}: metric '{args.metric}' not in {sorted(values)}")
        return None

    agg = {k: _stats(v) for k, v in sorted(values.items())}

    # best run by the chosen metric
    bi = max(range(len(per_run)), key=lambda i: per_run[i]["metrics"][args.metric])
    best = {"run": per_run[bi]["run"], "path": os.path.join(args.results, per_run[bi]["run"]),
            "split_seed": per_run[bi].get("split_seed"),
            "selection_metric": args.metric,
            "metrics": per_run[bi]["metrics"]}

    summary = {
        "experiment": os.path.basename(os.path.normpath(args.results)),
        "results_dir": args.results,
        "n_runs": len(runs),
        "split_seeds": sorted({r.get("split_seed") for r in per_run if r.get("split_seed")}),
        "backbone": per_run[0].get("backbone"),
        "readout_types": per_run[0].get("readout_types"),
        "has_metapath": per_run[0].get("has_metapath"),
        "aggregate": agg,
        "best_run": best,
        "per_run": per_run,
        "per_class": _per_class(runs),
    }

    out = args.out or os.path.join(args.results, "summary.json")
    json.dump(summary, open(out, "w"), indent=2)
    if quiet:
        s = agg[args.metric]
        sd = f"± {s['sd']:.4f}" if s["sd"] is not None else ""
        print(f"  {summary['experiment']:42} n={len(runs)}  {args.metric}="
              f"{s['mean']:.4f} {sd}   best={best['run']} ({best['metrics'][args.metric]:.4f})")
        return summary

    # --- readable report ---
    print(f"# {summary['experiment']}  |  {len(runs)} runs  |  backbone={summary['backbone']}")
    print(f"# split seeds: {summary['split_seeds']}  metapath={summary['has_metapath']}")
    print(f"# readout: {summary['readout_types']}\n")
    print(f"{'metric':22} {'mean':>8} {'sd':>8} {'95% CI':>18} {'min':>8} {'max':>8}")
    print("-" * 76)
    for k, s in agg.items():
        sd = f"{s['sd']:.4f}" if s["sd"] is not None else "  --  "
        ci = (f"[{s['ci95_low']:.4f},{s['ci95_high']:.4f}]"
              if s["ci95_low"] is not None else "        --        ")
        print(f"{k:22} {s['mean']:8.4f} {sd:>8} {ci:>18} {s['min']:8.4f} {s['max']:8.4f}")
    print(f"\nBEST RUN by {args.metric}: {best['run']} (seed {best['split_seed']}) "
          f"-> {best['metrics'][args.metric]:.4f}")
    print(f"  path: {best['path']}")
    if summary["per_class"]:
        pc = sorted(summary["per_class"], key=lambda r: r["f1_mean"])
        print(f"\nper-class F1 (mean over runs) — worst 3: "
              + ", ".join(f"{r['class']}{'/'+r['tumor'] if r['tumor'] and r['tumor']!='nan' else ''}"
                          f" {r['f1_mean']:.3f}" for r in pc[:3]))
        print(f"                                best 3: "
              + ", ".join(f"{r['class']}{'/'+r['tumor'] if r['tumor'] and r['tumor']!='nan' else ''}"
                          f" {r['f1_mean']:.3f}" for r in pc[-3:]))
    print(f"\n[saved] {out}")
    return summary


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate multi-seed runs into summary.json (one per experiment folder).")
    ap.add_argument("--results", default=None,
                    help="A single experiment folder. Omit to scan --root for all of them.")
    ap.add_argument("--root", default="results",
                    help="Tree to scan when --results is omitted (default: results).")
    ap.add_argument("--metric", default="test_macro_f1", help="Metric used to pick the best run.")
    ap.add_argument("--min-runs", type=int, default=2,
                    help="Only summarize folders with at least this many runs (default 2).")
    ap.add_argument("--out", default=None, help="Output path (single-folder mode only).")
    args = ap.parse_args()

    # single folder: full report
    if args.results:
        summarize(args.results, args.metric, args.out)
        return

    # scan mode: every experiment folder under --root, one line each
    if not os.path.isdir(args.root):
        print(f"[error] no such directory: {args.root}")
        return
    exps = find_experiment_dirs(args.root, args.min_runs)
    if not exps:
        print(f"No folder under {args.root}/ holds >= {args.min_runs} runs.")
        return
    print(f"Scanning {args.root}/ — {len(exps)} experiment folder(s) with >= "
          f"{args.min_runs} runs | metric={args.metric}\n")
    ok = 0
    for d in exps:
        if summarize(d, args.metric, out=None, quiet=True):
            ok += 1
    print(f"\n[done] wrote summary.json into {ok}/{len(exps)} folders.")
    print(f"       full report for one: python {os.path.relpath(__file__)} --results <folder>")


if __name__ == "__main__":
    main()
