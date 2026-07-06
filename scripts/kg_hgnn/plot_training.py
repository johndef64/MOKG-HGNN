"""Plot training curves for runs under results/.

Two modes:
  * per-run (default): one training_curves.png INSIDE each run folder
    (loss train/val, macro-F1 train/val with best-val marked, accuracy + test box).
  * aggregate (--aggregate): for each experiment with MULTIPLE runs (e.g. the 5
    seeds of a multi-seed evaluation), one aggregated_curves.png with the MEAN
    +/- s.d. band across runs. Curves are truncated to the SHORTEST run so every
    plotted epoch is a true average over all N runs (runs stop at different
    epochs via early stopping).

Usage:
    conda run -n gnn python scripts/kg_hgnn/plot_training.py               # per-run, all
    conda run -n gnn python scripts/kg_hgnn/plot_training.py --aggregate   # + aggregate per experiment
    conda run -n gnn python scripts/kg_hgnn/plot_training.py --run <dir>   # one run
    conda run -n gnn python scripts/kg_hgnn/plot_training.py --force       # redraw existing
"""

import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")  # headless (server-safe)
import matplotlib.pyplot as plt


def _read_history(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    cols = rows[0].keys()

    def col(name):
        return [float(r[name]) for r in rows] if name in cols else None

    return {
        "epoch": col("epoch"),
        "train_loss": col("train_loss"), "val_loss": col("val_loss"),
        "train_f1": col("train_macro_f1"), "val_f1": col("val_macro_f1"),
        "val_acc": col("val_accuracy"),
    }


def _read_metrics(run_dir):
    p = os.path.join(run_dir, "metrics.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def plot_run(run_dir, force=False):
    hist_path = os.path.join(run_dir, "history.csv")
    if not os.path.exists(hist_path):
        return False
    out_path = os.path.join(run_dir, "training_curves.png")
    if os.path.exists(out_path) and not force:
        print(f"[skip] {out_path} exists (use --force)")
        return True

    h = _read_history(hist_path)
    if h is None or not h["epoch"]:
        print(f"[warn] empty history: {hist_path}")
        return False
    m = _read_metrics(run_dir)
    ep = h["epoch"]

    # best validation epoch (macro-F1) — the checkpoint that was kept
    best_i = max(range(len(h["val_f1"])), key=lambda i: h["val_f1"][i]) if h["val_f1"] else None

    exp = os.path.basename(os.path.dirname(run_dir))
    stamp = os.path.basename(run_dir.rstrip("/\\"))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.suptitle(f"{exp}  ·  {stamp}", fontsize=13, fontweight="bold")

    # --- 1) loss ---
    ax = axes[0]
    if h["train_loss"]:
        ax.plot(ep, h["train_loss"], label="train", color="#4C72B0", lw=1.8)
    if h["val_loss"]:
        ax.plot(ep, h["val_loss"], label="val", color="#DD8452", lw=1.8)
    ax.set_title("Loss"); ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy")
    ax.legend(frameon=False); ax.grid(alpha=0.25)

    # --- 2) macro-F1 (the primary metric) ---
    ax = axes[1]
    if h["train_f1"]:
        ax.plot(ep, h["train_f1"], label="train", color="#4C72B0", lw=1.8)
    if h["val_f1"]:
        ax.plot(ep, h["val_f1"], label="val", color="#DD8452", lw=1.8)
    if best_i is not None:
        ax.axvline(ep[best_i], color="#55A868", ls="--", lw=1.2, alpha=0.8)
        ax.scatter([ep[best_i]], [h["val_f1"][best_i]], color="#55A868", zorder=5,
                   label=f"best val {h['val_f1'][best_i]:.3f} @ep{int(ep[best_i])}")
    ax.set_title("Macro-F1 (primary)"); ax.set_xlabel("epoch"); ax.set_ylabel("macro-F1")
    ax.legend(frameon=False); ax.grid(alpha=0.25); ax.set_ylim(0, 1)

    # --- 3) validation accuracy (+ final test annotation) ---
    ax = axes[2]
    if h["val_acc"]:
        ax.plot(ep, h["val_acc"], label="val accuracy", color="#C44E52", lw=1.8)
    ax.set_title("Accuracy"); ax.set_xlabel("epoch"); ax.set_ylabel("accuracy")
    ax.grid(alpha=0.25); ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")   # top-left: clear of the test box
    if m:
        txt = (f"TEST\nmacro-F1  {m.get('test_macro_f1', float('nan')):.3f}\n"
               f"accuracy  {m.get('test_accuracy', float('nan')):.3f}\n"
               f"best val  {m.get('best_val_macro_f1', float('nan')):.3f}")
        ax.text(0.97, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, family="monospace",
                bbox=dict(boxstyle="round", fc="#F0F0F0", ec="#BBBBBB"))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[saved] {out_path}")
    return True


def _mean_sd(series_list):
    """series_list: list of per-run lists (already truncated to equal length).
    Returns (mean, sd) as lists of length = common length."""
    import statistics as st
    n_epochs = len(series_list[0])
    mean, sd = [], []
    for i in range(n_epochs):
        vals = [s[i] for s in series_list]
        mean.append(st.mean(vals))
        sd.append(st.stdev(vals) if len(vals) > 1 else 0.0)
    return mean, sd


def plot_experiment_aggregate(exp_dir, force=False):
    """MEAN +/- s.d. curves across all runs of one experiment, truncated to the
    shortest run. Saves aggregated_curves.png in the experiment folder."""
    run_dirs = sorted(d for d in glob.glob(os.path.join(exp_dir, "*"))
                      if os.path.exists(os.path.join(d, "history.csv")))
    if len(run_dirs) < 2:
        return False  # aggregation only meaningful with >= 2 runs
    out_path = os.path.join(exp_dir, "aggregated_curves.png")
    if os.path.exists(out_path) and not force:
        print(f"[skip] {out_path} exists (use --force)")
        return True

    hists = [_read_history(os.path.join(d, "history.csv")) for d in run_dirs]
    hists = [h for h in hists if h and h["epoch"]]
    L = min(len(h["epoch"]) for h in hists)   # truncate to shortest run
    ep = list(range(1, L + 1))

    def stack(key):
        cols = [h[key][:L] for h in hists if h[key] is not None]
        return _mean_sd(cols) if len(cols) == len(hists) else (None, None)

    exp = os.path.basename(exp_dir.rstrip("/\\"))
    n = len(hists)
    # collect final test macro-F1 across runs for the annotation
    tests = []
    for d in run_dirs:
        m = _read_metrics(d)
        if "test_macro_f1" in m:
            tests.append(float(m["test_macro_f1"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle(f"{exp}  ·  mean ± s.d. over {n} runs (truncated to {L} epochs)",
                 fontsize=13, fontweight="bold")

    def band(ax, key, color, label):
        mean, sd = stack(key)
        if mean is None:
            return
        lo = [m - s for m, s in zip(mean, sd)]
        hi = [m + s for m, s in zip(mean, sd)]
        ax.plot(ep, mean, color=color, lw=2, label=label)
        ax.fill_between(ep, lo, hi, color=color, alpha=0.20)

    # --- loss ---
    ax = axes[0]
    band(ax, "train_loss", "#4C72B0", "train")
    band(ax, "val_loss", "#DD8452", "val")
    ax.set_title("Loss"); ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy")
    ax.legend(frameon=False); ax.grid(alpha=0.25)

    # --- macro-F1 ---
    ax = axes[1]
    band(ax, "train_f1", "#4C72B0", "train")
    band(ax, "val_f1", "#DD8452", "val")
    ax.set_title("Macro-F1 (primary)"); ax.set_xlabel("epoch"); ax.set_ylabel("macro-F1")
    ax.set_ylim(0, 1); ax.legend(frameon=False, loc="upper left"); ax.grid(alpha=0.25)
    if tests:
        import statistics as st
        mean_t = st.mean(tests)
        sd_t = st.stdev(tests) if len(tests) > 1 else 0.0
        ax.text(0.97, 0.05, f"TEST macro-F1\n{mean_t:.3f} ± {sd_t:.3f}\n(n={len(tests)})",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                family="monospace",
                bbox=dict(boxstyle="round", fc="#F0F0F0", ec="#BBBBBB"))

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[saved] {out_path}  ({n} runs, {L} epochs)")
    return True


def main():
    ap = argparse.ArgumentParser(description="Plot training curves for runs under results/.")
    ap.add_argument("--results", default="results", help="Root results directory.")
    ap.add_argument("--run", default=None, help="Plot a single run dir instead of all.")
    ap.add_argument("--aggregate", action="store_true",
                    help="Also plot mean±s.d. curves per experiment (>=2 runs).")
    ap.add_argument("--force", action="store_true", help="Redraw even if the PNG exists.")
    args = ap.parse_args()

    if args.run:
        plot_run(args.run, force=args.force)
        return

    run_dirs = sorted({os.path.dirname(p) for p in
                       glob.glob(os.path.join(args.results, "**", "history.csv"), recursive=True)})
    if not run_dirs:
        print(f"No runs with history.csv found under {args.results}/")
        return
    print(f"Found {len(run_dirs)} run(s).")
    ok = sum(plot_run(d, force=args.force) for d in run_dirs)
    print(f"Done: {ok}/{len(run_dirs)} per-run plotted.")

    if args.aggregate:
        # an experiment folder = the parent of a run folder
        exps = sorted({os.path.dirname(d.rstrip("/\\")) for d in run_dirs})
        n_agg = sum(plot_experiment_aggregate(e, force=args.force) for e in exps)
        print(f"Aggregate: {n_agg} experiment(s) with >=2 runs plotted.")


if __name__ == "__main__":
    main()
