"""Comparison figures for the no-shallow matched experiment (ticket §6).

Each figure is saved as vector PDF (thesis) and 300-dpi PNG (slides) in results/figures/.
Common style: multi-scale teal, solid, circles; molecular with shallow grey, dashed (the
published reference series); molecular without shallow near-black, solid, squares.
x = requested panel size, log scale, inverted (fewer genes to the right), ticks on the 7
levels; y = test macro-F1; +-1 s.d. over the 5 splits; sans-serif; no in-figure title;
legend outside the data area. Small multiplicative x offsets separate overlapping error
bars (stated in the captions).

  fig_scarcity_three_curves  three curves; faint shaded interval + label where each pair crosses
  fig_gap_vs_panel           (WS - MS) and (NS - MS) per level, per-split points behind the mean
  fig_shallow_effect         (NS - WS) per level: 5 paired points + mean segment
  fig_spread_vs_panel        s.d. across splits per level, three series

  python experiments/no_shallow_matched/plot_comparison.py [--allow-partial]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from compare import load_series, level_stats, crossings  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

TEAL, GREY, BLACK = "#1A7F7F", "#8C9198", "#1F1F1F"
STYLE = {
    "MS": dict(color=TEAL, ls="-", marker="o", label="Multi-scale (MOKG-HGNN)"),
    "WS": dict(color=GREY, ls="--", marker="o", mfc="white", label="Molecular, with shallow (MOGNN-TF)"),
    "NS": dict(color=BLACK, ls="-", marker="s", label="Molecular, no shallow"),
}
OFFSET = {"MS": 0.965, "WS": 1.0, "NS": 1.035}   # multiplicative x offset (log axis)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.4, "lines.markersize": 4,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
W = 3.5  # single-column width, inches


def _xaxis(ax):
    ax.set_xscale("log")
    ax.set_xlim(830, 16.5)   # inverted: richest panel on the left
    ax.xaxis.set_major_locator(FixedLocator(C.LEVELS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(g) for g in C.LEVELS])
    ax.set_xlabel("requested gene panel size")
    ax.grid(axis="y", alpha=0.25, lw=0.5)


def _legend_below(ax, fig, n_rows):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False,
              handlelength=2.6)
    fig.subplots_adjust(bottom=0.2 + 0.055 * n_rows)


def _save(fig, name):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(C.FIGURES / f"{name}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"[saved] {C.FIGURES / name}.pdf / .png")


def fig_three_curves(series, means):
    fig, ax = plt.subplots(figsize=(W, 3.6))
    pairs = (("WS", "MS", GREY), ("NS", "MS", BLACK), ("NS", "WS", "#B07A2A"))
    y_top = 0.0
    for k in ("MS", "WS", "NS"):
        pts = [(g, *level_stats(series[k], g)[:2]) for g in C.LEVELS if level_stats(series[k], g)[2]]
        if not pts:
            continue
        xs = [g * OFFSET[k] for g, _, _ in pts]
        st = dict(STYLE[k])
        ax.errorbar(xs, [m for _, m, _ in pts], yerr=[s for _, _, s in pts], capsize=2,
                    elinewidth=0.8, zorder=3, **st)
        y_top = max(y_top, max(m + s for _, m, s in pts))
    for i, (a, b, col) in enumerate(pairs):
        for c in crossings(means[a], means[b]):
            hi, lo = c["between"]
            ax.axvspan(hi, lo if lo != hi else hi * 0.98, color=col, alpha=0.07, lw=0, zorder=0)
            ax.axvline(c["approx_genes"], color=col, alpha=0.35, lw=0.7, ls=":", zorder=1)
            ax.text(c["approx_genes"], y_top + 0.004 + 0.009 * i, f"{a}={b}", color=col,
                    fontsize=6, ha="center", va="bottom")
    _xaxis(ax)
    ax.set_ylabel("test macro-F1")
    _legend_below(ax, fig, 3)
    _save(fig, "fig_scarcity_three_curves")


def _paired(series, a, b, g):
    return [series[a][(g, s)]["f1"] - series[b][(g, s)]["f1"] for s in C.SEEDS
            if (g, s) in series[a] and (g, s) in series[b]]


def fig_gap(series):
    fig, ax = plt.subplots(figsize=(W, 3.2))
    ax.axhline(0, color="0.3", lw=0.8, zorder=1)
    for k, lab in (("WS", "with shallow - multi-scale"), ("NS", "no shallow - multi-scale")):
        st = STYLE[k]
        xs, ms = [], []
        for g in C.LEVELS:
            ds = _paired(series, k, "MS", g)
            if not ds:
                continue
            ax.scatter([g * OFFSET[k]] * len(ds), ds, s=9, color=st["color"], alpha=0.3,
                       lw=0, zorder=2, marker=st["marker"])
            xs.append(g * OFFSET[k]); ms.append(sum(ds) / len(ds))
        ax.plot(xs, ms, color=st["color"], ls=st["ls"], marker=st["marker"],
                mfc=st.get("mfc", st["color"]), zorder=3, label=lab)
    _xaxis(ax)
    ax.set_ylabel(r"$\Delta$ test macro-F1 vs multi-scale")
    _legend_below(ax, fig, 2)
    _save(fig, "fig_gap_vs_panel")


def fig_shallow_effect(series):
    fig, ax = plt.subplots(figsize=(W, 3.0))
    ax.axhline(0, color="0.3", lw=0.8, zorder=1)
    first = True
    for g in C.LEVELS:
        ds = _paired(series, "NS", "WS", g)
        if not ds:
            continue
        ax.scatter([g] * len(ds), ds, s=12, color=BLACK, alpha=0.45, lw=0, zorder=2,
                   label="single split" if first else None)
        m = sum(ds) / len(ds)
        ax.plot([g * 0.9, g / 0.9], [m, m], color=BLACK, lw=2.0, zorder=3,
                label="mean over 5 splits" if first else None)
        first = False
    _xaxis(ax)
    ax.set_ylabel(r"$\Delta$ macro-F1 (no shallow - shallow)")
    _legend_below(ax, fig, 2)
    _save(fig, "fig_shallow_effect")


def fig_spread(series):
    fig, ax = plt.subplots(figsize=(W, 3.2))
    for k in ("MS", "WS", "NS"):
        pts = [(g, level_stats(series[k], g)[1]) for g in C.LEVELS if level_stats(series[k], g)[2] > 1]
        if pts:
            st = dict(STYLE[k])
            ax.plot([g for g, _ in pts], [s for _, s in pts], **st)
    _xaxis(ax)
    ax.set_ylabel("s.d. of test macro-F1")
    ax.set_ylim(bottom=0)
    _legend_below(ax, fig, 3)
    _save(fig, "fig_spread_vs_panel")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    series, _ = load_series(args.allow_partial)
    means = {k: {g: level_stats(v, g)[0] for g in C.LEVELS if level_stats(v, g)[2]}
             for k, v in series.items()}
    fig_three_curves(series, means)
    fig_gap(series)
    fig_shallow_effect(series)
    fig_spread(series)


if __name__ == "__main__":
    main()
