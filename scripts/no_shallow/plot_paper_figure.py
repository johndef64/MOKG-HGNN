"""Composite feature-scarcity figure for the paper (double column, three panels).

  (a) test macro-F1 vs requested panel size: multi-scale, molecular with shallow branch,
      molecular without shallow branch (mean +- 1 s.d. over the 5 splits, shaded);
  (b) paired difference from the multi-scale model, split by split, for the molecular
      model with and without the shallow branch; the band between the two means is the
      contribution of the shallow branch;
  (c) fraction of molecular nodes (gene, miRNA, TF) in the largest connected component
      of the graph each model propagates over (graph_topology.py).

Inputs: the three series (compare.load_series) and results/tables/graph_topology.csv.
Writes results/figures/fig_paper_collapse_composite.{pdf,png}.

  python scripts/no_shallow/plot_paper_figure.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from compare import load_series, level_stats  # noqa: E402
from plot_comparison import TEAL, GREY, BLACK, _xaxis, plt  # noqa: E402

# every level gets a tick; 500 and 100 are left unlabelled (too close to 700 and 150)
TICK_LABELS = {700: "700", 500: "", 300: "300", 150: "150", 100: "", 50: "50", 20: "20"}
NUDGE = {"MS": 0.035, "WS": -0.03, "NS": 0.0}   # vertical offset of the drop labels in (a)

NAME = "fig_paper_collapse_composite"
LAB = {"MS": "Multi-scale (MOKG-HGNN)",
       "WS": "Molecular, with shallow branch (MOGNN-TF)",
       "NS": "Molecular, graph pathway only (MOGNN-TF without shallow branch)"}
STY = {"MS": dict(color=TEAL, ls="-", marker="o"),
       "WS": dict(color=GREY, ls="--", marker="o", mfc="white"),
       "NS": dict(color=BLACK, ls="-", marker="s")}
OFF = {"MS": 0.97, "WS": 1.0, "NS": 1.03}


def _panel_label(ax, s):
    ax.text(-0.17, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def panel_a(ax, series):
    for k in ("WS", "MS", "NS"):
        pts = [(g, *level_stats(series[k], g)[:2]) for g in C.LEVELS]
        xs = [g * OFF[k] for g, _, _ in pts]
        m = [p[1] for p in pts]
        sd = [p[2] for p in pts]
        ax.fill_between(xs, [a - b for a, b in zip(m, sd)], [a + b for a, b in zip(m, sd)],
                        color=STY[k]["color"], alpha=0.13, lw=0)
        ax.plot(xs, m, label=LAB[k], **STY[k])
        drop = m[0] - m[-1]
        ax.text(20 * 0.86, m[-1] + NUDGE[k], f"$-${drop:.3f}", color=STY[k]["color"], fontsize=6.5,
                va="center", ha="left")
    _xaxis(ax)
    ax.set_xlim(830, 12.5)
    ax.set_ylim(0.0, 0.86)
    ax.set_ylabel("test macro-F1")


def panel_b(ax, series):
    ax.axhline(0, color="0.35", lw=0.8, zorder=1)
    means = {}
    for k in ("WS", "NS"):
        xs, ms = [], []
        for g in C.LEVELS:
            ds = [series[k][(g, s)]["f1"] - series["MS"][(g, s)]["f1"] for s in C.SEEDS]
            ax.scatter([g * OFF[k]] * len(ds), ds, s=8, color=STY[k]["color"], alpha=0.35, lw=0,
                       marker=STY[k]["marker"], zorder=2)
            xs.append(g); ms.append(sum(ds) / len(ds))
        means[k] = ms
        ax.plot([x * OFF[k] for x in xs], ms, zorder=3, **STY[k])
    ax.fill_between(C.LEVELS, means["NS"], means["WS"], color="#B07A2A", alpha=0.12, lw=0,
                    zorder=0, label="contribution of the shallow branch")
    ax.text(42, (means["NS"][5] + means["WS"][5]) / 2, "shallow\nbranch",
            color="#8A5A1A", fontsize=6.5, ha="center", va="center")
    _xaxis(ax)
    ax.set_ylabel(r"$\Delta$ macro-F1 vs multi-scale")


def panel_c(ax):
    rows = list(csv.DictReader(open(C.TABLES / "graph_topology.csv", newline="", encoding="utf-8")))
    for col, k, lab in (("ms_full_in_lcc_frac", "MS", "multi-scale"),
                        ("mol_in_lcc_frac", "NS", "molecular")):
        xs, m, lo, hi = [], [], [], []
        for g in C.LEVELS:
            v = [float(r[col]) for r in rows if int(r["genes"]) == g and r.get(col)]
            if not v:
                continue
            xs.append(g); m.append(sum(v) / len(v)); lo.append(min(v)); hi.append(max(v))
        ax.fill_between(xs, lo, hi, color=STY[k]["color"], alpha=0.13, lw=0)
        ax.plot(xs, m, color=STY[k]["color"], ls="-", marker=STY[k]["marker"], label=lab)
    _xaxis(ax)
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("fraction in largest component")
    ax.legend(loc="lower left", frameon=False, fontsize=6.5, borderaxespad=0.2)


def main():
    series, missing = load_series()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55))
    panel_a(axes[0], series)
    panel_b(axes[1], series)
    panel_c(axes[2])
    for ax, s in zip(axes, "abc"):
        _panel_label(ax, s)
        ax.set_xticklabels([TICK_LABELS[g] for g in C.LEVELS], fontsize=6.8)
    h, l = axes[0].get_legend_handles_labels()
    order = [l.index(LAB[k]) for k in ("MS", "WS", "NS")]
    hb, lb = axes[1].get_legend_handles_labels()
    fig.legend([h[i] for i in order] + hb, [l[i] for i in order] + lb, loc="lower center",
               ncol=2, frameon=False, fontsize=6.8,
               bbox_to_anchor=(0.5, -0.02), handlelength=2.6, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=1.6)
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(C.FIGURES / f"{NAME}.{ext}", bbox_inches="tight", **kw)
    print(f"[saved] {C.FIGURES / NAME}.pdf / .png")


if __name__ == "__main__":
    main()
