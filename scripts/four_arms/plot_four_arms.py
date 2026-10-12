"""Figures of the version-3 factorial experiment (plan §4). Arms with no data are skipped.

  fig_four_arms_composite  (a) macro-F1 vs panel size, six arms (mean +- 1 s.d.);
                           (b) shallow contribution in each architecture (S_MOL, S_KG);
                           (c) contribution of each graph on top of the features alone
                               (GAMMA_MOL, GAMMA_KG), with the fraction of molecular nodes
                               in the largest connected component (right axis, from
                               results/no_shallow/tables/graph_topology_by_level.csv)
  fig_decomposition_700    features only / graph only / graph + shallow, per architecture

Writes results/four_arms/figures/*.{pdf,png}.

  python scripts/four_arms/plot_four_arms.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
from compare import paired  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TEAL, GREY, BLACK = "#1b7f79", "#8c8c8c", "#222222"
STY = {
    "mol_s": dict(color=GREY, ls="-", marker="o"),
    "mol": dict(color=GREY, ls="--", marker="o", mfc="white"),
    "kg_s": dict(color=TEAL, ls="-", marker="s"),
    "kg": dict(color=TEAL, ls="--", marker="s", mfc="white"),
    "sh_tf": dict(color=BLACK, ls=":", marker="^", ms=4),
    "sh_kg": dict(color=TEAL, ls=":", marker="^", ms=4),
}
LAB = {"mol_s": "MOGNN-TF + shallow", "mol": "MOGNN-TF, graph only",
       "kg_s": "MOKG-HGNN + shallow", "kg": "MOKG-HGNN, graph only",
       "sh_tf": "shallow only (MOGNN-TF recipe)", "sh_kg": "shallow only (MOKG-HGNN recipe)"}
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})


def _xaxis(ax):
    ax.set_xscale("log")
    ax.set_xticks(core.LEVELS)
    ax.set_xticklabels([("" if g in (500, 100) else str(g)) for g in core.LEVELS])   # too close to 700 / 150
    ax.minorticks_off()
    ax.set_xlim(830, 16)
    ax.set_xlabel("genes in the panel")


def _band(ax, xs, vals, **sty):
    m = [sum(v) / len(v) for v in vals]
    sd = [core.NS.mean_sd(v)[1] for v in vals]
    ax.fill_between(xs, [a - b for a, b in zip(m, sd)], [a + b for a, b in zip(m, sd)],
                    color=sty["color"], alpha=0.12, lw=0)
    ax.plot(xs, m, lw=1.2, **sty)


def _series_band(ax, series, weights, label, **sty):
    xs, vals = [], []
    for g in core.LEVELS:
        _, d = paired(series, weights, g)
        if d:
            xs.append(g)
            vals.append(d)
    if xs:
        _band(ax, xs, vals, label=label, **sty)


def _lcc():
    f = core.REPO / "results" / "no_shallow" / "tables" / "graph_topology_by_level.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(open(f, newline="", encoding="utf-8")))
    return {int(r["genes"]): (float(r["mol_in_lcc_frac"]), float(r["ms_full_in_lcc_frac"])) for r in rows}


def _save(fig, name):
    core.FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(core.FIGURES / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {core.FIGURES / name}.pdf/.png")


def composite(series):
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.5), constrained_layout=True)
    ax = axs[0]
    for arm in ("mol_s", "kg_s", "kg", "mol", "sh_tf", "sh_kg"):
        xs = [g for g in core.LEVELS if any((g, s) in series[arm] for s in core.SEEDS)]
        if xs:
            vals = [[series[arm][(g, s)] for s in core.SEEDS if (g, s) in series[arm]] for g in xs]
            _band(ax, xs, vals, label=LAB[arm], **STY[arm])
    _xaxis(ax)
    ax.set_ylim(0, 0.86)
    ax.set_ylabel("test macro-F1")
    ax.set_title("(a) six arms", loc="left", fontsize=8)

    ax = axs[1]
    ax.axhline(0, color="k", lw=0.5)
    _series_band(ax, series, {"mol_s": 1, "mol": -1}, "MOGNN-TF: S$_{MOL}$", color=GREY, marker="o")
    _series_band(ax, series, {"kg_s": 1, "kg": -1}, "MOKG-HGNN: S$_{KG}$", color=TEAL, marker="s")
    _xaxis(ax)
    ax.set_ylabel("$\\Delta$ macro-F1, + shallow $-$ graph only")
    ax.set_title("(b) contribution of the shallow branch", loc="left", fontsize=8)
    ax.legend(frameon=False, fontsize=6.5)

    ax = axs[2]
    ax.axhline(0, color="k", lw=0.5)
    _series_band(ax, series, {"mol_s": 1, "sh_tf": -1}, "molecular graph: $\\Gamma_{MOL}$",
                 color=GREY, marker="o")
    _series_band(ax, series, {"kg_s": 1, "sh_kg": -1}, "knowledge graph: $\\Gamma_{KG}$",
                 color=TEAL, marker="s")
    _xaxis(ax)
    ax.set_ylabel("$\\Delta$ macro-F1, + graph $-$ shallow only")
    ax.set_title("(c) contribution of the graph", loc="left", fontsize=8)
    lcc = _lcc()
    if lcc:
        ax2 = ax.twinx()
        ax2.spines["right"].set_visible(True)
        gs = [g for g in core.LEVELS if g in lcc]
        ax2.plot(gs, [lcc[g][0] for g in gs], color=GREY, lw=0.8, ls=(0, (1, 1)), label="LCC, molecular")
        ax2.plot(gs, [lcc[g][1] for g in gs], color=TEAL, lw=0.8, ls=(0, (1, 1)), label="LCC, multi-scale")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("molecular nodes in largest component", fontsize=6.5)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.16))
    _save(fig, "fig_four_arms_composite")


def decomposition(series):
    fig, ax = plt.subplots(figsize=(3.4, 2.3), constrained_layout=True)
    groups = {"MOGNN-TF": ("sh_tf", "mol", "mol_s"), "MOKG-HGNN": ("sh_kg", "kg", "kg_s")}
    names = ("features only", "graph only", "graph + shallow")
    for i, (arch, arms) in enumerate(groups.items()):
        for j, arm in enumerate(arms):
            v = [series[arm][(700, s)] for s in core.SEEDS if (700, s) in series[arm]]
            if not v:
                continue
            m, sd = core.NS.mean_sd(v)
            x = i * 4 + j
            ax.bar(x, m, yerr=sd, color=STY[arm]["color"], alpha=[0.35, 0.6, 1.0][j],
                   capsize=2, width=0.8)
            ax.text(x, m + sd + 0.01, f"{m:.3f}", ha="center", fontsize=6)
    ax.set_xticks([0, 1, 2, 4, 5, 6])
    ax.set_xticklabels(list(names) * 2, rotation=30, ha="right", fontsize=6.5)
    for i, arch in enumerate(groups):
        ax.text(i * 4 + 1, -0.33, arch, ha="center", transform=ax.get_xaxis_transform(), fontsize=7.5)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("test macro-F1, 700 genes")
    _save(fig, "fig_decomposition_700")


def main():
    series = core.load_f1_series()
    composite(series)
    decomposition(series)


if __name__ == "__main__":
    main()
