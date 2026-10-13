"""Figures of the version-3 factorial experiment (plan §4). Arms with no data are skipped.

  fig_four_arms_composite  (a) macro-F1 vs panel size, six arms (mean +- 1 s.d.);
                           (b) graph alone vs features alone (A_MOL, A_KG);
                           (c) shallow contribution in each architecture (S_MOL, S_KG);
                           (d) contribution of each graph on top of the features (GAMMA);
                           (b-d) with the materiality band +-0.02 fixed before the runs
  fig_decomposition_700    features only / graph only / graph + shallow, per architecture

Writes results/four_arms/figures/*.{pdf,png}. With --labels thesis the same figures are written
with neutral model names (multi-scale / molecular scale, SH-mol) as *_thesis.{pdf,png}; the paper
figures are left untouched.

  python scripts/four_arms/plot_four_arms.py [--labels thesis]
"""

import argparse
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
LAB = {"mol_s": "MOL+S: MOGNN-TF with shallow branch", "mol": "MOL: MOGNN-TF, graph only",
       "kg_s": "KG+S: MOKG-HGNN with shallow branch", "kg": "KG: MOKG-HGNN, graph only",
       "sh_tf": "SH-tf: shallow branch only, MOGNN-TF recipe",
       "sh_kg": "SH-kg: shallow branch only, MOKG-HGNN recipe"}
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})

# neutral names for the thesis (--labels thesis); the paper keeps the model names
THESIS = (("SH-tf", "SH-mol"), ("MOKG-HGNN", "multi-scale"), ("MOGNN-TF", "molecular scale"))
MODE = {"labels": "paper"}


def T(text):
    if MODE["labels"] == "thesis":
        for a, b in THESIS:
            text = text.replace(a, b)
    return text


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
    if "label" in sty:
        sty = {**sty, "label": T(sty["label"])}
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


def _save(fig, name):
    core.FIGURES.mkdir(parents=True, exist_ok=True)
    if MODE["labels"] == "thesis":
        name += "_thesis"
    for ext in ("pdf", "png"):
        fig.savefig(core.FIGURES / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {core.FIGURES / name}.pdf/.png")


def _threshold(ax, lo=-0.02, hi=0.02):
    """Materiality threshold fixed before the runs (plan §3): |mean paired difference| < 0.02."""
    ax.axhspan(lo, hi, color="0.88", lw=0, zorder=0)
    ax.axhline(0, color="k", lw=0.5, zorder=1)


def composite(series):
    fig, axs = plt.subplots(2, 2, figsize=(7.16, 5.0), constrained_layout=True)
    ax = axs[0, 0]
    for arm in ("mol_s", "kg_s", "sh_tf", "sh_kg", "kg", "mol"):
        xs = [g for g in core.LEVELS if any((g, s) in series[arm] for s in core.SEEDS)]
        if xs:
            vals = [[series[arm][(g, s)] for s in core.SEEDS if (g, s) in series[arm]] for g in xs]
            _band(ax, xs, vals, label=LAB[arm], **STY[arm])
    _xaxis(ax)
    ax.set_ylim(0, 0.86)
    ax.set_ylabel("test macro-F1")
    ax.set_title("(a) the six arms", loc="left", fontsize=8)

    ax = axs[0, 1]
    _threshold(ax)
    _series_band(ax, series, {"mol": 1, "sh_tf": -1}, "molecular graph: MOL $-$ SH-tf",
                 color=GREY, marker="o", mfc="white")
    _series_band(ax, series, {"kg": 1, "sh_kg": -1}, "knowledge graph: KG $-$ SH-kg",
                 color=TEAL, marker="s", mfc="white")
    _xaxis(ax)
    ax.set_ylim(-0.66, 0.06)
    ax.set_ylabel("$\Delta$ macro-F1")
    ax.set_title("(b) graph alone vs. features alone", loc="left", fontsize=8)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")

    ax = axs[1, 0]
    _threshold(ax)
    _series_band(ax, series, {"mol_s": 1, "mol": -1}, "MOGNN-TF: MOL+S $-$ MOL", color=GREY, marker="o")
    _series_band(ax, series, {"kg_s": 1, "kg": -1}, "MOKG-HGNN: KG+S $-$ KG", color=TEAL, marker="s")
    _xaxis(ax)
    ax.set_ylim(-0.06, 0.66)
    ax.set_ylabel("$\Delta$ macro-F1")
    ax.set_title("(c) what the shallow branch adds to the graph", loc="left", fontsize=8)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")

    ax = axs[1, 1]
    _threshold(ax)
    _series_band(ax, series, {"mol_s": 1, "sh_tf": -1}, "molecular graph: MOL+S $-$ SH-tf",
                 color=GREY, marker="o")
    _series_band(ax, series, {"kg_s": 1, "sh_kg": -1}, "knowledge graph: KG+S $-$ SH-kg",
                 color=TEAL, marker="s")
    _xaxis(ax)
    ax.set_ylim(-0.066, 0.066)
    ax.set_ylabel("$\Delta$ macro-F1")
    ax.set_title("(d) what the graph adds to the features", loc="left", fontsize=8)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=1)
    ax.text(0.99, 0.03, "grey band: $\pm0.02$ (materiality threshold)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6, color="0.35")

    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.075))
    _save(fig, "fig_four_arms_composite")


def decomposition(series):
    fig, ax = plt.subplots(figsize=(3.4, 2.3), constrained_layout=True)
    groups = {"MOGNN-TF": ("sh_tf", "mol", "mol_s"), "MOKG-HGNN": ("sh_kg", "kg", "kg_s")}
    names = ("features\nonly", "graph\nonly", "graph +\nshallow")
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
    ax.set_xticklabels(list(names) * 2, fontsize=6.3)
    for i, arch in enumerate(groups):
        ax.text(i * 4 + 1, 0.93, T(arch), ha="center", fontsize=7.5, fontweight="bold")
    ax.set_ylim(0, 0.98)
    ax.set_ylabel("test macro-F1, 700 genes")
    _save(fig, "fig_decomposition_700")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", choices=("paper", "thesis"), default="paper")
    MODE["labels"] = ap.parse_args().labels
    series = core.load_f1_series()
    composite(series)
    decomposition(series)


if __name__ == "__main__":
    main()
