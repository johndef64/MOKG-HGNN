"""Per-subtype F1 difference of the multi-scale model from the molecular model, with and
without its shallow branch, at the full panel (single column, replaces the per-class
figure that compared against the published system only).

For each subtype: grey hollow circle = MS - WS (complete systems), black square = MS - NS
(graph against graph), joined by a thin segment whose length is the contribution of the
shallow branch to that subtype; a horizontal bar spans the per-split range of MS - NS.
Subtypes are sorted by MS - NS. Same panel and the same five splits for the three arms
(per_class_three_arms.load).

Writes results/no_shallow/figures/fig_per_class_three_arms.{pdf,png}.

  python scripts/no_shallow/plot_per_class_three_arms.py [--level 700]
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from per_class_three_arms import load  # noqa: E402
from plot_comparison import TEAL, GREY, BLACK, plt  # noqa: E402

ORANGE = "#D9822B"
# transversal / pan-squamous clusters, highlighted as in the original per-class figure
TRANSVERSAL = {"C10", "C17", "C20", "C25"}
MAPPING = C.REPO / "data" / "omics" / "TCGA_iCluster_C1-C28_mapping.csv"


def labels():
    out = {}
    with open(MAPPING, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            lab = r["iCluster_label"]
            if lab.lower().startswith("not individually"):
                lab = r["cluster_type"]   # e.g. C17: no published description
            out[r["iCluster"]] = f"{r['iCluster']} · {lab if len(lab) <= 22 else lab[:21] + '…'}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=700)
    args = ap.parse_args()
    names, seeds, per, _ = load(args.level)
    k = len(seeds)
    rows = []
    for n in names:
        d_ws = sum(a - b for a, b in zip(per["MS"][n], per["WS"][n])) / k
        d_ns_split = [a - b for a, b in zip(per["MS"][n], per["NS"][n])]
        rows.append((n, d_ws, sum(d_ns_split) / k, min(d_ns_split), max(d_ns_split)))
    rows.sort(key=lambda r: r[2])
    lab = labels()

    fig, ax = plt.subplots(figsize=(3.5, 5.6))
    ax.axvline(0, color="0.35", lw=0.8, zorder=1)
    for y, (n, dws, dns, lo, hi) in enumerate(rows):
        ax.plot([lo, hi], [y, y], color=BLACK, alpha=0.25, lw=2.2, solid_capstyle="butt", zorder=1)
        ax.plot([dws, dns], [y, y], color="0.6", lw=0.7, zorder=2)
        ax.scatter([dws], [y], s=16, facecolor="white", edgecolor=GREY, lw=1.0, zorder=3,
                   label="vs molecular with shallow branch (complete systems)" if y == 0 else None)
        ax.scatter([dns], [y], s=14, marker="s", color=TEAL if dns > 0.015 else BLACK, zorder=4,
                   label="vs molecular graph only (graph against graph)" if y == 0 else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab.get(r[0], r[0]) for r in rows], fontsize=6.3)
    for t, r in zip(ax.get_yticklabels(), rows):
        if r[0] in TRANSVERSAL:
            t.set_color(ORANGE)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel(r"$\Delta$ F1, multi-scale minus molecular")
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.tick_params(axis="y", length=0)
    h, l = ax.get_legend_handles_labels()
    import matplotlib.patches as mpatches
    h.append(mpatches.Patch(color=BLACK, alpha=0.25)); l.append("per-split range, graph against graph")
    ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.32, -0.07), frameon=False, fontsize=6.3,
              handlelength=1.6)
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    name = "fig_per_class_three_arms"
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(C.FIGURES / f"{name}.{ext}", bbox_inches="tight", **kw)
    print(f"[saved] {C.FIGURES / name}.pdf / .png | splits {seeds}")


if __name__ == "__main__":
    main()
