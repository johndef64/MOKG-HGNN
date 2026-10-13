"""Per-subtype F1 difference MOKG-HGNN minus MOGNN-TF at 700 genes, without and with the
shallow branch on both sides (single column; updates the per-class figure of v2).

For each subtype: grey hollow circle = KG - MOL (graph against graph, no shallow), teal or
black square = KG+S - MOL+S (both systems with the same shallow branch), joined by a thin
segment; a horizontal bar spans the per-split range of KG+S - MOL+S. Sorted by KG+S - MOL+S.
Same panel and the same five splits for all arms.

Reads results/four_arms/tables/per_class_four_arms.csv only for the order check; the values
come from the runs (per_class_four_arms.py logic). Writes
results/four_arms/figures/fig_per_class_four_arms.{pdf,png}; with --labels thesis, neutral model
names (multi-scale / molecular scale) in fig_per_class_four_arms_thesis.{pdf,png}.

  python scripts/four_arms/plot_per_class_four_arms.py [--labels thesis]
"""

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
from plot_four_arms import TEAL, GREY, BLACK, MODE, T, plt  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "ns_per_class", core.REPO / "scripts" / "no_shallow" / "per_class_three_arms.py")
ns_pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ns_pc)

ORANGE = "#D9822B"
TRANSVERSAL = {"C10", "C17", "C20", "C25"}    # transversal / pan-squamous, as in v2
MAPPING = core.REPO / "data" / "omics" / "TCGA_iCluster_C1-C28_mapping.csv"
LEVEL = 700


def labels():
    out = {}
    with open(MAPPING, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            lab = r["iCluster_label"]
            if lab.lower().startswith("not individually"):
                lab = r["cluster_type"]
            out[r["iCluster"]] = f"{r['iCluster']} · {lab if len(lab) <= 22 else lab[:21] + '…'}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", choices=("paper", "thesis"), default="paper")
    MODE["labels"] = ap.parse_args().labels
    names, seeds, per_old, support = ns_pc.load(LEVEL)
    kg, mol, mol_s = per_old["MS"], per_old["NS"], per_old["WS"]
    runs = core.read_runs("kg_s")
    kg_s = {n: [] for n in names}
    for s in seeds:
        pc = runs[(LEVEL, s)]["per_class"]
        for i, n in enumerate(names):
            assert pc[i][2] == support[n], f"support mismatch split {s} class {n}"
            kg_s[n].append(pc[i][1])
    k = len(seeds)
    rows = []
    for n in names:
        d0 = sum(a - b for a, b in zip(kg[n], mol[n])) / k
        ds = [a - b for a, b in zip(kg_s[n], mol_s[n])]
        rows.append((n, d0, sum(ds) / k, min(ds), max(ds)))
    rows.sort(key=lambda r: r[2])
    lab = labels()

    fig, ax = plt.subplots(figsize=(3.5, 5.6))
    ax.axvline(0, color="0.35", lw=0.8, zorder=1)
    for y, (n, d0, ds, lo, hi) in enumerate(rows):
        ax.plot([lo, hi], [y, y], color=BLACK, alpha=0.2, lw=2.2, solid_capstyle="butt", zorder=1)
        ax.plot([d0, ds], [y, y], color="0.6", lw=0.7, zorder=2)
        ax.scatter([d0], [y], s=16, facecolor="white", edgecolor=GREY, lw=1.0, zorder=3,
                   label="graph only on both sides (KG $-$ MOL)" if y == 0 else None)
        ax.scatter([ds], [y], s=14, marker="s", color=TEAL if ds > 0.015 else BLACK, zorder=4,
                   label="shallow branch on both sides (KG+S $-$ MOL+S)" if y == 0 else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab.get(r[0], r[0]) for r in rows], fontsize=6.3)
    for t, r in zip(ax.get_yticklabels(), rows):
        if r[0] in TRANSVERSAL:
            t.set_color(ORANGE)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel(T(r"$\Delta$ F1, MOKG-HGNN minus MOGNN-TF"))
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    h, l = ax.get_legend_handles_labels()
    import matplotlib.patches as mpatches
    h.append(mpatches.Patch(color=BLACK, alpha=0.2)); l.append("per-split range, with shallow branch")
    ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.32, -0.07), frameon=False, fontsize=6.3,
              handlelength=1.6)
    core.FIGURES.mkdir(parents=True, exist_ok=True)
    name = "fig_per_class_four_arms" + ("_thesis" if MODE["labels"] == "thesis" else "")
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(core.FIGURES / f"{name}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"[saved] {core.FIGURES / name}.pdf / .png | splits {seeds}")


if __name__ == "__main__":
    main()
