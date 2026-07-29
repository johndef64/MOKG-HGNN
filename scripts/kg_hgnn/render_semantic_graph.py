#!/usr/bin/env python
"""Render the multi-scale template as a publication figure (no Gephi needed).

Reads the Gephi node/edge tables written by export_gephi_graph.py and draws a
force-directed network: nodes coloured by biological scale, sized by degree,
top hubs labelled. Output: a single PNG.

Usage:
    python scripts/kg_hgnn/export_gephi_graph.py          # writes paper/gephi/*
    python scripts/kg_hgnn/render_semantic_graph.py \
        --gephi-dir paper/gephi --out paper/images/semantic_graph.png
"""
import argparse
import csv
import os
from collections import Counter

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

COLOR = {  # match the framework-figure palette
    "gene": "#1F9E9E", "miRNA": "#8ED0D0", "TF": "#0C5C5C",
    "pathway": "#7C5CA8", "GO": "#78AAD2", "disease": "#E08C3C",
}
ORDER = ["gene", "miRNA", "TF", "pathway", "GO", "disease"]
PERSCALE = {"gene": 3, "TF": 3, "miRNA": 3, "pathway": 3, "GO": 2, "disease": 0}


def clip(t, n=24):
    return t if len(t) <= n else t[:n - 1] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gephi-dir", default="paper/gephi")
    ap.add_argument("--out", default="paper/images/semantic_graph.png")
    ap.add_argument("--k", type=float, default=0.75)
    ap.add_argument("--iters", type=int, default=90)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    nodes = {}
    with open(os.path.join(args.gephi_dir, "gephi_nodes.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            nodes[r["Id"]] = (r["Label"], r["Scale"])
    edges = []
    with open(os.path.join(args.gephi_dir, "gephi_edges.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            edges.append((r["Source"], r["Target"]))

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    print("nodes", G.number_of_nodes(), "edges", G.number_of_edges())
    deg = dict(G.degree())

    print("laying out (force-directed)...")
    pos = nx.spring_layout(G, k=args.k, iterations=args.iters, seed=args.seed)

    nl = list(G.nodes())
    xs = np.array([pos[n][0] for n in nl])
    ys = np.array([pos[n][1] for n in nl])
    sizes = np.array([deg[n] for n in nl], float)
    sizes = 3 + 34 * np.sqrt(sizes / sizes.max())

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.add_collection(LineCollection([(pos[u], pos[v]) for u, v in G.edges()],
                      colors="#999999", linewidths=0.12, alpha=0.30, zorder=1))
    for s in ORDER:
        idx = [i for i, n in enumerate(nl) if nodes[n][1] == s]
        ax.scatter(xs[idx], ys[idx], s=sizes[idx], c=COLOR[s], linewidths=0,
                   alpha=0.9, zorder=2)

    label_nodes = []
    for s in ORDER:
        label_nodes += sorted([n for n in nl if nodes[n][1] == s],
                              key=lambda n: -deg[n])[:PERSCALE[s]]
    placed = []
    for n in sorted(label_nodes, key=lambda n: -deg[n]):
        x, y = pos[n]
        for px, py in placed:
            if abs(x - px) < 0.14 and abs(y - py) < 0.05:
                y += 0.06
        placed.append((x, y))
        ax.text(x, y, clip(nodes[n][0]), fontsize=6.4, zorder=5, va="center",
                ha="center", color="#111111",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.78))

    cnt = Counter(nodes[n][1] for n in nl)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR[s],
               markersize=8, label=f"{s} ({cnt[s]})") for s in ORDER]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8,
              handletextpad=0.2, borderpad=0.2, labelspacing=0.3)
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.tight_layout(pad=0.2)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor="white")
    print("saved", args.out)


if __name__ == "__main__":
    main()
