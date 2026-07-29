#!/usr/bin/env python
"""Render the multi-scale template as a publication figure (no Gephi needed).

Reads the Gephi node/edge tables written by export_gephi_graph.py and draws a
force-directed network: nodes coloured by biological scale, sized by degree so
hubs emerge, with example hubs labelled directly on the node. A second legend
box reports classic graph descriptors (LCC, degree, clustering, path length,
assortativity), computed on the fly.

Usage:
    python scripts/kg_hgnn/export_gephi_graph.py           # writes paper/gephi/*
    python scripts/kg_hgnn/render_semantic_graph.py \
        --gephi-dir paper/gephi --out paper/images/semantic_graph.png
"""
import argparse
import csv
import os

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

COLOR = {"gene": "#1F9E9E", "miRNA": "#8ED0D0", "TF": "#0C5C5C",
         "pathway": "#7C5CA8", "GO": "#78AAD2", "disease": "#E08C3C"}
ORDER = ["gene", "miRNA", "TF", "pathway", "GO", "disease"]
PERSCALE = {"gene": 3, "TF": 3, "miRNA": 2, "pathway": 2, "GO": 2, "disease": 0}


def clip(t, n=22):
    return t if len(t) <= n else t[:n - 1] + "…"


def descriptors(G):
    deg = np.array([d for _, d in G.degree()], float)
    lcc = max(nx.connected_components(G), key=len)
    L = G.subgraph(lcc)
    d = dict(nodes=G.number_of_nodes(), edges=G.number_of_edges(),
             density=nx.density(G), lcc=len(lcc) / G.number_of_nodes(),
             kmean=deg.mean(), kmax=int(deg.max()),
             clustering=nx.average_clustering(G),
             assort=nx.degree_assortativity_coefficient(G),
             apl=nx.average_shortest_path_length(L), diam=nx.diameter(L))
    return d


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
    G.remove_edges_from(nx.selfloop_edges(G))
    print("nodes", G.number_of_nodes(), "edges", G.number_of_edges())
    deg = dict(G.degree())

    print("computing descriptors...")
    D = descriptors(G)
    print("laying out (force-directed)...")
    pos = nx.spring_layout(G, k=args.k, iterations=args.iters, seed=args.seed)

    nl = list(G.nodes())
    xs = np.array([pos[n][0] for n in nl]); ys = np.array([pos[n][1] for n in nl])
    degarr = np.array([deg[n] for n in nl], float)
    sizes = 4 + 120 * (degarr / degarr.max()) ** 0.6
    sz = {n: s for n, s in zip(nl, sizes)}

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.add_collection(LineCollection([(pos[u], pos[v]) for u, v in G.edges()],
                      colors="#999999", linewidths=0.12, alpha=0.30, zorder=1))
    for s in ORDER:
        idx = [i for i, n in enumerate(nl) if nodes[n][1] == s]
        ax.scatter(xs[idx], ys[idx], s=sizes[idx], c=COLOR[s], linewidths=0,
                   alpha=0.9, zorder=2)

    # example hubs per scale, labelled ON the node; ring them to stand out
    label_nodes = []
    for s in ORDER:
        label_nodes += sorted([n for n in nl if nodes[n][1] == s],
                              key=lambda n: -deg[n])[:PERSCALE[s]]
    label_nodes = list(dict.fromkeys(label_nodes))
    center = np.array([xs.mean(), ys.mean()])
    Rmax = np.sqrt(((np.c_[xs, ys] - center) ** 2).sum(1)).max()
    ax.scatter([pos[n][0] for n in label_nodes], [pos[n][1] for n in label_nodes],
               s=[sz[n] * 1.6 for n in label_nodes], facecolors="none",
               edgecolors="#1a1a1a", linewidths=0.7, zorder=4)

    span = 2.3 * Rmax
    CW, CH = 0.0115 * span, 0.032 * span
    placed = []
    def hit(x, y, w, h):
        return any(abs(x - px) * 2 < (w + pw) and abs(y - py) * 2 < (h + ph)
                   for px, py, pw, ph in placed)
    for n in sorted(label_nodes, key=lambda n: -deg[n]):
        txt = clip(nodes[n][0]); w, h = len(txt) * CW, CH
        x0, y0 = pos[n]; best = (x0, y0)
        for r in np.arange(0.0, 0.55 * Rmax, 0.035 * Rmax):
            angs = [0.0] if r == 0 else np.linspace(0, 2 * np.pi, 16, endpoint=False)
            for a in angs:
                x, y = x0 + r * np.cos(a), y0 + r * np.sin(a)
                if not hit(x, y, w, h):
                    best = (x, y); break
            else:
                continue
            break
        x, y = best; placed.append((x, y, w, h))
        if (x - x0) ** 2 + (y - y0) ** 2 > (0.045 * Rmax) ** 2:
            ax.plot([x0, x], [y0, y], color="#555555", lw=0.4, alpha=0.75, zorder=3)
        ax.text(x, y, txt, fontsize=6.6, ha="center", va="center", zorder=6,
                color="#111111", bbox=dict(boxstyle="round,pad=0.13", fc="white",
                                           ec="#bbbbbb", lw=0.3, alpha=0.95))

    # scale legend
    from collections import Counter
    cnt = Counter(nodes[n][1] for n in nl)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR[s],
               markersize=8, label=f"{s} ({cnt[s]})") for s in ORDER]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8,
              handletextpad=0.2, borderpad=0.2, labelspacing=0.3)

    # graph-descriptor box
    asign = "−" if D['assort'] < 0 else "+"
    desc = (r"$\bf{Graph\ descriptors}$" "\n"
            f"nodes {D['nodes']}   edges {D['edges']:,}\n"
            f"density {D['density']:.3f}   LCC {D['lcc']*100:.1f}%\n"
            r"$\langle k \rangle$ " f"{D['kmean']:.1f}   " r"$k_{\max}$ " f"{D['kmax']}\n"
            f"clustering {D['clustering']:.2f}\n"
            f"avg. path {D['apl']:.1f}   diameter {D['diam']}\n"
            f"assortativity {asign}{abs(D['assort']):.2f}")
    ax.text(0.015, 0.015, desc, transform=ax.transAxes, fontsize=7.4, va="bottom",
            ha="left", linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#bbbbbb", lw=0.5))

    ax.set_axis_off(); ax.set_aspect("equal")
    fig.tight_layout(pad=0.2)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor="white")
    print("saved", args.out)


if __name__ == "__main__":
    main()
