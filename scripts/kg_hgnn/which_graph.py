"""Tell whether a run was trained WITH or WITHOUT metapaths (and how many
relations), for every run under results/. Prefers the explicit graph_info.json
written by new runs; falls back to inspecting the checkpoint weights for older
runs that predate graph_info.json.

Usage:
    conda run -n gnn python scripts/kg_hgnn/which_graph.py
    conda run -n gnn python scripts/kg_hgnn/which_graph.py --results results/kg_hgnn_hgt
"""

import argparse
import glob
import json
import os

import torch


def info_for_run(run_dir):
    gi = os.path.join(run_dir, "graph_info.json")
    if os.path.exists(gi):
        d = json.load(open(gi))
        return d["has_metapath"], d["num_relations"], "graph_info.json"

    # fallback: infer from the checkpoint weights
    ck = os.path.join(run_dir, "model_best.pt")
    if not os.path.exists(ck):
        return None, None, "no checkpoint"
    sd = torch.load(ck, map_location="cpu")
    has_mp = any("shares_target" in k for k in sd)
    # HGT stores per-relation weights as k_rel [num_rel, ...]; else count HeteroConv keys
    nrel = None
    for k, v in sd.items():
        if k.endswith("k_rel.weight"):
            nrel = int(v.shape[0]); break
    return has_mp, nrel, "checkpoint (inferred)"


def main():
    ap = argparse.ArgumentParser(description="WITH/WITHOUT metapath per run.")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    runs = sorted({os.path.dirname(p) for p in
                   glob.glob(os.path.join(args.results, "**", "model_best.pt"), recursive=True)}
                  | {os.path.dirname(p) for p in
                     glob.glob(os.path.join(args.results, "**", "graph_info.json"), recursive=True)})
    if not runs:
        print(f"No runs under {args.results}/")
        return
    print(f"{'metapath':10} {'rel':>4}  {'source':22} run")
    for r in runs:
        mp, nrel, src = info_for_run(r)
        tag = "?" if mp is None else ("METAPATH" if mp else "no-metap")
        print(f"{tag:10} {str(nrel):>4}  {src:22} {r}")


if __name__ == "__main__":
    main()
