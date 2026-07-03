#%%
# Smoke test: prove the hetero backbone template feeds the nets listed in
# docs/task_e_reti_da_usare.md. Injects dummy features (real features are
# per-patient, injected downstream) and runs one forward pass through each net,
# then the multi-scale readout -> 27-class graph-level logits.
#
#   conda run -n gnn python scripts/preprocessing/priors/check_hetero_graph.py
#
# Note: to_hetero(...) relies on FX tracing and currently breaks on this
# torch 2.7 / PyG 2.7 env (unrelated to the graph: a plain conv on a relation
# works). HeteroConv is the drop-in equivalent and is tested here instead.

import torch
import torch.nn as nn
from torch_geometric.nn import (SAGEConv, HGTConv, HeteroConv, RGCNConv,
                                global_mean_pool)

TEMPLATE = "data/prior_knowledge/hetero/hetero_graph_template.pt"
D, H, C = 16, 32, 27  # in-dim, hidden, num classes


def main():
    data = torch.load(TEMPLATE, weights_only=False)
    md = data.metadata()
    print("node types:", md[0])
    print("relations :", len(md[1]))

    # dummy per-node features (stand-in for per-patient omics / learned embeds)
    for nt in md[0]:
        data[nt].x = torch.randn(data[nt].num_nodes, D)
    xd, eid = data.x_dict, data.edge_index_dict

    # 1) HeteroConv (per-relation control; to_hetero equivalent)
    hc = HeteroConv({et: SAGEConv((-1, -1), H) for et in md[1]}, aggr="sum")
    o = hc(xd, eid)
    assert o["gene"].shape == (data["gene"].num_nodes, H)
    print(f"[HeteroConv/SAGE] gene -> {tuple(o['gene'].shape)}  OK")

    # 2) HGTConv (attention per meta-relation)
    o2 = HGTConv(D, H, md, heads=2)(xd, eid)
    assert o2["gene"].shape == (data["gene"].num_nodes, H)
    print(f"[HGTConv]         gene -> {tuple(o2['gene'].shape)}  OK")

    # 3) RGCNConv (KG-style, one weight per relation) via homogenization
    hom = data.to_homogeneous()
    r = RGCNConv(D, H, num_relations=int(hom.edge_type.max()) + 1)
    o3 = r(hom.x, hom.edge_index, hom.edge_type)
    print(f"[RGCNConv]        nodes -> {tuple(o3.shape)}  OK")

    # multi-scale readout -> graph-level 27-class logits
    batch = {nt: torch.zeros(data[nt].num_nodes, dtype=torch.long) for nt in md[0]}
    pooled = torch.cat(
        [global_mean_pool(o[nt], batch[nt]) for nt in ["gene", "pathway", "GO_term"]], dim=1)
    logits = nn.Linear(pooled.shape[1], C)(pooled)
    assert logits.shape == (1, C)
    print(f"[readout+MLP]     logits -> {tuple(logits.shape)}  OK ({C} classi)")
    print("\nALL NETS (HeteroConv / HGTConv / RGCNConv) + readout: OK")


if __name__ == "__main__":
    main()
