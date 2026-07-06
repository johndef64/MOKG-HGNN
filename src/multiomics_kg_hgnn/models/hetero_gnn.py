"""Heterogeneous multi-scale GNN for pan-cancer subtype classification.

Graph-level classification over per-patient HeteroData (see the datasets package).
Type-specific input encoding -> backbone message passing -> multi-scale readout
-> MLP -> 27-class logits.

Backbone is selected by name via BACKBONES. First iteration ships HGTConv; the
registry makes adding HeteroConv+SAGE / RGCN a one-function change (see the TODOs).
"""

import torch
import torch.nn as nn
from torch_geometric.nn import (HGTConv, HeteroConv, SAGEConv, RGCNConv,
                                global_mean_pool)


class HeteroInputEncoder(nn.Module):
    """Per-node-type encoding into a shared hidden space.
    Featured types get a concrete Linear projection (input dim from feature_dims);
    every other node type gets a learned Embedding (a single vector broadcast to
    all its nodes). Using concrete Linears — not LazyLinear — means the state_dict
    is fully materialized and portable (no UninitializedParameter to serialize)."""

    def __init__(self, node_types, hidden, feature_dims):
        super().__init__()
        self.node_types = list(node_types)
        self.hidden = hidden
        self.featured = dict(feature_dims)  # node_type -> in_channels
        self.proj = nn.ModuleDict(
            {self._key(nt): nn.Linear(c, hidden) for nt, c in self.featured.items()})
        self.emb = nn.ModuleDict(
            {self._key(nt): nn.Embedding(1, hidden) for nt in self.node_types
             if nt not in self.featured})

    @staticmethod
    def _key(nt):
        return nt.replace(".", "_")

    def forward(self, data):
        out = {}
        for nt in self.node_types:
            key = self._key(nt)
            if nt in self.featured:
                out[nt] = self.proj[key](data[nt].x)
            else:
                idx = torch.zeros(data[nt].num_nodes, dtype=torch.long, device=self.emb[key].weight.device)
                out[nt] = self.emb[key](idx)
        return out


# --------------------------------------------------------------------------
# Backbones (name -> builder). All three are HETEROGENEOUS, spanning a
# complexity spectrum on how they treat the typed relations:
#   hgt         -> attention per meta-relation           (most expressive)
#   hetero_sage -> one SAGEConv per relation, aggregated  (robust middle)
#   rgcn        -> one linear weight per relation         (simplest; KG-style)
# NB: MOGNN-TF's "GCN" is ChebConv on a COLLAPSED homogeneous graph — a different
# family. RGCNConv here is relational (per-relation weights), so it IS hetero;
# it is the simple end of the hetero spectrum, not the homogeneous baseline.
# Each builder returns an nn.Module whose forward is (x_dict, edge_index_dict) -> x_dict.
# --------------------------------------------------------------------------
def _build_hgt(metadata, hidden, num_layers, heads, dropout):
    return _HGTStack(metadata, hidden, num_layers, heads, dropout)


class _HGTStack(nn.Module):
    def __init__(self, metadata, hidden, num_layers, heads, dropout):
        super().__init__()
        self.convs = nn.ModuleList(
            [HGTConv(hidden, hidden, metadata, heads=heads) for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: self.drop(v.relu()) for k, v in x_dict.items()}
        return x_dict


def _build_hetero_sage(metadata, hidden, num_layers, heads, dropout):
    return _HeteroSAGEStack(metadata, hidden, num_layers, dropout)


class _HeteroSAGEStack(nn.Module):
    """HeteroConv with one SAGEConv per edge type, summed across relations per
    destination node type. The robust hetero baseline (== to_hetero(GraphSAGE))."""

    def __init__(self, metadata, hidden, num_layers, dropout):
        super().__init__()
        edge_types = metadata[1]
        self.convs = nn.ModuleList([
            HeteroConv({et: SAGEConv((-1, -1), hidden) for et in edge_types}, aggr="sum")
            for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: self.drop(v.relu()) for k, v in x_dict.items()}
        return x_dict


def _build_rgcn(metadata, hidden, num_layers, heads, dropout):
    return _RGCNStack(metadata, hidden, num_layers, dropout)


class _RGCNStack(nn.Module):
    """RGCNConv (one weight matrix per relation) run on the homogenized view.
    Converts the typed x_dict / edge_index_dict into a single (x, edge_index,
    edge_type) tensor, applies RGCN, then splits back per node type. Relation
    ids follow the fixed order of metadata[1]; node blocks follow metadata[0]."""

    def __init__(self, metadata, hidden, num_layers, dropout):
        super().__init__()
        self.node_types, self.edge_types = metadata
        self.rel2id = {et: i for i, et in enumerate(self.edge_types)}
        self.convs = nn.ModuleList([
            RGCNConv(hidden, hidden, num_relations=len(self.edge_types))
            for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        # concatenate node features in a fixed type order; record per-type offsets
        offsets, cursor, xs = {}, 0, []
        for nt in self.node_types:
            offsets[nt] = cursor
            xs.append(x_dict[nt])
            cursor += x_dict[nt].size(0)
        x = torch.cat(xs, dim=0)

        # remap each relation's edges into the global index space + tag edge_type
        ei_list, et_list = [], []
        for et, ei in edge_index_dict.items():
            src_t, _, dst_t = et
            shifted = ei.clone()
            shifted[0] = shifted[0] + offsets[src_t]
            shifted[1] = shifted[1] + offsets[dst_t]
            ei_list.append(shifted)
            et_list.append(torch.full((ei.size(1),), self.rel2id[et],
                                      dtype=torch.long, device=ei.device))
        edge_index = torch.cat(ei_list, dim=1)
        edge_type = torch.cat(et_list, dim=0)

        for conv in self.convs:
            x = self.drop(conv(x, edge_index, edge_type).relu())

        # split back into a per-type dict
        out = {}
        for nt in self.node_types:
            start = offsets[nt]
            out[nt] = x[start:start + x_dict[nt].size(0)]
        return out


BACKBONES = {
    "hgt": _build_hgt,
    "hetero_sage": _build_hetero_sage,
    "rgcn": _build_rgcn,
}


class HeteroMultiScaleGNN(nn.Module):
    """Full model: encoder -> backbone -> multi-scale readout -> classifier."""

    def __init__(self, metadata, num_classes, feature_dims, backbone="hgt", hidden=64,
                 num_layers=2, heads=2, dropout=0.2, readout_types=("gene", "pathway", "GO_term")):
        super().__init__()
        self.node_types, _ = metadata
        if backbone not in BACKBONES:
            raise ValueError(f"backbone '{backbone}' not in {list(BACKBONES)}")
        # readout only over scales that actually exist in this template
        self.readout_types = [t for t in readout_types if t in self.node_types]

        self.encoder = HeteroInputEncoder(self.node_types, hidden, feature_dims)
        self.backbone = BACKBONES[backbone](metadata, hidden, num_layers, heads, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * len(self.readout_types), hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, data):
        x_dict = self.encoder(data)
        x_dict = self.backbone(x_dict, data.edge_index_dict)
        pooled = [global_mean_pool(x_dict[t], data[t].batch) for t in self.readout_types]
        return self.classifier(torch.cat(pooled, dim=1))
