"""Model classes of the new arms (plan §2). The original classes are not modified.

Shallow branch of MOGNN-TF (benchmark/MOGNN-TF/multiomics_gnn/models/gcn_tf.py): per-type
projection of the node features to 8 dims + ReLU, flatten over all N molecular nodes
(gene, miRNA, TF), Linear(8N, 2N) + ReLU, Linear(2N, 64) + ReLU = theta2, late fusion with
the graph representation in the last linear layer, dropout before it.

  HeteroMultiScaleGNNShallow  MOKG-HGNN + the same branch (arm kg_s): the last layer of the
                              classifier reads [h (256) || theta2 (64)] instead of h.
  HeteroShallowOnly           the branch alone, MOKG-HGNN recipe (arm sh_kg).
  GCNShallowOnly              the branch alone inside MOGNN-TF (arm sh_tf).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from multiomics_kg_hgnn.models.hetero_gnn import HeteroMultiScaleGNN
from multiomics_gnn.models.gcn_tf import GCN

RAISED = 8          # MOGNN-TF raised_dimension
THETA = 64          # MOGNN-TF concate_layer
SHALLOW_TYPES = ("gene", "miRNA", "TF")   # MOGNN-TF node order


class ShallowBranch(nn.Module):
    """theta2 from a batch of per-patient HeteroData (all graphs share the node counts)."""

    def __init__(self, feature_dims, num_nodes):
        super().__init__()
        self.types = [t for t in SHALLOW_TYPES if t in feature_dims]
        self.proj = nn.ModuleDict({t: nn.Linear(feature_dims[t], RAISED) for t in self.types})
        n_in = RAISED * sum(num_nodes[t] for t in self.types)
        self.linear1 = nn.Linear(n_in, n_in // 4)
        self.linear2 = nn.Linear(n_in // 4, THETA)

    def forward(self, data):
        b = data.num_graphs
        h = torch.cat([F.relu(self.proj[t](data[t].x)).reshape(b, -1) for t in self.types], dim=1)
        return F.relu(self.linear2(F.relu(self.linear1(h))))


class HeteroMultiScaleGNNShallow(HeteroMultiScaleGNN):
    def __init__(self, num_nodes, **kw):
        super().__init__(**kw)
        # Built on a forked CPU RNG: the global stream is left exactly where the original
        # model leaves it, so encoder, backbone (lazy SAGEConv weights are drawn at the first
        # forward) and the first classifier layer start from the same weights as arm kg.
        with torch.random.fork_rng(devices=[]):
            self.shallow = ShallowBranch(kw["feature_dims"], num_nodes)
            hidden = self.classifier[0].out_features
            self.classifier[3] = nn.Linear(hidden + THETA, kw["num_classes"])

    def forward(self, data):
        x_dict = self.backbone(self.encoder(data), data.edge_index_dict)
        pooled = [global_mean_pool(x_dict[t], data[t].batch) for t in self.readout_types]
        h = self.classifier[1](self.classifier[0](torch.cat(pooled, dim=1)))   # Linear, ReLU
        z = torch.cat([h, self.shallow(data)], dim=1)
        return self.classifier[3](self.classifier[2](z))                         # Dropout, Linear


class HeteroShallowOnly(nn.Module):
    def __init__(self, num_nodes, feature_dims, num_classes, dropout=0.2, **_):
        super().__init__()
        self.shallow = ShallowBranch(feature_dims, num_nodes)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(THETA, num_classes)

    def forward(self, data):
        return self.classifier(self.drop(self.shallow(data)))


def mokg_factory(arm, num_nodes):
    """What the MOKG-HGNN runner instantiates for each arm (it passes keyword arguments)."""
    if arm == "kg":
        return HeteroMultiScaleGNN
    if arm == "kg_s":
        return lambda **kw: HeteroMultiScaleGNNShallow(num_nodes=num_nodes, **kw)
    if arm == "sh_kg":
        return lambda **kw: HeteroShallowOnly(num_nodes=num_nodes, **kw)
    raise ValueError(arm)


class GCNShallowOnly(GCN):
    """gcn_tf with the graph branch removed: pre_conv_linear + parallel_linear1/2 + Linear(64, C).

    Built exactly as the original of arm mol_s (parallel=True, decoder=True), so the
    projection and shallow layers get the same initial weights; then the graph layers and
    the decoder (it reads theta1, which no longer exists; configs/mognntf_shallow_only.yaml
    sets train.decoder false for the trainer) are dropped and the classifier replaced."""

    def __init__(self, method, parallel, l2, decoder, *args, **kw):
        # GCN.__init__ calls super(GCN, self) by global name, and core.swapped_gcn() rebinds
        # that name to this class: restore the original name while the parent builds.
        from multiomics_gnn.models import gcn_tf
        swapped, gcn_tf.GCN = gcn_tf.GCN, GCN
        try:
            super().__init__(method, True, l2, True, *args, **kw)
        finally:
            gcn_tf.GCN = swapped
        self.decoder = False
        for name in ("conv1", "conv2", "linear1", "linear2", "jk", "decoder_1", "decoder_2"):
            if hasattr(self, name):
                delattr(self, name)
        self.classifier = nn.Linear(self.concate_layer, self.num_classes)

    def forward(self, x, edge_index, edge_weight):
        b, G, M, T = x.shape[0], self.num_gene, self.num_mirna, self.num_tf
        if self.num_features == 1:
            parts = [F.relu(self.pre_conv_linear_gene(x))]
        else:
            exp, cnv = x[:, :, 0], x[:, :, 1]
            parts = [F.relu(self.pre_conv_linear_gene(torch.stack([exp[:, :G], cnv[:, :G]], 2)))]
            if M > 0:
                parts.append(F.relu(self.pre_conv_linear_mirna(exp[:, G:G + M].unsqueeze(2))))
            if T > 0:
                tf = torch.stack([exp[:, G + M:G + M + T], cnv[:, G + M:G + M + T]], 2)
                parts.append(F.relu(self.pre_conv_linear_tf(tf)))
        h = torch.cat(parts, dim=1).reshape(b, -1)
        h = F.relu(self.parallel_linear2(F.relu(self.parallel_linear1(h))))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        return self.classifier(h)

