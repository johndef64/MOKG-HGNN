"""Mechanistic explainability for a trained MOKG-HGNN run.

Runs GNNExplainer (edge masks) on a sample of test patients, aggregates the
importance of each pathway / GO_term node PER SUBTYPE, and writes tables mapping
subtype -> top biological mechanisms (with their real Reactome / GO ids). This is
the thesis selling point: mechanism-level attribution a gene-only model cannot give.

Why edge masks (not node masks): pathway / GO_term nodes are FEATURELESS (learned
embeddings, no per-patient x), so node-feature masks ignore them. Edge masks score
the gene->pathway / gene->GO / GO->GO edges; a node's importance is the aggregate
of the edge weights touching it.

The template used at train time may not be on disk (per-seed templates aren't
saved locally), so we REBUILD it from the run's feature selection + graph_info.json
(metapath / scales), which also regenerates node_*.csv for index->name mapping.

Usage:
    conda run -n gnn python scripts/kg_hgnn/explain.py --run results/ablation_hetero_sage/full/<ts>
    conda run -n gnn python scripts/kg_hgnn/explain.py --run <dir> --per-class 15 --epochs 50
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, GNNExplainer

from multiomics_kg_hgnn.pancancer_prediction.preprocessing.make_datasets import make_datasets
from multiomics_kg_hgnn.models.hetero_gnn import HeteroMultiScaleGNN

# scales to explain (featureless superior scales)
SCALES = ("pathway", "GO_term")
NODE_CSV = {"pathway": ("node_pathway.csv", "reactome_id"),
            "GO_term": ("node_GO_term.csv", "go_id")}
# id -> human-readable name tables, extracted once from the KG (PKT/nodes.zip) so
# the CSVs carry biological names, not bare R-HSA/GO ids. See pkt-name-tables note.
PKT_NAMES = {"pathway": "data/prior_knowledge/PKT/pathway_names.csv",
             "GO_term": "data/prior_knowledge/PKT/go_names.csv"}


def _load_name_map(scale):
    """{id: label} for a scale, or {} if the name table is missing."""
    p = PKT_NAMES.get(scale)
    if not p or not os.path.exists(p):
        print(f"[explain] WARN: {p} missing -> '{scale}' names left blank. "
              f"Regenerate with: python scripts/kg_hgnn/extract_pkt_names.py")
        return {}
    df = pd.read_csv(p)
    return dict(zip(df["id"].astype(str), df["label"].astype(str)))


# ---------------------------------------------------------------------------
def _infer_seed(split_dir):
    m = os.path.basename(split_dir.rstrip("/\\"))
    return m.replace("splits_seed_", "")


def rebuild_template(run_dir, cfg, out_dir):
    """Rebuild the exact template this run trained on (same metapath/scales),
    into out_dir, which also gets the node_*.csv vocabularies for name mapping."""
    gi = {}
    gip = os.path.join(run_dir, "graph_info.json")
    if os.path.exists(gip):
        gi = json.load(open(gip))
    seed = _infer_seed(cfg["data"]["split_dir"])
    fs = f"data/training/feature_selection/splits_seed_{seed}"
    if not os.path.exists(os.path.join(fs, "selected_genes.csv")):
        raise FileNotFoundError(
            f"Feature selection for seed {seed} not found ({fs}). Rebuild it first "
            f"(make_graph.sh / feature_selection) before explaining.")

    node_types = gi.get("node_types", ["gene", "miRNA", "TF", "pathway", "GO_term", "disease"])
    flags = ["--metapath"] if gi.get("has_metapath", True) else []
    if "disease" not in node_types:
        flags.append("--no-disease")
    if "pathway" not in node_types:
        flags.append("--no-pathway")
    if "GO_term" not in node_types:
        flags.append("--no-go")

    os.makedirs(out_dir, exist_ok=True)
    cmd = ["python", "scripts/preprocessing/priors/build_hetero_graph.py",
           "--gene-list", f"{fs}/selected_genes.csv", "--tf-list", f"{fs}/selected_tf.csv",
           "--mirna-list", f"{fs}/selected_mirna.txt", "--go-min-support", "3",
           "--out-dir", out_dir, "--force", *flags]
    print(f"[explain] rebuilding template ({' '.join(flags) or 'no extra flags'}) -> {out_dir}")
    subprocess.run(cmd, check=True)
    return os.path.join(out_dir, "hetero_graph_template.pt")


def load_model_and_data(run_dir, template_path, hetero_dir):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    d, m = cfg["data"], cfg["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, _, test_ds, ncls, classes = make_datasets(
        split_dir=d["split_dir"], template_path=template_path, hetero_dir=hetero_dir,
        use_cnv=d.get("use_cnv", True), use_mirna=d.get("use_mirna", True),
        scaler=d.get("scaler", "standard"))
    fdims = {nt: t.shape[-1] for nt, t in train_ds.features.items()}
    model = HeteroMultiScaleGNN(
        train_ds.template.metadata(), ncls, fdims, backbone=m.get("backbone", "hgt"),
        hidden=int(m["hidden"]), num_layers=int(m["num_layers"]), heads=int(m.get("heads", 2)),
        dropout=float(m["dropout"]), readout_types=tuple(m["readout_types"])).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, "model_best.pt"), map_location=device))
    model.eval()
    # `classes` maps each encoded index 0..C-1 back to the REAL iCluster label
    # (C24/LAML absent -> a gap): use it so the CSVs carry the true subtype, not a
    # renumbered C{idx+1}. See docs/explainability_report.md label note.
    return model, train_ds, test_ds, ncls, device, classes


class _Wrap(nn.Module):
    """Adapt forward(HeteroData) -> forward(x_dict, edge_index_dict) for the Explainer."""
    def __init__(self, model, template_batch):
        super().__init__(); self.m = model; self.tb = template_batch

    def forward(self, x_dict, edge_index_dict, **kw):
        d = self.tb.clone()
        for k, v in x_dict.items():
            d[k].x = v
        for k, v in edge_index_dict.items():
            d[k].edge_index = v
        return self.m(d)


def _sample_patients(model, test_ds, device, per_class, only_correct):
    """Return list of (dataset_index, true_label), balanced per class."""
    by_class = defaultdict(list)
    loader = DataLoader([test_ds[i] for i in range(len(test_ds))], batch_size=64)
    idx = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).argmax(1).cpu().numpy()
            y = batch.y.cpu().numpy()
            for j in range(len(y)):
                if (not only_correct) or (pred[j] == y[j]):
                    by_class[int(y[j])].append(idx + j)
            idx += len(y)
    chosen = []
    rng = np.random.default_rng(0)
    for c, items in sorted(by_class.items()):
        pick = items if len(items) <= per_class else list(rng.choice(items, per_class, replace=False))
        chosen += [(i, c) for i in pick]
    print(f"[explain] {len(chosen)} patients sampled across {len(by_class)} classes "
          f"({'correct only' if only_correct else 'all'})")
    return chosen


def _node_importance_from_edges(explanation, data):
    """Aggregate edge masks into per-node importance for each explained scale.
    A pathway/GO node's importance = sum of the mask weights of the edges incident
    to it. `data` is the single-graph batch the explanation was computed on, so its
    edge_index and the mask are aligned per relation.

    ToUndirected added reverse relations (e.g. rev_member_of): a scale node is the
    SRC of the reverse and the DST of the forward, so summing over both directions
    double-counts symmetrically — consistent across patients, fine for ranking."""
    scales = [s for s in SCALES if s in data.node_types]
    out = {s: np.zeros(int(data[s].num_nodes)) for s in scales}
    deg = {s: np.zeros(int(data[s].num_nodes)) for s in scales}  # edges per node
    for et, mask in explanation.edge_mask_dict.items():
        src, _, dst = et
        if src not in out and dst not in out:
            continue
        ei = data[et].edge_index.cpu().numpy()
        w = mask.detach().cpu().numpy()
        if ei.shape[1] != w.shape[0]:           # guard: mask must align with edges
            continue
        if dst in out:
            np.add.at(out[dst], ei[1], w); np.add.at(deg[dst], ei[1], 1)
        if src in out:
            np.add.at(out[src], ei[0], w); np.add.at(deg[src], ei[0], 1)
    # normalize by degree -> MEAN edge importance per node, not sum. Removes the
    # high-degree hub bias (generic pathways like Metabolism would otherwise always
    # win just by having more edges). Nodes with no edges stay 0.
    for s in scales:
        nz = deg[s] > 0
        out[s][nz] = out[s][nz] / deg[s][nz]
    return out


def explain_run(run_dir, per_class=15, only_correct=True, epochs=50, topk=15,
                work_dir=None, out_dir=None, reuse_template=False):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    work_dir = work_dir or os.path.join("data", "prior_knowledge", "hetero", "_explain_tmp")
    template_path = os.path.join(work_dir, "hetero_graph_template.pt")
    if reuse_template and os.path.exists(template_path):
        print(f"[explain] reusing existing template -> {template_path}")
    else:
        template_path = rebuild_template(run_dir, cfg, work_dir)
    model, train_ds, test_ds, ncls, device, classes = load_model_and_data(run_dir, template_path, work_dir)
    # real subtype name for an encoded index (fallback to C{idx+1} if unavailable)
    def _sub(cls):
        try:
            return f"C{int(classes[cls])}"
        except Exception:
            return f"C{cls + 1}"
    template = train_ds.template

    # index -> biological name, per scale
    names = {}
    for s, (csv, col) in NODE_CSV.items():
        p = os.path.join(work_dir, csv)
        names[s] = pd.read_csv(p)[col].astype(str).tolist() if os.path.exists(p) else None

    patients = _sample_patients(model, test_ds, device, per_class, only_correct)

    # accumulate per-subtype node importance. Use a factory bound to n_nodes to
    # avoid the classic late-binding bug (a lambda capturing the loop var `s`
    # would size every scale with the LAST s).
    def _zeros_factory(n):
        return lambda: np.zeros(n)
    accum = {s: defaultdict(_zeros_factory(int(template[s].num_nodes)))
             for s in SCALES if s in template.node_types}
    counts = defaultdict(int)

    for n, (di, cls) in enumerate(patients, 1):
        data = next(iter(DataLoader([test_ds[di]], batch_size=1))).to(device)
        wrap = _Wrap(model, data).to(device)
        expl = Explainer(
            model=wrap, algorithm=GNNExplainer(epochs=epochs), explanation_type="model",
            node_mask_type=None, edge_mask_type="object",
            model_config=dict(mode="multiclass_classification", task_level="graph", return_type="raw"))
        e = expl(data.x_dict, data.edge_index_dict)
        imp = _node_importance_from_edges(e, data)
        for s in accum:
            accum[s][cls] += imp[s]
        counts[cls] += 1
        if n % 20 == 0 or n == len(patients):
            print(f"[explain] {n}/{len(patients)} patients explained", flush=True)

    # write per-scale top-k tables (subtype -> top mechanisms)
    out_dir = out_dir or os.path.join("results", "explanations", os.path.basename(run_dir.rstrip("/\\")))
    os.makedirs(out_dir, exist_ok=True)
    for s in accum:
        name_map = _load_name_map(s)   # id -> biological name (empty if table absent)
        # per-class mean importance, and the across-class average (baseline)
        per_cls = {cls: accum[s][cls] / max(counts[cls], 1) for cls in accum[s]}
        overall = np.mean(np.stack(list(per_cls.values())), axis=0)
        rows = []
        for cls in sorted(per_cls):
            imp = per_cls[cls]
            distinctive = imp - overall       # how much MORE this subtype uses it
            # rank by distinctiveness: the mechanisms SPECIFIC to this subtype, not
            # the generically-important ones shared by every class.
            top = np.argsort(distinctive)[::-1][:topk]
            for rank, idx in enumerate(top, 1):
                node_id = names[s][idx] if names[s] else str(idx)
                rows.append({
                    "subtype": _sub(cls), "rank": rank, "node_idx": int(idx),
                    "id": node_id,
                    "name": name_map.get(node_id, ""),   # biological name (from PKT)
                    "distinctive_score": round(float(distinctive[idx]), 5),
                    "importance": round(float(imp[idx]), 5),
                    "overall_importance": round(float(overall[idx]), 5),
                    "n_patients": counts[cls],
                })
        df = pd.DataFrame(rows)
        fp = os.path.join(out_dir, f"top_{s}_by_subtype.csv")
        df.to_csv(fp, index=False)
        print(f"[saved] {fp}")

    _plot_heatmap(accum, names, counts, template, out_dir, sub=_sub)
    print(f"\n[explain] done -> {out_dir}")
    return out_dir


def _plot_heatmap(accum, names, counts, template, out_dir, top_per_scale=10, sub=None):
    # `sub`: encoded-index -> real subtype label. Fallback to C{c+1} if not given.
    if sub is None:
        sub = lambda c: f"C{c + 1}"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for s in accum:
        subs = sorted(accum[s])
        per_cls = {c: accum[s][c] / max(counts[c], 1) for c in subs}
        overall = np.mean(np.stack(list(per_cls.values())), axis=0)
        distinct = {c: per_cls[c] - overall for c in subs}   # subtype-specific signal
        # union of the most DISTINCTIVE mechanisms per subtype
        top_ids = set()
        for c in subs:
            top_ids.update(np.argsort(distinct[c])[::-1][:top_per_scale])
        top_ids = sorted(top_ids)
        M = np.array([[distinct[c][i] for i in top_ids] for c in subs])
        if M.size == 0:
            continue
        vmax = np.abs(M).max() or 1.0
        fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(top_ids)), max(5, 0.35 * len(subs))))
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)  # diverging
        ax.set_xticks(range(len(top_ids)))
        ax.set_xticklabels([names[s][i] if names[s] else str(i) for i in top_ids],
                           rotation=90, fontsize=7)
        ax.set_yticks(range(len(subs)))
        ax.set_yticklabels([sub(c) for c in subs], fontsize=8)
        ax.set_title(f"Distinctive {s} importance per subtype (vs mean)")
        fig.colorbar(im, ax=ax, fraction=0.02)
        fig.tight_layout()
        fp = os.path.join(out_dir, f"heatmap_{s}.png")
        fig.savefig(fp, dpi=140); plt.close(fig)
        print(f"[saved] {fp}")


def main():
    ap = argparse.ArgumentParser(description="Mechanistic explainability (GNNExplainer) for a run.")
    ap.add_argument("--run", required=True, help="A run dir with model_best.pt + config.json.")
    ap.add_argument("--per-class", type=int, default=15, help="Patients per subtype (default 15).")
    ap.add_argument("--all-patients", action="store_true", help="Include misclassified too.")
    ap.add_argument("--epochs", type=int, default=50, help="GNNExplainer epochs per patient.")
    ap.add_argument("--topk", type=int, default=15, help="Top mechanisms per subtype in the table.")
    ap.add_argument("--reuse-template", action="store_true",
                    help="Reuse the template already in the work dir (skip the ~90s KG rebuild).")
    args = ap.parse_args()
    explain_run(args.run, per_class=args.per_class, only_correct=not args.all_patients,
                epochs=args.epochs, topk=args.topk, reuse_template=args.reuse_template)


if __name__ == "__main__":
    main()
