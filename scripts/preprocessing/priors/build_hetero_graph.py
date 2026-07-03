#%%
# Build the heterogeneous multi-scale backbone graph (HeteroData) for the
# MOKG-HGNN project (Proposta B).
#
# It fuses the EXISTING single-scale molecular layer produced by
# load_interaction.py ...
#     gene <-> gene   (BioGRID, undirected, HGNC symbol)
#     miRNA -> gene   (miRDB,          HGNC symbol)
#     TF    -> gene   (TFLink,         HGNC symbol)
# ... with the SUPERIOR scales pulled from the PKT knowledge graph
# (PheKnowLator / OWL-NETS property graph, Entrez-keyed):
#     gene  -> pathway   (Reactome R-HSA)
#     gene  -> GO_term   (via the gene->protein->GO bridge, see below)
#     GO    -> GO_term   (is_a / part_of hierarchy)
#     gene  -> disease   (MONDO, optional)
#
# The output is a *topology-only* torch_geometric.data.HeteroData template:
# nodes carry NO per-patient features here (those are injected downstream,
# one HeteroData per patient), only the shared scaffold + node id vocabularies.
# This template is directly consumable by to_hetero(...), HeteroConv, RGCNConv,
# HGTConv and HANConv (see docs/task_e_reti_da_usare.md).
#
# Anchor identifier: everything is joined on the gene. The molecular layer is
# HGNC-symbol-keyed; the KG is Entrez-keyed. We bridge the two with the HGNC
# conversion table (symbol <-> Entrez), so the gene node vocabulary stays 1:1
# aligned with the MOGNN-TF omics matrices.
#
# KG streaming: nodes.json (~483 MB) and edges.json (~4.6 GB) never fit in RAM,
# so they are read with ijson in a single streaming pass each. See
# docs/piano-mapping-PKT-heterodata.md for the measured schema and rationale.

import os
import json
import zipfile
import argparse
import collections

import numpy as np
import pandas as pd
import ijson

# Repo-root-relative paths (cwd when invoked via Makefile / scripts README).
# The HGNC conversion table ships next to this script.
_HERE = os.path.dirname(os.path.abspath(__file__))

CONVERSION_TABLE = os.path.join(_HERE, "hgcn_mart_conversion_table.zip")

GGI_NODES = "data/prior_knowledge/GGI/GGI_nodes_list.csv"
GGI_EDGES = "data/prior_knowledge/GGI/GGI_edges_undirected.csv"
MIRNA_NODES = "data/prior_knowledge/miRNA/miRNA_nodes_list.csv"
MIRNA_GENE = "data/prior_knowledge/miRNA/miRNA_gene_interactions.csv"
METAPATH_MIRNA = "data/prior_knowledge/miRNA/metapath_mirna_mirna.csv"
TF_EDGES = "data/prior_knowledge/TF/TF_interactions.tsv"

PKT_NODES = "data/prior_knowledge/PKT/nodes.zip"
PKT_EDGES = "data/prior_knowledge/PKT/edges.zip"

OUT_DIR = "data/prior_knowledge/hetero/"

# --- KG relation routing (see docs/piano-mapping-PKT-heterodata.md, sez. 2) ---
# Only the predicates below are consumed; everything else in the 11.1M edges is
# ignored. Routing is keyed on the (source class_code, target class_code) PAIR,
# never on the predicate alone (e.g. "type" means is_a only when both endpoints
# share the class).
GENE_CC = "EntrezID"
PROTEIN_CC = "PR"
PATHWAY_CC = "R-HSA"
GO_CC = "GO"
DISEASE_CC = "MONDO"
KEEP_CC = {GENE_CC, PROTEIN_CC, PATHWAY_CC, GO_CC, DISEASE_CC}

# gene -> pathway
PRED_MEMBER_OF = {"participates in"}
# protein -> GO (bridged to gene). Covers BP / MF / CC aspects.
PRED_ANNOTATED = {"participates in", "has function", "located_in"}
# GO -> GO hierarchy
PRED_GO_HIER = {"type", "part_of"}
# gene -> disease
PRED_ASSOCIATED = {"causes or contributes to condition"}
# gene <-> protein bridge (either direction present in the KG)
PRED_HAS_PRODUCT = {"has gene product"}
PRED_PRODUCT_OF = {"gene product of"}


# ---------------------------------------------------------------------------
# 1. Molecular layer + gene<->Entrez vocabulary
# ---------------------------------------------------------------------------
def load_symbol_entrez_map(conversion_table=CONVERSION_TABLE):
    """symbol -> Entrez (str) from the HGNC conversion table (unique Entrez)."""
    conv = pd.read_csv(conversion_table, sep="\t", compression="zip")
    conv["NCBI gene ID"] = pd.to_numeric(conv["NCBI gene ID"], errors="coerce").astype("Int64")
    conv = conv.dropna(subset=["NCBI gene ID", "Approved symbol"]).drop_duplicates(subset=["Approved symbol"])
    return {sym: str(int(eid)) for sym, eid in zip(conv["Approved symbol"], conv["NCBI gene ID"])}


def load_gene_vocab(gene_nodes_file, sym2entrez):
    """Read the gene node list, keep genes that map to an Entrez.

    Returns (genes, sym2idx, entrez2idx). `genes` is the ordered list of HGNC
    symbols = row order of the downstream omics feature matrices.
    """
    df = pd.read_csv(gene_nodes_file)
    col = "symbol" if "symbol" in df.columns else ("gene" if "gene" in df.columns else df.columns[0])
    genes = df[col].astype(str).tolist()

    kept, dropped = [], 0
    for g in genes:
        if g in sym2entrez:
            kept.append(g)
        else:
            dropped += 1
    if dropped:
        print(f"[gene vocab] {dropped} genes without an Entrez mapping dropped "
              f"(kept {len(kept)}/{len(genes)})")
    sym2idx = {g: i for i, g in enumerate(kept)}
    entrez2idx = {sym2entrez[g]: i for g, i in sym2idx.items()}
    return kept, sym2idx, entrez2idx


def _shared_target_pairs(name_gene_pairs):
    """Metapath among {miRNA|TF}: connect two entities that share a target gene.
    Input: list of (name, gene_idx). Output: sorted unique (name_a, name_b) set.
    Mirrors MOGNN-TF's A @ A.T co-target metapath (get_tf_inner_connection /
    compute_mirna_metapath)."""
    from itertools import combinations
    by_gene = collections.defaultdict(set)
    for name, g in name_gene_pairs:
        by_gene[g].add(name)
    pairs = set()
    for names in by_gene.values():
        for a, b in combinations(sorted(names), 2):
            pairs.add((a, b))
    return pairs


def load_molecular_edges(sym2idx, mirna_keep=None, tf_keep=None, metapath=False):
    """gene<->gene, miRNA->gene, TF->gene edges restricted to the gene vocab, plus
    (optional) miRNA-miRNA and TF-TF co-target metapaths.

    mirna_keep / tf_keep: optional sets of miRNA / TF names to keep (feature-selected
    panels), so those node types match the selected omics.

    metapath: if True, add miRNA-miRNA (from metapath_mirna_mirna.csv, shared ANY
    target gene, filtered to the panel) and TF-TF (computed as shared SELECTED-gene
    target). A miRNA/TF node then survives if it has ANY edge (gene OR metapath),
    which recovers panel entities not directly linked to a selected gene.

    Returns (edges, vocabs). edges may include 'mirna_mirna' / 'tf_tf'.
    """
    # gene <-> gene (undirected, symbols u,v)
    gg = pd.read_csv(GGI_EDGES)
    gg = gg[gg["u"].isin(sym2idx) & gg["v"].isin(sym2idx)]
    gene_gene = np.array([[sym2idx[u], sym2idx[v]] for u, v in zip(gg["u"], gg["v"])], dtype=np.int64)
    print(f"[molecular] gene-gene edges: {len(gene_gene)}")

    # --- miRNA: gene edges (+ optional metapath) ---------------------------
    mg = pd.read_csv(MIRNA_GENE)
    mg = mg[mg["symbol"].isin(sym2idx)]
    if mirna_keep is not None:
        mg = mg[mg["miRNA"].isin(mirna_keep)]
    mirna_gene_names = [(str(m), sym2idx[s]) for m, s in zip(mg["miRNA"], mg["symbol"])]
    mirna_candidates = set(mirna_keep) if mirna_keep is not None else {m for m, _ in mirna_gene_names}

    mirna_mirna_names = set()
    if metapath:
        mm = pd.read_csv(METAPATH_MIRNA)
        mm = mm[mm["miRNA_1"].isin(mirna_candidates) & mm["miRNA_2"].isin(mirna_candidates)]
        mirna_mirna_names = {(str(a), str(b)) for a, b in zip(mm["miRNA_1"], mm["miRNA_2"]) if a != b}
        print(f"[molecular] miRNA-miRNA metapath edges: {len(mirna_mirna_names)}")

    # vocab = candidates with ANY edge (gene target or metapath)
    seen = {m for m, _ in mirna_gene_names}
    for a, b in mirna_mirna_names:
        seen.add(a); seen.add(b)
    mirnas = sorted(seen)
    mirna2idx = {m: i for i, m in enumerate(mirnas)}
    mirna_gene = np.array([[mirna2idx[m], g] for m, g in mirna_gene_names], dtype=np.int64).reshape(-1, 2)
    mirna_mirna = np.array([[mirna2idx[a], mirna2idx[b]] for a, b in mirna_mirna_names],
                           dtype=np.int64).reshape(-1, 2)
    print(f"[molecular] miRNA nodes: {len(mirnas)} | miRNA-gene edges: {len(mirna_gene)}")

    # --- TF: gene edges (+ optional metapath) ------------------------------
    tf = pd.read_csv(TF_EDGES, sep="\t")
    tf = tf[tf["HGNC.Target"].isin(sym2idx)]
    if tf_keep is not None:
        tf = tf[tf["HGNC.TF"].isin(tf_keep)]
    tf_gene_names = [(str(t), sym2idx[s]) for t, s in zip(tf["HGNC.TF"], tf["HGNC.Target"])]

    tf_tf_names = set()
    if metapath:
        tf_tf_names = _shared_target_pairs(tf_gene_names)  # shared selected-gene target
        print(f"[molecular] TF-TF metapath edges: {len(tf_tf_names)}")

    seen = {t for t, _ in tf_gene_names}
    for a, b in tf_tf_names:
        seen.add(a); seen.add(b)
    tfs = sorted(seen)
    tf2idx = {t: i for i, t in enumerate(tfs)}
    tf_gene = np.array([[tf2idx[t], g] for t, g in tf_gene_names], dtype=np.int64).reshape(-1, 2)
    tf_tf = np.array([[tf2idx[a], tf2idx[b]] for a, b in tf_tf_names], dtype=np.int64).reshape(-1, 2)
    print(f"[molecular] TF nodes: {len(tfs)} | TF-gene edges: {len(tf_gene)}")

    vocabs = {"miRNA": mirnas, "TF": tfs}
    edges = {"gene_gene": gene_gene, "mirna_gene": mirna_gene, "tf_gene": tf_gene}
    if metapath:
        edges["mirna_mirna"] = mirna_mirna
        edges["tf_tf"] = tf_tf
    return edges, vocabs


# ---------------------------------------------------------------------------
# 2. KG index (streaming) : uri -> class_code, and uri -> entity_id (Entrez etc.)
# ---------------------------------------------------------------------------
def _open_json_in_zip(zip_path):
    zf = zipfile.ZipFile(zip_path)
    name = [n for n in zf.namelist() if n.endswith(".json")][0]
    return zf.open(name)


def build_node_index(pkt_nodes=PKT_NODES):
    """Stream nodes.json -> uri2cc (only KEEP_CC classes) + gene uri->entrez."""
    print("[KG] indexing nodes (streaming)...")
    uri2cc = {}
    uri2eid = {}
    n = 0
    with _open_json_in_zip(pkt_nodes) as f:
        for node in ijson.items(f, "item"):
            cc = node.get("class_code")
            if cc in KEEP_CC:
                uri = node["uri"]
                uri2cc[uri] = cc
                uri2eid[uri] = node.get("entity_id", "")
            n += 1
    print(f"[KG] {n:,} nodes scanned | {len(uri2cc):,} kept "
          f"({', '.join(sorted(KEEP_CC))})")
    return uri2cc, uri2eid


# ---------------------------------------------------------------------------
# 3. KG edges (single streaming pass) : accumulate the relations we need
# ---------------------------------------------------------------------------
def extract_kg_edges(uri2cc, uri2eid, entrez2idx, pkt_edges=PKT_EDGES):
    """One pass over edges.json. Returns raw (still id-keyed) relation buffers.

    gene->pathway and gene->disease are resolved to gene index immediately.
    gene->GO needs the protein bridge, so PR->GO edges are buffered and resolved
    at the end.
    """
    print("[KG] extracting edges (single streaming pass)...")

    pr2entrez = {}                       # protein uri -> Entrez (bridge)
    gene_pathway = []                    # (gene_idx, pathway_id)
    gene_disease = []                    # (gene_idx, disease_id)
    pr_go = []                           # (pr_uri, go_id)   -> bridged later
    go_go = set()                        # (go_id_src, go_id_dst)

    n = 0
    for e in ijson.items(_open_json_in_zip(pkt_edges), "item"):
        n += 1
        if n % 2_000_000 == 0:
            print(f"    ...{n:,} edges")

        s_uri, t_uri = e["source_uri"], e["target_uri"]
        s_cc = uri2cc.get(s_uri)
        t_cc = uri2cc.get(t_uri)
        if s_cc is None or t_cc is None:
            continue  # at least one endpoint is a class we don't care about
        p = e.get("predicate_label", "")

        # gene <-> protein bridge (both directions)
        if s_cc == GENE_CC and t_cc == PROTEIN_CC and p in PRED_HAS_PRODUCT:
            pr2entrez[t_uri] = uri2eid[s_uri]
            continue
        if s_cc == PROTEIN_CC and t_cc == GENE_CC and p in PRED_PRODUCT_OF:
            pr2entrez[s_uri] = uri2eid[t_uri]
            continue

        # gene -> pathway
        if s_cc == GENE_CC and t_cc == PATHWAY_CC and p in PRED_MEMBER_OF:
            gi = entrez2idx.get(uri2eid[s_uri])
            if gi is not None:
                gene_pathway.append((gi, uri2eid[t_uri]))
            continue

        # gene -> disease
        if s_cc == GENE_CC and t_cc == DISEASE_CC and p in PRED_ASSOCIATED:
            gi = entrez2idx.get(uri2eid[s_uri])
            if gi is not None:
                gene_disease.append((gi, uri2eid[t_uri]))
            continue

        # protein -> GO  (bridged to gene after the pass)
        if s_cc == PROTEIN_CC and t_cc == GO_CC and p in PRED_ANNOTATED:
            pr_go.append((s_uri, uri2eid[t_uri]))
            continue

        # GO -> GO hierarchy (is_a via rdf:type on same class, or part_of)
        if s_cc == GO_CC and t_cc == GO_CC and p in PRED_GO_HIER:
            src, dst = uri2eid[s_uri], uri2eid[t_uri]
            if src != dst:
                go_go.add((src, dst))
            continue

    print(f"[KG] {n:,} edges scanned")
    print(f"     protein->gene bridge entries : {len(pr2entrez):,}")
    print(f"     gene->pathway (raw)          : {len(gene_pathway):,}")
    print(f"     gene->disease (raw)          : {len(gene_disease):,}")
    print(f"     protein->GO (to bridge)      : {len(pr_go):,}")
    print(f"     GO->GO hierarchy (raw)       : {len(go_go):,}")

    # bridge protein->GO into gene->GO (one gene may back several proteins)
    gene_go = []
    unbridged = 0
    for pr_uri, go_id in pr_go:
        entrez = pr2entrez.get(pr_uri)
        if entrez is None:
            unbridged += 1
            continue
        gi = entrez2idx.get(entrez)
        if gi is not None:
            gene_go.append((gi, go_id))
    print(f"     gene->GO after bridge        : {len(gene_go):,} "
          f"({unbridged:,} protein->GO dropped: no bridge)")

    return {
        "gene_pathway": gene_pathway,
        "gene_disease": gene_disease,
        "gene_go": gene_go,
        "go_go": go_go,
    }


# ---------------------------------------------------------------------------
# 4. Contain the superior scales + remap to contiguous per-type indices
# ---------------------------------------------------------------------------
def finalize_scale(gene_scale_pairs, go_min_support=1):
    """gene->X pairs (gene_idx, X_id) -> (edges int64, X vocab list, X_id->idx).

    Only X nodes actually connected to a kept gene survive. `go_min_support`
    drops X nodes annotated by fewer than N genes (anti blow-up, proposta 7-8).
    """
    support = collections.Counter(x for _, x in gene_scale_pairs)
    keep_x = {x for x, c in support.items() if c >= go_min_support}
    vocab = sorted(keep_x)
    x2idx = {x: i for i, x in enumerate(vocab)}
    edges = np.array(
        [[gi, x2idx[x]] for gi, x in gene_scale_pairs if x in keep_x],
        dtype=np.int64,
    ).reshape(-1, 2)
    return edges, vocab, x2idx


def finalize_hierarchy(hier_pairs, x2idx):
    """X->X hierarchy pairs restricted to nodes surviving in x2idx."""
    edges = np.array(
        [[x2idx[s], x2idx[d]] for s, d in hier_pairs if s in x2idx and d in x2idx],
        dtype=np.int64,
    ).reshape(-1, 2)
    return edges


# ---------------------------------------------------------------------------
# 5. Assemble HeteroData + persist
# ---------------------------------------------------------------------------
def build(gene_nodes_file=GGI_NODES, with_disease=True, go_min_support=1,
          make_undirected=True, mirna_keep=None, tf_keep=None, metapath=False,
          out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    # --- gene anchor + molecular layer -------------------------------------
    sym2entrez = load_symbol_entrez_map()
    genes, sym2idx, entrez2idx = load_gene_vocab(gene_nodes_file, sym2entrez)
    mol_edges, mol_vocabs = load_molecular_edges(
        sym2idx, mirna_keep=mirna_keep, tf_keep=tf_keep, metapath=metapath)

    # --- KG superior scales -------------------------------------------------
    uri2cc, uri2eid = build_node_index()
    kg = extract_kg_edges(uri2cc, uri2eid, entrez2idx)

    gene_pathway, pathway_vocab, _ = finalize_scale(kg["gene_pathway"])
    gene_go, go_vocab, go2idx = finalize_scale(kg["gene_go"], go_min_support)
    go_go = finalize_hierarchy(kg["go_go"], go2idx)
    if with_disease:
        gene_disease, disease_vocab, _ = finalize_scale(kg["gene_disease"])
    else:
        gene_disease, disease_vocab = np.empty((0, 2), np.int64), []

    print("\n=== BACKBONE SUMMARY ===")
    print(f"  gene    : {len(genes):>6} nodes")
    print(f"  miRNA   : {len(mol_vocabs['miRNA']):>6} nodes"
          + (f" | miRNA-miRNA {len(mol_edges['mirna_mirna'])}" if metapath else ""))
    print(f"  TF      : {len(mol_vocabs['TF']):>6} nodes"
          + (f" | TF-TF {len(mol_edges['tf_tf'])}" if metapath else ""))
    print(f"  pathway : {len(pathway_vocab):>6} nodes | gene->pathway {len(gene_pathway)}")
    print(f"  GO_term : {len(go_vocab):>6} nodes | gene->GO {len(gene_go)} | GO->GO {len(go_go)}")
    if with_disease:
        print(f"  disease : {len(disease_vocab):>6} nodes | gene->disease {len(gene_disease)}")

    # --- persist human-inspectable vocabs + edge lists ---------------------
    _save_vocabs(out_dir, genes, mol_vocabs, pathway_vocab, go_vocab, disease_vocab)
    _save_edge_lists(
        out_dir, genes, mol_vocabs, pathway_vocab, go_vocab, disease_vocab,
        mol_edges=mol_edges, gene_pathway=gene_pathway, gene_go=gene_go,
        go_go=go_go, gene_disease=gene_disease, with_disease=with_disease,
    )

    # --- assemble HeteroData (topology only) -------------------------------
    hetero = _assemble_heterodata(
        n_gene=len(genes), n_mirna=len(mol_vocabs["miRNA"]), n_tf=len(mol_vocabs["TF"]),
        n_pathway=len(pathway_vocab), n_go=len(go_vocab), n_disease=len(disease_vocab),
        mol_edges=mol_edges, gene_pathway=gene_pathway, gene_go=gene_go,
        go_go=go_go, gene_disease=gene_disease, with_disease=with_disease,
        make_undirected=make_undirected,
    )
    if hetero is not None:
        import torch
        out_pt = os.path.join(out_dir, "hetero_graph_template.pt")
        torch.save(hetero, out_pt)
        print(f"\n[saved] HeteroData template -> {out_pt}")
        print(hetero)
    return hetero


def _save_vocabs(out_dir, genes, mol_vocabs, pathway_vocab, go_vocab, disease_vocab):
    pd.DataFrame({"symbol": genes, "idx": range(len(genes))}).to_csv(
        os.path.join(out_dir, "node_gene.csv"), index=False)
    pd.DataFrame({"miRNA": mol_vocabs["miRNA"], "idx": range(len(mol_vocabs["miRNA"]))}).to_csv(
        os.path.join(out_dir, "node_miRNA.csv"), index=False)
    pd.DataFrame({"TF": mol_vocabs["TF"], "idx": range(len(mol_vocabs["TF"]))}).to_csv(
        os.path.join(out_dir, "node_TF.csv"), index=False)
    pd.DataFrame({"reactome_id": pathway_vocab, "idx": range(len(pathway_vocab))}).to_csv(
        os.path.join(out_dir, "node_pathway.csv"), index=False)
    pd.DataFrame({"go_id": go_vocab, "idx": range(len(go_vocab))}).to_csv(
        os.path.join(out_dir, "node_GO_term.csv"), index=False)
    if disease_vocab:
        pd.DataFrame({"mondo_id": disease_vocab, "idx": range(len(disease_vocab))}).to_csv(
            os.path.join(out_dir, "node_disease.csv"), index=False)
    print(f"[saved] node vocabularies -> {out_dir}node_*.csv")


def _save_edge_lists(out_dir, genes, mol_vocabs, pathway_vocab, go_vocab,
                     disease_vocab, mol_edges, gene_pathway, gene_go, go_go,
                     gene_disease, with_disease):
    """Write one human-readable CSV per relation (idx + resolved id) so the
    graph is fully analyzable without loading the .pt. Directional edges only;
    the reverse relations added by ToUndirected are just mirrors."""

    def dump(fname, edges, src_vocab, src_cols, dst_vocab, dst_cols):
        # edges: (N,2) int index pairs; *_cols = (idx_colname, id_colname)
        edges = np.asarray(edges).reshape(-1, 2)
        src_idx, dst_idx = edges[:, 0], edges[:, 1]
        df = pd.DataFrame({
            src_cols[0]: src_idx,
            src_cols[1]: [src_vocab[i] for i in src_idx],
            dst_cols[0]: dst_idx,
            dst_cols[1]: [dst_vocab[i] for i in dst_idx],
        })
        df.to_csv(os.path.join(out_dir, fname), index=False)
        return len(df)

    counts = {}
    counts["interacts"] = dump(
        "edge_gene_interacts_gene.csv", mol_edges["gene_gene"],
        genes, ("gene_u_idx", "gene_u"), genes, ("gene_v_idx", "gene_v"))
    counts["targets"] = dump(
        "edge_miRNA_targets_gene.csv", mol_edges["mirna_gene"],
        mol_vocabs["miRNA"], ("miRNA_idx", "miRNA"), genes, ("gene_idx", "gene"))
    counts["regulates"] = dump(
        "edge_TF_regulates_gene.csv", mol_edges["tf_gene"],
        mol_vocabs["TF"], ("TF_idx", "TF"), genes, ("gene_idx", "gene"))
    counts["member_of"] = dump(
        "edge_gene_member_of_pathway.csv", gene_pathway,
        genes, ("gene_idx", "gene"), pathway_vocab, ("pathway_idx", "reactome_id"))
    counts["annotated_with"] = dump(
        "edge_gene_annotated_with_GO_term.csv", gene_go,
        genes, ("gene_idx", "gene"), go_vocab, ("GO_idx", "go_id"))
    counts["is_a"] = dump(
        "edge_GO_term_is_a_GO_term.csv", go_go,
        go_vocab, ("GO_src_idx", "go_src"), go_vocab, ("GO_dst_idx", "go_dst"))
    if with_disease:
        counts["associated_with"] = dump(
            "edge_gene_associated_with_disease.csv", gene_disease,
            genes, ("gene_idx", "gene"), disease_vocab, ("disease_idx", "mondo_id"))
    if "mirna_mirna" in mol_edges:
        mirna_v, tf_v = mol_vocabs["miRNA"], mol_vocabs["TF"]
        counts["miRNA_shares_target"] = dump(
            "edge_miRNA_shares_target_miRNA.csv", mol_edges["mirna_mirna"],
            mirna_v, ("miRNA_a_idx", "miRNA_a"), mirna_v, ("miRNA_b_idx", "miRNA_b"))
        counts["TF_shares_target"] = dump(
            "edge_TF_shares_target_TF.csv", mol_edges["tf_tf"],
            tf_v, ("TF_a_idx", "TF_a"), tf_v, ("TF_b_idx", "TF_b"))
    print(f"[saved] edge lists       -> {out_dir}edge_*.csv")

    # manifest: node + edge counts in one place, for quick inspection
    node_counts = {
        "gene": len(genes), "miRNA": len(mol_vocabs["miRNA"]), "TF": len(mol_vocabs["TF"]),
        "pathway": len(pathway_vocab), "GO_term": len(go_vocab),
    }
    if with_disease:
        node_counts["disease"] = len(disease_vocab)
    rows = [{"kind": "node", "name": k, "count": v} for k, v in node_counts.items()]
    rows += [{"kind": "edge", "name": k, "count": v} for k, v in counts.items()]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "graph_manifest.csv"), index=False)
    print(f"[saved] manifest         -> {out_dir}graph_manifest.csv")


def _assemble_heterodata(n_gene, n_mirna, n_tf, n_pathway, n_go, n_disease,
                         mol_edges, gene_pathway, gene_go, go_go, gene_disease,
                         with_disease, make_undirected):
    try:
        import torch
        from torch_geometric.data import HeteroData
        import torch_geometric.transforms as T
    except Exception as exc:  # keep the CSV/edge-list output usable without PyG
        print(f"\n[warn] torch_geometric unavailable ({exc}); "
              f"skipped HeteroData assembly (vocabs/edge lists still written).")
        return None

    def ei(arr):  # (N,2) int64 -> [2,N] tensor
        return torch.as_tensor(arr.T if len(arr) else np.zeros((2, 0)), dtype=torch.long)

    data = HeteroData()
    # featureless template: only node counts; per-patient features injected later
    data["gene"].num_nodes = n_gene
    data["miRNA"].num_nodes = n_mirna
    data["TF"].num_nodes = n_tf
    data["pathway"].num_nodes = n_pathway
    data["GO_term"].num_nodes = n_go
    if with_disease:
        data["disease"].num_nodes = n_disease

    data["gene", "interacts", "gene"].edge_index = ei(mol_edges["gene_gene"])
    data["miRNA", "targets", "gene"].edge_index = ei(mol_edges["mirna_gene"])
    data["TF", "regulates", "gene"].edge_index = ei(mol_edges["tf_gene"])
    data["gene", "member_of", "pathway"].edge_index = ei(gene_pathway)
    data["gene", "annotated_with", "GO_term"].edge_index = ei(gene_go)
    data["GO_term", "is_a", "GO_term"].edge_index = ei(go_go)
    if with_disease:
        data["gene", "associated_with", "disease"].edge_index = ei(gene_disease)
    # optional co-target metapaths (MOGNN-TF molecular layer)
    if "mirna_mirna" in mol_edges:
        data["miRNA", "shares_target", "miRNA"].edge_index = ei(mol_edges["mirna_mirna"])
        data["TF", "shares_target", "TF"].edge_index = ei(mol_edges["tf_tf"])

    if make_undirected:
        # adds typed reverse relations (e.g. pathway__rev_member_of__gene) so the
        # message passing is bidirectional where it matters. gene<->gene stays as
        # is (already symmetric); the GO hierarchy also gets a reverse copy.
        data = T.ToUndirected()(data)
    return data


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the hetero multi-scale backbone (HeteroData).")
    ap.add_argument("--gene-list", default=GGI_NODES,
                    help="Gene node list (col 'symbol' or 'gene'). Default: full BioGRID vocab. "
                         "Pass a feature-selected list for the small full-batch graph.")
    ap.add_argument("--go-min-support", type=int, default=1,
                    help="Drop GO terms annotated by fewer than N genes (anti blow-up).")
    ap.add_argument("--no-disease", action="store_true", help="Exclude the gene->disease scale.")
    ap.add_argument("--directed", action="store_true",
                    help="Keep directional edges (skip ToUndirected).")
    ap.add_argument("--mirna-list", default=None,
                    help="File with the selected miRNA panel (one name per line, or a "
                         "TSV whose header row lists them). Restricts miRNA nodes to it "
                         "so the template matches the omics (avoids zero-featured miRNA).")
    ap.add_argument("--tf-list", default=None,
                    help="File with the selected TF panel (col 'TF'/'symbol', or one per "
                         "line). Restricts TF nodes to it (unified feature selection).")
    ap.add_argument("--metapath", action="store_true",
                    help="Add miRNA-miRNA and TF-TF co-target metapaths (MOGNN-TF molecular "
                         "layer). Lets panel miRNAs/TFs survive via metapath, not only via a "
                         "direct gene edge. Use with --mirna-list/--tf-list.")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--force", action="store_true", help="Rebuild even if output exists.")
    args = ap.parse_args()

    def _read_names(path, drop_first_col=False):
        """Parse a panel file: plain one-per-line, or CSV/TSV (uses a named
        'TF'/'symbol'/'miRNA' column if present, else the header columns)."""
        first = open(path).readline().strip()
        sep = "\t" if "\t" in first else ("," if "," in first else None)
        if sep is None:
            return {l.strip() for l in open(path) if l.strip()}
        df = pd.read_csv(path, sep=sep)
        for col in ("TF", "symbol", "gene", "miRNA"):
            if col in df.columns:
                return set(df[col].astype(str).str.strip())
        # otherwise treat the header itself as the list of names
        return {c for c in df.columns if c.lower() not in ("sample", "sample_id")}

    mirna_keep = _read_names(args.mirna_list) if args.mirna_list else None
    tf_keep = _read_names(args.tf_list) if args.tf_list else None
    if mirna_keep is not None:
        print(f"[cli] miRNA panel: {len(mirna_keep)} names from {args.mirna_list}")
    if tf_keep is not None:
        print(f"[cli] TF panel: {len(tf_keep)} names from {args.tf_list}")

    out_pt = os.path.join(args.out_dir, "hetero_graph_template.pt")
    if os.path.exists(out_pt) and not args.force:
        print(f"Hetero graph already exists ({out_pt}), skipping. Use --force to rebuild.")
    else:
        build(
            gene_nodes_file=args.gene_list,
            with_disease=not args.no_disease,
            go_min_support=args.go_min_support,
            make_undirected=not args.directed,
            mirna_keep=mirna_keep,
            tf_keep=tf_keep,
            metapath=args.metapath,
            out_dir=args.out_dir,
        )
