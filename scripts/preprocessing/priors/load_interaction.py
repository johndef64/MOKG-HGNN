#%%
# https://string-db.org/cgi/download?sessionId=b9LogbMlJZqq&species_text=Homo+sapiens&settings_expanded=0&min_download_score=0&filter_redundant_pairs=0&delimiter_type=txt
# https://downloads.thebiogrid.org/BioGRID/Release-Archive/BIOGRID-5.0.250/ (ORGANISM)
# https://mirdb.org/download.html
# Check networkX for graphs

# Load interaction data
import pandas as pd
import os
import numpy as np
from scipy.sparse import coo_matrix, save_npz, load_npz, eye

# All output paths are relative to the repository root (the cwd when invoked
# via the Makefile / scripts/README.md). The HGNC conversion table ships with
# this script folder, so we resolve it relative to __file__.
_HERE = os.path.dirname(os.path.abspath(__file__))

GGI_Interaction = "data/prior_knowledge/GGI/GGI.zip"
mirna_target_file = "data/prior_knowledge/miRNA/miRDB.gz"
conversion_table_gene = os.path.join(_HERE, "hgcn_mart_conversion_table.zip")
outputfolder_ggi = "data/prior_knowledge/GGI/"
outputfolder_mirna = "data/prior_knowledge/miRNA/"
refseq2gene_file = "data/prior_knowledge/refseq2gene_mappings.tsv"


def load_ggi_interactions(GGI_Interaction, outputfolder_ggi, conversion_table_gene):
    interaction_df = pd.read_csv(GGI_Interaction, sep="\t", compression='zip')

    # drop "-" rows in interaction_df, "-" are associated with entrez IDs in sarsCov2 and HPV studies
    interaction_df = interaction_df[interaction_df["Entrez Gene Interactor A"] != "-"]
    interaction_df = interaction_df[interaction_df["Entrez Gene Interactor B"] != "-"]
    # convertion int64 of entrez columns
    interaction_df["Entrez Gene Interactor A"] = pd.to_numeric(interaction_df["Entrez Gene Interactor A"], errors="coerce").astype("Int64")
    interaction_df["Entrez Gene Interactor B"] = pd.to_numeric(interaction_df["Entrez Gene Interactor B"], errors="coerce").astype("Int64")
    print("total interactions:", len(interaction_df))

    # Get unique genes in interaction_df
    unique_genes = pd.unique(interaction_df[["Entrez Gene Interactor A", "Entrez Gene Interactor B"]].values.ravel("K"))
    print("total unique genes in interactions:", len(unique_genes))

    conversion_table = pd.read_csv(conversion_table_gene, sep="\t", compression='zip')
    conversion_table["NCBI gene ID"] = pd.to_numeric(conversion_table["NCBI gene ID"], errors="coerce").astype("Int64")
    print("total genes in conversion table:", len(conversion_table))

    # filter interaction_df to keep only genes that are in conversion table
    print(interaction_df.columns.values)
    interaction_df = interaction_df[interaction_df["Entrez Gene Interactor A"].isin(conversion_table["NCBI gene ID"])]
    interaction_df = interaction_df[interaction_df["Entrez Gene Interactor B"].isin(conversion_table["NCBI gene ID"])]
    print("interactions after filtering:", len(interaction_df))

    # Add new columns for the Ensembl ids from conversion_table and hgcn symbol
    interaction_df = interaction_df.merge(conversion_table[["NCBI gene ID", "Ensembl gene ID", "Approved symbol"]], left_on="Entrez Gene Interactor A", right_on="NCBI gene ID", how="left")
    interaction_df = interaction_df.rename(columns={"Ensembl gene ID": "Ensembl Interactor A", "Approved symbol": "HGCN Symbol Interactor A"})
    interaction_df = interaction_df.drop(columns=["NCBI gene ID"])

    interaction_df = interaction_df.merge(conversion_table[["NCBI gene ID", "Ensembl gene ID", "Approved symbol"]], left_on="Entrez Gene Interactor B", right_on="NCBI gene ID", how="left")
    interaction_df = interaction_df.rename(columns={"Ensembl gene ID": "Ensembl Interactor B", "Approved symbol": "HGCN Symbol Interactor B"})
    interaction_df = interaction_df.drop(columns=["NCBI gene ID"])

    # print number of nan values for each column that are more than 0
    print(interaction_df.isna().sum()[interaction_df.isna().sum() > 0])
    # drop useless columns
    interaction_final = interaction_df[["#BioGRID Interaction ID","Entrez Gene Interactor A", "Entrez Gene Interactor B", "Ensembl Interactor A", "HGCN Symbol Interactor A", "Ensembl Interactor B", "HGCN Symbol Interactor B", "BioGRID ID Interactor A", "BioGRID ID Interactor B"]]
    # export
    interaction_final.to_csv(os.path.join(outputfolder_ggi, "GGI_interactions_processed.csv"), index=False)
    # removing self-loops
    interaction_df_2 = interaction_final[
        interaction_final["HGCN Symbol Interactor A"] != interaction_final["HGCN Symbol Interactor B"]
    ]

    edges = interaction_df_2[["HGCN Symbol Interactor A", "HGCN Symbol Interactor B"]].copy()

    # Normalizza ogni coppia: (min, max) per renderla non orientata
    edges["u"] = edges[["HGCN Symbol Interactor A", "HGCN Symbol Interactor B"]].min(axis=1)
    edges["v"] = edges[["HGCN Symbol Interactor A", "HGCN Symbol Interactor B"]].max(axis=1)

    # Togli duplicati
    edges_undirected = edges[["u", "v"]].dropna().drop_duplicates()
    # print len of edges_undirected
    print("Total undirected edges after removing self-loops and duplicates:", len(edges_undirected))

    # Questo è il tuo edge list finale (919833)
    edges_undirected.to_csv(os.path.join(outputfolder_ggi, "GGI_edges_undirected.csv"), index=False)

    # Extract gene list
    genes = pd.unique(edges_undirected[["u", "v"]].values.ravel("K"))
    genes = pd.Series(genes).sort_values().reset_index(drop=True)
    genes.to_csv(os.path.join(outputfolder_ggi, "GGI_nodes_list.csv"), index=False, header=["symbol"])
    print("Total unique genes in final network:", len(genes))
    # create adjacency matrix

    # mapping gene -> indice di nodo
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    # converto gli endpoint degli archi in indici numerici
    row = edges_undirected["u"].map(gene_to_idx).to_numpy()
    col = edges_undirected["v"].map(gene_to_idx).to_numpy()

    # siccome il grafo è non orientato, aggiungo entrambi i versi (u,v) e (v,u)
    row_idx = np.concatenate([row, col])
    col_idx = np.concatenate([col, row])

    data = np.ones(len(row_idx), dtype=np.float32)
    # print shape della matrice

    n = len(genes)
    A_sparse = coo_matrix((data, (row_idx, col_idx)), shape=(n, n))
    A_sparse.setdiag(0)                  # per sicurezza
    A_sparse = A_sparse + eye(n)      # aggiungi self-loops
    print(f"Adjacency matrix shape: {A_sparse.shape}")
    print(f"Number of edges in adjacency matrix: {A_sparse.nnz // 2}")  # diviso 2 perché ogni arco è contato due volte


    save_npz(os.path.join(outputfolder_ggi, "GGI_adjacency_sparse.npz"), A_sparse)
    return True
# check indici
def check_ggi_indices(outputfolder_ggi):
    print("Checking GGI indices...")
    # Carica la matrice e la lista geni
    A = load_npz(os.path.join(outputfolder_ggi, "GGI_adjacency_sparse.npz"))
    genes = pd.read_csv(os.path.join(outputfolder_ggi, "GGI_nodes_list.csv"))["symbol"]
    # Converti in formato COO per leggere facilmente le coordinate
    A_coo = A.tocoo()

    # Prendi le prime 5 connessioni e verifica
    for i in range(10):
        u_idx, v_idx = A_coo.row[i], A_coo.col[i]
        print(f"{u_idx} ({genes.iloc[u_idx]})  -  {v_idx} ({genes.iloc[v_idx]})")

def load_mirna_gene_interactions(mirna_target_file, refseq2gene_file, conversion_table_gene, outputfolder_mirna):
    print("Loading miRNA-gene interactions...")
    # Load miRNA target data
    mirna_target_df = pd.read_csv(mirna_target_file, sep="\t", compression="gzip")
    refseq2gene_df = pd.read_csv(refseq2gene_file, sep="\t")
    # Add header columns
    mirna_target_df.columns = ["miRNA", "Target", "Score"]
    # Filter huuman miRNA-target interactions
    human_targets = mirna_target_df[mirna_target_df["miRNA"].str.startswith("hsa-")]
    print("Number of duplicate interactions in original data:", human_targets.duplicated().sum())
    # Merge with refseq2gene_df to get gene symbols
    merged_df = human_targets.merge(refseq2gene_df, left_on="Target", right_on="query", how="left")
    merged_df = merged_df[merged_df["symbol"].notna()]
    merged_df['entrezgene'] = merged_df['entrezgene'].astype('Int64')

    mirna_gene_df = merged_df[["miRNA", "symbol", "entrezgene", "Score"]]
    print("Number of duplicate interactions after merging with gene symbols:", mirna_gene_df.duplicated().sum())
    print("Number of interactions after merging with gene symbols:", len(mirna_gene_df))
    # Drop duplicates
    mirna_gene_df = mirna_gene_df.drop_duplicates()
    print("Number of duplicate interactions after dropping duplicates:", mirna_gene_df.duplicated().sum())
    print("Number of interactions after dropping duplicates:", len(mirna_gene_df))
    # Optional: filter out scores below a threshold
    print("Filtering interactions by score...")
    print("Head:", mirna_gene_df.head(5))
    threshold = 90
    mirna_gene_filtered = mirna_gene_df[mirna_gene_df["Score"] >= threshold]
    print("Number of interactions after score filtering:", len(mirna_gene_filtered))
    print("Number of duplicate interactions after score filtering:", mirna_gene_filtered.duplicated().sum())
    print("Head after filtering:", mirna_gene_filtered.head(5))
    # Drop scores column
    mirna_gene_filtered = mirna_gene_filtered.drop(columns=["Score"])

    # Map unique Entrez gene IDs to conversion table
    conversion_table = pd.read_csv(conversion_table_gene, sep="\t", compression='zip')
    conversion_table["NCBI gene ID"] = pd.to_numeric(conversion_table["NCBI gene ID"], errors="coerce").astype("Int64")
    conversion_table = conversion_table.drop_duplicates(subset=["NCBI gene ID"])
    mapped_conversion = conversion_table[conversion_table["NCBI gene ID"].isin(mirna_gene_filtered["entrezgene"])]
    print("Number of unique Entrez gene IDs found in conversion table:", len(mapped_conversion))

    # Filter mirna_gene_filtered to keep only genes present in conversion table
    mirna_gene_final = mirna_gene_filtered[mirna_gene_filtered["entrezgene"].isin(conversion_table["NCBI gene ID"])]
    print("Number of duplicate interactions after merging with gene symbols:", mirna_gene_final.duplicated().sum())
    mirna_gene_final = mirna_gene_final.merge(conversion_table[["NCBI gene ID", "Approved symbol"]], left_on="entrezgene", right_on="NCBI gene ID", how="left")
    mirna_gene_final = mirna_gene_final.rename(columns={"Approved symbol": "HGCN Symbol"}).drop(columns=["NCBI gene ID"])
    print("Number of interactions after filtering by conversion table:", len(mirna_gene_final))

    # Drop duplicates
    mirna_gene_final = mirna_gene_final.drop_duplicates()
    print("Number of interactions after dropping duplicates:", len(mirna_gene_final))
    # Export final miRNA-gene interactions
    mirna_gene_final.to_csv(os.path.join(outputfolder_mirna, "miRNA_gene_interactions.csv"), index=False)
    return True
# ---------------------------------
# Calcolo dei metapath miRNA-miRNA

def compute_mirna_metapath(outputfolder_mirna):
    print("Computing miRNA-miRNA metapath...")
    mirna_gene_final_def = pd.read_csv(os.path.join(outputfolder_mirna, "miRNA_gene_interactions.csv"))
    # ---------------------------------
    # Trova i collegamenti tra miRNA associati allo stesso gene
    print("Calcolo dei metapath miRNA-miRNA...")
    # Raggruppa per gene
    mirna_gene_grouped = mirna_gene_final_def.groupby("symbol")

    # Crea una lista di collegamenti miRNA-miRNA
    metapath_mirna = []
    for gene, group in mirna_gene_final_def.groupby("symbol"):
        # miRNA unici per quel gene
        mirnas = sorted(group["miRNA"].unique())
        # tutte le coppie i < j
        for i in range(len(mirnas)):
            for j in range(i + 1, len(mirnas)):
                metapath_mirna.append((mirnas[i], mirnas[j]))
    # Converte in DataFrame
    metapath_mirna_df = pd.DataFrame(metapath_mirna, columns=["miRNA_1", "miRNA_2"])
    # Rompzione self loops
    metapath_mirna_df = metapath_mirna_df[
        metapath_mirna_df["miRNA_1"] != metapath_mirna_df["miRNA_2"]
    ]
    print("Number of metapath before removing self loops:", len(metapath_mirna_df))
    # Rimozione duplicati
    before = len(metapath_mirna_df)
    metapath_mirna_df = metapath_mirna_df.drop_duplicates()
    after = len(metapath_mirna_df)
    print("Removed exact duplicates:", before - after)

    # Grafo non orientato (miRNA_1, miRNA_2) è lo stesso di (miRNA_2, miRNA_1)
    before_undir = len(metapath_mirna_df)
    metapath_mirna_df = pd.DataFrame(
        np.sort(metapath_mirna_df.values, axis=1),
        columns=["miRNA_1", "miRNA_2"]
    ).drop_duplicates()
    after_undir = len(metapath_mirna_df)
    print("Removed undirected duplicates:", before_undir - after_undir)


    # Esporta i metapath in un file CSV
    metapath_mirna_df.to_csv(
        os.path.join(outputfolder_mirna, "metapath_mirna_mirna.csv"),
        index=False
    )
    # Esportazione degli mirna node list
    unique_mirnas = pd.unique(mirna_gene_final_def[["miRNA"]].values.ravel("K"))
    unique_mirnas_df = pd.DataFrame(unique_mirnas, columns=["miRNA"])
    unique_mirnas_df.to_csv(os.path.join(outputfolder_mirna, "miRNA_nodes_list.csv"), index=False)
    
    # Crea la matrice di adiacenza miRNA–miRNA (non orientata, non pesata)
    # mapping miRNA -> indice
    mirna_to_idx = {m: i for i, m in enumerate(unique_mirnas)}

    # converti le coppie in indici
    row = metapath_mirna_df["miRNA_1"].map(mirna_to_idx).to_numpy()
    col = metapath_mirna_df["miRNA_2"].map(mirna_to_idx).to_numpy()

    # aggiungi entrambi i versi (u,v) e (v,u) per grafo non orientato
    row_idx = np.concatenate([row, col])
    col_idx = np.concatenate([col, row])

    data = np.ones(len(row_idx), dtype=np.float32)
    n = len(unique_mirnas)

    A_mirna = coo_matrix((data, (row_idx, col_idx)), shape=(n, n))

    print(f"Adjacency miRNA-miRNA shape: {A_mirna.shape}")
    print(f"Numero di archi (non-direzionati): {A_mirna.nnz // 2}")

    # Salva la matrice di adiacenza
    adj_path = os.path.join(outputfolder_mirna, "miRNA_adjacency_sparse.npz")
    save_npz(adj_path, A_mirna)
    print(f"Matrice di adiacenza miRNA-miRNA salvata in: {adj_path}")

    return True

def compute_gene_mirna_adjacency(
    outputfolder_mirna,
    gene_nodes_file,
    mirna_nodes_file,
    gene_col="symbol",
    mirna_col="miRNA",
    adjacency_filename="gene_mirna_adjacency_sparse.npz",
    filter_by_variance=False,
):

    print("=== Computing GENE miRNA adjacency matrix ===")
    # 1. Carica interazioni miRNA–gene
    mirna_gene_path = os.path.join(outputfolder_mirna, "miRNA_gene_interactions.csv")
    if not os.path.exists(mirna_gene_path):
        raise FileNotFoundError(f"File not found: {mirna_gene_path}")

    mirna_gene_df = pd.read_csv(mirna_gene_path)
    print(f"Loaded miRNA–gene interactions: {len(mirna_gene_df)} rows")

    # 2. Carica lista geni (ordine = righe della matrice)
    if not os.path.exists(gene_nodes_file):
        raise FileNotFoundError(f"Gene nodes file not found: {gene_nodes_file}")

    gene_nodes_df = pd.read_csv(gene_nodes_file)
    if gene_col not in gene_nodes_df.columns:
        raise ValueError(f"Column '{gene_col}' not found in gene_nodes_file")

    gene_list = gene_nodes_df[gene_col].astype(str).tolist()
    n_genes = len(gene_list)
    print(f"Number of genes in node list: {n_genes}")

    # 3. Carica lista miRNA (ordine = colonne della matrice)
    if not os.path.exists(mirna_nodes_file):
        raise FileNotFoundError(f"miRNA nodes file not found: {mirna_nodes_file}")

    mirna_nodes_df = pd.read_csv(mirna_nodes_file)
    if mirna_col not in mirna_nodes_df.columns:
        raise ValueError(f"Column '{mirna_col}' not found in mirna_nodes_file")

    mirna_list = mirna_nodes_df[mirna_col].astype(str).tolist()
    n_mirna = len(mirna_list)
    print(f"Number of miRNAs in node list: {n_mirna}")

    # 4. Crea mapping nome -> indice
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    mirna_to_idx = {m: j for j, m in enumerate(mirna_list)}

    # 5. Filtra il dataframe di interazioni alle sole coppie gene/miRNA presenti in entrambe le liste
    #    NOTA: le interazioni sono sulle colonne 'miRNA' e 'symbol'
    if "miRNA" not in mirna_gene_df.columns:
        raise ValueError("Column 'miRNA' not found in miRNA_gene_interactions.csv")
    if "symbol" not in mirna_gene_df.columns:
        raise ValueError("Column 'symbol' not found in miRNA_gene_interactions.csv")

    interactions = mirna_gene_df[
        mirna_gene_df["symbol"].isin(gene_to_idx.keys())
        & mirna_gene_df["miRNA"].isin(mirna_to_idx.keys())
    ].copy()

    print(f"Interactions after filtering by gene/miRNA lists: {len(interactions)}")

    if interactions.empty:
        raise ValueError("No overlapping interactions between gene/miRNA lists and miRNA_gene_interactions.csv")

    # 6. Converte in indici (righe = geni, colonne = miRNA)
    row_idx = interactions["symbol"].map(gene_to_idx).to_numpy()
    col_idx = interactions["miRNA"].map(mirna_to_idx).to_numpy()

    # 7. Pesi sugli archi: binari (1.0) perché 'Score' è stato droppato prima del salvataggio
    data = np.ones(len(row_idx), dtype=np.float32)

    # 8. Costruisce la matrice sparse (GENE × miRNA)
    A_gene_mirna = coo_matrix(
        (data, (row_idx, col_idx)),
        shape=(n_genes, n_mirna),
        dtype=np.float32,
    ).tocsr()

    print(f"GENE miRNA adjacency shape: {A_gene_mirna.shape}")
    print(f"Number of non-zero entries (edges): {A_gene_mirna.nnz}")

    # 9. Salva in .npz
    adj_path = os.path.join(outputfolder_mirna, adjacency_filename)
    save_npz(adj_path, A_gene_mirna)
    print(f"Saved GENE - miRNA adjacency matrix to: {adj_path}")

    return True

def load_tf_interactions(tf_target_file, conversion_table_gene, outputfolder_tf):
    """Load and process TF-gene interactions."""
    print("Loading TF-gene interactions...")
    TF_df = pd.read_csv(tf_target_file, sep="\t")
    # ---------- 2) explode target entrez multipli + map to hgnc ----------
    e = TF_df.copy()

    # split TARGET "801;805;808" -> righe multiple
    e["NCBI.GeneID.Target"] = (
        e["NCBI.GeneID.Target"].fillna("")
        .astype(str).str.split(";")
    )
    e = e.explode("NCBI.GeneID.Target")
    e["NCBI.GeneID.Target"] = e["NCBI.GeneID.Target"].astype(str).str.strip()
    e = e[e["NCBI.GeneID.Target"] != ""]  # elimina vuoti
    # ----------- explode TF ------------------
    # split "801;805;808" -> righe multiple
    e["NCBI.GeneID.TF"] = (
        e["NCBI.GeneID.TF"].fillna("")
        .astype(str).str.split(";")
    )
    e = e.explode("NCBI.GeneID.TF")
    e["NCBI.GeneID.TF"] = e["NCBI.GeneID.TF"].astype(str).str.strip()
    e = e[e["NCBI.GeneID.TF"] != ""]  # elimina vuoti

    # drop "-" rows in interaction_df, "-" are associated with entrez IDs in sarsCov2 and HPV studies
    e = e[e["NCBI.GeneID.TF"] != "-"]
    e = e[e["NCBI.GeneID.Target"] != "-"]
    # convertion int64 of entrez columns
    e["NCBI.GeneID.TF"] = pd.to_numeric(e["NCBI.GeneID.TF"], errors="coerce").astype("Int64")
    e["NCBI.GeneID.Target"] = pd.to_numeric(e["NCBI.GeneID.Target"], errors="coerce").astype("Int64")
    print("total interactions:", len(e))

    # drop duplicates row
    e = e.drop_duplicates(subset=["NCBI.GeneID.TF", "NCBI.GeneID.Target"])
    print("total unique interactions:", len(e))
    # Get unique genes in TF_df
    unique_genes = pd.unique(e[["NCBI.GeneID.TF", "NCBI.GeneID.Target"]].values.ravel("K"))
    print("total unique genes in interactions:", len(unique_genes))
    
    conversion_table = pd.read_csv(conversion_table_gene, sep="\t", compression='zip')
    conversion_table["NCBI gene ID"] = pd.to_numeric(conversion_table["NCBI gene ID"], errors="coerce").astype("Int64")
    print("total genes in conversion table:", len(conversion_table))

    # filter interaction_df to keep only genes that are in conversion table
    print(e.columns.values)
    conv_ids = conversion_table["NCBI gene ID"].dropna()

    interaction_df = e[
        e["NCBI.GeneID.TF"].isin(conv_ids) &
        e["NCBI.GeneID.Target"].isin(conv_ids)
    ].copy()

    print("interactions after filtering:", len(interaction_df))

    #interaction_df = e[e["NCBI.GeneID.TF"].isin(conversion_table["NCBI gene ID"])]
    #interaction_df = e[e["NCBI.GeneID.Target"].isin(conversion_table["NCBI gene ID"])]
    print("interactions after filtering:", len(interaction_df))
    
    # Add new columns for the Ensembl ids from conversion_table and hgcn symbol
    interaction_df = interaction_df.merge(conversion_table[["NCBI gene ID", "Ensembl gene ID", "Approved symbol"]], left_on="NCBI.GeneID.TF", right_on="NCBI gene ID", how="left")
    interaction_df = interaction_df.rename(columns={"Ensembl gene ID": "Ensembl Interactor A", "Approved symbol": "HGCN Symbol Interactor A"})
    interaction_df = interaction_df.drop(columns=["NCBI gene ID"])

    interaction_df = interaction_df.merge(conversion_table[["NCBI gene ID", "Ensembl gene ID", "Approved symbol"]], left_on="NCBI.GeneID.Target", right_on="NCBI gene ID", how="left")
    interaction_df = interaction_df.rename(columns={"Ensembl gene ID": "Ensembl Interactor B", "Approved symbol": "HGCN Symbol Interactor B"})
    interaction_df = interaction_df.drop(columns=["NCBI gene ID"])

    # print number of nan values for each column that are more than 0
    print(interaction_df.isna().sum()[interaction_df.isna().sum() > 0])
    
    interaction_final = interaction_df[['NCBI.GeneID.TF', 'HGCN Symbol Interactor A',
                                    'NCBI.GeneID.Target', 'HGCN Symbol Interactor B', 'Name.TF', 'Name.Target']]
    # rename columns
    interaction_final = interaction_final.rename(columns={
        'NCBI.GeneID.TF': 'Entrez.TF',
        'HGCN Symbol Interactor A': 'HGNC.TF',
        'NCBI.GeneID.Target': 'Entrez.Target',
        'HGCN Symbol Interactor B': 'HGNC.Target'
    })
    print("total interactions final:", len(interaction_final))
    # check for genes in both TF and Target
    #export_df = interaction_final[["Name.TF", "Name.Target"]]
    export_df = interaction_final[["HGNC.TF", "HGNC.Target"]]
    export_df.to_csv(os.path.join(outputfolder_tf, "TF_interactions.tsv"), sep="\t", index=False)
    print("Number of unique TF", len(export_df["HGNC.TF"].unique()))
    print("Number of unique Target", len(export_df["HGNC.Target"].unique()))

    return True


if __name__ == "__main__":
    # GGI
    # check if files exist, if yes skip
    if os.path.exists(os.path.join(outputfolder_ggi, "GGI_nodes_list.csv")):
        print("GGI data already exists, skipping loading.")
    else:
        ok_ggi = load_ggi_interactions(
            GGI_Interaction=GGI_Interaction,
            outputfolder_ggi=outputfolder_ggi,
            conversion_table_gene=conversion_table_gene,
        )
        if ok_ggi:
            check_ggi_indices(outputfolder_ggi)
    if os.path.exists(os.path.join(outputfolder_mirna, "miRNA_nodes_list.csv")):
        print("miRNA data already exists, skipping loading.")
    else:
        # miRNA–gene interactions
        ok_mirna = load_mirna_gene_interactions(
            mirna_target_file=mirna_target_file,
            refseq2gene_file=refseq2gene_file,
            conversion_table_gene=conversion_table_gene,
            outputfolder_mirna=outputfolder_mirna,
        )
        # miRNA–miRNA metapath: this step ALSO writes miRNA_nodes_list.csv,
        # which is required by compute_gene_mirna_adjacency below, so it must
        # come first.
        if ok_mirna:
            compute_mirna_metapath(outputfolder_mirna)
        # gene–miRNA adjacency (depends on both GGI_nodes_list.csv and
        # miRNA_nodes_list.csv generated above).
        compute_gene_mirna_adjacency(
            outputfolder_mirna=outputfolder_mirna,
            gene_nodes_file=os.path.join(outputfolder_ggi, "GGI_nodes_list.csv"),
            mirna_nodes_file=os.path.join(outputfolder_mirna, "miRNA_nodes_list.csv"),
            adjacency_filename="gene_miRNA_adjacency_sparse.npz",
            filter_by_variance=False,
        )
    if os.path.exists(os.path.join("data/prior_knowledge/TF", "TF_interactions.tsv")):
        print("TF data already exists, skipping loading.")
    else:
        load_tf_interactions(
            tf_target_file="data/prior_knowledge/TF/TFLink_simple.tsv",
            conversion_table_gene=conversion_table_gene,
            outputfolder_tf="data/prior_knowledge/TF/"
        )
