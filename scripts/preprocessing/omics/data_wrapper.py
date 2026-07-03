# %%
# ------------------------------------------------------------------------------
# Process TCGA BRCA data from omics folder to adapt to pipeline
# Files requirements:
# - brca_subtype.csv: patient_id, subtype from clinical_processed.csv
# - cnv_data.csv: sample_id, gene1, gene2, ..., geneN from cnv_processed.csv. Z-score normalized, From Gistic2_CopyNumber_Gistic2_all_data_by_genes
# - expression_data.csv: sample_id, gene1, gene2, ..., geneN from fpkm_processed.csv. Min-max normalized
# - mirna_data.csv: sample_id, mirna1, mirna2, ..., mirnaN from mirna_processed.csv. Min-max normalized
# - brca_shuffle_index.csv: node list shuffle index from GGI_node_list.csv
# - adj_matrix_biogrid.npz: adjacency matrix from GGI_adjacency_sparse.npz, not filtered, preprocessing in utils.py downsample functions
# - standardized_mirna_mrna_edge_filtred_at_eight_with_top_100_mirna.npz: mirna-mrna edges, filtered with top 100 mirna selected by variance??
# - expression_variance.csv: variance of each gene in expression data
# - Adesso ci sta anche TF-gene interactions
# ------------------------------------------------------------------------------
# Da chiedere: se inserisco anche i dati did metilazione devo includerli anche in multimodal gnn?
# Se inserico i dati di metilazione, come li normalizzo? min-max come espressione o z-score come cnv?
# Se inserisco TF-gene interactions, come le rappresento? come un altro grafo? o come attributi dei nodi? 
# I TF vengono dal file TF_TF_interactions_undirected.tsv, una volta filtrati per i geni di biogrid
# si salva il file di interazione con tutti i geni, nel preprocessing del modello poi "src/multiomics_gnn/experiment_pancan/preprocessing/preprocessing_pancan.py"
# quando si selezionano i top n geni per varianza, si filtrano anche i TF e si ricostruisce la matrice di adiacenza
# ------------------------------------------------------------------------------
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import scipy.sparse as sp

def build_dtypese_deprecated(file) -> dict:
    """
    Legge solo l'header del file di esempio e costruisce un dizionario dtypes:
    - 'sample_id' come string (pandas StringDtype)
    - eventuale 'cohort' come category
    - tutte le altre colonne come float32
    """
    header = pd.read_csv(file, nrows=0)
    cols = header.columns.tolist()
    print(f"Columns list: {cols}")
    dtypes = {}
    for c in cols:
        if c == "sample" or c == "icluster_cluster_assignment" or c == "sample_id":
            dtypes[c] = "string"
        elif c == "cohort":
            dtypes[c] = "category"
        else:
            dtypes[c] = "float16"
    return dtypes

def filter_data_with_genes(cnv_df, expression_df, biogrid_gene_list):
    """Filter CNV and expression dataframes to keep only common genes in gene_list."""
    cnv_genes = set(cnv_df.columns)
    expr_genes = set(expression_df.columns)
    biogrid_genes = list(biogrid_gene_list)

    common = cnv_genes & expr_genes & set(biogrid_genes)
    ordered_common_genes = [g for g in biogrid_genes if g in common]

    print("Number of common genes found:", len(ordered_common_genes))
    print("Number of common genes found:", len(common))
    print(f"Original CNV shape: {cnv_df.shape}, Filtered CNV shape: ({cnv_df.shape[0]}, {len(common) + 1})")
    #print(f"Original Expression shape: {expression_df.shape}, Filtered Expression shape: ({expression_df.shape[0]}, {len(common_genes) + 1})")
    #print(f"Original Biogrid gene list length: {len(biogrid_gene_list)}, Filtered Biogrid gene list length: {len(common_genes)}")
    #filtered_cnv = cnv_df[[gene for gene in common_genes if gene in cnv_df.columns]]
    #filtered_expression = expression_df[[gene for gene in common_genes if gene in expression_df.columns]]
    filtered_cnv = cnv_df.reindex(columns=ordered_common_genes)
    filtered_expression = expression_df.reindex(columns=ordered_common_genes)
    #filtered_node_list = [gene for gene in biogrid_gene_list if gene in common_genes]
    #return filtered_cnv, filtered_expression, filtered_node_list
    
    return filtered_cnv, filtered_expression, ordered_common_genes


def filter_data_with_mirna(mirna_df, mirna_list):
    """Filter miRNA dataframe to keep only miRNAs in mirna_list."""
    mirna_mirnas = set(mirna_df.columns)
    mirdb_mirnas = list(mirna_list)

    common = mirna_mirnas & set(mirdb_mirnas)
    ordered_common_mirnas = [m for m in mirdb_mirnas if m in common]

    print("Number of common miRNAs found:", len(ordered_common_mirnas))
    print("Number of common miRNAs found:", len(common))
    print(f"Original miRNA shape: {mirna_df.shape}, Filtered miRNA shape: ({mirna_df.shape[0]}, {len(common)})")
    
    filtered_mirna = mirna_df.reindex(columns=ordered_common_mirnas)
    return filtered_mirna, ordered_common_mirnas

print("Working directory:", os.getcwd())
# All paths are relative to the repository root (the cwd from which the
# pipeline is invoked via the Makefile).
data_paths = {
    "clinical_file": "data/omics/clinical.zip",
    "cnv_file": "data/omics/cnv.zip",
    "mirna_file": "data/omics/mirna.zip",
    "fpkm_file": "data/omics/fpkm.zip",
    "adj_file": "data/prior_knowledge/GGI/GGI_adjacency_sparse.npz",
    "node_list": "data/prior_knowledge/GGI/GGI_nodes_list.csv",
    "gene_mirna_adj_file": "data/prior_knowledge/miRNA/gene_miRNA_adjacency_sparse.npz",
    "mirna_node_list": "data/prior_knowledge/miRNA/miRNA_nodes_list.csv",
}

def build_dtype(data_file):
    """Build dtype dictionary for loading omics data."""
    sample_dtype = {'sample_id': str}
    print(f"Building dtype for {data_file}...")
    # load first row to get column names
    if data_file.endswith('.zip'):
        temp_df = pd.read_csv(data_file, nrows=1, compression='zip')
    else:
        temp_df = pd.read_csv(data_file, nrows=1)
    for col in temp_df.columns:
        if col != 'sample_id':
            sample_dtype[col] = np.float16
    return sample_dtype

def load_omics_data(file_path, dtype=None):
    """Load omics data from a CSV file."""
    print(f"Loading omics data from {file_path}...")
    if file_path.endswith('.zip'):
        df = pd.read_csv(file_path, dtype=dtype, compression='zip')
    else:
        df = pd.read_csv(file_path, dtype=dtype)
    df.set_index('sample_id', inplace=True)
    return df

#minmax normalization to single column
def min_max_normalize(df):
    print("Applying min-max normalization...")
    normalized_df = df.apply(
            lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0,
            axis=0
        )    # check if any value is nan after normalization
    if normalized_df.isna().sum().sum() > 0:
        print("Warning: NaN values found after normalization.")
    return normalized_df


makedir_path = "data/training"
os.makedirs(makedir_path, exist_ok=True)
#%%
# loading all data and filtering by common genes between cnv, expression and biogrid
# ----------------- LOAD DATA -------------------
print("Loading omics data...")
print("Loading CNV data...")
cnv_dtype = build_dtype(data_paths["cnv_file"])
cnv_data_start = load_omics_data(data_paths["cnv_file"], dtype=cnv_dtype)
#%%
print("Loading expression data...")
expression_dtype = build_dtype(data_paths["fpkm_file"])
expression_data_start = load_omics_data(data_paths["fpkm_file"], dtype=expression_dtype)
#%%
print("Loading miRNA data...")
mirna_dtype = build_dtype(data_paths["mirna_file"])
mirna_data_start = load_omics_data(data_paths["mirna_file"], dtype=mirna_dtype)
#%%
print("Loading adjacency matrix and node lists...")
adj_matrix = sp.load_npz(data_paths["adj_file"])
node_list = pd.read_csv(data_paths["node_list"])
mirna_node_list = pd.read_csv(data_paths["mirna_node_list"])
#%%
cnv_data, expression_data, filtered_biogrid_genes = filter_data_with_genes(
    cnv_data_start,
    expression_data_start,
    node_list['symbol'].tolist()
)
#%%
mirna_data, filtered_mirna_list = filter_data_with_mirna(
    mirna_data_start,
    mirna_node_list['miRNA'].tolist()
)


#%%

#----------------- CLINICAL DATA-------------------
# Import
print("preprocessing clinical data...")
clinical_data = load_omics_data(data_paths["clinical_file"])
# Renaming columns
clinical_data = clinical_data.rename(columns={"sample_id": "sample", "Molecular_Subtype": "icluster_cluster_assignment"})
clinical_data.index.name = "sample"
# delete "C" from icluster_cluster_assignment values
clinical_data["icluster_cluster_assignment"] = clinical_data["icluster_cluster_assignment"].str.replace("C", "")
clinical_data = clinical_data.sort_index()

print("Clinical data loaded with shape:", clinical_data.shape)
clinical_data.reset_index(inplace=True)
print("Clinical data head:")
print(clinical_data.head())

clinical_data["icluster_cluster_assignment"] = pd.to_numeric(clinical_data["icluster_cluster_assignment"], errors='coerce').fillna(-1).astype(int)
print("Clinical data after processing icluster_cluster_assignment:")
print(clinical_data.head())
# No need to export clinical data
clinical_data.to_csv(os.path.join(makedir_path, "molecular_subtype.csv"), )
#%%
# ------------------- PLOT SUBTYPE DISTRIBUTION -------------------
# plot histogram of icluster_cluster_assignment with count values on the bars sorting by icluster_cluster_assignment
from matplotlib import pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.countplot(data=clinical_data, x="icluster_cluster_assignment", order=clinical_data["icluster_cluster_assignment"].sort_values().unique())
plt.title("Distribution of ICluster Cluster Assignment in BRCA Samples")
plt.xlabel("ICluster Cluster Assignment")
plt.ylabel("Number of Samples")
# add count values on the bars
for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), 
                       textcoords='offset points')
plt.savefig(os.path.join(makedir_path, "brca_subtype_distribution.png"))
# plt.show()


# %%
# -------------------- CNV DATA -------------------
# CNV data originariamente derivano dal file /Gistic2_CopyNumber_Gistic2_all_data_by_genes
# contiene valori di copy number log2 ratio, non thresholded
# load cnv data
print("CNV data before normalization:")
print(cnv_data.head())
print("index:", cnv_data.index)
print("CNV data loaded with shape:", cnv_data.shape)

# z score normalize each gene column
#cnv_data_normalized = pd.DataFrame(StandardScaler().fit_transform(cnv_data), columns=cnv_data.columns, index=cnv_data.index)
#cnv_data = cnv_data_normalized
# rename index column to sample
cnv_data.index.name = "sample"
# add icluster_cluster_assignment column if not present matching clinical data sample ids
cnv_data = cnv_data.sort_index()
cnv_data = cnv_data.merge(clinical_data.set_index("sample")["icluster_cluster_assignment"], left_index=True, right_index=True, how="left")
# check for nan values in icluster_cluster_assignment
nan_count = cnv_data["icluster_cluster_assignment"].isna().sum()
print("CNV data after normalization:")
print(cnv_data.head())
print(f"Number of NaN values in icluster_cluster_assignment column of CNV data: {nan_count}")
# sort columns by names, keep sample as index
# mantieni l'ordine contrattuale dei geni
gene_cols = filtered_biogrid_genes  # lista ordinata e unica di riferimento

# riordina esplicitamente: geni nell’ordine master + label in fondo
cnv_data = cnv_data.reindex(columns=gene_cols + ["icluster_cluster_assignment"])

assert list(cnv_data.columns[:-1]) == filtered_biogrid_genes, \
    "CNV columns not aligned with filtered_biogrid_genes"
assert cnv_data.columns[-1] == "icluster_cluster_assignment"
#%%
# EXPORT CNV DATA
# reset index to have sample_id as column and export cnv data
# esporta come Gli espression data
# EXPORT CON INDICE
cnv_data_export = cnv_data.copy()
# export expression data
cnv_data_export.index.name = "sample"
cnv_data_export.reset_index(inplace=True)

cnv_data_export.index = range(len(cnv_data_export))
print("CNV data to be exported:")
print(cnv_data_export.head())
#%%
print("Exporting CNV data...")
cnv_data_export.to_csv(os.path.join(makedir_path, "cnv_data_pancan.tsv"), sep='\t', index=True)
# Secondo me il problema è che il dataset di multimodal è stato normalizzato
# a partire dal dataset completo pan-cancer, quindi i valori di espressione
# sono diversi rispetto a quelli che si ottengono normalizzando solo il dataset BRCA.
# infatti non sempre i valori di espressione sono minimo 0 massimo 1.

# -Normalizzo i dati di espressione con min-max normalization
# -Aggiungo la colonna icluster_cluster_assignment
# -Esporto i dati normalizzati
# -Calcolo la varianza di ogni gene e la salvo in un file a parte
# -Genero l'indice di shuffle per i campioni BRCA e lo salvo in un file a parte
# %%

# ---------------------EXPRESSION DATA---------------------------

print("Expression data loaded with shape:", expression_data.shape)
# get nan values
nan_count = expression_data.isna().sum().sum()
print(f"Number of NaN values in expression data before normalization: {nan_count}")

# Normalizzazione
#expression_data_normalized = min_max_normalize(expression_data)
#nan_count_after = expression_data_normalized.isna().sum().sum()
#print(f"Number of NaN values in expression data after normalization: {nan_count_after}")
#print("Expression data after normalization:")
#print(expression_data_normalized.head())
#%%
# Aggiunta colonna icluster_cluster_assignment
#expression_data = expression_data_normalized
expression_data.index.name = "sample"

expression_data = expression_data.sort_index()
expression_data = expression_data.merge(clinical_data.set_index("sample")["icluster_cluster_assignment"], left_index=True, right_index=True, how="left")
#%%
print("Expression data after adding icluster_cluster_assignment:")
print(expression_data.head())
# mantieni l'ordine contrattuale dei geni
gene_cols = filtered_biogrid_genes

# riordina esplicitamente: geni nell’ordine master + label in fondo
expression_data = expression_data.reindex(columns=gene_cols + ["icluster_cluster_assignment"])
assert list(expression_data.columns[:-1]) == filtered_biogrid_genes, \
    "Expression columns not aligned with filtered_biogrid_genes"
assert expression_data.columns[-1] == "icluster_cluster_assignment"
# reset index to have sample_id as column
expression_data.reset_index(inplace=True)

nan_count_expr = expression_data["icluster_cluster_assignment"].isna().sum()
print(f"Number of NaN values in icluster_cluster_assignment column of expression data: {nan_count_expr}")
#%%
# EXPORT CON INDICE
expression_data_export = expression_data.copy()
# export expression data
print(expression_data_export.head())
#%%
expression_data_export.to_csv(
    os.path.join(makedir_path, "expression_data_pancan.tsv"),
    sep="\t",
    index=True
)
# --------------------Calcolo expression variance
# print nans or infs in expression_data
print("Checking for NaNs or Infs in expression data...")
print(expression_data.isna().sum().sum(), "NaNs found.")
# excude icluster_cluster_assignment column for variance calculation and sample column
expression_data_variance_calc = expression_data.drop(columns=["sample", "icluster_cluster_assignment"])
expression_variance = expression_data_variance_calc.var(axis=0, skipna=True)
print("Expression data variance:")
print(expression_variance)

# costruisci DataFrame nel formato richiesto
expression_variance_df = pd.DataFrame({
    "index": expression_variance.index.astype(str),
    "variance": expression_variance.values
})
# %%
print("Expression variance data to be exported:")
print(expression_variance_df.head())
# export expression variance
print("Exporting expression variance data...")
expression_variance_df.to_csv(
    os.path.join(makedir_path, "expression_variance.tsv"),
    sep="\t",
    index=False
)
#%%
# ------------------------Calcolo expression f-values for ANOVA-----------------------
from sklearn.feature_selection import f_classif
# prepare data
print("Calculating expression F-values for ANOVA...")
expression_data_anova_calc = expression_data.drop(columns=["sample"])
# get dtype of columns
print("Dtypes of expression data columns:")
print(expression_data_anova_calc.dtypes)
X = expression_data_anova_calc.select_dtypes(include=['float16','float32', 'float64']).to_numpy()
print("Feature matrix X shape:", X.shape)
Y = expression_data_anova_calc["icluster_cluster_assignment"].to_numpy()
print("Target vector Y shape:", Y.shape)
# calcola f-values
f_values, p_values = f_classif(X, Y)
# costruisci DataFrame nel formato richiesto
print("len(f_values):", len(f_values))
print("len(p_values):", len(p_values))
print("n_genes (cols):", len(expression_data_anova_calc.select_dtypes(include=['float16','float32', 'float64']).columns))

expression_fvalues_df = pd.DataFrame({
    "index": expression_data_anova_calc.select_dtypes(include=['float16','float32', 'float64']).columns.astype(str),
    "f_value": f_values,
    "p_value": p_values
})

expression_fvalues_df.to_csv(
    os.path.join(makedir_path, "expression_fvalues.tsv"),
    sep="\t",
    index=False
)

# %%
# -------------------------MIRNA DATA-----------------------


print("miRNA data loaded with shape:", mirna_data.shape)
# min-max normalize each mirna column
#mirna_data_normalized = min_max_normalize(mirna_data)
#mirna_data = mirna_data_normalized
mirna_data.index.name = "sample"
mirna_data = mirna_data.sort_index()

print("miRNA data after normalization:")
print(mirna_data.head())
#%%
# export miRNA data
mirna_export = mirna_data.copy()
# export expression data
mirna_export.index.name = "sample"
mirna_export.reset_index(inplace=True)

mirna_export.index = range(len(mirna_export))
# check miRNA if column sample exists
if "sample" in mirna_export.columns:
    #drop it
    mirna_export = mirna_export.drop(columns=["sample"])
print("miRNA data to be exported:")
print(mirna_export.head())
#%%
print("Exporting miRNA data...")
mirna_export.to_csv(os.path.join(makedir_path, "mirna_data_brca.tsv"), sep='\t', index=True)

#%%
# ------------------------ TOP 100 MIRNA BY VARIANCE -----------------------
print("Calculating miRNA variance...")
if "sample_id" in mirna_data.columns:
    mirna_only  = mirna_data.drop(columns=["sample_id"])
else:
    mirna_only = mirna_data

# calcola varianza di ogni mirna
mirna_only = mirna_only.replace(-1, np.nan)
mirna_variance = mirna_only.var(axis=0, skipna=True)
print("miRNA data variance:")
print(mirna_variance)

top_100_mirnas = mirna_variance.sort_values(ascending=False).head(100).index.tolist()
# filter mirna data with top 100 mirnas
mirna_data_top_100 = mirna_data[top_100_mirnas]
# export top 100 mirna list to txt file
print(mirna_data_top_100.head())

mirna_data_top_100.to_csv(os.path.join(makedir_path, "mirna_data_top_100.tsv"),
                          sep='\t', index=True)
with open(os.path.join(makedir_path, "top_100_mirna.txt"), 'w') as f:
    for mirna in top_100_mirnas:
        f.write(f"{mirna}\n")

print("Top 100 miRNAs exported.")
print(mirna_data_top_100.head())
#%%
# Questa è la tua lista master definitiva per i nodi miRNA
filtered_mirnas = list(mirna_data_top_100.columns)

# controllo immediato
assert filtered_mirnas == top_100_mirnas, \
    "Mismatch: top_100_mirnas order differs from mirna_data_top_100 columns"

# %%
# ---------------------- GENE-GENE ADJACENCY MATRIX FILTERING -----------------------
# al posto delle matrici di adiacenza carico i file di interazione, filtro e costruisco le matrici di adiacenza
interactions_df = pd.read_csv("data/prior_knowledge/GGI/GGI_edges_undirected.csv")
print("Interactions data loaded with shape:", interactions_df.shape)
# da questo dataframe filtra le interazioni per tenere solo quelle tra geni presenti in filtered_biogrid_genes
filtered_interactions = interactions_df[(interactions_df['u'].isin(filtered_biogrid_genes)) & (interactions_df['v'].isin(filtered_biogrid_genes))]
print("Filtered interactions shape:", filtered_interactions.shape)

print(f"Unique genes in filtered interactions: {len(set(filtered_interactions['u']).union(set(filtered_interactions['v'])))}")
# stampa poi quelli del filtered_biogrid_genes unici
print(f"Number of filtered biogrid genes: {len(filtered_biogrid_genes)}")
# costruisci la matrice di adiacenza a partire dalle interazioni filtrate

#%%



# ---------------------- GENE GENE ADJACENCY MATRIX AND NODE LIST -----------------------
# ADJACENCY MATRIX AND NODE LIST FOR GGI
# build adjacency matrix FROM filtered interactions node list
import scipy.sparse as sp
print("Building gene-gene adjacency matrix...")
# load interactions
interactions_df = pd.read_csv("data/prior_knowledge/GGI/GGI_edges_undirected.csv")
print("Interactions data loaded with shape:", interactions_df.shape)


# 1) node list 
genes = pd.Series(filtered_biogrid_genes).reset_index(drop=True)
print(f"Number of genes in node list: {len(genes)}")
# 2) mapping stabile
gene_to_idx = {g: i for i, g in enumerate(genes)}

# 1) filtra archi solo tra geni ammessi
edges = interactions_df[
    interactions_df["u"].isin(gene_to_idx) &
    interactions_df["v"].isin(gene_to_idx)
].copy()

row = edges["u"].map(gene_to_idx).to_numpy()
col = edges["v"].map(gene_to_idx).to_numpy()

# 4) simmetria esplicita
row_idx = np.concatenate([row, col])
col_idx = np.concatenate([col, row])
data = np.ones(len(row_idx), dtype=np.float32)

n = len(genes)
A = sp.coo_matrix((data, (row_idx, col_idx)), shape=(n, n))

# 5) comprimi duplicati e pulisci diagonale
A = A.tocsr()
A.setdiag(0)
A.eliminate_zeros()

# 6) opzionale: aggiungi self-loops (decidi tu)
A = A + sp.eye(n, format='csr')

print(f"Adjacency matrix shape: {A.shape}")

# export new adjacency matrix
adj_matrix = A
node_list = genes.to_frame(name="symbol")

# export adjacency matrix
sp.save_npz(os.path.join(makedir_path, "adj_matrix_biogrid.npz"), adj_matrix)

#%%
# ----------------------- TF-GENE ADJACENCY MATRIX AND NODE LIST -----------------------
print("Building TF-gene adjacency matrix...")
# load tf-gene interactions
tf_gene_interactions_path = "data/prior_knowledge/TF/TF_interactions.tsv"
tf_gene_df = pd.read_csv(tf_gene_interactions_path, sep="\t")
print("TF-gene interactions loaded with shape:", tf_gene_df.shape)
filtered_genes = [str(g).strip() for g in filtered_biogrid_genes]
#%%
# --- drop NA / empty ---
tf_gene_df = tf_gene_df.dropna(subset=["HGNC.TF", "HGNC.Target"])
tf_gene_df = tf_gene_df[(tf_gene_df["HGNC.TF"] != "") & (tf_gene_df["HGNC.Target"] != "")]
print(f"TF-gene interactions after dropping NA/empty: {tf_gene_df.shape}")
#%%
# Tutti i TF sono geni.
# standardizza le stringhe
tf_gene_df["HGNC.TF"] = tf_gene_df["HGNC.TF"].astype(str).str.strip()
tf_gene_df["HGNC.Target"] = tf_gene_df["HGNC.Target"].astype(str).str.strip()
print(f"Unique TFs (global, raw): {tf_gene_df['HGNC.TF'].nunique()}")
print(f"Unique Targets (global, raw): {tf_gene_df['HGNC.Target'].nunique()}")
print(f"TF-gene interactions (raw): {tf_gene_df.shape}")
#%%
# --- deduplicate edges ---
tf_gene_df = tf_gene_df.drop_duplicates(subset=["HGNC.TF", "HGNC.Target"]).reset_index(drop=True)
print(f"Unique TFs (global, cleaned): {tf_gene_df['HGNC.TF'].nunique()}")
print(f"Unique Targets (global, cleaned): {tf_gene_df['HGNC.Target'].nunique()}")
print(f"TF-gene interactions after deduplication: {tf_gene_df.shape}")
#%%
# --- filtro al vocabolario geni del grafo ---
filtered_genes = [str(g).strip() for g in filtered_biogrid_genes]
gene_set = set(filtered_genes)
if len(gene_set) == 0:
    raise ValueError("filtered_biogrid_genes is empty")

tf_gene_edges_filtered = tf_gene_df[
    tf_gene_df["HGNC.TF"].isin(gene_set) &
    tf_gene_df["HGNC.Target"].isin(gene_set)
].copy()
print("TF-gene edges after filtering to gene vocab:", tf_gene_edges_filtered.shape)
print(f"Unique TFs (in vocab): {tf_gene_edges_filtered['HGNC.TF'].nunique()}")
print(f"Unique Targets (in vocab): {tf_gene_edges_filtered['HGNC.Target'].nunique()}")
#%%
# --- costruzione matrice di adiacenza TF-gene ---
tf_list = tf_gene_edges_filtered["HGNC.TF"].unique().tolist()
# ordered tf list by alphabetical order
tf_list.sort()
print(f"Total unique TFs in filtered edges: {len(tf_list)}")

# esporta lista TF con gli indici di matrice
tf_to_idx = {g: i for i, g in enumerate(tf_list)}
gene_to_idx = {g: i for i, g in enumerate(filtered_genes)}
#%%
# La matrice è TF x Gene
row_idx = tf_gene_edges_filtered["HGNC.TF"].map(tf_to_idx).to_numpy()
col_idx = tf_gene_edges_filtered["HGNC.Target"].map(gene_to_idx).to_numpy()
data = np.ones(len(row_idx), dtype=np.float32)
n_genes = len(filtered_genes)
n_tf = len(tf_list)
#%%
A_tf_gene = sp.coo_matrix(
    (data, (row_idx, col_idx)),
    shape=(n_tf, n_genes),
    dtype=np.float32
).tocsr()
# forza binario (utile se residuano duplicati)
A_tf_gene.data[:] = 1.0
A_tf_gene.eliminate_zeros()
print("TF-gene adjacency shape:", A_tf_gene.shape)
print("TF-gene non-zero edges:", A_tf_gene.nnz)
#%%
# --- salva artefatti ---
os.makedirs(makedir_path, exist_ok=True)

tf_adj_path = os.path.join(makedir_path, "tf_gene_adj_global.npz")
sp.save_npz(tf_adj_path, A_tf_gene)
print("Saved TF-gene adjacency:", tf_adj_path)

tf_nodes_df = pd.DataFrame({"TF": tf_list, "matrix_index": np.arange(n_tf, dtype=int)})
tf_nodes_path = os.path.join(makedir_path, "tf_nodes_all_in_vocab.csv")
tf_nodes_df.to_csv(tf_nodes_path, index=False)
print("Saved TF nodes list:", tf_nodes_path)
#%%
n_genes = len(filtered_genes)
gene_nodes_df = pd.DataFrame({
    "gene": filtered_genes,
    "matrix_index": np.arange(n_genes, dtype=int)
})
gene_nodes_df.to_csv(os.path.join(makedir_path, "gene_nodes_filtered_for_tf.csv"), index=False)
print("Saved gene nodes list for TF-gene matrix:", os.path.join(makedir_path, "gene_nodes_filtered_for_tf.csv"))
# %%
# ----------------------- GENE-MIRNA ADJACENCY MATRIX AND NODE LIST -----------------------
# gene-mirna adjacency matrix with only top 100 mirna by variance
print("Building gene-miRNA adjacency matrix...")
# ------------------ LOAD NODE LISTS ------------------
mirna_gene_interactions_path = "data/prior_knowledge/miRNA/miRNA_gene_interactions.csv"
filtered_genes = [str(g).strip() for g in filtered_biogrid_genes]
filtered_mirnas = list(mirna_data_top_100.columns) # oppure la tua lista filtrata finale
#%%

print("miRNA node list loaded with shape:", mirna_node_list.shape)
# ------------------ SANITY CHECK LISTE ------------------
if len(filtered_genes) == 0:
    raise ValueError("filtered_genes is empty")
if len(filtered_mirnas) == 0:
    raise ValueError("filtered_mirnas is empty")
#%%
# ------------------ LOAD INTERACTIONS ------------------
if not os.path.exists(mirna_gene_interactions_path):
    raise FileNotFoundError(f"File not found: {mirna_gene_interactions_path}")

df = pd.read_csv(mirna_gene_interactions_path)
print("Interactions loaded:", df.shape)

# controlla colonne necessarie
# la colonna dei simboli genici si chiama "HGCN Symbol"
if "HGCN Symbol" not in df.columns:
    raise ValueError(f"Missing column 'HGCN Symbol'. Found: {list(df.columns)}")
if "miRNA" not in df.columns:
    raise ValueError(f"Missing column 'miRNA'. Found: {list(df.columns)}")

# pulizia stringhe
df["HGCN Symbol"] = df["HGCN Symbol"].astype(str).str.strip()
df["miRNA"] = df["miRNA"].astype(str).str.strip()

#%%
# ------------------ MAPPING STABILE DA LISTE FILTRATE ------------------
gene_to_idx = {g: i for i, g in enumerate(filtered_genes)}
mirna_to_idx = {m: j for j, m in enumerate(filtered_mirnas)}

#%%
# ------------------ FILTER INTERACTIONS ------------------
interactions = df[
    df["HGCN Symbol"].isin(gene_to_idx) &
    df["miRNA"].isin(mirna_to_idx)
].copy()

print("Interactions after filtering:", interactions.shape)

# deduplica coppie per matrice binaria pulita
interactions = interactions.drop_duplicates(subset=["HGCN Symbol", "miRNA"])

if interactions.empty:
    raise ValueError("No overlapping interactions with the filtered gene/miRNA lists")
#%%
# ------------------ BUILD INDICES ------------------
row_idx = interactions["HGCN Symbol"].map(gene_to_idx).to_numpy()
col_idx = interactions["miRNA"].map(mirna_to_idx).to_numpy()

data = np.ones(len(row_idx), dtype=np.float32)

n_genes = len(filtered_genes)
n_mirnas = len(filtered_mirnas)
#%%
# ------------------ BUILD SPARSE MATRIX (GENE x miRNA) ------------------
A_gene_mirna = sp.coo_matrix(
    (data, (row_idx, col_idx)),
    shape=(n_genes, n_mirnas),
    dtype=np.float32
).tocsr()

# forza binario (utile se residuano duplicati)
A_gene_mirna.data[:] = 1.0
A_gene_mirna.eliminate_zeros()

print("Adjacency shape:", A_gene_mirna.shape)
print("Non-zero edges:", A_gene_mirna.nnz)

# ------------------ EXPORT ADJ ------------------
sp.save_npz(
    os.path.join(
        makedir_path,
        "standardized_mirna_mrna_edge_filtered_at_eight_with_top_100_mirna.npz"
    ),
    A_gene_mirna
)


#%%
# ------------------ EXPORT NODE LISTS CON MATRIX INDEX ------------------
gene_nodes_df = pd.DataFrame({
    "gene": filtered_genes,
    "matrix_index": np.arange(n_genes, dtype=int)
})
mirna_nodes_df = pd.DataFrame({
    "miRNA": filtered_mirnas,
    "matrix_index": np.arange(n_mirnas, dtype=int)
})
#%%




