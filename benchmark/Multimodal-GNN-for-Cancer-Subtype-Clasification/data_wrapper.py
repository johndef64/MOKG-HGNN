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
# ------------------------------------------------------------------------------
# Da chiedere: se inserisco anche i dati did metilazione devo includerli anche in multimodal gnn?
# Se inserico i dati di metilazione, come li normalizzo? min-max come espressione o z-score come cnv?
# Se inserisco TF-gene interactions, come le rappresento? come un altro grafo? o come attributi dei nodi? 
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
data_paths = {
    "clinical_file": "../../datasets/data/omics/clinical.csv",
    "cnv_file": "../../datasets/data/omics/cnv_gistic.csv",
    "mirna_file": "../../datasets/data/omics/mirna.csv",
    "fpkm_file": "../../datasets/data/omics/fpkm.csv",
    "adj_file": "../../datasets/data/prior_knowledge/GGI/GGI_adjacency_sparse.npz",
    "node_list": "../../datasets/data/prior_knowledge/GGI/GGI_nodes_list.csv",
    "gene_mirna_adj_file": "../../datasets/data/prior_knowledge/miRNA/gene_miRNA_adjacency_sparse.npz",
    "mirna_node_list": "../../datasets/data/prior_knowledge/miRNA/miRNA_nodes_list.csv",
    "top_100_mirna_file": "../../datasets/data/train_test_split/top_100_variance_mirnas.txt"
}

# %%

def load_omics_data(file_path):
    """Load omics data from a CSV file."""
    df = pd.read_csv(file_path)
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

# %%
makedir_path = "data_new"
os.makedirs(makedir_path, exist_ok=True)
# CLINICAL DATA
clinical_data = load_omics_data(data_paths["clinical_file"])
# process in this format "patient","subtype""TCGA-3C-AAAU","LumA""TCGA-3C-AALI","Her2"
clinical_data = clinical_data.rename(columns={"sample_id": "patient", "BRCA_Subtype": "subtype"})
print("Clinical data loaded with shape:", clinical_data.shape)
# export clinical data
#save also index
clinical_data.to_csv(os.path.join(makedir_path, "brca_subtype.csv"), )
# %%
# CNV DATA
# CNV data originariamente derivano dal file /Gistic2_CopyNumber_Gistic2_all_data_by_genes
# contiene valori di copy number log2 ratio, non thresholded
cnv_data = load_omics_data(data_paths["cnv_file"])
print("index:", cnv_data.index)
print("CNV data loaded with shape:", cnv_data.shape)
# z score normalize each gene column
cnv_data_normalized = pd.DataFrame(StandardScaler().fit_transform(cnv_data), columns=cnv_data.columns, index=cnv_data.index)
cnv_data = cnv_data_normalized
# rename index column to sample
cnv_data.index.name = "sample"
# add icluster_cluster_assignment column if not present
if "icluster_cluster_assignment" not in cnv_data.columns:
    cnv_data["icluster_cluster_assignment"] = -1
print("CNV data after normalization:")
print(cnv_data.head())
# export CNV data
#%%
# reset index to have sample_id as column
cnv_data.reset_index(inplace=True)
cnv_data.to_csv(os.path.join(makedir_path, "cnv_data_brca.tsv"), sep='\t', index=True)

# %%
# Secondo me il problema è che il dataset di multimodal è stato normalizzato
# a partire dal dataset completo pan-cancer, quindi i valori di espressione
# sono diversi rispetto a quelli che si ottengono normalizzando solo il dataset BRCA.
# infatti non sempre i valori di espressione sono minimo 0 massimo 1.
# EXPRESSION DATA
expression_data = load_omics_data(data_paths["fpkm_file"])
print("Expression data loaded with shape:", expression_data.shape)
# get nan values
nan_count = expression_data.isna().sum().sum()
print(f"Number of NaN values in expression data before normalization: {nan_count}")
# min-max normalize each gene column
#%%
expression_data_normalized = min_max_normalize(expression_data)
nan_count_after = expression_data_normalized.isna().sum().sum()
print(f"Number of NaN values in expression data after normalization: {nan_count_after}")
#%%
expression_data = expression_data_normalized
expression_data.index.name = "sample"
# add icluster_cluster_assignment column if not present
if "icluster_cluster_assignment" not in expression_data.columns:
    expression_data["icluster_cluster_assignment"] = -1
print("Expression data after normalization:")
print(expression_data.head())
# export expression data
# reset index to have sample_id as column
expression_data.reset_index(inplace=True)
expression_data.to_csv(os.path.join(makedir_path, "expression_data_brca.tsv"), sep='\t', index=True)
# %%
# expression variance
# print nans or infs in expression_data
print("Checking for NaNs or Infs in expression data...")
print(expression_data.isna().sum().sum(), "NaNs found.")
expression_variance = expression_data.var(axis=0)
print("Expression data variance:")
print(expression_variance)
# %%
# get shuffle inxed for brca samples
num_samples = expression_data.shape[0]   # es. 989
np.random.seed(42)   # opzionale, per riproducibilità
shuffle_index = np.random.permutation(num_samples)
print("Shuffle index for BRCA samples:")
print(shuffle_index)
# export shuffle index
df_shuffle = pd.DataFrame({
    "index": shuffle_index     # la vera permutazione
})

df_shuffle.to_csv(
    os.path.join(makedir_path, "brca_shuffle_index.tsv"),
    sep='\t',
    index=True
)
# %%
# MIRNA DATA
mirna_data = load_omics_data(data_paths["mirna_file"])
print("miRNA data loaded with shape:", mirna_data.shape)
# min-max normalize each mirna column
mirna_data_normalized = min_max_normalize(mirna_data)
mirna_data = mirna_data_normalized
mirna_data.index.name = "sample"

print("miRNA data after normalization:")
print(mirna_data.head())
# export miRNA data
mirna_data.to_csv(os.path.join(makedir_path, "mirna_data_brca.tsv"), sep='\t')
# %%
# ADJACENCY MATRIX AND NODE LIST
import scipy.sparse as sp
# load adjacency matrix
adj_matrix = sp.load_npz(data_paths["adj_file"])
print("Adjacency matrix loaded with shape:", adj_matrix.shape)
# export adjacency matrix
sp.save_npz(os.path.join(makedir_path, "adj_matrix_biogrid.npz"), adj_matrix)
# load node list
node_list = pd.read_csv(data_paths["node_list"])
print("Node list loaded with shape:", node_list.shape)
# filter expression variance to have only genes in node list
expression_variance = expression_variance[expression_variance.index.isin(node_list['symbol'])]
expression_variance.to_csv(os.path.join(makedir_path, "expression_variance.tsv"), sep='\t')

# shuffle node list and export shuffle index
# shuffled_indices = np.random.permutation(len(node_list))
# pd.DataFrame(shuffled_indices, columns=["index"]).to_csv(os.path.join(makedir_path, "brca_shuffle_index.tsv"), sep='\t', index=False)
# biogrid node list export as biogrid non null with this gene,matrix_index as columns, 
# now is a single column with gene names called symbol
# add matrix_index column
node_list['matrix_index'] = range(len(node_list))
node_list = node_list.rename(columns={"symbol": "gene"})
node_list.to_csv(os.path.join(makedir_path, "biogrid_non_null.csv"), index=False)
# %%
# gene-mirna adjacency matrix with only top 100 mirna by variance
mirna_adj_matrix = sp.load_npz(data_paths["gene_mirna_adj_file"])
print("Gene-miRNA adjacency matrix loaded with shape:", mirna_adj_matrix.shape)
# load mirna node list
mirna_node_list = pd.read_csv(data_paths["mirna_node_list"])
print("miRNA node list loaded with shape:", mirna_node_list.shape)
# load top 100 mirna list txt
with open(data_paths["top_100_mirna_file"], 'r') as f:
    top_100_mirnas = [line.strip() for line in f.readlines()]
print("Top 100 miRNAs loaded:", top_100_mirnas)
# %%
# filter mirna_node_list to have only top 100 mirnas
print("Filtering miRNA node list to have only top 100 miRNAs...")
filtered_mirna_nodes = mirna_node_list[mirna_node_list['miRNA'].isin(top_100_mirnas)]
print("Filtered miRNA node list shape:", filtered_mirna_nodes.shape)
# get indices of top 100 mirnas
top_100_mirna_indices = filtered_mirna_nodes.index.tolist()
# filter adjacency matrix to have only columns corresponding to top 100 mirnas
filtered_mirna_adj_matrix = mirna_adj_matrix[:, top_100_mirna_indices]
print("Filtered Gene-miRNA adjacency matrix shape:", filtered_mirna_adj_matrix.shape)
# %%
# export filtered adjacency matrix
sp.save_npz(os.path.join(makedir_path, "standardized_mirna_mrna_edge_filtred_at_eight_with_top_100_mirna.npz"), filtered_mirna_adj_matrix)
# %%
