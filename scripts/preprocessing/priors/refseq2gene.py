#%%
#load miRDB
import mygene
import pandas as pd
mirna_df = pd.read_csv("data/prior_knowledge/miRNA/miRDB.gz", sep="\t", compression="gzip", header=None)
mirna_df.columns = ["miRNA", "Target", "Score"]
human80 = mirna_df[mirna_df["miRNA"].str.startswith("hsa-")]
# %%
# count how many NM_ ids are there
nm_ids = human80["Target"].str.startswith("NM_").sum()
print(f"Number of NM_ ids: {nm_ids}")
#%%
# get unique NM_ ids
unique_nm_ids = human80["Target"].unique()
print(f"Number of unique NM_ ids: {len(unique_nm_ids)}")
# %%
mg = mygene.MyGeneInfo()
# conversione NM_ → symbol first 3 values
res = mg.querymany(unique_nm_ids.tolist(),
                   scopes="refseq",
                   fields="symbol, entrezgene, ensembl.gene",
                   species="human")
# %%
# convert to dataframe
res_df = pd.DataFrame(res)
# keep only relevant columns
res_df = res_df[["query", "symbol", "entrezgene", "ensembl"]]
# Estrazione dell'ensembl id, {'gene': 'ENSGXXX'}, se ce ne sono più di uno tieni solo il primo
res_df["ensembl"] = res_df["ensembl"].apply(lambda x: x[0]["gene"] if isinstance(x, list) else x['gene'] if isinstance(x, dict) else None)

# %%
# check how many mappings were not found
not_found = res_df[res_df["symbol"].isna()]
print(f"Number of NM_ ids not found: {len(not_found)}")

# %%
# export not found ids
# Sono 86 NM_ ids non trovati in mygene
# %%
# esportazione mappature trovate
found = res_df[res_df["symbol"].notna()]
import os
os.makedirs("data/prior_knowledge", exist_ok=True)
found.to_csv("data/prior_knowledge/refseq2gene_mappings.tsv", sep="\t", index=False)
# %%