#%%
import os
import requests

os.chdir(os.path.dirname(__file__))  # Cambia directory alla posizione dello script
os.makedirs("data", exist_ok=True)
os.chdir("data")

# source= https://figshare.com/articles/dataset/Figs1_png/19248078 

urls = [
    "https://figshare.com/ndownloader/files/58617895",
    "https://figshare.com/ndownloader/files/58617898",
    "https://figshare.com/ndownloader/files/58617901",
    "https://figshare.com/ndownloader/files/58617904",
    "https://figshare.com/ndownloader/files/58617916",
    "https://figshare.com/ndownloader/files/58617907",
    "https://figshare.com/ndownloader/files/58617910",
    "https://figshare.com/ndownloader/files/58617913",
    "https://figshare.com/ndownloader/files/58617919",
]

filenames = [
    "adj_matrix_biogrid.npz",
    "biogrid_non_null.csv",
    "brca_shuffle_index.tsv",
    "brca_subtype.csv",
    "cnv_data_brca.tsv",
    "expression_variance.tsv",
    "mirna_data_brca.tsv",
    "standardized_mirna_mrna_edge_filtered_at_eight_with_top_100_mirna.npz",
    "expression_data_brca.tsv",
]


# Use requests to download files
if __name__ == '__main__':
    for url, filename in zip(urls, filenames):
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            response = requests.get(url)
            with open(filename, "wb") as f:
                f.write(response.content)
        else:
            print(f"{filename} already exists, skipping download.")
    print("All files are ready.")




