#%%
import os
import requests

os.chdir(os.path.dirname(__file__))  # Cambia directory alla posizione dello script
os.makedirs("data", exist_ok=True)
os.chdir("data")

# source= https://figshare.com/articles/dataset/Figs1_png/19248078 

urls = [
    "https://figshare.com/ndownloader/files/34202496",
    "https://figshare.com/ndownloader/files/34202499",
    "https://figshare.com/ndownloader/files/34202502",
    "https://figshare.com/ndownloader/files/34202505",
    "https://figshare.com/ndownloader/files/34202652"
]

filenames = [
    "fpkm_data.csv",
    "gistic_data.csv",
    "rppa_data.csv",
    "sample_classes.csv",
    "test_sample.csv"
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




