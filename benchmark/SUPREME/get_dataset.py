#%%
import requests
import os
import json
from urllib.parse import urlparse
import zipfile

def download_github_folder(folder_url, output_dir="./data"):
    """
    Scarica una cartella specifica da GitHub usando l'API
    """
    # Estrai informazioni dall'URL
    parts = folder_url.replace("https://github.com/", "").split("/")
    owner = parts[0]
    repo = parts[1]
    
    # Trova il percorso della cartella dall'URL
    if "tree" in parts:
        branch = parts[3]
        folder_path = "/".join(parts[4:])
    else:
        branch = "main"  # default branch
        folder_path = ""
    
    # URL dell'API GitHub
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{folder_path}"
    if branch != "main":
        api_url += f"?ref={branch}"
    
    # Crea la directory di output
    os.makedirs(output_dir, exist_ok=True)
    
    def download_contents(api_url, local_path):
        response = requests.get(api_url)
        if response.status_code != 200:
            print(f"Errore: {response.status_code} - {response.text}")
            return
        
        contents = response.json()
        
        for item in contents:
            # Salta il download della cartella models, che contiene i modelli salvati usati in
            # feat_importanc.py
            if item['name'] == 'models':
                print(f"Saltato: {item['name']} (cartella modelli)")
                continue
                
            item_path = os.path.join(local_path, item['name'])
            
            if item['type'] == 'file':
                # Scarica il file
                file_response = requests.get(item['download_url'])
                with open(item_path, 'wb') as f:
                    f.write(file_response.content)
                print(f"Scaricato: {item_path}")
                
            elif item['type'] == 'dir':
                # Crea directory e scarica ricorsivamente
                os.makedirs(item_path, exist_ok=True)
                download_contents(item['url'], item_path)
    
    download_contents(api_url, output_dir)
    print(f"Download completato in: {output_dir}")

    # Unzip files in the output directory
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isfile(item_path) and item_path.endswith('.zip'):
            with zipfile.ZipFile(item_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            print(f"Unzipped: {item_path}")

# URL della cartella che vuoi scaricare
data = "https://github.com/bozdaglab/SUPREME/tree/main/data/sample_data"

# Scarica la cartella
download_github_folder(data, "./data/sample_data")
