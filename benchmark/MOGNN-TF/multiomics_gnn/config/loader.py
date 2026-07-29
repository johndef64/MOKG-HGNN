import yaml

def load_config(file_path: str) -> dict:
    """
    Carica un file di configurazione YAML e lo converte in un dizionario Python.

    Args:
        file_path (str): Il percorso del file YAML da caricare.

    Returns:
        dict: Un dizionario contenente i dati del file YAML.
    """
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return config_dict