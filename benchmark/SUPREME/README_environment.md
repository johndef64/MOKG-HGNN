# SUPREME Environment Setup Guide

## Creare l'environment conda

```bash
# Crea l'environment da file
conda env create -f environment.yml

# Attiva l'environment
conda activate supreme

# Verifica l'installazione
python -c "import torch; import pandas; import xgboost; print('Environment pronto!')"
```

## Comandi alternativi per setup manuale

Se il file environment.yml non funziona, puoi creare l'environment manualmente:

```bash
# Crea environment con Python 3.6
conda create -n supreme python=3.6.13

# Attiva l'environment
conda activate supreme

# Installa PyTorch (CPU only)
conda install pytorch=1.10.2 torchvision=0.11.3 torchaudio=0.10.2 cpuonly -c pytorch

# Installa dipendenze via pip
pip install -r requirements.txt
```

## Setup R per rpy2 (necessario per feature selection)

Se usi Windows, potresti aver bisogno di installare R separatamente:

1. Scarica e installa R da https://cran.r-project.org/
2. Installa i pacchetti R necessari:
```r
install.packages(c("rFerns", "Boruta", "pracma", "dplyr"))
```

## Verifica installazione

```bash
conda activate supreme
python SUPREME.py -data sample_data
```

## Risoluzione problemi comuni

- **rpy2 non funziona**: Assicurati che R sia installato e nel PATH
- **torch-geometric errori**: Potrebbe essere necessario compilare da source
- **pickle5 errori**: Su Python 3.8+ non è necessario, usa pickle standard

## Environment alternativo (Python 3.8+)

Per una versione più moderna:
```bash
conda create -n supreme-modern python=3.8
conda activate supreme-modern
pip install torch torchvision torchaudio torch-geometric xgboost pandas scikit-learn
```