"""
@Project: MPK-GNN
@File   : ops_loss.py
@Author : Yonglong Tian (yonglong@mit.edu)
@Date   : May 07, 2020
@Desc   : Implementazione della Supervised Contrastive Loss

Questo file implementa la Supervised Contrastive Loss per l'apprendimento
di rappresentazioni discriminative. La loss combina il contrastive learning
con la supervisione delle etichette di classe.

Principio della Supervised Contrastive Loss:
1. Per ogni campione "anchor", identifica campioni positivi (stessa classe) e negativi (classi diverse)
2. Avvicina l'anchor ai campioni positivi nello spazio delle rappresentazioni
3. Allontana l'anchor dai campioni negativi
4. Utilizza temperatura per controllare la "durezza" della distribuzione

Formula: L = -log(sum(exp(sim(z_i, z_p)/τ)) / sum(exp(sim(z_i, z_k)/τ)))
dove:
- z_i: rappresentazione dell'anchor
- z_p: rappresentazioni dei campioni positivi
- z_k: rappresentazioni di tutti i campioni (positivi + negativi)
- τ: parametro di temperatura

Reference: https://arxiv.org/pdf/2004.11362.pdf (Supervised Contrastive Learning)
          https://arxiv.org/pdf/2002.05709.pdf (SimCLR)
"""
from __future__ import print_function
import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss
    ====================================
    
    Implementa la funzione di loss per il contrastive learning supervisionato.
    Questa loss migliora l'apprendimento delle rappresentazioni utilizzando
    le etichette di classe per guidare il processo di contrastive learning.
    
    Principio di funzionamento:
    - Per ogni campione anchor, identifica tutti i campioni positivi (stessa classe)
    - Avvicina l'anchor ai campioni positivi nello spazio delle rappresentazioni
    - Allontana l'anchor dai campioni negativi (classi diverse)
    - Supporta anche unsupervised contrastive loss (SimCLR) senza etichette
    
    Vantaggi rispetto a CrossEntropy standard:
    1. Impara rappresentazioni più robuste e discriminative
    2. Migliora la generalizzazione su nuovi dati
    3. Riduce l'overfitting attraverso la regolarizzazione implicita
    """
    
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, cuda_device='0'):
        """
        Inizializza la Supervised Contrastive Loss.
        
        Args:
            temperature (float): Parametro di temperatura per scalare i logits.
                               Valori più bassi (es. 0.07) rendono la distribuzione più "sharp",
                               valori più alti la rendono più "smooth"
            contrast_mode (str): Modalità di contrasto:
                               'all' - usa tutte le viste come anchor (default)
                               'one' - usa solo la prima vista come anchor
            base_temperature (float): Temperatura base per normalizzazione della loss
            cuda_device (str): ID del dispositivo CUDA da utilizzare
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.cuda_device = cuda_device

    def forward(self, features, labels=None, mask=None):
        """
        Calcola la Supervised Contrastive Loss.
        
        Se sia labels che mask sono None, degenera nella unsupervised loss di SimCLR.
        
        Args:
            features (torch.Tensor): Rappresentazioni nascoste con shape [batch_size, n_views, feature_dim].
                                    n_views è il numero di viste/augmentazioni per ogni campione.
                                    In MPK-GNN: n_views=3 (corrispondenti alle 3 matrici di adiacenza)
            labels (torch.Tensor): Etichette ground truth con shape [batch_size].
                                  Se None, viene utilizzata la modalità unsupervised
            mask (torch.Tensor): Maschera contrastiva con shape [batch_size, batch_size].
                                mask[i,j]=1 se il campione j ha la stessa classe del campione i
        
        Returns:
            torch.Tensor: Valore scalare della loss
        """
        
        # === CONFIGURAZIONE DISPOSITIVO ===
        device = ("cuda:" + self.cuda_device if features.is_cuda else torch.device('cpu'))

        # === VALIDAZIONE INPUT ===
        if len(features.shape) < 3:
            raise ValueError('`features` deve avere shape [bsz, n_views, ...], '
                           'sono richieste almeno 3 dimensioni')
        if len(features.shape) > 3:
            # Appiattisce dimensioni extra mantenendo [batch_size, n_views, -1]
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        
        # === COSTRUZIONE MASCHERA DI SIMILARITÀ ===
        if labels is not None and mask is not None:
            raise ValueError('Non è possibile definire sia `labels` che `mask`')
        elif labels is None and mask is None:
            # Modalità unsupervised: ogni campione è positivo solo con se stesso
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            # Modalità supervised: costruisce maschera dalle etichette
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Numero di etichette non corrisponde al numero di feature')
            # Crea maschera: mask[i,j] = 1 se labels[i] == labels[j]
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            # Usa maschera fornita dall'utente
            mask = mask.float().to(device)

        # === PREPARAZIONE FEATURE PER CONTRASTIVE LEARNING ===
        contrast_count = features.shape[1]  # Numero di viste (n_views = 3 in MPK-GNN)
        
        # Concatena tutte le viste in un unico tensore
        # Da [batch_size, n_views, feature_dim] a [batch_size * n_views, feature_dim]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # === SELEZIONE ANCHOR FEATURES ===
        if self.contrast_mode == 'one':
            # Usa solo la prima vista come anchor
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            # Usa tutte le viste come anchor (default)
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Modalità sconosciuta: {}'.format(self.contrast_mode))

        # === CALCOLO SIMILARITÀ (DOT PRODUCT) ===
        # Calcola similarità coseno tra anchor e tutte le feature di contrasto
        # anchor_dot_contrast[i,j] = sim(anchor_i, contrast_j) / temperature
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # === STABILITÀ NUMERICA ===
        # Sottrae il massimo per evitare overflow nell'esponenziale
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # === ESTENSIONE MASCHERA PER MULTIPLE VISTE ===
        # Replica la maschera per gestire anchor e contrast multipli
        # Da [batch_size, batch_size] a [batch_size * anchor_count, batch_size * contrast_count]
        mask = mask.repeat(anchor_count, contrast_count)

        # === MASCHERA PER ESCLUDERE AUTO-CONTRASTI ===
        # Crea una maschera per evitare che un campione contrasti con se stesso
        # logits_mask[i,j] = 0 se i-esimo anchor coincide con j-esimo contrast (stessa posizione)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,  # Dimensione lungo cui fare lo scatter (colonne)
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),  # Indici diagonali
            0   # Valore da inserire (0 per mascherare)
        )  # Imposta la diagonale a 0 per escludere auto-contrasti
        
        # Applica entrambe le maschere: similarità di classe E non auto-contrasto
        mask = mask * logits_mask  # Moltiplica elemento per elemento

        # === CALCOLO LOG-PROBABILITÀ ===
        # Calcola il denominatore del softmax (somma di tutti gli esponenziali)
        exp_logits = torch.exp(logits) * logits_mask  # Applica maschera per escludere auto-contrasti
        
        # Calcola log-probabilità: log(exp(logit_ij) / sum_k(exp(logit_ik)))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # === CALCOLO LOSS FINALE ===
        # Calcola la media delle log-probabilità sui campioni positivi
        # Per ogni anchor, calcola: (1/|P(i)|) * sum_{p in P(i)} log(p_ip)
        # dove P(i) è l'insieme dei campioni positivi per l'anchor i
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # |P(i)| = mask.sum(1)

        # Calcola la loss finale con normalizzazione della temperatura
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        
        # Prende la media su tutti gli anchor nel batch
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
