# Cambiamento a pytorch geometric
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) per la classificazione di campioni su grafi
    
    Questa classe implementa una GCN a due layer che opera su grafi di similarità
    per classificare campioni biologici basandosi su caratteristiche multi-omiche integrate.
    La rete utilizza convoluzioni grafiche per aggregare informazioni dai nodi vicini
    nel grafo di similarità.
    
    Architettura:
    - Due layer di convoluzione grafica (GraphConvolution)
    - Funzioni di attivazione ELU per non linearità
    - Dropout per regolarizzazione e prevenzione dell'overfitting
    - Layer lineare finale per la classificazione
    """
    
    def __init__(self, n_in, n_hid, n_out, dropout=None):
        """
        Inizializza la Graph Convolutional Network
        
        Args:
            n_in (int): Dimensione delle caratteristiche di input per ciascun nodo
            n_hid (int): Dimensione del layer nascosto (hidden layer)
            n_out (int): Numero di classi di output per la classificazione
            dropout (float): Tasso di dropout per la regolarizzazione (default: None)
        
        Note:
            Il dropout aiuta a prevenire l'overfitting rimuovendo casualmente
            una frazione dei neuroni durante l'addestramento
        """
        super(GCN, self).__init__()
        
        # Primo layer di convoluzione grafica: input -> hidden
        self.gc1 = GCNConv(n_in, n_hid)

        # Secondo layer di convoluzione grafica: hidden -> hidden
        self.gc2 = GCNConv(n_hid, n_hid)

        # Layer di dropout per regolarizzazione dopo ogni convoluzione
        self.dp1 = nn.Dropout(dropout)  # Dropout dopo il primo layer
        self.dp2 = nn.Dropout(dropout)  # Dropout dopo il secondo layer
        
        # Layer lineare finale per la classificazione: hidden -> output_classes
        self.fc = nn.Linear(n_hid, n_out)
        
        # Salva il tasso di dropout come attributo
        self.dropout = dropout

    def forward(self, input, edge_index):
        """
        Metodo forward per il passaggio in avanti attraverso la GCN
        
        Questo metodo implementa il flusso completo di dati attraverso la rete:
        1. Prima convoluzione grafica con attivazione ELU e dropout
        2. Seconda convoluzione grafica con attivazione ELU e dropout  
        3. Classificazione finale tramite layer lineare
        
        Args:
            input (torch.Tensor): Caratteristiche dei nodi del grafo
                                 Forma: (n_nodes, n_features)
            adj (torch.Tensor): Matrice di adiacenza normalizzata del grafo
                               Forma: (n_nodes, n_nodes)
        
        Returns:
            torch.Tensor: Logits per la classificazione di ciascun nodo
                         Forma: (n_nodes, n_classes)
        
        Note:
            La funzione ELU (Exponential Linear Unit) è utilizzata per introdurre
            non linearità mantenendo output negativi per una migliore convergenza.
            Il dropout è applicato solo durante l'addestramento.
        """
        # Primo layer di convoluzione grafica
        # Aggrega le caratteristiche dai nodi vicini secondo la matrice di adiacenza
        x = self.gc1(input, edge_index)
        
        # Applica la funzione di attivazione ELU per introdurre non linearità
        # ELU: f(x) = x se x > 0, α(e^x - 1) se x ≤ 0 (con α=1)
        x = F.elu(x)
        
        # Applica dropout per regolarizzazione (solo durante l'addestramento)
        x = self.dp1(x)
        
        # Secondo layer di convoluzione grafica
        # Raffina ulteriormente le rappresentazioni aggregate
        x = self.gc2(x, edge_index)
        
        # Seconda attivazione ELU
        x = F.elu(x)
        
        # Secondo dropout per regolarizzazione
        x = self.dp2(x)

        # Layer lineare finale per la classificazione
        # Trasforma le caratteristiche apprese in logits per ciascuna classe
        x = self.fc(x)

        return x