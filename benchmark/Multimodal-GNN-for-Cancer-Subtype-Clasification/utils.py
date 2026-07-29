## IMPORTAZIONE DELLE LIBRERIE NECESSARIE
# Librerie per manipolazione dati e calcolo numerico
import numpy as np               # operazioni numeriche su array multidimensionali
import pandas as pd              # manipolazione di dataframe e dati tabulari
import scipy.sparse as sp        # gestione efficiente di matrici sparse (con molti zeri)
from random import sample        # campionamento casuale per dropout

# Librerie PyTorch per tensori e deep learning
import torch                     # framework per deep learning
import argparse                  # parsing argomenti da linea di comando

## ===== FUNZIONI PER LA SELEZIONE DEI GENI AD ALTA VARIANZA =====

def high_variance_expression_gene(expression_variance_path, non_null_path, num_gene, singleton=False):
    """
    Seleziona i geni con maggiore varianza nell'espressione genica.
    I geni con alta varianza sono più informativi per distinguere tra diversi tipi di campioni.
    
    Args:
        expression_variance_path: percorso del file con le varianze dell'espressione genica
        non_null_path: percorso del file con geni validi nella rete biologica
        num_gene: numero di geni da selezionare (es. top 1000)
        singleton: se True, considera solo geni presenti nella rete biologica
    
    Returns:
        gene_list: nomi dei geni selezionati
        gene_list_index: indici numerici dei geni selezionati
    """
    # Carica il file con le varianze precalcolate per ogni gene
    gene_variance = pd.read_csv(expression_variance_path, sep='\t', index_col=0, header=0)

    if singleton:
        # Modalità singleton: considera solo geni presenti nella rete biologica
        non_null_row = pd.read_csv(non_null_path, sep=',', header=0)
        gene_variance['id'] = range(gene_variance.shape[0])  # aggiungi indici numerici
        
        # Filtra per mantenere solo i geni presenti nella rete biologica
        gene_variance_non_null = gene_variance.loc[gene_variance.index.isin(non_null_row['gene']),:]
        
        # Seleziona i top N geni per varianza tra quelli validi
        gene_list = gene_variance_non_null.nlargest(num_gene, 'variance').index
        gene_variance_non_null.index = gene_variance_non_null['id']
        gene_list_index = gene_variance_non_null.nlargest(num_gene, 'variance').index
    else:
        # Modalità normale: seleziona i top N geni per varianza da tutti i geni
        gene_list = gene_variance.nlargest(num_gene, 'variance').index      # nomi dei geni
        gene_variance.index = range(gene_variance.shape[0])                 # reindicizza con numeri
        gene_list_index = gene_variance.nlargest(num_gene, 'variance').index  # indici numerici
    
    return gene_list, gene_list_index

def get_mirna_inner_connection(mirna_connection):
    """
    Crea connessioni interne tra miRNA basate sui loro target comuni.
    Se due miRNA regolano gli stessi geni, vengono considerati correlati e connessi tra loro.
    
    Args:
        mirna_connection: matrice miRNA-gene dove righe=geni, colonne=miRNA
                         Valore 1 indica che il miRNA regola quel gene
    
    Returns:
        matrice di adiacenza miRNA-miRNA dove 1 indica connessione tra miRNA
    """
    mirna_connection_row = []  # lista degli indici di riga per le connessioni
    mirna_connection_col = []  # lista degli indici di colonna per le connessioni
    
    # Per ogni gene (riga), trova quali miRNA lo regolano
    for i in range(mirna_connection.shape[0]):
        current_indices = np.nonzero(mirna_connection[i,])  # trova miRNA che regolano questo gene
        column_indexes = current_indices[1]                 # estrai gli indici dei miRNA
        
        # Se più di un miRNA regola questo gene, li connettiamo tra loro
        if len(column_indexes) > 1:
            # Crea tutte le possibili coppie di miRNA che condividono questo gene
            for x in range(len(column_indexes)):
                for y in range(x, len(column_indexes)):  # evita duplicati usando x come limite inferiore
                    mirna_connection_row.append(column_indexes[x])
                    mirna_connection_col.append(column_indexes[y])
    
    # Crea matrice sparse con le connessioni miRNA-miRNA
    mirna_connection_data = [1] * len(mirna_connection_row)  # tutti i pesi = 1 (connessione binaria)
    mirna_connection_adj = sp.csr_matrix(
        (mirna_connection_data, (mirna_connection_row, mirna_connection_col)),
        shape=(mirna_connection.shape[1], mirna_connection.shape[1])  # dimensione: num_mirna × num_mirna
    )
    
    # Estrai le connessioni non zero e ricrea la matrice per assicurare simmetria
    mirna_index = mirna_connection_adj.nonzero()
    mirna_row = mirna_index[0]
    mirna_col = mirna_index[1]
    mirna_data = [1] * len(mirna_row)
    
    # Matrice finale simmetrica delle connessioni miRNA-miRNA
    mirna_adj = sp.csr_matrix(
        (mirna_data, (mirna_row, mirna_col)),
        shape=(mirna_connection.shape[1], mirna_connection.shape[1])
    )
    
    return mirna_adj.toarray()  # restituisce matrice densa


## ===== FUNZIONI PER CARICAMENTO DEI DATI OMICI =====

def load_exp_and_real_mirna_data(expression_data_path, mirna_data_path):
    """
    Carica dati di espressione genica e miRNA da file TSV.
    
    Args:
        expression_data_path: percorso del file con dati di espressione genica
        mirna_data_path: percorso del file con dati miRNA
    
    Returns:
        expression_data: DataFrame con espressione genica (campioni × geni)
        mirna_data: DataFrame con livelli miRNA (campioni × miRNA)
    """
    # Carica dati di espressione genica: ogni riga = campione, ogni colonna = gene
    expression_data = pd.read_csv(expression_data_path, sep='\t', index_col=0, header=0)
    
    # Carica dati miRNA: ogni riga = campione, ogni colonna = miRNA specifico
    mirna_data = pd.read_csv(mirna_data_path, sep='\t', index_col=0, header=0)
    
    return expression_data, mirna_data

def load_exp_cnv_and_real_mirna_data(expression_data_path, cnv_data_path, mirna_data_path):
    """
    Carica dati multimodali: espressione genica, CNV (Copy Number Variations) e miRNA.
    
    Args:
        expression_data_path: percorso dati espressione genica
        cnv_data_path: percorso dati CNV (variazioni numero copie cromosomiche)
        mirna_data_path: percorso dati miRNA
    
    Returns:
        expression_data: DataFrame espressione genica
        cnv_data: DataFrame CNV normalizzato [0,1]
        mirna_data: DataFrame miRNA
    """
    # Carica dati di espressione genica
    expression_data = pd.read_csv(expression_data_path, sep='\t', index_col=0, header=0)
    print(f"Expression data shape before dropping columns: {expression_data.shape}")
    
    # Carica dati CNV (Copy Number Variations)
    cnv_data = pd.read_csv(cnv_data_path, sep='\t', index_col=0, header=0)
    print(f"CNV data shape before dropping columns: {cnv_data.shape}")
    
    # Rimuovi colonne non necessarie (etichette e identificatori campioni)
    cnv_data = cnv_data.drop(['icluster_cluster_assignment','sample'], axis=1)
    print(f"CNV data shape before normalization: {cnv_data.shape}")
    
    # Normalizza dati CNV nell'intervallo [0,1] usando min-max scaling
    cnv_data = (cnv_data - cnv_data.min().min()) / (cnv_data.max().max() - cnv_data.min().min())
    
    # Carica dati miRNA
    mirna_data = pd.read_csv(mirna_data_path, sep='\t', index_col=0, header=0)
    
    return expression_data, cnv_data, mirna_data

## ===== FUNZIONI PER PREPROCESSING E COSTRUZIONE GRAFI =====

def down_sampling_exp_and_real_mirna_data(expression_variance_path, 
                                            expression_data, 
                                            mirna_data, 
                                            omic_mode, 
                                            non_null_index_path, 
                                            shuffle_index_path, 
                                            adjacency_matrix_path, 
                                            mirna_to_gene_matrix_path, 
                                            gene_gene, 
                                            mirna_gene, 
                                            mirna_mirna, 
                                            number_gene,  
                                            singleton=False):
    """
    Preprocessa i dati omici e costruisce la rete biologica per modalità 0,1,2.
    
    Args:
        expression_variance_path: percorso file varianze geni
        expression_data: dati espressione genica
        mirna_data: dati miRNA
        omic_mode: 0=solo mRNA, 1=solo miRNA, 2=mRNA+miRNA
        non_null_index_path: geni validi nella rete biologica
        shuffle_index_path: indici per randomizzazione dataset
        adjacency_matrix_path: matrice connessioni gene-gene
        mirna_to_gene_matrix_path: connessioni miRNA-gene
        gene_gene: usa connessioni gene-gene
        mirna_gene: usa connessioni miRNA-gene
        mirna_mirna: usa connessioni miRNA-miRNA
        number_gene: numero di geni da selezionare
        singleton: include nodi isolati
    
    Returns:
        supra_adj: matrice di adiacenza finale della rete
        data: dati processati
        labels: etichette dei campioni
        shuffle_index: indici per randomizzazione
    """
    ## SELEZIONE DEI GENI AD ALTA VARIANZA
    # Seleziona i geni più informativi basandosi sulla varianza dell'espressione
    high_variance_gene_list, high_variance_gene_index = high_variance_expression_gene(
        expression_variance_path, non_null_index_path, number_gene, singleton)
    # esporta i geni ad alta varianza
    print(f'Selezionati {len(high_variance_gene_list)} geni ad alta varianza.')
    high_variance_gene_list_series = pd.Series(high_variance_gene_list)
    high_variance_gene_list_series.to_csv('high_variance_genes.csv', index=False)

    ## ESTRAZIONE DELLE ETICHETTE
    # Estrae le etichette prima di filtrare le colonne
    labels = expression_data['icluster_cluster_assignment']
    labels = labels - 1  # converte etichette da 1-based a 0-based per PyTorch
    
    # Verifica coerenza dimensionale tra i dataset
    if expression_data.shape[0] == mirna_data.shape[0]:
        print('Numero di campioni in Espressione e miRNA coincidenti.')
    
    ## FILTRAGGIO DEI DATI PER GENI SELEZIONATI
    # Mantiene solo i geni ad alta varianza selezionati
    expression_data = expression_data.loc[:,high_variance_gene_list]
    expression_data.index = range(expression_data.shape[0])  # reindicizza numericamente
    
    ## COMBINAZIONE DEI DATI OMICI BASATA SULLA MODALITÀ
    if omic_mode > 0:
        # Modalità che includono miRNA (1: solo miRNA, 2: mRNA+miRNA)
        mirna_data.index = range(mirna_data.shape[0])
        data = pd.concat([expression_data, mirna_data], axis=1)  # concatena orizzontalmente
        num_samples = mirna_data.shape[0]
        
        # Reshape per formato compatibile con GNN: [num_campioni, num_features, 1]
        data = np.asarray(data).reshape(num_samples, -1, 1)
        print(f'Shape dati combinati: {data.shape}')
    else:
        # Modalità 0: solo dati di espressione genica
        data = np.asarray(expression_data).reshape(expression_data.shape[0], -1, 1)

    ## CARICAMENTO E PREPROCESSING DELLA RETE GENE-GENE
    if gene_gene:
        # Carica matrice di adiacenza gene-gene precomputata (es. da BioGRID database)
        gene_gene_adj = sp.load_npz(adjacency_matrix_path)
        gene_gene_adj_mat = gene_gene_adj.todense()  # converte in matrice densa
        
        # Normalizza i pesi delle connessioni nell'intervallo [0,1]
        gene_gene_adj_mat = gene_gene_adj_mat / (gene_gene_adj_mat.max() - gene_gene_adj_mat.min())
        
        # Filtra per mantenere solo connessioni tra geni selezionati ad alta varianza
        gene_gene_adj_selected = gene_gene_adj_mat[high_variance_gene_index,:]  # filtra righe
        gene_gene_adj_selected = gene_gene_adj_selected[:,high_variance_gene_index]  # filtra colonne
    else:
        # Se non usiamo connessioni gene-gene, crea matrice identità (solo self-loops)
        gene_gene_adj_selected = np.identity(number_gene)
    
    ## CARICAMENTO DELLA RETE miRNA-GENE
    if mirna_gene or mirna_mirna:
        # Carica matrice delle interazioni miRNA-gene validate sperimentalmente
        mirna_gene_adj = sp.load_npz(mirna_to_gene_matrix_path)
        mirna_gene_adj = mirna_gene_adj.todense()
        
        # Filtra per mantenere solo connessioni ai geni selezionati
        mirna_gene_adj_selected = mirna_gene_adj[high_variance_gene_index,:]
    else:
        # Se non usiamo connessioni miRNA-gene, crea matrice zero
        mirna_gene_adj_selected = np.zeros((number_gene, 100))  # 100 = numero di miRNA

    ## COSTRUZIONE DELLA RETE SUPRA-ADIACENTE BASATA SULLA MODALITÀ
    if omic_mode == 2:
        # Modalità 2: mRNA + miRNA - crea rete eterogenea completa
        
        # Parte superiore: [connessioni gene-gene | connessioni gene-miRNA]
        top_supra_adj = np.concatenate((gene_gene_adj_selected, mirna_gene_adj_selected), axis=1)
        
        if mirna_mirna:
            # Include connessioni interne miRNA-miRNA basate su target comuni
            print(f'Shape matrice miRNA-gene: {mirna_gene_adj_selected.shape}')
            print(f'Shape connessioni miRNA-miRNA: {get_mirna_inner_connection(mirna_gene_adj_selected).shape}')
            
            # Parte inferiore: [connessioni miRNA-gene | connessioni miRNA-miRNA]
            bottom_supra_adj = np.concatenate((np.transpose(mirna_gene_adj_selected), 
                                             get_mirna_inner_connection(mirna_gene_adj_selected)), axis=1)
        else:
            # Solo self-loops per miRNA (matrice identità)
            bottom_supra_adj = np.concatenate((np.transpose(mirna_gene_adj_selected), 
                                             np.identity(100)), axis=1)
    
        # Combina verticalmente per creare la rete supra-adiacente completa
        # Struttura finale: [ gene-gene   gene-miRNA ]
        #                  [ miRNA-gene  miRNA-miRNA]
        supra_adj = np.concatenate((top_supra_adj, bottom_supra_adj), axis=0)

        # Converte in formato sparse per efficienza computazionale
        supra_adj = sp.csr_matrix(supra_adj)

        if singleton:
            # Aggiunge self-loops per tutti i nodi (inclusi quelli isolati)
            print('Includendo nodi singleton con self-loops')
            supra_adj = supra_adj + sp.eye(supra_adj.shape[0])
            
    elif omic_mode == 0:
        # Modalità 0: solo mRNA - usa solo connessioni gene-gene
        supra_adj = sp.csr_matrix(gene_gene_adj_selected)
        
    elif omic_mode == 1:
        # Modalità 1: solo miRNA - usa solo connessioni miRNA-miRNA
        supra_adj = sp.csr_matrix(get_mirna_inner_connection(mirna_gene_adj_selected))

    ## CARICAMENTO DEGLI INDICI DI RANDOMIZZAZIONE
    shuffle_index = pd.read_csv(shuffle_index_path, sep='\t', index_col=0, header=0)
    
    return supra_adj, np.asarray(data), labels.to_numpy(), shuffle_index.to_numpy()

def down_sampling_exp_cnv_and_real_mirna_data(expression_variance_path, 
                                            expression_data, 
                                            cnv_data,
                                            mirna_data, 
                                            omic_mode, 
                                            non_null_index_path, 
                                            shuffle_index_path, 
                                            adjacency_matrix_path, 
                                            mirna_to_gene_matrix_path, 
                                            gene_gene, 
                                            mirna_gene, 
                                            mirna_mirna, 
                                            number_gene,  
                                            singleton=False):
    """
    Preprocessa dati omici e costruisce la rete biologica per modalità 3,4 che includono CNV.
    
    Args:
        Similar alla funzione precedente ma include cnv_data
        omic_mode: 3=mRNA+CNV, 4=mRNA+CNV+miRNA
    
    Returns:
        Same outputs as previous function but with CNV data incorporated
    """
    print('Processing expression, CNV and miRNA data...')
    ## SELEZIONE DEI GENI AD ALTA VARIANZA
    high_variance_gene_list, high_variance_gene_index = high_variance_expression_gene(
        expression_variance_path, non_null_index_path, number_gene, singleton)
    # esporta i geni ad alta varianza
    print(f'Selezionati {len(high_variance_gene_list)} geni ad alta varianza.')
    high_variance_gene_list_series = pd.Series(high_variance_gene_list)
    print(high_variance_gene_list_series.head())
    print(high_variance_gene_list_series.shape)
    high_variance_gene_list_series.to_csv('high_variance_genes.csv', index=False)

    ## ESTRAZIONE DELLE ETICHETTE
    labels = expression_data['icluster_cluster_assignment']
    labels = labels - 1  # conversione 1-based -> 0-based
    
    if expression_data.shape[0] == mirna_data.shape[0]:
        print('Exp and miRNA sample numbers match.')
    ## filter multi-omics data by gene list
    expression_data = expression_data.loc[:,high_variance_gene_list]
    cnv_data = cnv_data.loc[:,high_variance_gene_list]
    expression_data.index = range(expression_data.shape[0])
    mirna_data.index = range(mirna_data.shape[0])
    cnv_data.index = range(cnv_data.shape[0])

    ##  only pad CNV data when using Exp, CNV and miRNA
    if omic_mode == 4:
        cnv_padding = pd.DataFrame(np.zeros((mirna_data.shape[0],mirna_data.shape[1])))
        cnv_data_padded = pd.concat([cnv_data, cnv_padding], axis=1)

        data = pd.concat([expression_data, mirna_data], axis=1)
        
        num_samples = mirna_data.shape[0]
        ## concatenate expr and mirna
        data= np.asarray(data).reshape(num_samples, -1 ,1)
        cnv_data_padded = np.asarray(cnv_data_padded).reshape(num_samples, -1 ,1)
        data = np.concatenate([data,cnv_data_padded], axis=2)
        print(data.shape)
    else:
        num_samples = expression_data.shape[0]
        data = np.array(expression_data).reshape(num_samples, -1 ,1)
        print(data.shape)
        cnv_data = np.asarray(cnv_data).reshape(num_samples, -1 ,1)
        print(data.shape)
        data = np.concatenate([data,cnv_data], axis=2)
        print(data.shape)

    ## ===== CARICAMENTO E PREPROCESSING MATRICE DI ADIACENZA GENE-GENE =====
    if gene_gene:
        # Carica la matrice di adiacenza gene-gene precomputata (formato sparse .npz)
        # Questa matrice rappresenta le interazioni biologiche note tra geni (es. da BioGRID)
        gene_gene_adj = sp.load_npz(adjacency_matrix_path)
        
        # Converte da formato sparse a denso per manipolazioni numeriche
        gene_gene_adj_mat = gene_gene_adj.todense()
        
        # Normalizza i pesi delle connessioni nell'intervallo [0,1] usando min-max scaling
        # Questo assicura che tutti i pesi abbiano la stessa scala indipendentemente dai valori originali
        gene_gene_adj_mat = gene_gene_adj_mat / (gene_gene_adj_mat.max() - gene_gene_adj_mat.min())
        
        # Filtra la matrice per mantenere solo i geni ad alta varianza selezionati
        # Prima filtra le righe (geni sorgente), poi le colonne (geni target)
        gene_gene_adj_selected = gene_gene_adj_mat[high_variance_gene_index,:]  # Seleziona righe
        gene_gene_adj_selected = gene_gene_adj_selected[:,high_variance_gene_index]  # Seleziona colonne
    else:
        # Se non usiamo connessioni gene-gene, crea una matrice identità
        # Questo significa che ogni gene è connesso solo a se stesso (self-loops)
        gene_gene_adj_selected = np.identity(number_gene)
    
    ## ===== CARICAMENTO E PREPROCESSING MATRICE miRNA-GENE =====
    if mirna_gene or mirna_mirna:
        # Carica la matrice delle interazioni miRNA-gene validate (es. da miRDB, TargetScan)
        # Questa matrice indica quali miRNA regolano quali geni
        mirna_gene_adj = sp.load_npz(mirna_to_gene_matrix_path)
        mirna_gene_adj = mirna_gene_adj.todense()
        
        # Filtra per mantenere solo le connessioni ai geni ad alta varianza selezionati
        # Struttura: righe=geni selezionati, colonne=tutti i miRNA
        mirna_gene_adj_selected = mirna_gene_adj[high_variance_gene_index,:]
    else:
        # Se non usiamo connessioni miRNA-gene, crea matrice zero
        # Dimensioni: [numero_geni_selezionati × 100_miRNA]
        mirna_gene_adj_selected = np.zeros((number_gene,100))

    ## ===== COSTRUZIONE RETE SUPRA-ADIACENTE MULTI-LAYER =====
    if omic_mode > 3:
        # Modalità 4: mRNA + CNV + miRNA - costruisce rete eterogenea completa
        
        if mirna_mirna:
            # Debug: stampa dimensioni delle matrici per verifica coerenza
            print(mirna_gene_adj_selected.shape)
            print(get_mirna_inner_connection(mirna_gene_adj_selected).shape)
            
            if mirna_gene:
                # CASO 1: Include sia connessioni miRNA-gene che miRNA-miRNA
                # Parte superiore della matrice supra-adiacente: [gene-gene | gene-miRNA]
                top_supra_adj = np.concatenate((gene_gene_adj_selected, mirna_gene_adj_selected), axis=1)
                
                # Parte inferiore: [miRNA-gene | miRNA-miRNA_interne]
                # miRNA-miRNA_interne = connessioni tra miRNA basate su target comuni
                bottom_supra_adj = np.concatenate((np.transpose(mirna_gene_adj_selected), 
                                                 get_mirna_inner_connection(mirna_gene_adj_selected)), axis=1)
            else:
                # CASO 2: Solo connessioni miRNA-miRNA (no miRNA-gene)
                # Parte superiore: [gene-gene | zero_padding]
                top_supra_adj = np.concatenate((gene_gene_adj_selected, np.zeros((number_gene,100))), axis=1)
                
                # Parte inferiore: [zero_padding | miRNA-miRNA_interne]
                bottom_supra_adj = np.concatenate((np.transpose(np.zeros((number_gene,100))), 
                                                 get_mirna_inner_connection(mirna_gene_adj_selected)), axis=1)
        else:
            # Non include connessioni miRNA-miRNA interne
            if mirna_gene:
                # CASO 3: Solo connessioni miRNA-gene (no miRNA-miRNA)
                # Parte superiore: [gene-gene | gene-miRNA]
                top_supra_adj = np.concatenate((gene_gene_adj_selected, mirna_gene_adj_selected), axis=1)
                
                # Parte inferiore: [miRNA-gene | identità_miRNA]
                # Identità per miRNA = solo self-loops, no connessioni tra miRNA diversi
                bottom_supra_adj = np.concatenate((np.transpose(mirna_gene_adj_selected), 
                                                 np.identity(100)), axis=1)
            else:
                # CASO 4: Configurazione di default (dovrebbe essere equivalente al CASO 3)
                top_supra_adj = np.concatenate((gene_gene_adj_selected, mirna_gene_adj_selected), axis=1)
                bottom_supra_adj = np.concatenate((np.transpose(mirna_gene_adj_selected), 
                                                 np.identity(100)), axis=1)
        
        # Combina verticalmente le parti superiore e inferiore per creare la matrice completa
        # Struttura finale della rete supra-adiacente:
        #                    [Geni]    [miRNA]
        #        [Geni]   [gene-gene] [gene-miRNA]
        #        [miRNA]  [miRNA-gene][miRNA-miRNA]
        supra_adj = np.concatenate((top_supra_adj, bottom_supra_adj), axis=0)

        # Converte la matrice densa in formato sparse per efficienza computazionale
        # Le GNN lavorano meglio con matrici sparse per grafi con molti nodi
        supra_adj = sp.csr_matrix(supra_adj)
        
    elif omic_mode == 3:
        # Modalità 3: solo mRNA + CNV - usa solo connessioni gene-gene
        # Non ci sono miRNA, quindi la rete è solo mono-layer genica
        supra_adj = sp.csr_matrix(gene_gene_adj_selected)

    ## ===== GESTIONE NODI ISOLATI (SINGLETON) =====
    if singleton:
        print('including singleton')
        # Aggiunge self-loops a tutti i nodi per includere anche quelli isolati
        # Questo previene problemi durante il training GNN con nodi non connessi
        # sp.eye crea una matrice identità sparse delle stesse dimensioni
        supra_adj = supra_adj + sp.eye(supra_adj.shape[0])

    ## ===== CARICAMENTO INDICI DI RANDOMIZZAZIONE =====
    # Carica gli indici per la randomizzazione del dataset (train/validation/test split)
    # Questi indici sono precomputati per garantire riproducibilità degli esperimenti
    shuffle_index = pd.read_csv(shuffle_index_path, sep='\t', index_col=0, header=0)
    
    # Ritorna tutti i componenti necessari per il training della GNN:
    # - supra_adj: matrice di adiacenza della rete biologica multi-layer
    # - data: dati omici processati e concatenati
    # - labels: etichette dei sottotipi tumorali (0-based)
    # - shuffle_index: indici per split dataset
    return supra_adj, np.asarray(data), labels.to_numpy(), shuffle_index.to_numpy()

## ===== FUNZIONI UTILITY DI SUPPORTO =====

def dropout_data(data, labels, drop_out=0.6):
    """
    Riduce casualmente il dataset di training per bilanciamento o test.
    
    Args:
        data: dati di input
        labels: etichette corrispondenti
        drop_out: percentuale di dati da mantenere (0.6 = mantiene 60%)
    
    Returns:
        dropped_data: sottoinsieme casuale dei dati
        dropped_labels: etichette corrispondenti
    """
    # Campiona casualmente gli indici dei campioni da mantenere
    dropout_index = sample(range(len(labels)), round(len(labels)*drop_out))
    dropped_data = data[dropout_index,:]
    dropped_labels = labels[dropout_index]
    return dropped_data, dropped_labels

def accuracy(output, labels):
    """
    Calcola l'accuratezza delle predizioni rispetto alle etichette vere.
    
    Args:
        output: tensor delle probabilità predette per ogni classe [batch_size, num_classes]
        labels: tensor delle etichette vere [batch_size]
    
    Returns:
        accuratezza come frazione dei campioni classificati correttamente
    """
    # Trova la classe con probabilità massima per ogni campione
    preds = output.max(1)[1].type_as(labels)  # indici delle classi predette
    correct = preds.eq(labels).double()       # confronta con etichette vere (True/False)
    correct = correct.sum()                   # conta le predizioni corrette
    return correct / len(labels)              # percentuale di accuratezza

def edge_filter(adj, top=10, which_axis=1):
    """
    Filtra una matrice di adiacenza mantenendo solo le top N connessioni per nodo.
    
    Args:
        adj: matrice di adiacenza
        top: numero di top connessioni da mantenere per nodo
        which_axis: asse lungo cui ordinare (1 = per riga)
    
    Returns:
        matrice filtrata con solo le connessioni più forti
    """
    adj = np.array(adj)
    print(f'Shape matrice originale: {adj.shape}')
    # Maschera che mantiene solo i top N valori più alti per ogni riga/colonna
    new_adj = adj * (np.argsort(np.argsort(adj, axis=which_axis)) >= adj.shape[1] - top)
    return new_adj

def disassemble_edge_weights(edge_weights, edge_index, num_gene, num_attributes):
    """
    Converte pesi scalari delle connessioni in vettori di attributi multidimensionali.
    Distingue tra diversi tipi di connessioni biologiche.
    
    Args:
        edge_weights: pesi scalari delle connessioni
        edge_index: indici delle connessioni [2, num_edges]
        num_gene: numero di geni (per distinguere da miRNA)
        num_attributes: dimensioni del vettore attributi
    
    Returns:
        tensor di attributi multidimensionali per ogni connessione
    """
    edge_index_transposed = edge_index.T
    edge_attributes = np.zeros((edge_index.shape[1], num_attributes))
    
    for idx, x in enumerate(edge_index_transposed):
        # Classifica il tipo di connessione basandosi sugli indici dei nodi
        if x[0] < num_gene and x[1] < num_gene:
            # Connessione gene-gene: attributo dimensione 0
            edge_attributes[idx, 0] = edge_weights[idx]
        elif x[0] < num_gene and x[1] >= num_gene:
            # Connessione gene-miRNA: attributo dimensione 1
            edge_attributes[idx, 1] = edge_weights[idx]
        elif x[0] >= num_gene and x[1] < num_gene:
            # Connessione miRNA-gene: attributo dimensione 1
            edge_attributes[idx, 1] = edge_weights[idx]
        else:
            # Connessione miRNA-miRNA: attributo dimensione 0
            edge_attributes[idx, 0] = edge_weights[idx]
    
    return torch.Tensor(edge_attributes)

## ===== FUNZIONI DI CONFIGURAZIONE E VALIDAZIONE =====

def str2bool(v):
    """
    Converte stringhe in valori booleani per argparse.
    Utile per parametri da linea di comando tipo --parallel true/false.
    
    Args:
        v: valore da convertire (stringa o booleano)
    
    Returns:
        valore booleano corrispondente
    
    Raises:
        ArgumentTypeError: se il valore non è riconosciuto come booleano
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Valore booleano atteso.')

def omic_mode_translation(omic_mode):
    """
    Traduce la modalità omica nel numero di tipi di dati utilizzati.
    
    Args:
        omic_mode: codice modalità (0-4)
    
    Returns:
        numero di tipi di dati omici utilizzati
    """
    if omic_mode == 0:
        print('Utilizzando solo dati di Espressione Genica.')
        return 1
    elif omic_mode == 1:
        print('Utilizzando solo dati miRNA.')
        return 1
    elif omic_mode == 2:
        print('Utilizzando dati di Espressione e miRNA.')
        return 2
    elif omic_mode == 3:
        print('Utilizzando dati di Espressione e CNV.')
        return 2
    elif omic_mode == 4:
        print('Utilizzando dati di Espressione, CNV e miRNA.')
        return 3
    
def validate_network_choice(omic_mode, gene_gene, mirna_gene, mirna_mirna):
    """
    Valida e corregge le scelte di rete biologica in base alla modalità omica.
    Alcune connessioni non sono disponibili per certe modalità.
    
    Args:
        omic_mode: modalità dati omici (0-4)
        gene_gene: flag per connessioni gene-gene
        mirna_gene: flag per connessioni miRNA-gene
        mirna_mirna: flag per connessioni miRNA-miRNA
    
    Returns:
        tuple: (gene_gene_corrected, mirna_gene_corrected, mirna_mirna_corrected, num_mirna)
    """
    if omic_mode == 0:
        # Solo espressione genica: non ci sono miRNA disponibili
        if mirna_gene or mirna_mirna:
            print('Connessioni miRNA-Gene o miRNA-miRNA non disponibili usando solo dati di espressione.')
            return True, False, False, 0
        else:
            return gene_gene, mirna_gene, mirna_mirna, 0
            
    elif omic_mode == 1:
        # Solo miRNA: non ci sono geni disponibili
        if gene_gene:
            print('Connessioni Gene-Gene non disponibili usando solo dati miRNA.')
            return False, True, True, 100
        else:
            return gene_gene, mirna_gene, mirna_mirna, 100
            
    elif omic_mode == 3:
        # Espressione + CNV: non ci sono miRNA disponibili
        if mirna_gene or mirna_mirna:
            print('Connessioni miRNA-Gene o miRNA-miRNA non disponibili usando dati di espressione e CNV.')
            return True, False, False, 0
        else:
            return gene_gene, mirna_gene, mirna_mirna, 0
    else:
        # Modalità 2 o 4: tutti i tipi di connessioni disponibili
        return gene_gene, mirna_gene, mirna_mirna, 100

def filter_data_by_cancer_type(cancer_subtype_label_path, data, expression_data):
    """
    Filtra i dati per un tipo specifico di cancro e associa le etichette dei sottotipi.
    Utilizzata per classificazione di sottotipi tumorali (es. sottotipi di cancro al seno).
    
    Args:
        cancer_subtype_label_path: percorso file con etichette sottotipi per paziente
        data: dati omici processati
        expression_data: dati di espressione originali con ID pazienti
    
    Returns:
        tuple: (dati_filtrati, etichette_sottotipi)
    """
    # Carica file con etichette dei sottotipi per ogni paziente
    cancer_subtype_label = pd.read_csv(cancer_subtype_label_path, sep=',', header=0)
    
    # Filtra per mantenere solo pazienti presenti nei dati di espressione
    cancer_subtype_label = cancer_subtype_label[
        cancer_subtype_label['patient'].isin(expression_data['sample'].tolist())
    ]
    
    # Estrai lista dei pazienti dall'espressione genica che hanno etichette sottotipo
    expression_index = expression_data[
        expression_data['sample'].isin(cancer_subtype_label['patient'].tolist())
    ]['sample']

    ## VERIFICA COERENZA ORDINE PAZIENTI
    # Assicura che l'ordine dei pazienti sia identico in entrambi i dataset
    for i in range(cancer_subtype_label.shape[0]):
        if expression_index.iloc[i] != cancer_subtype_label.iloc[i, 0]:
            print('ERRORE: Ordine pazienti non corrispondente!')
            quit()

    ## FILTRAGGIO DEI DATI
    # Crea maschera booleana per selezionare solo campioni con etichette sottotipo
    subtype_sample_index = expression_data['sample'].isin(cancer_subtype_label['patient'].tolist())

    # Applica il filtro ai dati omici
    data = data[subtype_sample_index]
    
    # Estrai le etichette dei sottotipi (es. 'luminal_A', 'HER2+', 'basal')
    labels = cancer_subtype_label['subtype']

    return np.asarray(data), labels.to_numpy()
