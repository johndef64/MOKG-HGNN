import numpy as np
import pandas as pd
import scipy.sparse as sp
from random import sample
import torch
import argparse
import sklearn.metrics
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from multiomics_gnn.pancancer_prediction.preprocessing.feature_selection import gini_feature_selection

def community_detection_feature_selection(expression_data, biogrid_adj, num_gene):
    import networkx as nx
    adj = sp.load_npz(biogrid_adj).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    n_adj = adj.shape[0]
    print("Adjacency matrix loaded. Shape:", adj.shape, "Non-zero entries:", adj.nnz)
    # Assumiamo che le colonne di expression_data corrispondano ai nodi della matrice adj
    genes = expression_data.columns.tolist()
    G = nx.from_scipy_sparse_array(adj)
    
    print(f"Grafo creato con {G.number_of_nodes()} nodi e {G.number_of_edges()} archi.")

    # 2. Esegui la Community Detection (Louvain)
    # Trova le comunità di geni (es. pathway metabolici, ciclo cellulare)
    communities = nx.community.louvain_communities(G, seed=42)
    print(f"Trovate {len(communities)} comunità biologiche nel BioGRID.")

    # 3. Calcola la MAD (Median Absolute Deviation) per tutti i geni
    # Molto più robusta della varianza per i dataset sbilanciati
    mad_values = (expression_data - expression_data.median()).abs().median()

    selected_genes = []
    total_nodes = len(genes)

    # 4. Estrazione proporzionale da ogni comunità
    for i, comm in enumerate(communities):
        comm_list = list(comm) # Indici dei nodi nella comunità
        comm_size = len(comm_list)
        
        # Calcola quanti geni estrarre da questa comunità (quota proporzionale)
        quota = int(np.ceil((comm_size / total_nodes) * num_gene))
        
        # Recupera i nomi dei geni e le loro MAD
        comm_genes = [genes[idx] for idx in comm_list]
        comm_mad = mad_values[comm_genes].sort_values(ascending=False)
        
        # Prendi i top 'quota' geni per MAD in questa specifica comunità
        top_genes = comm_mad.head(quota).index.tolist()
        selected_genes.extend(top_genes)

    # 5. Pulizia finale (per assicurarci di avere ESATTAMENTE num_gene)
    selected_genes = list(set(selected_genes)) # rimuove eventuali duplicati
    
    if len(selected_genes) > num_gene:
        # Se abbiamo sbordato per via degli arrotondamenti (np.ceil), tagliamo i più deboli
        final_mad = mad_values[selected_genes].sort_values(ascending=False)
        selected_genes = final_mad.head(num_gene).index.tolist()
    elif len(selected_genes) < num_gene:
        # Se ne mancano alcuni, ripeschiamo i migliori in assoluto tra gli esclusi
        missing = num_gene - len(selected_genes)
        remaining_genes = list(set(genes) - set(selected_genes))
        top_remaining = mad_values[remaining_genes].sort_values(ascending=False).head(missing).index.tolist()
        selected_genes.extend(top_remaining)
    gene_list = selected_genes[:num_gene] # assicurati di avere esattamente num_gene
    gene_index = [expression_data.columns.get_loc(gene) for gene in gene_list]
    print(f"Selected {len(gene_list)} genes after community detection feature selection.")
    return gene_list, gene_index

def new_FDS(expression_data, biogrid_adj, num_gene):
    adj = sp.load_npz(biogrid_adj).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    n_adj = adj.shape[0]
    print("Adjacency matrix loaded. Shape:", adj.shape, "Non-zero entries:", adj.nnz)
    #2) calc variance
    variances = expression_data.var().sort_values(ascending=False)
    # 1. Identifica i nomi delle feature che superano la soglia
    high_variance_features = variances[variances > 4].index.tolist()

    # 2. Crea il nuovo dataset filtrato
    X_filtered = expression_data[high_variance_features]

    print(f"Feature rimosse: {len(expression_data.columns) - len(high_variance_features)}")
    print(f"Feature mantenute: {len(high_variance_features)}")

    x_filtered_idx = [expression_data.columns.get_loc(col) for col in X_filtered.columns]
    
    # 3) FILTRA LA MATRICE DI ADIACENZA
    # check the degree of these idx in the adj matrix
    adj_filtered = adj[x_filtered_idx, :][:, x_filtered_idx]
    print(adj_filtered.shape)
    import numpy as np
    import matplotlib.pyplot as plt

    # Calcolo del grado per ogni nodo (somma delle righe)
    # Se il grafo è diretto, questo è l'out-degree. 
    # Se è simmetrico (non diretto), riga o colonna è uguale.
    degrees = np.asarray(adj_filtered.sum(axis=1)).flatten()
    variances_filtered = variances.iloc[x_filtered_idx].values

    # Creiamo un DataFrame per facilitare l'analisi
    import pandas as pd
    node_analysis = pd.DataFrame({
        'feature_idx': x_filtered_idx,
        'degree': degrees,
        'variance': variances[expression_data.columns[x_filtered_idx]],
        'symbol': expression_data.columns[x_filtered_idx]
    }).sort_values(by='degree', ascending=False)
    # 4) PROPAGAZIONE DEL SEGNALE SUL GRAFO (GSP)
    from scipy import sparse

    # Aggiungiamo i self-loops per mantenere l'informazione locale del nodo durante la diffusione
    #adj_with_self = adj_filtered + sp.eye(adj_filtered.shape[0])
    adj_with_self = adj_filtered
    # Normalizzazione simmetrica della matrice di adiacenza: D^-0.5 * A * D^-0.5
    # Questo garantisce che la propagazione non "esploda" sui nodi ad altissimo grado
    d = np.array(adj_with_self.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(d, -0.5, where=d!=0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_norm = D_inv_sqrt @ adj_with_self @ D_inv_sqrt

    # Propagazione: calcoliamo lo "Smooth Variance Score"
    # Moltiplicando la matrice per il vettore varianza, ogni nodo riceve una quota 
    # della varianza dei suoi vicini pesata dalla forza della connessione
    v_signal = node_analysis['variance'].values
    propagated_variance = adj_norm.dot(v_signal)

    # Aggiungiamo il nuovo score al DataFrame
    node_analysis['propagated_variance'] = propagated_variance

    # Ranking Finale: Feature che sono importanti sia per segnale proprio che per posizione nel grafo
    node_analysis = node_analysis.sort_values(by='propagated_variance', ascending=False)

    print("Top 10 Feature dopo Propagazione su Grafo:")
    print(node_analysis[['symbol', 'variance', 'degree', 'propagated_variance']].head(10))

    # 5) SELEZIONE FINALE
    # Ora puoi tagliare su 'propagated_variance' invece che sulla varianza grezza
    # Ad esempio, prendiamo le top 500 feature "strutturalmente" significative
    #final_features = node_analysis.head(700)['symbol'].tolist()
    #X_final = expression_data[final_features]
    gene_list = node_analysis.head(num_gene)['symbol'].tolist()
    gene_index = [expression_data.columns.get_loc(gene) for gene in gene_list]
    return gene_list, gene_index

def variance_FS(expression_data, num_gene):
    ## filter gene list by expression variance
    X = expression_data
    #X = expression_data.drop(columns=['icluster_cluster_assignment'])
    gene_variance = X.var(axis=0, ddof=1).sort_values(ascending=False)
    gene_list = top_genes = gene_variance.index[:num_gene].tolist()
    # 3) indici numerici coerenti con X.columns (gene-only)
    col_index = {g: i for i, g in enumerate(X.columns)}
    top_idx = [col_index[g] for g in top_genes]
    
    return gene_list, top_idx

def high_variance_expression_gene(expression_variance_path, num_gene):
    ## filter gene list by expression variance
    gene_variance = pd.read_csv(expression_variance_path, sep='\t', index_col=0, header=0)
    ## load expression data
    gene_list = gene_variance.nlargest(num_gene, 'variance').index
    gene_variance.index = range(gene_variance.shape[0])
    gene_list_index = gene_variance.nlargest(num_gene, 'variance').index

    return gene_list, gene_list_index

def high_fvalue_expression_gene(expression_fvalue_path, num_gene):
    ## filter gene list by expression f-value
    gene_fvalue = pd.read_csv(expression_fvalue_path, sep='\t', index_col=0, header=0)
    print("Gene f-value data shape:", gene_fvalue.shape)
    print("Gene f-value columns:", gene_fvalue.columns)
    ## load expression data
    gene_list = gene_fvalue.nlargest(num_gene, 'f_value').index
    gene_fvalue.index = range(gene_fvalue.shape[0])
    gene_list_index = gene_fvalue.nlargest(num_gene, 'f_value').index

    return gene_list, gene_list_index

def high_fvalue_per_class(expression_data, num_gene):
    from sklearn.feature_selection import SelectKBest
    ## filter gene list by expression f-value per class
    from sklearn.feature_selection import f_classif
    print("Expression data shape:", expression_data.shape)
    print("expression columns:", expression_data.columns)
    labels = expression_data['icluster_cluster_assignment'].values.astype(int) - 1
    print("Performing feature selection based on f-value per class")
    expression_data_temp = expression_data.drop(columns=['icluster_cluster_assignment'])
    selected_features = set()
    #decide number of feature per class with weighs for each class, the sum must be num_gene
    weighs = np.bincount(labels)
    weighs = weighs / np.sum(weighs)
    # the sum of all feature per class must be num_gene and the number of feature mus be proportional to the weighs
    num_features_per_class = np.round(weighs * num_gene).astype(int)
    # count total features
    total_features = np.sum(num_features_per_class)
    print("Total features selected per class:", total_features)
    # if is different from num_gene adjust the number of features for the class with the highest weigh
    if total_features != num_gene:
        diff = num_gene - total_features
        max_weigh_class = np.argmax(weighs)
        num_features_per_class[max_weigh_class] += diff
        print(f"Adjusting number of features for class {max_weigh_class} by {diff} to match total {num_gene}")
    print("Number of features per class:", num_features_per_class)

    for class_label in np.unique(labels):
        binary_labels = (labels == class_label).astype(int)
        print(f"Performing feature selection for class {class_label}")
        print("Number of samples in class:", np.sum(binary_labels))
        print("Number of samples not in class:", len(binary_labels) - np.sum(binary_labels))
        selector = SelectKBest(score_func=f_classif, k=num_features_per_class[class_label])
        selector.fit(expression_data_temp.values, binary_labels)
        selected_indices = selector.get_support(indices=True)
        selected_features.update(expression_data_temp.columns[selected_indices])
        for idx in selected_indices:
            print(f"Class {class_label}: Selected feature {expression_data_temp.columns[idx]}")
    selected_features = list(selected_features)
    print("Number of selected features:", len(selected_features))
    if len(selected_features) < num_gene:
        print(f"Warning: only {len(selected_features)} unique features selected, less than requested {num_gene}.")
        # to fill up to num_gene, add top features by f-value
        additional_needed = num_gene - len(selected_features)
        # get top features by f-value
        # compute f-values for all features
        # re-fit selector on all data
        selector_full = SelectKBest(score_func=f_classif, k='all')
        selector_full.fit(expression_data_temp.values, labels)
        f_values = selector_full.scores_
        feature_fvalue_pairs = list(zip(expression_data_temp.columns, f_values))
        # sort by f-value descending
        feature_fvalue_pairs.sort(key=lambda x: x[1], reverse=True)
        # add top features not already selected
        print("Filling up selected features to reach requested number...")
        for feat, fval in feature_fvalue_pairs:
            if feat not in selected_features:
                selected_features.append(feat)
                print(f"Added feature {feat} with f-value {fval}")
                if len(selected_features) >= num_gene:
                    break     
    
    # check duplicates in selected features
    print("Number of unique selected features:", len(set(selected_features)))
    # get indices of selected features in the original expression data
    if len(selected_features) > num_gene:
        selected_features = selected_features[:num_gene]
    selected_feature_indices = [expression_data_temp.columns.get_loc(feat) for feat in selected_features]
    print("Selected feature indices:", selected_feature_indices)

    return selected_features, selected_feature_indices

def get_mirna_inner_connection(mirna_connection):
    mirna_connection_row = []
    mirna_connection_col = []
    print("Building miRNA-miRNA connections...")
    print("miRNA-gene connection matrix shape:", mirna_connection.shape)
    for i in range(mirna_connection.shape[0]):
        current_indices = np.nonzero(mirna_connection[i,])
        # print(current_indices)
        column_indexes = current_indices[1]
        if len(column_indexes) > 1:
            for x in range(len(column_indexes)):
                for y in range(x, len(column_indexes)):
                    mirna_connection_row += [column_indexes[x]]
                    mirna_connection_col += [column_indexes[y]]
    
    mirna_connection_data = [1] * len(mirna_connection_row)
    mirna_connection_adj = sp.csr_matrix((mirna_connection_data, (mirna_connection_row, mirna_connection_col)),shape=(mirna_connection.shape[1], mirna_connection.shape[1]))
    mirna_index = mirna_connection_adj.nonzero()
    mirna_row = mirna_index[0]
    mirna_col = mirna_index[1]
    mirna_data = [1] * len(mirna_row)
    mirna_adj = sp.csr_matrix((mirna_data, (mirna_row, mirna_col)),shape=(mirna_connection.shape[1], mirna_connection.shape[1]))
    # check symmetry
    print("check symmetry miRNA-miRNA adjacency matrix before symmetrization:", (mirna_adj != mirna_adj.T).nnz)
    mirna_sym = mirna_adj + mirna_adj.T
    mirna_sym[mirna_sym > 1] = 1
    # check symmetry
    A = mirna_sym
    if not sp.isspmatrix(A):
        A = sp.csr_matrix(np.asarray(A))   # np.asarray evita numpy.matrix
    A = A.tocsr()
    print("check symmetry miRNA-miRNA adjacency matrix after symmetrization:", (A != A.T).nnz)
    mirna_adj = mirna_sym
    # add self-loops
    mirna_adj.setdiag(1)
    return(mirna_adj.toarray())

#def get_tf_mirna_connection(tf_gene_adj, gene_mirna_adj):
    # matrix multiplication TF-gene (TF x Gene) * Gene-miRNA (Gene x miRNA) = TF-miRNA (TF x miRNA)
    assert tf_gene_adj.shape[1] == gene_mirna_adj.shape[0], "Gene axis non allineata: controlla ordine/lista geni."
    A_tf_mirna = (tf_gene_adj @ gene_mirna_adj).tocsr()        # (TF, miRNA)
    A_tf_mirna.eliminate_zeros()
    print("TF-miRNA adjacency shape:", A_tf_mirna.shape)
    print("TF-miRNA non-zero edges:", A_tf_mirna.nnz)
    A_tf_mirna = tf_gene_adj.dot(gene_mirna_adj)
    # make binary, da modificare se con edge weights
    A_tf_mirna.data[:] = 1.0
    print("TF-miRNA adjacency matrix after binarization data range:", A_tf_mirna.data.min(), "to", A_tf_mirna.data.max())
    return A_tf_mirna

def get_tf_mirna_connection(tf_gene_adj, gene_mirna_adj):
    # forza CSR (così @ ritorna sparse e .data è valido)
    if not sp.isspmatrix(tf_gene_adj):
        tf_gene_adj = sp.csr_matrix(np.asarray(tf_gene_adj, dtype=np.float32))
    else:
        tf_gene_adj = tf_gene_adj.tocsr().astype(np.float32)

    if not sp.isspmatrix(gene_mirna_adj):
        gene_mirna_adj = sp.csr_matrix(np.asarray(gene_mirna_adj, dtype=np.float32))
    else:
        gene_mirna_adj = gene_mirna_adj.tocsr().astype(np.float32)

    assert tf_gene_adj.shape[1] == gene_mirna_adj.shape[0], (
        f"Gene axis non allineata: TFxG={tf_gene_adj.shape} vs GxM={gene_mirna_adj.shape}"
    )

    A_tf_mirna = (tf_gene_adj @ gene_mirna_adj).tocsr()  # (T, M)
    A_tf_mirna.eliminate_zeros()

    if A_tf_mirna.nnz > 0:
        A_tf_mirna.data[:] = 1.0  # binarizza solo nnz

    print("TF-miRNA adjacency shape:", A_tf_mirna.shape)
    print("TF-miRNA non-zero edges:", A_tf_mirna.nnz)
    return A_tf_mirna

def get_tf_inner_connection(tf_gene_adj):
    print("Building TF-TF connections...")
     # 1) Assicura formato sparse CSR
    if sp.isspmatrix(tf_gene_adj):
        A = tf_gene_adj.tocsr()
    else:
        # numpy array / matrix -> CSR
        A = sp.csr_matrix(np.asarray(tf_gene_adj, dtype=np.float32))
     # make binary
    A_bin = A.copy()
    if A_bin.nnz > 0:
        A_bin.data[:] = 1.0
        A_bin.eliminate_zeros()

    S = (A_bin @ A_bin.T).tocsr()
    S.eliminate_zeros()
    S.data[:] = 1.0  # binarize


    print("TF-TF meta-path adjacency shape:", S.shape)
    print("TF-TF meta-path non-zero edges:", S.nnz)
    print("TF-TF meta-path adjacency matrix data range:", S.data.min(), "to", S.data.max())
    print("symmetry check:", (S != S.T).nnz)
    return S

def load_exp_cnv_and_mirna_data(expression_data_path, cnv_data_path, mirna_data_path):
    ## load multi-omics data
    expression_data = None
    cnv_data = None
    mirna_data = None
    if expression_data_path is not None:
        expression_data = pd.read_csv(expression_data_path, sep='\t', index_col=0, header=0)
        # convet to float32 only numerical columns
        expression_data = expression_data.select_dtypes(include=[np.number]).astype(np.float32)
    if cnv_data_path is not None:
        cnv_data = pd.read_csv(cnv_data_path, sep='\t', index_col=0, header=0)
        cnv_data = cnv_data.drop(['icluster_cluster_assignment','sample'], axis=1)
        cnv_data = cnv_data.astype(np.float32)
    if mirna_data_path is not None:
        mirna_data = pd.read_csv(mirna_data_path, sep='\t', index_col=0, header=0)
        mirna_data = mirna_data.astype(np.float32)
    
    return expression_data, cnv_data, mirna_data
    
def filter_tf_adjacency_matrix(tf_gene_matrix_path, top_gene_variance_index, tf_nodes_in_vocab_path, n_top_tf, min_deg=5, max_deg=None, percentile=98):
    # Function to filter TF-gene adjacency matrix based on gene variance and TF degree
    ## load tf-gene adjacency matrix
    A_tf_gene = sp.load_npz(tf_gene_matrix_path).tocsr()  # (T_all x G_all)
    print("Original TF-gene adjacency matrix shape:", A_tf_gene.shape)
    # load tf nodes in vocab
    tf_nodes_in_vocab = pd.read_csv(tf_nodes_in_vocab_path)
    tf_list = tf_nodes_in_vocab["TF"].astype(str).str.strip().tolist()
    print("Total TFs in vocab:", len(tf_list))
    # filter genes by variance
    top_gene_idx = np.array(top_gene_variance_index)
    A_tf_gene_filtered = A_tf_gene[:, top_gene_idx].tocsr()  # (T_all x G_top)
    print("TF-gene adjacency matrix after gene filtering shape:", A_tf_gene_filtered.shape)
    # compute tf degrees and filter by min/max degree
    tf_degrees = np.array(A_tf_gene_filtered.getnnz(axis=1)).flatten()
    print("TF degrees stats before filtering: min {}, max {}, mean {}, median {}".format(
        tf_degrees.min(), tf_degrees.max(), tf_degrees.mean(), np.median(tf_degrees)
    ))
    # apply degree filtering
    if max_deg is None:
        max_deg = np.percentile(tf_degrees, percentile)
    # --- filter TFs by degree range ---
    print("tf_degrees unique (first 20):", np.unique(tf_degrees)[:20])
    print("percentile", np.percentile, "=> max_deg", max_deg)
    print("count degrees >= 5:", np.sum(tf_degrees >= 5))
    print("A_tf_gene_filtered nnz:", A_tf_gene_filtered.nnz)
    valid = np.where((tf_degrees >= min_deg) & (tf_degrees <= max_deg))[0]
    
    if len(valid) == 0:
        raise ValueError("No TFs left after degree filtering. Adjust min_deg and max_deg parameters.")
    # get top n_top_tf TFs by degree
    valid_sorted = valid[np.argsort(tf_degrees[valid])[::-1]]
    if n_top_tf is not None and n_top_tf > 0:
        top_tf_rows = valid_sorted[:int(n_top_tf)]
    else:
        top_tf_rows = valid_sorted
    print("Number of TFs after degree filtering:", len(top_tf_rows))
    A_tf_gene_selected = A_tf_gene_filtered[top_tf_rows, :].tocsr()
    
    print("A_tf_gene_selected shape:", A_tf_gene_selected.shape)
    
    print("TF-gene adjacency matrix after TF degree filtering shape:", A_tf_gene_selected.shape)
    # get filtered tf list
    filtered_tf_list = [tf_list[i] for i in top_tf_rows]

    # check: riga 0 della matrice filtrata deve corrispondere al TF filtered_tf_list[0]
    i_global = top_tf_rows[0]
    print("Global TF for selected row 0:", tf_list[i_global], "==", filtered_tf_list[0])
    print("first 10  TFs list:", tf_list[:10])
    print("first 10 filtered TFs list:", filtered_tf_list[:10])
    print("first 10 top TF degrees:", tf_degrees[top_tf_rows][:10])
    print("Number of TFs after filtering:", len(filtered_tf_list))
    print("TF-gene adjacency matrix after TF filtering shape:", A_tf_gene_selected.shape)
    print("Number of TFs after filtering:", len(filtered_tf_list))
    print("Selected TF degree stats: min {}, max {}, mean {}, median {}".format(
        int(tf_degrees[top_tf_rows].min()), int(tf_degrees[top_tf_rows].max()),
        float(tf_degrees[top_tf_rows].mean()), float(np.median(tf_degrees[top_tf_rows]))
    ))
    # print selected TFs sybols
    print("Selected TFs symbols (first 20):", filtered_tf_list[:20])
    return A_tf_gene_selected, filtered_tf_list

def fileter_tf_per_variance(tf_nodes_in_vocab_path, n_top_tf, expression_data_train):
    # Filter all TF per variance of expression data.
    tf_vocab = pd.read_csv(tf_nodes_in_vocab_path)
    tf_list = tf_vocab["TF"].astype(str).str.strip().tolist()
    tf_vocab["matrix_index"] = tf_vocab["matrix_index"].astype(int)

    # get the variance of tf genes in the expression data
    tf_gene_variance = expression_data_train[tf_list].var().sort_values(ascending=False)
    # get top n_top_tf TFs by variance
    top_tf_list = tf_gene_variance.head(n_top_tf).index.tolist()
    print("Top TFs by variance:", top_tf_list[:20])
    
    # === Prendi gli indici riga (matrix_index) dei TF selezionati ===
    tf_row_idx = (
        tf_vocab.set_index("TF")
               .loc[top_tf_list, "matrix_index"]     # mantiene l'ordine di top_tf_list
               .to_numpy(dtype=int)
    )

    return top_tf_list, tf_row_idx

def down_unified_data_with_TF(
                                            expression_data, 
                                            cnv_data,
                                            mirna_data,
                                            selected_gene_list,
                                            selected_gene_index, 
                                            selected_tf_list,
                                            selected_tf_index,
                                            omic_mode,                                              
                                            adjacency_matrix_path, 
                                            mirna_to_gene_matrix_path, 
                                            tf_gene_matrix_path,
                                            enable_tf,
                                            gene_gene, 
                                            mirna_gene, 
                                            mirna_mirna,
                                            number_gene,
                                            tf_gene=False,
                                            tf_mirna=False,
                                            tf_tf=False,
                                            num_mirna=100,
                                            num_tf=100
                                            ):  
    # 0) Normalizza flags (config keys assenti -> None)
    # ----------------------------
    ## mode 0: mRNA
    ## mode 1: miRNA
    ## mode 2: mRNA + miRNA
    ## mode 3: mRNA + CNV
    ## mode 4: mRNA + CNV + miRNA
    ## nuove modalità Con TF-gene:
    ##0 + TF: mRNA + TF network
    ##2 + TF: mRNA + miRNA + TF network
    ##3 + TF: mRNA + CNV + TF network
    ##4 + TF: mRNA + CNV + miRNA + TF network
    high_variance_gene_list = selected_gene_list
    high_variance_gene_index = selected_gene_index
    print("Omic mode:", omic_mode)
    gene_gene = bool(gene_gene) if gene_gene is not None else False
    mirna_gene = bool(mirna_gene) if mirna_gene is not None else False
    mirna_mirna = bool(mirna_mirna) if mirna_mirna is not None else False
    tf_gene = bool(tf_gene) if tf_gene is not None else False
    tf_mirna = bool(tf_mirna) if tf_mirna is not None else False
    tf_tf = bool(tf_tf) if tf_tf is not None else False

    print("Number of high variance genes selected:", len(high_variance_gene_list))
    ## get TF expression data
    if enable_tf:
        # --- TF features (expr) ---
        expression_data_full = expression_data.copy()
        # getting TF expression data
        filtered_tf_list = [str(tf).strip().rstrip(".") for tf in selected_tf_list]
        # estrai i TF nello stesso ordine di filtered_tf_list
        X_tf_expr_df = expression_data_full.reindex(columns=filtered_tf_list)  # mantiene ordine, mette NaN se manca
        X_tf_expr = X_tf_expr_df.fillna(0).to_numpy(dtype=np.float32)         # (N, T)
        
        # --- TF features (cnv) solo se mode 3/4 ---
        if omic_mode in (3,4):
            cnv_data_full = cnv_data.copy()
            X_tf_cnv_df = cnv_data_full.reindex(columns=filtered_tf_list)  # mantiene ordine, mette NaN se manca
            X_tf_cnv = X_tf_cnv_df.fillna(0).to_numpy(dtype=np.float32)         # (N, T)

        # --- TF-gene adjacency: (T_sel x G_top) ---    
        # load tf-gene adjacency matrix and filter by tf list
        tf_gene_adj = sp.load_npz(tf_gene_matrix_path).tocsr()  # (T_all x G_all)
        print("Original TF-gene adjacency matrix shape:", tf_gene_adj.shape)
        # filter TF-gene adjacency matrix by selected TFs idx
        A_tf_gene_selected = tf_gene_adj[selected_tf_index, :][:, high_variance_gene_index].tocsr()
        print("TF-gene adjacency matrix after TF filtering shape:", A_tf_gene_selected.shape)
        A_tf_gene_selected = A_tf_gene_selected.todense()

        print("Number of TFs selected:", len(selected_tf_list))
        # save expression data with TF nodes
        #expression_data_full = expression_data.drop(columns=['icluster_cluster_assignment'], errors='ignore').copy()
        

    
    number_tf = number_tf = int(num_tf) if enable_tf else 0
    
    ## get labels before filtering columns
    #labels = expression_data['icluster_cluster_assignment']
    #labels = labels - 1
    
    if expression_data.shape[0] == mirna_data.shape[0] and (omic_mode in (1,2,4)):
        print('Exp and miRNA sample numbers match.')
    ## filter multi-omics data by gene list

    expression_data = expression_data.loc[:,high_variance_gene_list]
    expression_data.index = range(expression_data.shape[0])
    print("expression_data.shape:", expression_data.shape)

    if omic_mode in (3,4):
        cnv_data = cnv_data.loc[:,high_variance_gene_list]
        cnv_data.index = range(cnv_data.shape[0])
        print("cnv_data.shape:", cnv_data.shape)
    if omic_mode in (1,2,4):
        mirna_data.index = range(mirna_data.shape[0])
        print("mirna_data.shape:", mirna_data.shape)
    
    # Validate sample numbers
    if expression_data.shape[0] == mirna_data.shape[0] and (omic_mode in (1,2,4)):
        print("Exp and miRNA sample numbers match.")        

    if omic_mode == 0:
        N = expression_data.shape[0]
        if enable_tf:
            # append TF expression data
            # Nodi sono G geni + T TF
            X_exp = expression_data.to_numpy(dtype=np.float32)  # (N, G)
            expression_data_with_tf = np.concatenate((X_exp, X_tf_expr), axis=1)  # (N, G+T)
            data = np.array(expression_data_with_tf).reshape(N, -1 ,1)
        else:
            data = np.array(expression_data).reshape(N, -1 ,1)
        print(data.shape)

    elif omic_mode == 1:
        # miRNA only
        N = mirna_data.shape[0]
        data = np.asarray(mirna_data, dtype=np.float32).reshape(N, -1, 1)   
    elif omic_mode == 2:
        # concatenate expr and mirna
        N = expression_data.shape[0]
        if enable_tf:
            # append TF expression data
            # Nodi sono G geni + M miRNA + T TF in questo ordine

            #X_exp = expression_data.to_numpy(dtype=np.float32)  # (N, G)
            data = pd.concat([expression_data, mirna_data], axis=1) # (N, G+M)

            #expression_data_with_tf = np.concatenate((X_exp, X_tf_expr), axis=1)  # (N, G+T)
            data = pd.concat([data, pd.DataFrame(X_tf_expr)], axis=1) # (N, G+M+T)
        # concattenate along features axis
        else:
            data = pd.concat([expression_data, mirna_data], axis=1)
        data = np.asarray(data).reshape(N, -1 ,1)
        print(data.shape)
        # stop execution here
        # return data, X_tf_expr, high_variance_gene_list
    elif omic_mode ==3:
        # concatenate expr and cnv
        N = expression_data.shape[0]
        if enable_tf:
            # append TF expression and CNV data
            # Nodi sono G geni + T TF
            X_exp = expression_data.to_numpy(dtype=np.float32)  # (N, G)
            X_cnv = cnv_data.to_numpy(dtype=np.float32)        # (N, G)
            expression_data_with_tf = np.concatenate((X_exp, X_tf_expr), axis=1)  # (N, G+T)
            cnv_data_with_tf = np.concatenate((X_cnv, X_tf_cnv), axis=1)           # (N, G+T)
            data =  np.concatenate([
                expression_data_with_tf.reshape(N, -1 ,1),
                cnv_data_with_tf.reshape(N, -1 ,1)
            ], axis=2)
        else:
            data = np.array(expression_data).reshape(N, -1 ,1)
            print(data.shape)
            cnv_data = np.asarray(cnv_data).reshape(N, -1 ,1)
            print(cnv_data.shape)
            data =  np.concatenate([data,cnv_data], axis=2)
        print(data.shape)
    else:
        # omic mode 4: concatenate expr, cnv and mirna
        N = mirna_data.shape[0]
        if enable_tf:
            # 1) Canale 0: expr + mirna + tf_expr
            # 2) Canale 1: cnv + padding mirna + tf_cnv
            # Calcolo cnv
            cnv_padding = pd.DataFrame(np.zeros((mirna_data.shape[0],mirna_data.shape[1])))
            cnv_data_pad_exp_mir = pd.concat([cnv_data, cnv_padding], axis=1) # (N, G+M)
            cnv_data_padded = pd.concat([cnv_data_pad_exp_mir, pd.DataFrame(X_tf_cnv)], axis=1) # (N, G+M+T)
            # Calcolo expr + mirna
            data_exp_mir = pd.concat([expression_data, mirna_data], axis=1) # (N, G+M)
            # append TF expression data
            data = pd.concat([data_exp_mir, pd.DataFrame(X_tf_expr)], axis=1)  # (N, G+M+T)
        else:
            cnv_padding = pd.DataFrame(np.zeros((mirna_data.shape[0],mirna_data.shape[1])))
            cnv_data_padded = pd.concat([cnv_data, cnv_padding], axis=1)
            data = pd.concat([expression_data, mirna_data], axis=1)

        ## reshape e concatenate dei canali 1 e 2
        data= np.asarray(data).reshape(N, -1 ,1)
        cnv_data_padded = np.asarray(cnv_data_padded).reshape(N, -1 ,1)
        data = np.concatenate([data,cnv_data_padded], axis=2)
        print(data.shape)
    # ----------------------------
    # 1) Load and process adjacency matrices
    # ----------------------------
    # Support function for min-max normalization of sparse matrix
    def _minmax_sparse(A: sp.spmatrix) -> sp.csr_matrix:
        A = A.tocoo(copy=True).astype(np.float32)
        if A.nnz == 0:
            return A.tocsr()
        dmin = float(A.data.min())
        dmax = float(A.data.max())
        if dmax > dmin:
            A.data = (A.data - dmin) / (dmax - dmin)
        return A.tocsr()
    
    ## load adjacency matrix
    if gene_gene and number_gene > 0:
        gene_gene_adj = sp.load_npz(adjacency_matrix_path)
        print("gene_gene_adj shape:", gene_gene_adj.shape)
        print('Gene-Gene matrix max value:', np.max(gene_gene_adj.data))
        print('Gene-Gene matrix min value:', np.min(gene_gene_adj.data))
        # print node degrees before normalization
        if sp.isspmatrix(gene_gene_adj):
            degrees = np.array(gene_gene_adj.getnnz(axis=0)).flatten()
            print("Gene-Gene adjacency matrix node degree stats before normalization: min {}, max {}, mean {}, median {}".format(
                degrees.min(), degrees.max(), degrees.mean(), np.median(degrees)
            ))
        gene_gene_adj = _minmax_sparse(gene_gene_adj)
        gene_gene_adj = gene_gene_adj.todense()
        gene_gene_adj_selected = gene_gene_adj[high_variance_gene_index,:][:,high_variance_gene_index]
        print("gene_gene_adj_selected shape:", gene_gene_adj_selected.shape)
        print("gene_gene_adj_selected max value:", np.max(gene_gene_adj_selected))
        print("gene_gene_adj_selected min value:", np.min(gene_gene_adj_selected))
        print("Non-zero count in gene_gene_adj_selected:", np.count_nonzero(gene_gene_adj_selected))
        print("Total elements in gene_gene_adj_selected:", gene_gene_adj_selected.size)
    else:
        gene_gene_adj_selected = np.identity(number_gene)
    
    ## load mirna_to_gene matrix
    if (omic_mode in (1, 2, 4)) and (mirna_gene or mirna_mirna) and number_gene > 0:
        mirna_gene_adj = sp.load_npz(mirna_to_gene_matrix_path)
        print("mirna_to_gene_matrix shape:", mirna_gene_adj.shape)
        print('miRNA-Gene matrix max value:', np.max(mirna_gene_adj.data))
        print('miRNA-Gene matrix min value:', np.min(mirna_gene_adj.data))
        # print node degrees before normalization
        print("degree:", mirna_gene_adj.shape)
        if sp.isspmatrix(mirna_gene_adj):
            degrees = np.array(mirna_gene_adj.getnnz(axis=0)).flatten()
            print("miRNA-Gene adjacency matrix gene node degree stats before normalization: min {}, max {}, mean {}, median {}".format(
                degrees.min(), degrees.max(), degrees.mean(), np.median(degrees)
            ))
        mirna_gene_adj = _minmax_sparse(mirna_gene_adj)
        mirna_gene_adj = mirna_gene_adj.todense()
        mirna_gene_adj_selected = mirna_gene_adj[high_variance_gene_index,:]
        print("mirna_gene_adj_selected shape:", mirna_gene_adj_selected.shape)
        print("mirna_gene_adj_selected max value:", np.max(mirna_gene_adj_selected))
        print("mirna_gene_adj_selected min value:", np.min(mirna_gene_adj_selected))
        print("Non-zero count in mirna_gene_adj_selected:", np.count_nonzero(mirna_gene_adj_selected))
        print("Total elements in mirna_gene_adj_selected:", mirna_gene_adj_selected.size)
    else:
        mirna_gene_adj_selected = np.zeros((number_gene, num_mirna), dtype=np.float32)

    # load tf-gene adjacency matrix
    if enable_tf and tf_gene and number_gene > 0:
        # la matrice è TF x G
        print("Including TF-gene connections...")
        tf_gene_adj_selected = A_tf_gene_selected
        print("tf_gene_adj_selected shape:", tf_gene_adj_selected.shape)
    else:
        tf_gene_adj_selected = np.zeros((number_tf, number_gene), dtype=np.float32)

    ## construct supra-adjacency matrix
    ## mode 0: mRNA
    ## mode 1: miRNA
    ## mode 2: mRNA + miRNA
    ## mode 3: mRNA + CNV
    ## mode 4: mRNA + CNV + miRNA
    # - mode 0/3: solo geni -> supra = GG (GxG)
    # - mode 1: solo miRNA -> supra = MM (MxM) o I
    # - mode 2/4: nodi gene+miRNA -> supra = [[GG, GM],[GM^
        # blocco cross da usare nella supra-adjacency
    # Supra adj with gene and mirna
    # |Agg  Agm|
    # |Amg  Amm|
    # Supra adj with gene, mirna and TF
    # |Agg  Agm  Agt|
    # |Amg  Amm  Amt|
    # |Atg  Atm  Att|
    # Mode 0/3 with TF
    # |Agg  Agt|
    # |Atg  Att|
    if omic_mode == 0 or omic_mode == 3:
        print("Supra adj omic mode", omic_mode)
        if enable_tf:
            print("Including TF nodes in supra-adjacency matrix...")
            top_supra_adj = np.concatenate((gene_gene_adj_selected, tf_gene_adj_selected.T), axis=1)  # (G, G+T)
            if tf_tf:
                print("Including TF-TF connections...")
                # create TF-TF adjacency as identity matrix for now
                #============================MODIFICA


                tf_inner_adj = get_tf_inner_connection(A_tf_gene_selected)
            else:
                tf_inner_adj = np.identity(number_tf)

            if sp.isspmatrix(tf_inner_adj):
                tf_inner_adj = tf_inner_adj.toarray()
            bottom_supra_adj = np.concatenate((tf_gene_adj_selected, tf_inner_adj), axis=1)  # (T, G+T)
            supra_matrix = np.concatenate((top_supra_adj, bottom_supra_adj), axis=0)  # (G+T, G+T)
            supra_adj = sp.csr_matrix(supra_matrix)
        else:
            supra_adj = sp.csr_matrix(gene_gene_adj_selected)
    elif omic_mode == 1:
        print("Supra adj omic mode 1")
        mirna_inner_adj = get_mirna_inner_connection(mirna_gene_adj_selected)
        supra_adj = sp.csr_matrix(mirna_inner_adj)
    else:
        print("Supra adj omic mode", omic_mode)
        # omic mode 2 and 4
        if mirna_mirna:
            print("Including miRNA-miRNA connections...")
            print("Check symmetry miRNA-gene adjacency matrix:")
            A = get_mirna_inner_connection(mirna_gene_adj_selected)
            print("check symmetry miRNA-miRNA adjacency matrix after symmetrization:", np.any(A != A.T))
            mirna_inner_adj = A
        else:
            mirna_inner_adj = np.identity(num_mirna)
        if mirna_gene:
            print("Including miRNA-gene connections...")
        else:
            mirna_gene_adj_selected = np.zeros((number_gene, num_mirna), dtype=np.float32)
            print("Excluding miRNA-gene connections...")
        if enable_tf:
            print("Including TF-gene connections in supra-adjacency matrix...")
            top_supra_adj = np.concatenate((
                np.concatenate((gene_gene_adj_selected, mirna_gene_adj_selected), axis=1),
                tf_gene_adj_selected.T
            ), axis=1)  # (G, G+M+T)
            print("top_supra_adj shape:", top_supra_adj.shape)

            if tf_tf:
                print("Including TF-TF connections...")
                tf_inner_adj = get_tf_inner_connection(A_tf_gene_selected)
                print("tf_inner_adj shape:", tf_inner_adj.shape)
            else:
                tf_inner_adj = np.identity(number_tf)
            if tf_mirna:
                print("Including TF-miRNA connections...")
                # create TF-miRNA adjacency from TF-gene and miRNA-gene
                tf_mirna_adj = get_tf_mirna_connection(tf_gene_adj_selected, mirna_gene_adj_selected)  # (T, M)
                print("tf_mirna_adj shape:", tf_mirna_adj.shape)
            else:
                tf_mirna_adj = np.zeros((number_tf, num_mirna), dtype=np.float32)
            # check if is np array
            if sp.isspmatrix(tf_mirna_adj):
                tf_mirna_adj = tf_mirna_adj.toarray()
            if sp.isspmatrix(mirna_inner_adj):
                mirna_inner_adj = mirna_inner_adj.toarray()
            if sp.isspmatrix(tf_inner_adj):
                tf_inner_adj = tf_inner_adj.toarray()
            print("type mirna_inner_adj:", type(mirna_inner_adj))
            print("type tf_inner_adj:", type(tf_inner_adj))
            print("type tf_mirna_adj:", type(tf_mirna_adj))

            print("type tf_mirna_adj:", type(tf_mirna_adj))
            print("mirna gene adj selected shape:", mirna_gene_adj_selected.shape)
            print("mirna inner adj shape:", mirna_inner_adj.shape)
            print("tf mirna adj shape:", tf_mirna_adj.shape)
            print("tf mirna adj transpose shape:", tf_mirna_adj.T.shape)
            mid_supra_adj = np.concatenate((
                np.concatenate((np.transpose(mirna_gene_adj_selected), mirna_inner_adj), axis=1),
                tf_mirna_adj.T
            ), axis=1)  # (M, G+M+T)
            print("mid_supra_adj shape:", mid_supra_adj.shape)

            bottom_supra_adj = np.concatenate((np.concatenate((tf_gene_adj_selected, tf_mirna_adj), axis=1), tf_inner_adj), axis=1)  # (T, G+M+T)
            print("bottom_supra_adj shape:", bottom_supra_adj.shape)
            supra_matrix = np.concatenate((top_supra_adj, mid_supra_adj, bottom_supra_adj), axis=0)
            print("supra_matrix shape:", supra_matrix.shape)
        else:
            top_supra_adj = np.concatenate((gene_gene_adj_selected, mirna_gene_adj_selected), axis=1)
            bottom_supra_adj = np.concatenate((np.transpose(mirna_gene_adj_selected), mirna_inner_adj), axis=1)
            supra_matrix = np.concatenate((top_supra_adj, bottom_supra_adj), axis=0)
        supra_adj = sp.csr_matrix(supra_matrix)

    print("Supra-adjacency matrix shape:", supra_adj.shape)
        

    return supra_adj, np.asarray(data)

def dropout_data(data, labels, drop_out=0.6):
    dropout_index = sample(range(len(labels)), round(len(labels)*drop_out))
    dropped_data = data[dropout_index,:]
    dropped_labels = labels[dropout_index]
    return dropped_data, dropped_labels

def disassemble_edge_weights(edge_weights, edge_index, num_gene, num_attributes):
    edge_index_transposed = edge_index.T
    edge_attributes = np.zeros((edge_index.shape[1], num_attributes))
    for idx, x in enumerate(edge_index_transposed):
        # print(x)
        if x[0] < num_gene and x[1] < num_gene: ## gene-gene connection
            edge_attributes[idx,0] = edge_weights[idx]
        elif x[0] < num_gene and x[1] >= num_gene: ## mirna-gene connection
            edge_attributes[idx,1] = edge_weights[idx]
        elif x[0] >= num_gene and x[1] < num_gene: ## mirna-gene connection
            edge_attributes[idx,1] = edge_weights[idx]
        else: ## mirna-mirna connections
            edge_attributes[idx,0] = edge_weights[idx]
    return(torch.Tensor(edge_attributes))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def omic_mode_translation(omic_mode):
    if omic_mode == 0:
        print('Using Expression Data Only.')
        return 0
    elif omic_mode == 1:
        print('Using miRNA Data Only.')
        return 1
    elif omic_mode == 2:
        print('Using Expression and miRNA Data.')
        return 2
    elif omic_mode == 3:
        print('Using Expression and CNV Data.')
        return 3
    elif omic_mode == 4:
        print('Using Expression, CNV and miRNA Data.')
        return 4
    
def validate_network_choice(omic_mode, gene_gene, mirna_gene, mirna_mirna):
    if omic_mode == 0:
        if mirna_gene or mirna_mirna:
            print('miRNA-Gene or miRNA-miRNA network not available when only using expression data.')
            return True, False, False
        else:
            return gene_gene, mirna_gene, mirna_mirna
    elif omic_mode == 1:
        if gene_gene:
            print('Gene-Gene not available when only using miRNA data.')
            return False, True, True
        else:
            return gene_gene, mirna_gene, mirna_mirna
    elif omic_mode == 3:
        if mirna_gene or mirna_mirna:
            print('miRNA-Gene or miRNA-miRNA network not available when using expression and CNV data.')
            return True, False, False
        else:
            return gene_gene, mirna_gene, mirna_mirna
    else:
        return gene_gene, mirna_gene, mirna_mirna
    
def validate_tf_network_choice(
    omic_mode, gene_gene, mirna_gene, mirna_mirna,
    tf_gene, tf_mirna, tf_tf,
    enable_tf=True, num_tf=0
):
    gene_gene, mirna_gene, mirna_mirna = validate_network_choice(
        omic_mode, gene_gene, mirna_gene, mirna_mirna)

    # Se TF non abilitati o non ci sono nodi TF, spegni tutto
    if (not enable_tf) or (int(num_tf) <= 0):
        if tf_gene or tf_mirna or tf_tf:
            print("TF disabled (enable_tf=False or num_tf=0). Disabling TF networks.")
        return gene_gene, mirna_gene, mirna_mirna, False, False, False

    # Nessuna rete TF selezionata
    if not tf_gene and not tf_mirna and not tf_tf:
        print("No TF network selected.")
        return gene_gene, mirna_gene, mirna_mirna, False, False, False

    # Compatibilità per omic_mode
    if omic_mode in (0, 3):  # niente miRNA
        if tf_mirna:
            print('TF-miRNA network not available when no miRNA data is used.')
            tf_mirna = False
        return gene_gene, mirna_gene, mirna_mirna, tf_gene, tf_mirna, tf_tf

    if omic_mode == 1:  # solo miRNA (niente geni)
        if tf_gene:
            print('TF-gene network not available when only using miRNA data.')
            tf_gene = False
        # tf_tf è ok (TF nodes esistono), tf_mirna è ok (miRNA esistono)
        return gene_gene, mirna_gene, mirna_mirna, tf_gene, tf_mirna, tf_tf

    # omic_mode 2 o 4: geni + miRNA presenti
    return gene_gene, mirna_gene, mirna_mirna, tf_gene, tf_mirna, tf_tf
    
def process_adj(cfg, adj, logger=None):

    adj = adj/np.max(adj)
    adj = adj.astype('float32')
    adj.setdiag(0)
    adj = adj + sp.eye(adj.shape[0])

    adj = sp.coo_matrix(adj)

    G = cfg['data']['num_gene']
    M = cfg['data']['num_mirna']
    A = adj.tocsr()
    
    edge_index = torch.tensor(
    np.vstack((adj.row, adj.col)),
    dtype=torch.long
    )

    edge_weight = torch.tensor(adj.data, dtype=torch.float32)
    # debug info
    if logger is not None:
        logger.info(f"Adjacency matrix shape: {adj.shape}")
        logger.info(f"Edge index shape: {edge_index.shape}")
        logger.info(f"Edge weight shape: {edge_weight.shape}")
    return adj, edge_index, edge_weight