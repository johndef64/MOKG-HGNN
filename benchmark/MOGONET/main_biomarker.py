"""
Script principale per l'identificazione di biomarcatori utilizzando MOGONET.

Questo script calcola l'importanza delle features (geni, miRNA, ecc.) attraverso 
multiple ripetizioni di modelli pre-addestrati e identifica i biomarcatori più 
significativi per la classificazione dei campioni.

Dataset supportati:
- BRCA: Breast Invasive Carcinoma (5 classi)
- ROSMAP: Religious Orders Study and Memory and Aging Project (2 classi)
"""

# Importazione delle librerie necessarie
import os
import copy
# Importazione delle funzioni personalizzate per il calcolo dell'importanza delle features
from feat_importance import cal_feat_imp, summarize_imp_feat

if __name__ == "__main__":
    # =============================================================================
    # CONFIGURAZIONE PARAMETRI DATASET
    # =============================================================================
    
    # Specifica il dataset da analizzare
    data_folder = 'BRCA'  # Può essere 'BRCA' o 'ROSMAP'
    
    # Percorso alla cartella contenente i modelli pre-addestrati
    model_folder = os.path.join(data_folder, 'models')
    
    # Lista delle viste/omiche da analizzare:
    # 1 = Dati di espressione genica (mRNA)
    # 2 = Dati di espressione dei microRNA (miRNA) 
    # 3 = Dati di metilazione del DNA
    view_list = [1, 2, 3]
    
    # Configurazione del numero di classi basata sul dataset
    if data_folder == 'ROSMAP':
        num_class = 2  # Classificazione binaria: cognitivamente normale vs declino cognitivo
    if data_folder == 'BRCA':
        num_class = 5  # Classificazione multi-classe: 5 sottotipi molecolari del cancro al seno

    # =============================================================================
    # CALCOLO DELL'IMPORTANZA DELLE FEATURES ATTRAVERSO MULTIPLE RIPETIZIONI
    # =============================================================================
    
    # Lista per raccogliere i risultati di importanza da tutte le ripetizioni
    featimp_list_list = []
    
    # Esecuzione del calcolo su 5 modelli diversi (cross-validation a 5 fold)
    for rep in range(5):
        print(f"Processando ripetizione {rep+1}/5...")
        
        # Calcola l'importanza delle features usando il modello della ripetizione corrente
        featimp_list = cal_feat_imp(data_folder, 
                                    os.path.join(model_folder, str(rep+1)), 
                                    view_list, 
                                    num_class)
        
        # Aggiunge una copia profonda dei risultati alla lista principale
        featimp_list_list.append(copy.deepcopy(featimp_list))

    # =============================================================================
    # AGGREGAZIONE E RANKING DEI BIOMARCATORI
    # =============================================================================
    
    # Aggrega i risultati di tutte le ripetizioni e identifica i top biomarcatori
    summarize_imp_feat(featimp_list_list)
    
# =============================================================================
# SEZIONE JUPYTER NOTEBOOK: FUNZIONE DI RIEPILOGO E ANALISI AVANZATA
# =============================================================================

#%%
import numpy as np
import pandas as pd

def summarize_imp_feat(featimp_list_list, topn=30):
    """
    Aggrega e riassume l'importanza delle features da multiple ripetizioni.
    
    Questa funzione combina i risultati di importanza delle features calcolati
    su diversi modelli (fold di cross-validation) e identifica i biomarcatori
    più consistentemente importanti.
    
    Args:
        featimp_list_list (list): Lista di liste contenenti i risultati di importanza
                                 per ogni ripetizione e vista
        topn (int): Numero di top features da restituire (default: 30)
    
    Returns:
        df_featimp_top (DataFrame): DataFrame con le top features ordinate per importanza
    """
    # Numero di ripetizioni (fold di cross-validation)
    num_rep = len(featimp_list_list)
    # Numero di viste/omiche analizzate
    num_view = len(featimp_list_list[0])
    
    # =============================================================================
    # INIZIALIZZAZIONE: Processamento della prima ripetizione
    # =============================================================================
    df_tmp_list = []
    for v in range(num_view):
        # Copia i dati della prima ripetizione per ogni vista
        df_tmp = copy.deepcopy(featimp_list_list[0][v])
        # Aggiunge un identificatore della vista omica (0, 1, 2 per mRNA, miRNA, metilazione)
        df_tmp['omics'] = np.ones(df_tmp.shape[0], dtype=int) * v
        df_tmp_list.append(df_tmp.copy(deep=True))
    
    # Concatena tutti i risultati della prima ripetizione
    df_featimp = pd.concat(df_tmp_list).copy(deep=True)
    
    # =============================================================================
    # AGGREGAZIONE: Aggiunge i risultati delle altre ripetizioni
    # =============================================================================
    for r in range(1, num_rep):
        for v in range(num_view):
            df_tmp = copy.deepcopy(featimp_list_list[r][v])
            # Aggiunge l'identificatore della vista omica
            df_tmp['omics'] = np.ones(df_tmp.shape[0], dtype=int) * v
            # Concatena con i risultati precedenti
            # Nota: usa pd.concat invece del deprecato append
            df_featimp = pd.concat([df_featimp, df_tmp.copy(deep=True)], ignore_index=True)
    
    # =============================================================================
    # RANKING: Calcola l'importanza totale e ordina le features
    # =============================================================================
    
    # Raggruppa per nome della feature e vista omica, sommando l'importanza
    df_featimp_top = df_featimp.groupby(['feat_name', 'omics'])['imp'].sum()
    df_featimp_top = df_featimp_top.reset_index()
    
    # Ordina per importanza decrescente
    df_featimp_top = df_featimp_top.sort_values(by='imp', ascending=False)
    
    # Seleziona solo le top N features
    df_featimp_top = df_featimp_top.iloc[:topn]
    
    # =============================================================================
    # OUTPUT: Stampa i risultati del ranking
    # =============================================================================
    print('{:}\t{:}'.format('Rank', 'Feature name'))
    for i in range(len(df_featimp_top)):
        print('{:}\t{:}'.format(i+1, df_featimp_top.iloc[i]['feat_name']))
    
    return df_featimp_top

# Esecuzione della funzione di riepilogo e salvataggio dei risultati
df_featimp_top = summarize_imp_feat(featimp_list_list)

# Salvataggio dei risultati in un file CSV per analisi future
df_featimp_top.to_csv(f'{data_folder}_feat_importance.csv', index=False)

# =============================================================================
# INTERPRETAZIONE DEI RISULTATI: Top 30 biomarcatori identificati per BRCA
# =============================================================================

# I risultati mostrano i biomarcatori più importanti per la classificazione
# dei sottotipi molecolari del cancro al seno (BRCA):

# Rank	Feature name
# 1	SOX11|6664        - Gene SOX11: fattore di trascrizione, ruolo nel sviluppo tumorale
# 2	hsa-mir-205       - microRNA-205: regolatore post-trascrizionale, biomarcatore noto 
# 3	GPR37L1           - Recettore accoppiato a proteina G, possibile target terapeutico
# 4	AMY1A|276         - Gene dell'amilasi, metabolismo dei carboidrati
# 5	SLC6A15|55117     - Trasportatore di aminoacidi, neurobiologia
# 6	FABP7|2173        - Proteina legante acidi grassi, metabolismo lipidico
# 7	MIR563            - microRNA 563: regolazione dell'espressione genica
# 8	SLC6A14|11254     - Trasportatore di aminoacidi, uptake cellulare
# 9	hsa-mir-187       - microRNA-187: oncosoppressore in vari tumori
# 10	SLC6A2|6530      - Trasportatore della noradrenalina
# 11	FGFBP1|9982      - Proteina legante FGF, angiogenesi e crescita tumorale
# 12	DSG1|1828        - Desmogleina 1: adesione cellulare, integrità epiteliale
# 13	UGT8|7368        - UDP glucuronosiltransferasi: detossificazione
# 14	ANKRD45|339416   - Proteina contenente domini ankyrin repeat
# 15	OR1J4             - Recettore olfattivo
# 16	ATP10B            - ATPasi trasportatrice di fosfolipidi
# 17	PI3|5266          - Inibitore della peptidasi 3
# 18	hsa-mir-452       - microRNA-452: regolazione metabolismo e proliferazione
# 19	hsa-mir-20b       - microRNA-20b: controllo del ciclo cellulare
# 20	SERPINB5|5268     - Serpin B5 (maspin): oncosoppressore nei tumori epiteliali
# 21	KRTAP3-3          - Proteina associata alla cheratina
# 22	COL11A2|1302      - Collagene XI: componente della matrice extracellulare
# 23	hsa-mir-224       - microRNA-224: oncogene in vari tipi di cancro
# 24	FLJ41941          - Gene ipotetico
# ...
# 27	TMEM207           - Proteina transmembrana 207
# 28	CDH26             - Caderina 26: adesione cellulare
# 29	MT1DP             - Pseudogene della metallotionina
# 30	hsa-mir-204       - microRNA-204: oncosoppressore, regola EMT
# %%
