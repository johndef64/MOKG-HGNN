import argparse
import json
import os
import sys
import copy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import wandb
import yaml
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing

from torch_geometric.utils import from_scipy_sparse_matrix

from multiomics_gnn.config.loader import load_config
from multiomics_gnn.pancancer_prediction.preprocessing.preprocess_pancan_traning import (
    down_unified_data_with_TF,
    load_exp_cnv_and_mirna_data,
    omic_mode_translation,
    process_adj,
    validate_tf_network_choice,
    variance_FS,
    fileter_tf_per_variance,
    new_FDS,
    community_detection_feature_selection
)
from multiomics_gnn.pancancer_prediction.datasets.OmicDataset import OmicsGraphDataset
from multiomics_gnn.pancancer_prediction.training.builder import build_all
from multiomics_gnn.utils.logger import get_logger, set_log_to_file
from multiomics_gnn.utils.paths import make_run_dir
from multiomics_gnn.utils.seed import set_seed
from multiomics_gnn.pancancer_prediction.utils.split import generate_stratified_shuffle_indices
from multiomics_gnn.pancancer_prediction.utils.save_metrics import save_curves, get_all_metrics, save_confusion_matrix_pretty, save_metrics_pretty_txt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
@dataclass
class AllPaths:
    expression_data_path: Path
    cnv_data_path: Path
    mirna_data_path: Path
    expression_variance_path: Path
    clinical_path: Path
    adj_path: Path
    mirna_adj_path: Path
    tf_adj_path: Path
    tf_nodes_path: Path

class ExperimentDataLoader:
    def __init__(self, config: Optional[dict] = None):
        self.config = config
    def _resolve_paths(self) -> AllPaths:
            td = Path(self.config["paths"]["training_data_dir"])
            pp = self.config["pancan_paths"]

            return AllPaths(
                expression_data_path=td / pp["expression_path"],
                cnv_data_path=td / pp["cnv_path"],
                mirna_data_path=td / pp["mirna_path"],
                expression_variance_path=td / pp["expression_variance_path"],
                clinical_path=td / pp["clinical_data_path"],
                adj_path=td / pp["adj_path"],
                mirna_adj_path=td / pp["mirna_gene_matrix_path"],
                tf_adj_path=td / pp["tf_gene_matrix_path"],
                tf_nodes_path=td / pp["tf_nodes_path"]
        )
        
    def load_raw_data(self):
        self.paths = self._resolve_paths()
        exp_data, cnv_data, mirna_data = load_exp_cnv_and_mirna_data(
            expression_data_path=str(self.paths.expression_data_path),
            cnv_data_path=str(self.paths.cnv_data_path),
            mirna_data_path=str(self.paths.mirna_data_path)
        )
        return exp_data, cnv_data, mirna_data

class ExperimentRunner():
    def __init__(self, exp_data = None, cnv_data = None, mirna_data = None):
        #self.config = load_config(default_config)
        self.logger = get_logger()
        self.exp_data = exp_data
        self.cnv_data = cnv_data
        self.mirna_data = mirna_data

    def _resolve_paths(self) -> AllPaths:
        td = Path(self.config["paths"]["training_data_dir"])
        pp = self.config["pancan_paths"]

        return AllPaths(
            expression_data_path=td / pp["expression_path"],
            cnv_data_path=td / pp["cnv_path"],
            mirna_data_path=td / pp["mirna_path"],
            expression_variance_path=td / pp["expression_variance_path"],
            clinical_path=td / pp["clinical_data_path"],
            adj_path=td / pp["adj_path"],
            mirna_adj_path=td / pp["mirna_gene_matrix_path"],
            tf_adj_path=td / pp["tf_gene_matrix_path"],
            tf_nodes_path=td / pp["tf_nodes_path"]
        )

    def _wandb_init(self, run_dir: str, experiment_name: str):
        wb = self.config["wandb"]
        if not wb["enabled"]:
            return None

        # group: utile per raggruppare tutte le run di uno sweep
        group = self.config.get("sweep", {}).get("name") or experiment_name

        run = wandb.init(
            project=wb.get("project", "multimodal_gnn_fresh"),
            entity=wb.get("entity", None),
            name=os.path.basename(run_dir),  # oppure f"{experiment_name}_{timestamp}"
            group=group,
            tags=wb.get("tags", []),
            config=self.config,                      # logga TUTTO il config finale
            dir="temp_wandb",                     # scrive i file di wandb dentro results/...
            save_code=False
        )
        return run

# genera un grafo random con pesi = 1 di dimensione uguale a quello della matrice adj post downsampling per verificare se il grafo serve
    
    def _generate_random_graph(self, adj) -> np.ndarray:
        import scipy.sparse as sp
        # 1. Rilevamento flessibile del numero di archi
        if sp.issparse(adj):
            # Se è una matrice sparsa SciPy (CSR, COO, ecc.)
            num_nodes = adj.shape[0]
            num_edges = adj.nnz  # Attributo specifico per matrici sparse
        else:
            # Se è un array NumPy denso
            num_nodes = adj.shape[0]
            num_edges = np.count_nonzero(adj)

        # 2. Creazione della matrice casuale densa
        # Creiamo una matrice di zeri (float32 è standard per le GNN)
        random_adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        
        # Campioniamo indici lineari unici per evitare duplicati
        total_slots = num_nodes * num_nodes
        chosen_indices = np.random.choice(
            total_slots, 
            size=num_edges, 
            replace=False
        )
        
        # Trasformiamo gli indici "flat" in coordinate (i, j)
        rows = chosen_indices // num_nodes
        cols = chosen_indices % num_nodes
        
        # Assegniamo peso 1 agli archi estratti
        random_adj[rows, cols] = 1.0
        # add self loops
        np.fill_diagonal(random_adj, 1.0)
        return random_adj
    
    def run_experiment(self, cfg: dict, experiment_name: Optional[str] = None, trial: Optional[optuna.Trial]=None):
        self.config = cfg
        if experiment_name is None:
            experiment_name = self.config['project']['experiment_name']
        print(f"Running experiment: {experiment_name}")

        paths = self._resolve_paths()
        
        self.logger.info("Starting experiment with config:")
        #self.logger.info(self.config)
        self.logger.info(self.config['paths']['training_data_dir'])
        set_seed(self.config['project']['seed'])
        # check if seed is set in torch
        self.logger.info(f"Seed used: {torch.initial_seed()}")

        #self.logger.info("Data paths set:")
        #self.logger.info(f"Expression data path: {paths.expression_data_path}")
        #self.logger.info(f"CNV data path: {paths.cnv_data_path}")
        #self.logger.info(f"miRNA data path: {paths.mirna_data_path}")
        #self.logger.info(f"Adjacency matrix path: {paths.adj_path}")
        #self.logger.info(f"miRNA-Gene matrix path: {paths.mirna_adj_path}")
        #self.logger.info(f"Clinical data path: {paths.clinical_path}")
        #self.logger.info(f"Expression variance path: {paths.expression_variance_path}")
        #self.logger.info(f"TF-Gene matrix path: {paths.tf_adj_path}")
        #self.logger.info(f"TF nodes path: {paths.tf_nodes_path}")
        # create run directory
        
        run_dir = make_run_dir(self.config['paths']['results_dir'], experiment_name, True)
        self.logger.info(f"Run directory created at: {run_dir}")
        set_log_to_file(self.logger, os.path.join(run_dir, "experiment.log"))

        # wandb init
        run = self._wandb_init(run_dir, experiment_name)

        # split creation
        split_dir = Path(self.config["paths"]["training_data_dir"]) / "splits"
        split_seed = int(self.config["project"].get("split_seed", 42))
        seed_dir = split_dir / f"splits_seed_{split_seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        # check seed split files, if not exist generate them with fixed seed (42) and stratified by label distribution (icluster_cluster_assignment)

        train_idx_f = seed_dir / "common_trimmed_shuffle_index_train.tsv"
        val_idx_f   = seed_dir / "common_trimmed_shuffle_index_val.tsv"
        test_idx_f  = seed_dir / "common_trimmed_shuffle_index_test.tsv"

        train_fraction = float(self.config['tuning'].get('train_fraction', 1.0))
        fraction_pct = int(round(train_fraction * 100))
        fraction_idx_f = seed_dir / f"common_trimmed_shuffle_index_train_fraction_{fraction_pct}.tsv"

        # Regenerate splits if (a) the base files are missing, or
        # (b) we need a train-fraction file that has not been produced yet.
        # NB: `generate_stratified_shuffle_indices` writes both the base
        # files AND, when train_fraction < 1.0, the fraction-specific file.
        need_base = not (train_idx_f.exists() and val_idx_f.exists() and test_idx_f.exists())
        need_fraction = train_fraction < 1.0 and not fraction_idx_f.exists()
        if need_base or need_fraction:
            if need_base:
                self.logger.info("Split files not found. Generating fixed stratified split.")
            else:
                self.logger.info(
                    f"Train-fraction split for {fraction_pct}% not found. "
                    "Generating it (base splits already exist)."
                )
            generate_stratified_shuffle_indices(
                labels_csv=str(paths.clinical_path),
                out_dir=str(seed_dir),
                train_fraction=train_fraction,
                random_state=self.config['project']['split_seed'],
                logger=self.logger,
            )
        if train_fraction < 1.0:
            self.logger.info(
                f"Using reduced train fraction: {train_fraction}. "
                f"Loading from {fraction_idx_f}."
            )
            train_idx_f = fraction_idx_f
        print(f"train path: {train_idx_f}, val path: {val_idx_f}, test path: {test_idx_f}")
        # Load and preprocess data
        train_idx = np.loadtxt(train_idx_f, dtype=int, delimiter="\t")
        val_idx   = np.loadtxt(val_idx_f, dtype=int, delimiter="\t")
        test_idx  = np.loadtxt(test_idx_f, dtype=int, delimiter="\t")

        self.logger.info(f"Split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
        self.logger.info("Loading expression, CNV, and miRNA data...")

        if self.exp_data is not None and self.cnv_data is not None:
            self.logger.info("Preloaded data found. Using preloaded data instead of loading from disk.")
            exp_data, cnv_data, mirna_data = self.exp_data, self.cnv_data, self.mirna_data
        else:
            self.logger.info("No preloaded data found. Loading from disk...")
            
            exp_data, cnv_data, mirna_data = load_exp_cnv_and_mirna_data(
                expression_data_path=str(paths.expression_data_path),
                cnv_data_path=str(paths.cnv_data_path),
                mirna_data_path=str(paths.mirna_data_path)
            )
        self.logger.info("Data loaded successfully.")
        self.logger.info(f"Expression data shape: {exp_data.shape}, CNV data shape: {cnv_data.shape}, miRNA data shape: {mirna_data.shape if mirna_data is not None else 'N/A'}")
        
        label_col = "icluster_cluster_assignment"

        labels = exp_data[label_col].to_numpy(dtype=np.int64) - 1
        exp_data = exp_data.drop(columns=[label_col]).copy()

        # get train data for feature selection and normalization
        exp_train = exp_data.iloc[train_idx]
        
        # ---------------------------- feature selection and normalization
        # Feature-collapse alignment: if the config provides explicit gene/TF panels
        # (data.gene_list / data.tf_list), MOGNN-TF uses EXACTLY those instead of its
        # own variance FS, so it shares the identical molecular panel with MOKG-HGNN
        # at this gene level. miRNA is handled separately via data.mirna_keep below.
        override_genes = self.config['data'].get('gene_list', None)
        override_tf = self.config['data'].get('tf_list', None)
        top_genes, top_gene_idx = None, None
        top_tf_list, top_tf_index = None, None

        if override_genes is not None:
            col_index = {g: i for i, g in enumerate(exp_train.columns)}
            missing = [g for g in override_genes if g not in col_index]
            if missing:
                self.logger.warning(f"[collapse] {len(missing)} genes in gene_list absent from "
                                    f"expression, e.g. {missing[:5]} — dropped")
            top_genes = [g for g in override_genes if g in col_index]
            top_gene_idx = [col_index[g] for g in top_genes]
            self.logger.info(f"[collapse] using explicit gene panel: {len(top_genes)} genes")
            if override_tf is not None and int(self.config['data']['num_tf']) > 0:
                # map the explicit TF list to matrix indices via the TF vocab
                tf_vocab = pd.read_csv(str(paths.tf_nodes_path))
                tf_vocab["matrix_index"] = tf_vocab["matrix_index"].astype(int)
                vocab_map = tf_vocab.set_index(tf_vocab["TF"].astype(str).str.strip())["matrix_index"]
                keep_tf = [t for t in override_tf if t in vocab_map.index]
                top_tf_list = keep_tf
                top_tf_index = vocab_map.loc[keep_tf].to_numpy(dtype=int)
                self.logger.info(f"[collapse] using explicit TF panel: {len(top_tf_list)} TFs")

        if top_genes is None and self.config['data'].get('feature_selection_method', None) == 'variance':
            top_genes, top_gene_idx = variance_FS(exp_train, num_gene=int(self.config['data']['num_gene']))
            # check if top_gene_idx and top_genes are consistent with exp_train columns
            self.logger.info(f"Expression train columns: {exp_train.columns.tolist()[:10]} ...")
            self.logger.info(f"Top gene indices: {top_gene_idx[:10]} ...")
            # check if top_gene_idx and top_genes are consistent with exp_train columns
            assert all(exp_train.columns[i] == top_genes[j] for j, i in enumerate(top_gene_idx)), "Mismatch between top_gene_idx and top_genes"

            self.logger.info(f"Top genes selected by variance: {top_genes[:10]} ...")
            # feature selection for TF
            if  self.config['data']['num_tf'] > 0:
                top_tf_list, top_tf_index = fileter_tf_per_variance(
                    tf_nodes_in_vocab_path=str(paths.tf_nodes_path),
                    n_top_tf=int(self.config['data']['num_tf']),
                    expression_data_train=exp_train
                )

        if top_genes is None and self.config['data'].get('feature_selection_method', None) == 'GSP':
            # se è GSP, allora faccio prima la feature selection by variance con una soglia più alta (es. 4) per ridurre il numero di feature e poi applico GSP
            top_genes, top_gene_idx = new_FDS(exp_train, num_gene=int(self.config['data']['num_gene']), biogrid_adj=str(paths.adj_path))
            # check if top_gene_idx and top_genes are consistent with exp_train columns
            self.logger.info(f"Expression train columns: {exp_train.columns.tolist()[:10]} ...")
            self.logger.info(f"Top gene indices: {top_gene_idx[:10]} ...")
            # check if top_gene_idx and top_genes are consistent with exp_train columns
            assert all(exp_train.columns[i] == top_genes[j] for j, i in enumerate(top_gene_idx)), "Mismatch between top_gene_idx and top_genes"

            self.logger.info(f"Top genes selected by variance: {top_genes[:10]} ...")
            # feature selection for TF
            if  self.config['data']['num_tf'] > 0:
                top_tf_list, top_tf_index = fileter_tf_per_variance(
                    tf_nodes_in_vocab_path=str(paths.tf_nodes_path),
                    n_top_tf=int(self.config['data']['num_tf']),
                    expression_data_train=exp_train
                )
        if top_genes is None and self.config['data'].get('feature_selection_method', None) == "community":
            top_genes, top_gene_idx = community_detection_feature_selection(exp_train, biogrid_adj=str(paths.adj_path), num_gene=int(self.config['data']['num_gene']))
            self.logger.info(f"Top genes selected by community detection: {top_genes[:10]} ...")
             # feature selection for TF
            if  self.config['data']['num_tf'] > 0:
                top_tf_list, top_tf_index = fileter_tf_per_variance(
                    tf_nodes_in_vocab_path=str(paths.tf_nodes_path),
                    n_top_tf=int(self.config['data']['num_tf']),
                    expression_data_train=exp_train
                )

        
        # ====================== Normalizzazione per tutti i dati (gene expression, CNV, miRNA):
        # fit solo sul training set, applicata anche a validation/test (no leakage).
        # Default = min-max, come riportato nella tesi.
        normalization = self.config['data'].get('normalization', 'min-max')
        if normalization == 'min-max':
            from sklearn.preprocessing import MinMaxScaler as _Scaler
        elif normalization == 'mad':
            from sklearn.preprocessing import RobustScaler as _Scaler
        elif normalization == 'z-score':
            from sklearn.preprocessing import StandardScaler as _Scaler
        else:
            raise ValueError(
                f"Unsupported data.normalization='{normalization}'. "
                "Use one of: 'min-max', 'mad', 'z-score'."
            )
        scaler_exp = _Scaler()
        scaler_cnv = _Scaler()
        if mirna_data is not None:
            scaler_mirna = _Scaler()

        scaler_exp.fit(exp_data.iloc[train_idx]) # fit solo sui dati di train
        #exp_data_scaled = scaler_exp.transform(exp_data[exp_data.columns]) # applica la trasformazione a tutto il dataset
        scaler_cnv.fit(cnv_data.iloc[train_idx]) # fit solo sui dati di train
        cnv_data_scaled = scaler_cnv.transform(cnv_data[cnv_data.columns]) # applica la trasformazione a tutto il dataset
        if mirna_data is not None:

            scaler_mirna.fit(mirna_data.iloc[train_idx]) # fit solo sui dati di train
            mirna_data_scaled = scaler_mirna.transform(mirna_data[mirna_data.columns]) # applica la trasformazione a tutto il dataset

        exp_data_scaled = pd.DataFrame(
            scaler_exp.transform(exp_data),
            index=exp_data.index,
            columns=exp_data.columns
        )
        cnv_data_scaled = pd.DataFrame(
            scaler_cnv.transform(cnv_data),
            index=cnv_data.index,
            columns=cnv_data.columns
        )
        if mirna_data is not None:
            mirna_data_scaled = pd.DataFrame(
                scaler_mirna.transform(mirna_data),
                index=mirna_data.index,
                columns=mirna_data.columns
            )
        self.logger.info(f"Data normalization completed using {self.config['data']['normalization']} scaler.")
        self.logger.info(f"Expression data shape: {exp_data_scaled.shape}, CNV data shape: {cnv_data_scaled.shape}, miRNA data shape: {mirna_data_scaled.shape if mirna_data is not None else 'N/A'}")
        self.logger.info(f"Expression data train min and max sample values (before scaling): min={exp_data.iloc[train_idx].min().min():.4f}, max={exp_data.iloc[train_idx].max().max():.4f}")
        self.logger.info(f"Expression data train min and max sample values (after scaling): min={exp_data_scaled.iloc[train_idx].min().min():.4f}, max={exp_data_scaled.iloc[train_idx].max().max():.4f}")
        self.logger.info(f"Expression data Validation and Test min and max sample values (after scaling): val_min={exp_data_scaled.iloc[val_idx].min().min():.4f}, val_max={exp_data_scaled.iloc[val_idx].max().max():.4f}, test_min={exp_data_scaled.iloc[test_idx].min().min():.4f}, test_max={exp_data_scaled.iloc[test_idx].max().max():.4f}")
        
        self.logger.info(f"CNV data train min and max sample values (before scaling): min={cnv_data.iloc[train_idx].min().min():.4f}, max={cnv_data.iloc[train_idx].max().max():.4f}")
        self.logger.info(f"CNV data train min and max sample values (after scaling): min={cnv_data_scaled.iloc[train_idx].min().min():.4f}, max={cnv_data_scaled.iloc[train_idx].max().max():.4f}")
        self.logger.info(f"CNV data Validation and Test min and max sample values (after scaling): val_min={cnv_data_scaled.iloc[val_idx].min().min():.4f}, val_max={cnv_data_scaled.iloc[val_idx].max().max():.4f}, test_min={cnv_data_scaled.iloc[test_idx].min().min():.4f}, test_max={cnv_data_scaled.iloc[test_idx].max().max():.4f}")
        if mirna_data is not None:
            self.logger.info(f"miRNA data train min and max sample values (before scaling): min={mirna_data.iloc[train_idx].min().min():.4f}, max={mirna_data.iloc[train_idx].max().max():.4f}")
            self.logger.info(f"miRNA data train min and max sample values (after scaling): min={mirna_data_scaled.iloc[train_idx].min().min():.4f}, max={mirna_data_scaled.iloc[train_idx].max().max():.4f}")
            self.logger.info(f"miRNA data Validation and Test min and max sample values (after scaling): val_min={mirna_data_scaled.iloc[val_idx].min().min():.4f}, val_max={mirna_data_scaled.iloc[val_idx].max().max():.4f}, test_min={mirna_data_scaled.iloc[test_idx].min().min():.4f}, test_max={mirna_data_scaled.iloc[test_idx].max().max():.4f}")
        #debug prints
        self.logger.info(f"Omic mode readed from config: {self.config['data']['omic_mode']}")
        omic_mode = omic_mode_translation(int(self.config['data']['omic_mode']))
        self.logger.info(f"Using omic mode: {omic_mode}")
        self.logger.info(f"Number of TFs to use: {self.config['data']['num_tf']}")

        #gene_gene, mirna_gene, mirna_mirna = validate_network_choice(
        #    omic_mode=omic_mode,
        #    gene_gene=bool(self.config['data']['gene_gene']),
        #    mirna_gene=bool(self.config['data']['mirna_gene']),
        #    mirna_mirna=bool(self.config['data']['mirna_mirna']),
        #)
        # Modificato per TF
        enable_tf = int(self.config['data'].get('num_tf', 0)) > 0
        self.logger.info(f"TF integration enabled: {enable_tf}")
        gene_gene, mirna_gene, mirna_mirna, tf_gene, tf_mirna, tf_tf = validate_tf_network_choice(
            omic_mode=omic_mode,
            gene_gene=bool(self.config['data']['gene_gene']),
            mirna_gene=bool(self.config['data']['mirna_gene']),
            mirna_mirna=bool(self.config['data']['mirna_mirna']),
            tf_gene=bool(self.config['data'].get('tf_gene', False)),
            tf_tf=bool(self.config['data'].get('tf_tf', False)),
            tf_mirna=bool(self.config['data'].get('tf_mirna', False)),
            enable_tf=enable_tf,
            num_tf=int(self.config['data'].get('num_tf', 0))
        )

        #self.logger.info(f"Network choices - gene_gene: {gene_gene}, mirna_gene: {mirna_gene}, mirna_mirna: {mirna_mirna}")
        self.logger.info(f"Network choices - gene_gene: {gene_gene}, mirna_gene: {mirna_gene}, mirna_mirna: {mirna_mirna}, tf_gene: {tf_gene}, tf_tf: {tf_tf}, tf_mirna: {tf_mirna}")
       

        #===============temp
        # not apply normalization for now, to check the effect of normalization on the graph construction and model performance. Use raw data for now.
        # exp_data_scaled = exp_data
        # cnv_data_scaled = cnv_data
        # if mirna_data is not None:
        #     mirna_data_scaled = mirna_data
        

        self.logger.info("train data shape before graph construction: " + str(exp_data_scaled.iloc[train_idx].shape))

        #sys.exit("Feature selection by variance completed. Exiting for debugging purposes.")
        print("Preprocessing data and building graph...")
        print("Feature selection mode:", self.config['data'].get('feature_selection_method', 'None'))

        adj, train_data_all = down_unified_data_with_TF(
            expression_data=exp_data_scaled,
            cnv_data=cnv_data_scaled,
            mirna_data=mirna_data_scaled if mirna_data is not None else None,
            selected_gene_index=top_gene_idx,
            selected_gene_list=top_genes,
            selected_tf_list=top_tf_list if self.config['data']['num_tf'] > 0 else None,
            selected_tf_index=top_tf_index if self.config['data']['num_tf'] > 0 else None,
            omic_mode= omic_mode,
            enable_tf=self.config['data']['num_tf'] > 0,
            adjacency_matrix_path=str(paths.adj_path),
            mirna_to_gene_matrix_path=str(paths.mirna_adj_path),
            tf_gene_matrix_path=str(paths.tf_adj_path),
            gene_gene= gene_gene,
            mirna_gene= mirna_gene,
            mirna_mirna= mirna_mirna,
            tf_gene= tf_gene,
            tf_tf= tf_tf,
            tf_mirna= tf_mirna,
            number_gene=int(self.config['data']['num_gene']),
            num_mirna=int(self.config['data']['num_mirna']),
            num_tf=int(self.config['data']['num_tf']),
            # feature-collapse: restrict miRNA to the exact survived panel (list of
            # names) so MOGNN-TF uses the same miRNAs as MOKG-HGNN at this gene level.
            mirna_keep=self.config['data'].get('mirna_keep', None)
            )
        # ===========================================
        # test se serve il grafo.
        #import scipy.sparse as sp
        #self.logger.info("adj type: " + str(type(adj)) + " adj shape: " + str(adj.shape) + " adj nnz (num edges): " + str(adj.nnz))
        #random_adj = self._generate_random_graph(adj)
        # check se adj e random_adj hanno la stessa shape e lo stesso numero di edge sparse (non zero entries)
        #self.logger.info(f"rand adj type: {type(random_adj)}")
        #self.logger.info(f"Original adj - shape: {adj.shape}, num edges: {adj.nnz}")
        #self.logger.info(f"Random adj - shape: {random_adj.shape}, num edges: {np.count_nonzero(random_adj)}")\
        # controlla se i valori all'interno effettivamente sono diversi, gli edge weight sono tutti 1.

        #self.logger.info(f"Original adj - min: {adj.min()}, max: {adj.max()}")
        #self.logger.info(f"Random adj - min: {random_adj.min()}, max: {random_adj.max()}")
        #frobenius_dist = np.linalg.norm(adj - random_adj)
        #self.logger.info(f"Frobenius distance between original adj and random adj: {frobenius_dist:.4f}")
        # convert random_adj to <class 'scipy.sparse._csr.csr_matrix'>
        #from scipy.sparse import csr_matrix
        #random_adj_sparse = csr_matrix(random_adj)
        #frobenius_dist_sparse = sp.linalg.norm(adj - random_adj_sparse)
        #self.logger.info(f"Frobenius distance between original adj and random adj (sparse): {frobenius_dist_sparse:.4f}")
        #adj = random_adj_sparse

        # ===========================================
        # debug prints
        self.logger.info(f"Adjacency matrix shape: {adj.shape}, number of non-zero entries (edges): {adj.nnz}")
        self.logger.info(f"All data shape after graph construction: {train_data_all.shape}")
        # shapes and min and max of train data after graph construction
        self.logger.info(f"Train data sample values after graph construction: min={train_data_all.min().min():.4f}, max={train_data_all.max().max():.4f}")

        train_data = train_data_all[train_idx]
        val_data   = train_data_all[val_idx]
        test_data  = train_data_all[test_idx]

        # channel 0 = expression, channel 1 = cnv (nel tuo caso)
        for name, arr in [("TRAIN", train_data), ("VAL", val_data), ("TEST", test_data)]:
            self.logger.info(f"{name} expr min/max: {arr[:,:,0].min():.4f} / {arr[:,:,0].max():.4f}")
            if arr.shape[2] > 1:  # se c'è anche il canale CNV
                self.logger.info(f"{name} cnv  min/max: {arr[:,:,1].min():.4f} / {arr[:,:,1].max():.4f}")


        #sys.exit("Graph construction completed. Exiting for debugging purposes.")

        # export adj
        from scipy.sparse import save_npz
        save_npz(os.path.join(run_dir, "adj_used.npz"), adj)

        # print graph structure
        self.logger.info(f"Graph structure - num_gene: {int(self.config['data']['num_gene'])}, num_mirna: {int(self.config['data']['num_mirna'])}, num_tf: {int(self.config['data'].get('num_tf', 0))}")
        self.logger.info(f"Graph structure - num_nodes in adj: {adj.shape[0]}, num_edges in adj: {adj.nnz}")


        self.logger.info(f"[DEBUG] config num_gene={self.config['data']['num_gene']} num_mirna={self.config['data']['num_mirna']}")
        self.logger.info(f"[DEBUG] train_data_all.shape={train_data_all.shape}")
        self.logger.info(f"[DEBUG] num_nodes_real={train_data_all.shape[1]} expected={int(self.config['data']['num_gene']) + int(self.config['data']['num_mirna']) + int(self.config['data'].get('num_tf', 0))}")
        self.logger.info("Data loaded and preprocessed.")
        #debug prints
        self.logger.info(f"Adjacency matrix shape: {adj.shape}")
        self.logger.info(f"Training data shape: {train_data_all.shape}")

        self.logger.info(f"Total number of samples after filtering: {len(labels)}")
        #self.logger.info(f"Training data: {train_data_all}")
        #self.logger.info(f"Labels: {labels}")

        #Labels processing
        #elf.logger.info(f"Labels before encoding: {np.unique(labels, return_counts=True)}")
        le = preprocessing.LabelEncoder()
        labels_encoded = le.fit_transform(labels)
        #elf.logger.info(f"Labels after encoding: {np.unique(labels_encoded, return_counts=True)}")

        adj_processed, edge_index, edge_weight = process_adj(cfg=self.config, adj=adj, logger=self.logger)
        
        #debug prints
        self.logger.info(f'Number of nodes: {adj_processed.shape[0]}')
        self.logger.info(f'Number of edges: {edge_index.size(1)}')
        #self.logger.info(f"[DEBUG] adj.nnz={adj_processed.nnz}  edge_index.E={edge_index.size(1)}  (devono essere uguali)")
        #self.logger.info(f"Processed adjacency matrix with {adj.shape[0]} nodes and {edge_index.size(1)} edges.")
        #self.logger.info(f'Edge index shape: {edge_index.shape}')
        #self.logger.info(f'Edge weight shape: {edge_weight.shape}')
        #self.logger.info(f'Edge weight values: {edge_weight}')
        #self.logger.info(f'Edge weight min: {edge_weight.min().item()}')
        #self.logger.info(f'Edge weight max: {edge_weight.max().item()}')
        self.logger.info(f'Number of nodes: {adj.shape[0]}')
        self.logger.info(f'Number of edges: {edge_index.size(1)}')
        self.logger.info(f'Number of features per node: {train_data_all[0].shape[1]}')

        # Prepare datasets

        #train_data = np.asarray(train_data_all).astype(np.float32)[train_idx]
        #val_data = np.asarray(train_data_all).astype(np.float32)[val_idx]
        #test_data = np.asarray(train_data_all).astype(np.float32)[test_idx]
        train_labels = labels_encoded[train_idx]
        val_labels = labels_encoded[val_idx]
        test_labels = labels_encoded[test_idx]
        self.logger.info(f'Train data shape: {train_data.shape}')
        self.logger.info(f'Validation data shape: {val_data.shape}')
        self.logger.info(f'Test data shape: {test_data.shape}')

        self.logger.info(f'Train labels distribution: {np.unique(train_labels, return_counts=True)}')
        self.logger.info(f'Validation labels distribution: {np.unique(val_labels, return_counts=True)}')
        self.logger.info(f'Test labels distribution: {np.unique(test_labels, return_counts=True)}')
        train_size = train_data.shape[0]
        val_size = val_data.shape[0]
        test_size = test_data.shape[0]

        #self.logger.info(f'Train labels distribution: {np.unique(train_labels, return_counts=True)}')

        nclass = len(np.unique(labels_encoded))


        train_labels = train_labels.astype(np.int64)
        test_labels = test_labels.astype(np.int64)
        val_labels = val_labels.astype(np.int64)
        train_data = torch.FloatTensor(train_data)
        test_data = torch.FloatTensor(test_data)
        val_data = torch.FloatTensor(val_data)
        train_labels = torch.LongTensor(train_labels)
        test_labels = torch.LongTensor(test_labels)
        val_labels = torch.LongTensor(val_labels)

        train_ds = OmicsGraphDataset(train_data, train_labels, edge_index, edge_weight=edge_weight, edge_attr=None, transform=None, pre_transform=None)
        val_ds = OmicsGraphDataset(val_data, val_labels, edge_index, edge_weight=edge_weight, edge_attr=None, transform=None, pre_transform=None)
        test_ds = OmicsGraphDataset(test_data, test_labels, edge_index, edge_weight=edge_weight, edge_attr=None, transform=None, pre_transform=None)

        num_nodes = train_data.shape[1]
        self.logger.info(f"Number of classes: {nclass}")
        #self.logger.info(f"Number of nodes: {num_nodes}")
        # Debug prints to inspect the dataset class behavior
        self.logger.info('=== Dataset debug info ===')
        self.logger.info(f'Train dataset length: {len(train_ds)}')
        self.logger.info(f'Val dataset length: {len(val_ds)}')
        self.logger.info(f'Test dataset length: {len(test_ds)}')
        # Build model and training components
        #model, optimizer, loss_fn = build_all(cfg=self.config, self.config['model'], self.config['training'])
        #weights = train_ds.get_weight_pancan(gamma=1)
        #self.logger.info(f'Weights tensor shape: {weights.shape}')
        #self.logger.info(f'Weights tensor sample values: {weights[:10]}')
        

        # Build all components, data loaders, model, optimizer
        trainer = build_all(
            cfg=self.config,
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
            num_classes=nclass,
            device= self.config['project']['device'],
            logger=self.logger,
            results_dir=run_dir,
            wandb_run=run
        )
        # Training loop (simplified)
        self.logger.info("Starting training...")
        trainer.fit(trial=trial)
        history = trainer.history

        self.logger.info("Training completed.")
        self.logger.info(f"Best validation metric: {trainer.early_stopping.best_metric:.4f} at epoch {trainer.early_stopping.best_epoch}")
        self.logger.info("Evaluating on best model...")
        # ===== VALIDATION metrics (da mettere nel summary per statistiche) =====
        # ===== VALIDATION EVAL (per statistica + selezione config) =====
        y_true_val, y_pred_val, y_prob_val = trainer.predict(
            trainer.val_loader, load_best=True
        )

        # metriche scalari (da pred discrete)
        val_global_metrics, val_per_class_metrics, val_confusion_mat, val_clf_report = get_all_metrics(
            y_true_val, y_pred_val, nclass
        )

        # AUC-ROC multiclass OvR (da probabilità)
        # Robustezza: calcolo solo sulle classi presenti in validation (evita crash se qualche classe è assente)
        present = np.unique(y_true_val)
        # binarize su tutte le classi, poi seleziono le colonne delle classi presenti
        y_true_oh = label_binarize(y_true_val, classes=list(range(nclass)))
        present_cols = present.astype(int)

        val_auc_roc_ovr_macro = roc_auc_score(
            y_true_oh[:, present_cols], y_prob_val[:, present_cols],
            average="macro", multi_class="ovr"
        )
        val_auc_roc_ovr_weighted = roc_auc_score(
            y_true_oh[:, present_cols], y_prob_val[:, present_cols],
            average="weighted", multi_class="ovr"
        )
        # (opzionale) salva anche artifacts validation come fai per il test
        save_confusion_matrix_pretty(val_confusion_mat, filepath=os.path.join(run_dir, "val_confusion_matrix"))
        save_metrics_pretty_txt(
            val_global_metrics, val_per_class_metrics, val_clf_report,
            filepath=os.path.join(run_dir, "val_metrics.txt"),
            num_classes=nclass,
            class_names=None
        )

        self.logger.info("Starting testing...")
        acc, f1, y_true, y_pred = trainer.test()
        self.logger.info(f"Test Accuracy: {acc:.4f}, Test F1 Score: {f1:.4f}")
        # Optional: delete all checkpoints except the best one to save space
        checkpoin_path = run_dir / "checkpoints"
        if checkpoin_path.exists() and checkpoin_path.is_dir():
            # delete folder
            import shutil
            shutil.rmtree(checkpoin_path)



        # Wandb logging
        if run is not None:
            wandb.log({"test/acc": float(acc), "test/f1": float(f1)})
            class_names = [str(i) for i in range(nclass)]
            wandb.log({
                "test/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None, y_true=y_true.tolist(), preds=y_pred.tolist(), class_names=class_names
                )
            })
            wandb.summary["test/acc"] = float(acc)
            wandb.summary["test/f1"] = float(f1)

        # Save final results
        self.logger.info(f"Saving results to {run_dir}")
        save_curves(history, filepath=run_dir)
        global_metrics, per_class_metrics, confusion_mat, clf_report = get_all_metrics(y_true, y_pred, nclass)
        save_confusion_matrix_pretty(confusion_mat, filepath=os.path.join(run_dir, "confusion_matrix"))
        save_metrics_pretty_txt(global_metrics, per_class_metrics, clf_report, filepath=os.path.join(run_dir, "metrics.txt"), num_classes=nclass, class_names=None)
        self.logger.info("Experiment completed.")
        self.logger.info(f"Saving config and results to {run_dir}")
        # saving the config used for the experiment
        with open(os.path.join(run_dir, "used_config.yaml"), 'w') as f:
            yaml.dump(self.config, f)
        # Optuna: ritorna la metrica di interesse per l'ottimizzazione (es. best val f1)
        best_val = None
        best_epoch = None
        if trainer.early_stopping is not None:
            best_val = float(trainer.early_stopping.best_metric)
            best_epoch = int(trainer.early_stopping.best_epoch)

        summary = {
            "best_val_metric": best_val,
            "best_epoch": best_epoch,

            # --- validation (PRIMARY) ---
            "val_auc_roc_ovr_macro": float(val_auc_roc_ovr_macro),
            "val_auc_roc_ovr_weighted": float(val_auc_roc_ovr_weighted),
            "val_per_class": json.dumps(val_per_class_metrics),

            # --- test (SECONDARY / reporting) ---
            "test_f1": float(f1),
            "test_acc": float(acc),
        }

        # aggiungi tutte le metriche globali di validation (accuracy + f1 micro/macro/weighted)
        for k, v in val_global_metrics.items():
            summary[f"val_{k}"] = float(v)

        # (opzionale) aggiungi tutte le metriche globali di test nello stesso stile
        for k, v in global_metrics.items():
            summary[f"test_{k}"] = float(v)


        # close wandb run
        if run is not None:
            run.finish()
        return summary
        

    def run_single_split(self, cfg: dict, experiment_name: Optional[str] = None):
        return self.run_experiment(cfg, experiment_name)