# Script to build training components for pancancer prediction using multi-omics GNN
import os
from pathlib import Path
import sys
from typing import Dict, Any, Optional, Tuple

import torch
from torch_geometric.loader import DataLoader

from multiomics_gnn.models import gat, gcn_tf, baseline, gat_tf
from multiomics_gnn.base_ml.sampler import OmicSampler
from multiomics_gnn.base_ml.trainer import Trainer
from multiomics_gnn.base_ml.scheduler import OmicScheduler
from multiomics_gnn.utils.paths import make_run_dir

# ----------------------------
# MODEL BUILDER
# ----------------------------

def build_model(cfg: Dict[str, Any], num_classes: int, device: str = None):
    
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    # sezioni opzionali
    train_cfg = cfg.get("train", {})
    graph_cfg = cfg.get("graph", {})
    runtime_cfg = cfg.get("runtime", {})

    # device
    if device is None:
        device = runtime_cfg.get("device", None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # parametri (minimi) dal config
    parallel = bool(model_cfg.get("parallel", False))

    # nel tuo progetto questi sono più "train" che "model"
    l2 = bool(train_cfg.get("l2", False))
    decoder = bool(train_cfg.get("decoder", False))

    poolsize = int(model_cfg.get("poolsize", 8))
    poolrate = float(model_cfg.get("poolrate", 0.8))
    dropout = float(model_cfg.get("dropout", 0.2))

    edge_weight = bool(graph_cfg.get("edge_weight", False))
    edge_attribute = bool(graph_cfg.get("edge_attribute", False))

    num_gene = int(data_cfg["num_gene"])
    omic_mode = int(data_cfg["omic_mode"])
    num_mirna = int(data_cfg.get("num_mirna", 100)) 
    num_tf = int(data_cfg.get("num_tf", 0))
    
    jumping_knowledge = bool(model_cfg.get("jumping_knowledge", False))
    jk_mode = str(model_cfg.get("jk_mode", "cat"))
    print(f"Building model with num_gene={num_gene}, num_mirna={num_mirna}, omic_mode={omic_mode}, num_classes={num_classes}, dropout={dropout}, jk_mode={jk_mode if jumping_knowledge else 'N/A'}")
    name = cfg["model"]["name"].lower()
    if name == 'gat':
        model = gat.GAT(
            name,
            parallel,
            l2,
            decoder,
            poolsize,
            poolrate,
            edge_weight,
            edge_attribute,
            num_gene,
            num_mirna,
            omic_mode,
            num_classes,
            dropout,
        )
    elif name == 'gcn_tf':
        model = gcn_tf.GCN(
            name,
            parallel,
            l2,
            decoder,
            poolsize,
            poolrate,
            edge_weight,
            edge_attribute,
            num_gene,
            num_mirna,
            num_tf, 
            omic_mode,
            num_classes,
            jumping_knowledge,
            jk_mode,
            dropout,
        )
    elif name == 'baseline':
        model = baseline.Baseline(
            name,
            parallel,
            l2,
            decoder,
            poolsize,
            poolrate,
            edge_weight,
            edge_attribute,
            num_gene,
            num_mirna,
            omic_mode,
            num_classes,
            dropout,
        )
    elif name == 'gat_tf':
        model = gat_tf.GAT(
            name,
            parallel,
            l2,
            decoder,
            poolsize,
            poolrate,
            edge_weight,
            edge_attribute,
            num_gene,
            num_mirna,
            num_tf, 
            omic_mode,
            num_classes,
            jumping_knowledge,
            jk_mode,
            dropout,
        )
    else:
        raise ValueError(f"model.name non supportato: {name}")
    return model.to(device)

# ----------------------------
# OPTIMIZER BUILDER
# ----------------------------
def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    opt_cfg = cfg.get("optimizer", {})
    train_cfg = cfg.get("train", {})
    name = str(opt_cfg.get("name", "adamw")).lower()
    lr = float(train_cfg.get("learning_rate", 0.001))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    if name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=lr,
                                      weight_decay=weight_decay)
    elif name == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)
    elif name == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay,
                                    momentum=momentum)
    return optimizer

# ----------------------------
# SCHEDULER BUILDER
# ----------------------------

def build_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer) -> Optional[OmicScheduler]:
    sched_cfg = cfg.get("scheduler", {})
    train_cfg = cfg.get("train", {})

    name = sched_cfg.get("name", None)
    if name is None:
        return None
    if str(name).lower() in {"none", "null", ""}:
        return None

    total_epochs = int(train_cfg["num_epochs"])
    return OmicScheduler(optimizer, str(name), total_epochs)
# ----------------------------
# DATA LOADER BUILDER
# ----------------------------

def build_data_loader(cfg: Dict[str, Any], train_dataset, val_dataset, test_dataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
    print("Building data loaders...")
    data_cfg = cfg.get("data", {})
    batch_size = int(data_cfg.get("batch_size", 32))
    print(f"Using batch size: {batch_size}")
    num_workers = int(data_cfg.get("num_workers", 4))
    sampler_strategy = str(cfg.get("sampler_strategy", "none")).lower()
    sampler_gamma = float(cfg.get("sampler_gamma", 1.0))
    sampler_alpha = float(cfg.get("sampler_alpha", 1.0))
    sampler_mode = str(cfg.get("sampler_mode", "nulite")).lower()
    print(f"Sampler strategy: {sampler_strategy}, mode: {sampler_mode}, gamma: {sampler_gamma}, alpha: {sampler_alpha}")

    sampler = OmicSampler(train_dataset, batch_size=batch_size, strategy=sampler_strategy,alpha=sampler_alpha, gamma=sampler_gamma, seed=cfg["project"].get("seed", 42), mode=sampler_mode).sampler



    train_data_loader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=num_workers,
                                   sampler=sampler)

    val_data_loader = DataLoader(val_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)

    test_data_loader = DataLoader(test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)
    # for each batch in train_data_loader count the number of samples for each class
    # log log to a file the prints 
    #batch_class_counts = []
    #batch_idx = 0
    #for batch in train_data_loader:
    #    labels = batch.y
    #    unique, counts = torch.unique(labels, return_counts=True)
    #    class_counts = dict(zip(unique.tolist(), counts.tolist()))
    #    print(f"Batch {batch_idx}:")
        # log to file
        
    #    print(f"Class distribution in a training batch: {class_counts}")
    #    batch_idx += 1
    #    with open("batch_class_counts.txt", "a") as f:
    #        f.write(f"Batch {batch_idx}: {class_counts}\n")
        # append to list for later analysis
    #    batch_class_counts.append(class_counts)
    # plot the mean count for each class across all batches
    #sys.exit(0)
    #import matplotlib.pyplot as plt
    #from collections import defaultdict
    #mean_class_counts = defaultdict(list)
    #for class_count in batch_class_counts:
    #    for cls, count in class_count.items():
    #        mean_class_counts[cls].append(count)
    #mean_counts = {cls: sum(counts)/len(counts) for cls, counts in mean_class_counts.items()}
    #plt.bar(mean_counts.keys(), mean_counts.values())
    #plt.xlabel("Class")
    #plt.ylabel("Mean Count per Batch")
    #plt.title("Mean Class Distribution in Training Batches")
    #plt.show()    


    return train_data_loader, val_data_loader, test_data_loader
# ----------------------------
# TRAINER BUILDER
# ----------------------------

def build_trainer(
    cfg: Dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[OmicScheduler],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    wandb_run: Optional[Any] = None,
    logger: Optional[Any] = None,
    results_dir: Optional[Path] = None
) -> Trainer:
    paths_cfg = cfg.get("paths", {})
    misc_cfg = cfg.get("misc", {})
    train_cfg = cfg.get("train", {})

    # 2. Definizione del percorso di salvataggio checkpoint (Result Path)
    #results_dir = paths_cfg.get("results_dir", "./results")
    if results_dir is None:
        results_dir = make_run_dir(cfg['paths']['results_dir'], cfg['project']['experiment_name'], True)
    logger.info(f"Results directory: {results_dir}")

    # Crea un nome file per il modello salvato (puoi renderlo dinamico se necessario)
    model_name = cfg.get("model", {}).get("name", "model")
    result_path = os.path.join(results_dir, f"{model_name}_best_weight.pt")

    # 3. Parametri opzionali
    debug = bool(misc_cfg.get("debug", False))
    # Il trainer usa un default di 5e-4, ma controlliamo se è nel config (sotto train o optimizer)
    l2_reg_factor = float(train_cfg.get("l2_reg_factor", 5e-4))

    # 4. Istanziazione Trainer
    # Nota: Trainer inizializza internamente EarlyStopping basandosi su 'args' (cfg)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        args=cfg,             # Passiamo l'intero dizionario config come 'args'
        result_path=result_path,
        l2_reg_factor=l2_reg_factor,
        debug=debug,
        logger=logger,
        wandb_run=wandb_run
    )
    return trainer







def build_all(cfg: Dict[str, Any],
              train_dataset,
              val_dataset,
              test_dataset,
              num_classes: int,
              device: str,
              wandb_run: Optional[Any] = None,
              logger: Optional[Any] = None,
              results_dir: Optional[Path] = None) -> Tuple[torch.nn.Module,
                                    torch.optim.Optimizer,
                                    Optional[OmicScheduler],
                                    DataLoader,
                                    DataLoader,
                                    DataLoader,
                                    Trainer]:
    model = build_model(cfg, num_classes=num_classes, device=device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    train_data_loader, val_data_loader, test_data_loader = build_data_loader(
        cfg, train_dataset, val_dataset, test_dataset)
    # 5. Build Trainer
    trainer = build_trainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_data_loader,
        val_loader=val_data_loader,
        test_loader=test_data_loader,
        device=device,
        logger=logger,
        results_dir=results_dir,
        wandb_run=wandb_run
    )
    return trainer