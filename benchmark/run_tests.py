import os
import subprocess
import pandas as pd
import numpy as np

models = [
    'li24'
    #'mpk'
]

paths = [
    'Multimodal-GNN-for-Cancer-Subtype-Clasification'
    #'MPK-GNN'
]

args = {
        "lr": [0.01],
        "big_lr": [False],
        "num_gene": [300,500,700,1000],
        "omic_mode": [4],
        "cancer_subtype": [True],
        "specific_type": ["brca"],
        "shuffle_index": [0],
        "batch_size": [16],
        "epochs": [200],
        "dropout": [0.6],
        "model": ["gat","gcn"],
        "decay": [0.9],
        "poolsize": [8],
        "poolrate": [0.8],
        "gene_gene": [True],
        "mirna_gene": [True],
        "mirna_mirna": [True],
        "parallel": [True],
        "l2": [True],
        "decoder": [True],
        "edge_attribute": [False],
        "edge_weight": [True],
        "train_ratio": [0.8],
        "test_ratio": [0.1]
    }

SCRIPT_NAME = "cancer_test.py"
N_RUNS = 5

def build_base_config(args_dict):
    """
    Prende il dict args (con liste) e restituisce una config
    "base" prendendo il primo valore di ogni lista.
    """
    cfg = {}
    for k, v in args_dict.items():
        if isinstance(v, (list, tuple)):
            cfg[k] = v[0]
        else:
            cfg[k] = v
    return cfg

def config_to_cli_list(cfg):
    """
    Converte il dict dei parametri in lista di argomenti da CLI
    compatibili con argparse (cancer_test.py).
    """
    cli = []
    for k, v in cfg.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            # cancer_test usa str2bool, quindi passiamo "True"/"False"
            cli.append(f"{flag}={str(v)}")
        else:
            cli.append(f"{flag}={v}")
    return cli

def compute_per_class_metrics(cm_df, f1_df):
    """
    Calcola metriche per classe (precision, recall, f1, support)
    a partire dalla confusion matrix e dal file degli F1 per classe.
    """
    cm = cm_df.values
    num_classes = cm.shape[0]

    per_class_metrics = []

    for c in range(num_classes):
        tp = cm[c, c]
        support = cm[c, :].sum()
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan

        # F1 per classe: lo leggiamo dal CSV di cancer_test.py
        row_f1 = f1_df[f1_df["class"] == c]
        if not row_f1.empty:
            f1 = float(row_f1["f1_score"].values[0])
        else:
            f1 = np.nan

        per_class_metrics.append(
            {
                "class": int(c),
                "support": int(support),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return per_class_metrics


def run_single_experiment(run_id, project_path, base_cfg, results_dir):
    """
    Esegue una singola run di cancer_test.py con una certa config,
    legge i file di output e restituisce:
    - metrics_global: dict con accuracy, F1 macro/weighted, ecc.
    - per_class_metrics: lista di dict per ogni classe
    Inoltre salva le confusion matrix / f1_scores con suffisso run_id.
    """
    # --------------------------------------------------------------------
    # 1) Costruisci la config per questa run (modifichiamo solo shuffle_index)
    # --------------------------------------------------------------------
    cfg = base_cfg.copy()
    cfg["shuffle_index"] = run_id  # qui decidi come variare il seed / split

    cli_args = config_to_cli_list(cfg)

    cmd = ["python", SCRIPT_NAME] + cli_args

    print(f"\n=== RUN {run_id+1}/{N_RUNS} ===")
    print("Comando:", " ".join(cmd))
    print("Working dir:", project_path)

    # --------------------------------------------------------------------
    # 2) Lancia cancer_test.py
    # --------------------------------------------------------------------
    completed = subprocess.run(cmd, cwd=project_path)
    if completed.returncode != 0:
        raise RuntimeError(f"Run {run_id} fallita con return code {completed.returncode}")

    # --------------------------------------------------------------------
    # 3) Leggi i file prodotti da cancer_test.py
    #    (devono già esistere in project_path)
    # --------------------------------------------------------------------
    cm_path = os.path.join(project_path, "confusion_matrix_test.csv")
    cm_norm_path = os.path.join(project_path, "confusion_matrix_test_normalized.csv")
    f1_path = os.path.join(project_path, "f1_scores_test.csv")

    if not (os.path.exists(cm_path) and os.path.exists(f1_path)):
        raise FileNotFoundError("Non trovo i file di output (confusion_matrix_test.csv / f1_scores_test.csv).")

    cm_df = pd.read_csv(cm_path, index_col=0)
    cm_norm_df = pd.read_csv(cm_norm_path, index_col=0) if os.path.exists(cm_norm_path) else None
    f1_df = pd.read_csv(f1_path)

    # --------------------------------------------------------------------
    # 4) Calcola metriche globali
    # --------------------------------------------------------------------
    cm = cm_df.values
    total = cm.sum()
    correct = np.trace(cm)
    global_accuracy = correct / total if total > 0 else np.nan

    # F1 macro e weighted dal CSV
    row_macro = f1_df[f1_df["class"] == "macro_avg"]
    row_weighted = f1_df[f1_df["class"] == "weighted_avg"]

    f1_macro = float(row_macro["f1_score"].values[0]) if not row_macro.empty else np.nan
    f1_weighted = float(row_weighted["f1_score"].values[0]) if not row_weighted.empty else np.nan

    metrics_global = {
        "run": run_id,
        "global_accuracy": global_accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }

    # --------------------------------------------------------------------
    # 5) Metriche per classe
    # --------------------------------------------------------------------
    per_class_metrics = compute_per_class_metrics(cm_df, f1_df)
    for m in per_class_metrics:
        m["run"] = run_id

    # --------------------------------------------------------------------
    # 6) Salva i file specifici di questa run (confusion matrix / f1)
    # --------------------------------------------------------------------
    cm_run_path = os.path.join(results_dir, f"confusion_matrix_test_run{run_id}.csv")
    f1_run_path = os.path.join(results_dir, f"f1_scores_test_run{run_id}.csv")
    cm_df.to_csv(cm_run_path)
    f1_df.to_csv(f1_run_path, index=False)
    if cm_norm_df is not None:
        cm_norm_run_path = os.path.join(results_dir, f"confusion_matrix_test_normalized_run{run_id}.csv")
        cm_norm_df.to_csv(cm_norm_run_path)

    return metrics_global, per_class_metrics


def main():
    # --------------------------------------------------------------------
    # 0) Path di base
    # --------------------------------------------------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Se run_tests.py è nella cartella padre del repo Li24:
    project_path = os.path.join(base_dir, paths[0])

    # Se invece run_tests.py è nella stessa cartella di cancer_test.py,
    # sostituisci la riga sopra con:
    # project_path = base_dir

    results_dir = os.path.join(project_path, "results_run_tests")
    os.makedirs(results_dir, exist_ok=True)

    base_cfg = build_base_config(args)

    all_global_rows = []     # una riga per run con accuracy / f1 macro / weighted
    all_per_class_rows = []  # una riga per run X classe

    for run_id in range(N_RUNS):
        metrics_global, per_class_metrics = run_single_experiment(
            run_id=run_id,
            project_path=project_path,
            base_cfg=base_cfg,
            results_dir=results_dir,
        )

        all_global_rows.append(metrics_global)
        all_per_class_rows.extend(per_class_metrics)

    # --------------------------------------------------------------------
    # 7) Salva tabelle riassuntive
    # --------------------------------------------------------------------
    df_global = pd.DataFrame(all_global_rows)
    df_per_class = pd.DataFrame(all_per_class_rows)

    summary_global_path = os.path.join(results_dir, "summary_global_metrics_10runs.csv")
    summary_per_class_path = os.path.join(results_dir, "summary_per_class_metrics_10runs.csv")

    df_global.to_csv(summary_global_path, index=False)
    df_per_class.to_csv(summary_per_class_path, index=False)

    print("\n=== RISULTATI RIASSUNTIVI ===")
    print("Global metrics (per run):")
    print(df_global)
    print("\nTabelle salvate in:")
    print(" -", summary_global_path)
    print(" -", summary_per_class_path)


if __name__ == "__main__":
    main()