import copy
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import optuna
import torch

from multiomics_gnn.config.loader import load_config
from multiomics_gnn.pancancer_prediction.experiments.experiment_runner import ExperimentRunner


def objective_factory(base_cfg: dict, fixed_seed: int = 42):
    runner = ExperimentRunner()

    def objective(trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(base_cfg)

        # Seed fisso durante Optuna per confrontabilità
        cfg["project"]["seed"] = fixed_seed

        # Disabilita wandb durante tuning (consigliato)
        if "wandb" in cfg:
            cfg["wandb"]["enabled"] = False

        # velocizza tuning (coerente con i tuoi plot: plateau presto)
        cfg["train"]["num_epochs"] = 50
        cfg["early_stopping"]["enabled"] = True
        cfg["early_stopping"]["patience"] = 7
        cfg["early_stopping"]["metric"] = "f1"
        cfg["early_stopping"]["strategy"] = "maximize"

        # ---- Search space (aderente al tuo config.yml) ----
        cfg["train"]["learning_rate"] = trial.suggest_categorical("lr",[1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 3e-3])
        cfg["train"]["weight_decay"] = trial.suggest_categorical("weight_decay",[0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3])


        cfg["model"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.6, step=0.2)
        cfg["model"]["poolsize"] = trial.suggest_categorical("poolsize", [4, 8, 16])

        # jk_mode solo se JK è attivo nel tuo config
        cfg["model"]["jumping_knowledge"] = trial.suggest_categorical("jumping_knowledge", [True, False])

        if cfg["model"].get("jumping_knowledge", False):
            cfg["model"]["jk_mode"] = trial.suggest_categorical("jk_mode", ["max", "cat"])

        # NODI: nel tuo setup = num_gene (per te è il vero vincolo VRAM)
        cfg["data"]["num_gene"] = trial.suggest_categorical("num_gene", [500, 700, 1000])

        # opzionale: sampler strategy (se vuoi includerlo)
        cfg["sampler_strategy"] = trial.suggest_categorical("sampler_strategy", ["weighted", "random"])
        # weight sampler mode (se includi sampler strategy)
        if cfg["sampler_strategy"] == "weighted":
            cfg["sampler_mode"] = trial.suggest_categorical("sampler_mode", ["nulite", "alpha"])
        # gamma per weighted sampler (se includi questa strategia)
        if cfg["sampler_strategy"] == "weighted":
            if cfg["sampler_mode"] == "nulite":
                cfg["sampler_gamma"] = trial.suggest_categorical("sampler_gamma", [0.95, 0.97, 0.99])
            elif cfg["sampler_mode"] == "alpha":
                cfg["sampler_alpha"] = trial.suggest_categorical("sampler_alpha", [2,3])

        # opzionale: decoder on/off
        cfg["train"]["decoder"] = trial.suggest_categorical("decoder", [True, False])

        # parallel GAT: on/off
        cfg["model"]["parallel"] = trial.suggest_categorical("parallel", [True, False])

        # opzionale: feature selection mode
        cfg["data"]["feature_selection_method"] = trial.suggest_categorical("feature_selection_method", ["variance"])

        # batch size: attenzione alla VRAM, magari meglio non includerlo o limitare a 2-3 valori
        cfg["data"]["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        # ----------------------------------------------------

        try:
            run_name = f"optuna_trial_{trial.number}"
            summary = runner.run_experiment(cfg, experiment_name=run_name, trial=trial)

            # --- salva in tabella: nome run + metriche best/test (se presenti nel summary) ---
            trial.set_user_attr("run_name", run_name)
            trial.set_user_attr("best_val_metric", summary.get("best_val_metric"))
            trial.set_user_attr("best_epoch", summary.get("best_epoch"))
            trial.set_user_attr("test_f1", summary.get("test_f1"))
            trial.set_user_attr("test_acc", summary.get("test_acc"))

            # Obiettivo: best val metric secondo early stopping (f1)
            score = summary["best_val_metric"]
            if score is None:
                return 0.0
            return float(score)

        except RuntimeError as e:
            # gestione OOM: non far crashare lo studio
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return 0.0
            raise


    return objective


def run_study(
    config_path: str = "configs/config.yml",
    n_trials: int = 35,
    timeout_hours: float = 10.0,
    fixed_seed: int = 42,
    study_name: str = "pancan_optuna_simple",
    out_dir: str = "./results/pancan/optuna"
):
    base_cfg = load_config(config_path)
    objective = objective_factory(base_cfg, fixed_seed=fixed_seed)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=15)
    study = optuna.create_study(direction="maximize", pruner=pruner, study_name=study_name)

    study.optimize(objective, n_trials=n_trials, timeout=int(timeout_hours * 3600), )

    import pandas as pd

    df = study.trials_dataframe(attrs=("number", "state", "value", "params", "user_attrs", "duration"))

    # colonne "core" che vuoi vedere in testa
    core_cols = [
        "number",
        "user_attrs_run_name",
        "value",
        "user_attrs_best_val_metric",
        "user_attrs_best_epoch",
        "user_attrs_test_f1",
        "user_attrs_test_acc",
        "state",
        "duration",
    ]
    core_cols = [c for c in core_cols if c in df.columns]

    # tutte le colonne parametri scelte da Optuna
    param_cols = [c for c in df.columns if c.startswith("params_")]

    df = df[core_cols + param_cols].sort_values("value", ascending=False)

    # stampa tabella completa a console
    print("\n=== OPTUNA TRIALS REPORT (sorted by value desc) ===")
    print(df.to_string(index=False))

    # salva report nella stessa out_dir dello studio (lo crei sotto)


    out_dir = Path(out_dir) / study_name
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "optuna_trials_report.csv", index=False)
    df.to_excel(out_dir / "optuna_trials_report.xlsx", index=False)

    payload = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
    }
    (out_dir / "best.json").write_text(json.dumps(payload, indent=2))

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
    print("Saved best params to:", out_dir / "best.json")
    return study


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="Optuna hyperparameter search for MOGNN-TF")
    ap.add_argument("--config", default="configs/config.yml",
                    help="Base config to start from (Optuna overrides a subset of fields)")
    ap.add_argument("--n-trials", type=int, default=35,
                    help="Number of Optuna trials (paper used 35)")
    ap.add_argument("--timeout-hours", type=float, default=10.0,
                    help="Hard timeout in hours (paper used 10)")
    ap.add_argument("--fixed-seed", type=int, default=42)
    ap.add_argument("--study-name", default="pancan_optuna_simple")
    ap.add_argument("--out-dir", default="./results/pancan/optuna")
    args = ap.parse_args()
    run_study(
        config_path=args.config,
        n_trials=args.n_trials,
        timeout_hours=args.timeout_hours,
        fixed_seed=args.fixed_seed,
        study_name=args.study_name,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    _cli()
