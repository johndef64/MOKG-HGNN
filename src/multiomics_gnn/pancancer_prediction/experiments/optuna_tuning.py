import copy
import json
from pathlib import Path

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
        cfg["train"]["learning_rate"] = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
        cfg["train"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True)

        cfg["model"]["dropout"] = trial.suggest_float("dropout", 0.1, 0.5)
        cfg["model"]["poolsize"] = trial.suggest_categorical("poolsize", [4, 8, 16])

        # jk_mode solo se JK è attivo nel tuo config
        if cfg["model"].get("jumping_knowledge", False):
            cfg["model"]["jk_mode"] = trial.suggest_categorical("jk_mode", ["max", "cat", "sum"])

        # NODI: nel tuo setup = num_gene (per te è il vero vincolo VRAM)
        cfg["data"]["num_gene"] = trial.suggest_categorical("num_gene", [500, 700, 1000])

        # opzionale: sampler strategy (se vuoi includerlo)
        cfg["sampler_strategy"] = trial.suggest_categorical("sampler_strategy", ["weighted", "random"])

        # opzionale: decoder on/off
        cfg["train"]["decoder"] = trial.suggest_categorical("decoder", [True, False])

        # opzionale: feature selection mode
        cfg["data"]["feature_selection_mode"] = trial.suggest_categorical("feature_selection_mode", ["fvalue", "fvalue_per_class"])

        try:
            summary = runner.run_experiment(cfg, experiment_name=f"optuna_trial_{trial.number}", trial=trial)

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
    n_trials: int = 36,
    timeout_hours: float = 8.0,
    fixed_seed: int = 42,
    study_name: str = "pancan_optuna_simple",
    out_dir: str = "./results/pancan/optuna"
):
    base_cfg = load_config(config_path)
    objective = objective_factory(base_cfg, fixed_seed=fixed_seed)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction="maximize", pruner=pruner, study_name=study_name)

    study.optimize(objective, n_trials=n_trials, timeout=int(timeout_hours * 3600))

    out_dir = Path(out_dir) / study_name
    out_dir.mkdir(parents=True, exist_ok=True)

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


if __name__ == "__main__":
    run_study()
