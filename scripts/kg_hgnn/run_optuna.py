"""Optuna hyperparameter search for the heterogeneous multi-scale model.

Same skeleton as MOGNN-TF's scripts/run_optuna.py (objective with trial.suggest_*,
MedianPruner, best.json + CSV/xlsx report, OOM guard) but with THIS model's real
hyperparameters. The graph template is held fixed — only model/training knobs are
tuned (top_genes / metapath are a separate sweep, since they'd force a template
rebuild per trial). Objective = validation macro-F1 (never the test set).

    conda run -n gnn python scripts/kg_hgnn/run_optuna.py \
        --config configs/config_kg_hgnn.yml --n-trials 35 --timeout-hours 10
"""

import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import optuna
import torch

from multiomics_gnn.config.loader import load_config  # reuse the simple YAML loader
from multiomics_kg_hgnn.pancancer_prediction.experiments.runner import run_experiment


def objective_factory(base_cfg, fixed_seed=42, tune_epochs=60, tune_patience=12,
                      verbose=True):
    def objective(trial):
        cfg = copy.deepcopy(base_cfg)

        # fixed seed during tuning -> trials are comparable
        cfg["project"]["seed"] = fixed_seed
        cfg["project"]["experiment_name"] = f"optuna_trial_{trial.number}"
        # shorter schedule during tuning (val plateaus well before 300)
        cfg["train"]["num_epochs"] = tune_epochs
        cfg.setdefault("early_stopping", {})["patience"] = tune_patience

        # ---- search space (this model's real knobs) ----
        cfg["model"]["backbone"] = "hgt"     # only hgt implemented; sage/rgcn are TODO
        cfg["model"]["hidden"] = trial.suggest_categorical("hidden", [32, 64, 128, 256])
        cfg["model"]["num_layers"] = trial.suggest_int("num_layers", 1, 3)
        cfg["model"]["heads"] = trial.suggest_categorical("heads", [1, 2, 4, 8])
        cfg["model"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.6, step=0.1)
        # multi-scale readout: which scales feed the classifier
        cfg["model"]["readout_types"] = trial.suggest_categorical(
            "readout_types",
            [["gene"], ["gene", "pathway"], ["gene", "pathway", "GO_term"],
             ["gene", "pathway", "GO_term", "disease"]])

        cfg["train"]["learning_rate"] = trial.suggest_categorical(
            "lr", [1e-2, 5e-3, 3e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4])
        cfg["train"]["weight_decay"] = trial.suggest_categorical(
            "weight_decay", [0.0, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3])
        cfg["train"]["class_weighted_loss"] = trial.suggest_categorical(
            "class_weighted_loss", [True, False])

        cfg["data"]["batch_size"] = trial.suggest_categorical("batch_size", [8, 16, 32, 64])

        # imbalance handling on the sampler
        cfg["sampler_strategy"] = trial.suggest_categorical("sampler_strategy", ["none", "weighted"])
        if cfg["sampler_strategy"] == "weighted":
            cfg["sampler_gamma"] = trial.suggest_categorical("sampler_gamma", [0.90, 0.95, 0.98, 0.99])

        # heads must divide hidden for HGTConv — skip invalid combos cleanly
        if cfg["model"]["hidden"] % cfg["model"]["heads"] != 0:
            if verbose:
                print(f"[trial {trial.number}] pruned (hidden {cfg['model']['hidden']} "
                      f"not divisible by heads {cfg['model']['heads']})", flush=True)
            raise optuna.TrialPruned()

        if verbose:
            print(f"\n[trial {trial.number}] START params={trial.params}", flush=True)

        # during tuning: don't write per-epoch spam, but DO print one line/epoch so
        # you can see progress. flush=True so it streams through nohup/tee.
        def tlog(msg):
            s = str(msg)
            if verbose and ("epoch" in s or "[TEST]" in s or "training done" in s):
                print(f"  [t{trial.number}] {s}", flush=True)

        try:
            summary = run_experiment(cfg, logger=tlog)
            score = summary.get("best_val_macro_f1")
            trial.set_user_attr("test_macro_f1", summary.get("macro_f1"))
            trial.set_user_attr("test_accuracy", summary.get("accuracy"))
            trial.set_user_attr("run_dir", summary.get("run_dir"))
            return float(score) if score is not None else 0.0
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                if verbose:
                    print(f"[trial {trial.number}] OOM -> score 0.0", flush=True)
                return 0.0
            raise

    return objective


def _progress_callback(study, trial):
    """Printed after every finished trial: score, best-so-far, elapsed."""
    dur = trial.duration.total_seconds() if trial.duration else 0.0
    val = "pruned" if trial.value is None else f"{trial.value:.4f}"
    try:
        best = f"{study.best_value:.4f} (trial {study.best_trial.number})"
    except ValueError:
        best = "n/a"
    done = len([t for t in study.trials if t.state.is_finished()])
    print(f"[optuna] trial {trial.number} done: val_macroF1={val} | "
          f"best={best} | {dur:.0f}s | finished {done} trials", flush=True)


def run_study(config_path="configs/config_kg_hgnn.yml", n_trials=35, timeout_hours=10.0,
              fixed_seed=42, tune_epochs=60, tune_patience=12,
              study_name="kg_hgnn_optuna", out_dir="./results/optuna"):
    base_cfg = load_config(config_path)
    objective = objective_factory(base_cfg, fixed_seed, tune_epochs, tune_patience)

    # make Optuna's own logs visible (it defaults to quiet under some setups)
    optuna.logging.set_verbosity(optuna.logging.INFO)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=15)
    study = optuna.create_study(direction="maximize", pruner=pruner, study_name=study_name)
    print(f"[optuna] starting study '{study_name}': {n_trials} trials, "
          f"{timeout_hours}h timeout, {tune_epochs} epochs/trial", flush=True)
    study.optimize(objective, n_trials=n_trials, timeout=int(timeout_hours * 3600),
                   callbacks=[_progress_callback])

    import pandas as pd
    df = study.trials_dataframe(attrs=("number", "state", "value", "params", "user_attrs", "duration"))
    core = [c for c in ["number", "value", "user_attrs_test_macro_f1", "user_attrs_test_accuracy",
                        "state", "duration", "user_attrs_run_dir"] if c in df.columns]
    params = [c for c in df.columns if c.startswith("params_")]
    df = df[core + params].sort_values("value", ascending=False)

    print("\n=== OPTUNA TRIALS (sorted by val macro-F1) ===")
    print(df.to_string(index=False))

    out = Path(out_dir) / study_name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "optuna_trials_report.csv", index=False)
    try:
        df.to_excel(out / "optuna_trials_report.xlsx", index=False)
    except Exception as e:
        print(f"[warn] xlsx not written ({e}); CSV is available.")
    (out / "best.json").write_text(json.dumps(
        {"best_value": study.best_value, "best_params": study.best_params,
         "best_trial_number": study.best_trial.number}, indent=2))

    print(f"\nBest val macro-F1: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"Saved -> {out}/best.json")
    return study


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="Optuna hyperparameter search for MOKG-HGNN")
    ap.add_argument("--config", default="configs/config_kg_hgnn.yml")
    ap.add_argument("--n-trials", type=int, default=35)
    ap.add_argument("--timeout-hours", type=float, default=10.0)
    ap.add_argument("--fixed-seed", type=int, default=42)
    ap.add_argument("--tune-epochs", type=int, default=60,
                    help="Epoch cap during tuning (shorter than a final run).")
    ap.add_argument("--tune-patience", type=int, default=12)
    ap.add_argument("--study-name", default="kg_hgnn_optuna")
    ap.add_argument("--out-dir", default="./results/optuna")
    args = ap.parse_args()
    run_study(args.config, args.n_trials, args.timeout_hours, args.fixed_seed,
              args.tune_epochs, args.tune_patience, args.study_name, args.out_dir)


if __name__ == "__main__":
    _cli()
