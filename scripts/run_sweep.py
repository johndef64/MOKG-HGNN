import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import wandb
from multiomics_gnn.pancancer_prediction.experiments.experiment_runner import ExperimentRunner, ExperimentDataLoader
from multiomics_gnn.config.loader import load_config
import copy
import yaml
import itertools

def deep_merge(base: dict, patch: dict) -> dict:
    """serve a unire due dizionari Python in modo ricorsivo, 
    assicurandosi che i dati annidati vengano combinati anziché sovrascritti"""
    out = copy.deepcopy(base)
    stack = [(out, patch)]
    while stack:
        dst, src = stack.pop()
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                stack.append((dst[k], v))
            else:
                dst[k] = copy.deepcopy(v)
    return out
# support for grid mode implementation so for example train.num_genes = [500, 1000, 2000]
def set_nested(d: dict, dotted_key: str, value):
    """Imposta d['a']['b']['c']=value partendo da 'a.b.c'."""
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def expand_grid(grid: dict):
    """
    grid = {"train.num_genes":[500,1000], "training.lr":[1e-3, 5e-4]}
    -> genera overrides per ogni combinazione (cartesiano).
    """
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        ov = {}
        for k, v in zip(keys, combo):
            set_nested(ov, k, v)
        yield ov

def main():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True,
                    help="Path to a sweep YAML (with 'experiments:' and/or 'grid:' top-level keys)")
    ap.add_argument("--config", default="configs/config_final.yml",
                    help="Base config to apply the sweep overrides on top of "
                         "(default: configs/config_final.yml = thesis final setup)")
    args = ap.parse_args()

    base_config_path = args.config
    print(f"Loading base config from: {base_config_path}")
    base_config = load_config(base_config_path)

    sweep_config_path = args.sweep
    sweep_config = load_config(sweep_config_path)
    print(f"Loaded sweep config from: {sweep_config_path}")

    # load data
    print("Initializing data loader...")
    exp_data_loader = ExperimentDataLoader(config=base_config)
    print("Data loader initialized. Loading data...")
    expression_data, cnv_data, mirna_data = exp_data_loader.load_raw_data()
    print("Data loaded successfully.")
    print(f"Expression data shape: {expression_data.shape}, CNV data shape: {cnv_data.shape}, miRNA data shape: {mirna_data.shape if mirna_data is not None else 'N/A'}")
    runner = ExperimentRunner(expression_data, cnv_data, mirna_data)

    print("Initialized ExperimentRunner.")
    sweep_summary = {}
    if "experiments" in sweep_config and sweep_config["experiments"]:
        for exp in sweep_config["experiments"]:
            name = exp["name"]
            print(f"Running experiment: {name}")

            overrides = exp.get("overrides", {}) or {}

            cfg_finale = deep_merge(base_config, overrides)

            #try:
            summary = runner.run_experiment(cfg_finale, experiment_name=name)
            print(f"Summary for experiment {name}: {summary}")
            #except Exception as e:
            #    print(f"Experiment {name} failed with error: {e}")
            #    print(f"riga: {e.__traceback__.tb_lineno}")
            sweep_summary[name] = summary
    if "grid" in sweep_config and sweep_config["grid"]:
        grid = sweep_config["grid"]
        fixed = sweep_config.get("fixed_overrides", {}) or {}
        base_name = sweep_config.get("name", "grid")

        i = 0
        for grid_overrides in expand_grid(grid):
            i += 1
            overrides = deep_merge(fixed, grid_overrides)
            name = f"{base_name}_{i:03d}"

            print(f"Running grid experiment: {name}")
            print(f"With overrides: {overrides}")
            cfg_finale = deep_merge(base_config, overrides)
            summary = runner.run_experiment(cfg_finale, experiment_name=name)
            sweep_summary[name] = summary
    # Salva i risultati in una tabella CSV
    import pandas as pd
    df = pd.DataFrame.from_dict(sweep_summary, orient="index")
    
    results_dir = base_config["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    output_csv = os.path.join(results_dir, "sweep_summary.csv")
    df.to_csv(output_csv)
    print(f"Saved sweep summary to: {output_csv}")

if __name__ == "__main__":
    main()

