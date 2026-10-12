"""Shared definitions for the version-3 factorial experiment (graph x shallow branch).

Plan: docs/versione_3/piano_v3_quattro_bracci.md. Six arms on the paired feature-scarcity
grid (7 gene levels x 5 splits, model seed 2025, identical panel per level):

  existing  mol_s  MOGNN-TF with shallow      results/feature_collapse/mognntf_g*
            mol    MOGNN-TF without shallow   results/no_shallow/runs
            kg     MOKG-HGNN                  results/feature_collapse/mokghgnn_g*
  new       kg_s   MOKG-HGNN + shallow        (this experiment)
            sh_kg  shallow MLP only, MOKG-HGNN recipe
            sh_tf  shallow MLP only, MOGNN-TF recipe

New code only: the panels, splits and MOGNN-TF runner come from scripts/no_shallow/common.py
(imported); the MOKG-HGNN runner is the original one, with the model class swapped for the
duration of a run (see run_mokg). No existing file is modified.
"""

import contextlib
import copy
import glob
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "no_shallow"))
import common as NS  # noqa: E402  (sets sys.path and cwd to the repository root)
# scripts/no_shallow also has compare.py, sanity_check.py, ...: this folder must win
sys.path.remove(str(HERE.parent / "no_shallow"))
sys.path.append(str(HERE.parent / "no_shallow"))
if str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))

REPO = NS.REPO
LEVELS, SEEDS, MODEL_SEED = NS.LEVELS, NS.SEEDS, NS.MODEL_SEED

MOKG_BASE_CONFIG = REPO / "configs" / "best_model_config.yml"
SHALLOW_ONLY_TF_CONFIG = REPO / "configs" / "mognntf_shallow_only.yaml"
NS.VARIANTS["shallow_only"] = SHALLOW_ONLY_TF_CONFIG

OUT = REPO / "results" / "four_arms"
HETERO = OUT / "hetero"          # node vocabularies per (level, split), verified on the template
RUNS = OUT / "runs"              # one JSON per completed run
RUNS_RAW = OUT / "runs_raw"      # run folders written by the original runners
SMOKE = OUT / "smoke"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
SANITY_JSON = OUT / "sanity_check.json"

ARMS = {
    "mol_s": "MOGNN-TF + shallow",
    "mol": "MOGNN-TF (graph only)",
    "kg": "MOKG-HGNN (graph only)",
    "kg_s": "MOKG-HGNN + shallow",
    "sh_tf": "Shallow only (MOGNN-TF recipe)",
    "sh_kg": "Shallow only (MOKG-HGNN recipe)",
}
NEW_ARMS = ("kg_s", "sh_kg", "sh_tf")
MOKG_ARMS = ("kg", "kg_s", "sh_kg")    # "kg" here = re-run through this code (sanity only)


def template_path(level, seed):
    return REPO / "data" / "prior_knowledge" / "hetero" / f"collapse_g{level}_seed{seed}.pt"


def run_json(arm, level, seed, runs_dir=RUNS):
    return Path(runs_dir) / f"{arm}_g{level}_seed{seed}.json"


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    NS.write_text_atomic(path, json.dumps(obj, indent=1))


# ----------------------------------------------------------------------------------
# MOKG-HGNN arms
# ----------------------------------------------------------------------------------
def mokg_config(arm, level, seed, hetero_dir, results_dir, smoke_epochs=None):
    """best_model_config.yml + the overrides of scripts/kg_hgnn/collapse_mokghgnn.sh, plus
    the per-run node vocabulary (data.hetero_dir) and the arm name (recorded only)."""
    import yaml
    cfg = yaml.safe_load(open(MOKG_BASE_CONFIG, encoding="utf-8"))
    cfg["project"]["seed"] = MODEL_SEED
    cfg["project"]["experiment_name"] = f"{arm}/g{level}_seed{seed}"
    cfg["paths"]["results_dir"] = str(Path(results_dir).relative_to(REPO)).replace("\\", "/")
    cfg["data"]["split_dir"] = f"data/training/splits/splits_seed_{seed}"
    cfg["data"]["template_path"] = str(template_path(level, seed).relative_to(REPO)).replace("\\", "/")
    cfg["data"]["hetero_dir"] = str(Path(hetero_dir).relative_to(REPO)).replace("\\", "/")
    cfg["model"]["backbone"] = "hetero_sage"
    cfg["model"]["arm"] = arm
    if smoke_epochs:
        cfg["train"]["num_epochs"] = int(smoke_epochs)
    return cfg


def num_nodes_of(level, seed):
    import torch
    t = torch.load(template_path(level, seed), weights_only=False)
    return {nt: int(t[nt].num_nodes) for nt in t.node_types}


@contextlib.contextmanager
def swapped_model(arm, num_nodes):
    """Replace the model class the MOKG-HGNN runner instantiates, for one run only."""
    from multiomics_kg_hgnn.pancancer_prediction.experiments import runner
    import models
    original = runner.HeteroMultiScaleGNN
    runner.HeteroMultiScaleGNN = models.mokg_factory(arm, num_nodes)
    try:
        yield runner
    finally:
        runner.HeteroMultiScaleGNN = original


def run_mokg(arm, level, seed, raw_root=RUNS_RAW, smoke_epochs=None):
    import hetero_dirs
    hdir = hetero_dirs.ensure(level, seed)
    cfg = mokg_config(arm, level, seed, hdir, raw_root, smoke_epochs)
    nn_ = num_nodes_of(level, seed)
    t0 = time.perf_counter()
    with swapped_model(arm, nn_) as runner:
        res = runner.run_experiment(copy.deepcopy(cfg))
    duration = time.perf_counter() - t0
    run_dir = Path(res["run_dir"])
    pc = json.loads((run_dir / "per_class_metrics.json").read_text())["per_class"]
    pc.sort(key=lambda r: int(r["class_idx"]))
    with open(run_dir / "history.csv") as fh:
        epochs = sum(1 for _ in fh) - 1
    return {
        "arm": arm, "level": int(level), "split_seed": int(seed), "model_seed": MODEL_SEED,
        "test_macro_f1": float(res["macro_f1"]), "test_accuracy": float(res["accuracy"]),
        "val_macro_f1": float(res["best_val_macro_f1"]), "epochs_trained": epochs,
        "per_class": [[r["class"], float(r["f1"]), int(r["support"])] for r in pc],
        "num_nodes": nn_, "smoke_epochs": smoke_epochs, "duration_s": round(duration, 1),
        "run_dir": str(run_dir).replace("\\", "/"), "env": NS.environment(),
    }


# ----------------------------------------------------------------------------------
# MOGNN-TF shallow-only arm (original runner, model class swapped in the builder)
# ----------------------------------------------------------------------------------
@contextlib.contextmanager
def swapped_gcn():
    from multiomics_gnn.models import gcn_tf
    import models
    original = gcn_tf.GCN
    gcn_tf.GCN = models.GCNShallowOnly
    try:
        yield
    finally:
        gcn_tf.GCN = original


def run_sh_tf(level, seed, raw_root=RUNS_RAW, smoke_epochs=None, num_workers=None):
    with swapped_gcn():
        rec = NS.run_variant("shallow_only", level, seed, raw_root=raw_root,
                             smoke_epochs=smoke_epochs, num_workers=num_workers)
    names = list(rec["per_class_f1"])
    rec["per_class"] = [[n, rec["per_class_f1"][n], rec["per_class_support"][n]] for n in names]
    rec["arm"] = "sh_tf"
    return rec


def run_arm(arm, level, seed, raw_root=RUNS_RAW, smoke_epochs=None, num_workers=None):
    if arm == "sh_tf":
        return run_sh_tf(level, seed, raw_root, smoke_epochs, num_workers)
    if arm in MOKG_ARMS:
        return run_mokg(arm, level, seed, raw_root, smoke_epochs)
    raise ValueError(f"unknown arm {arm}")


# ----------------------------------------------------------------------------------
# results of all six arms: {arm: {(level, seed): macro-F1}}
# ----------------------------------------------------------------------------------
def read_runs(arm, runs_dir=RUNS):
    out = {}
    for f in glob.glob(str(Path(runs_dir) / f"{arm}_g*_seed*.json")):
        d = json.loads(Path(f).read_text())
        if d.get("smoke_epochs"):
            continue
        out[(int(d["level"]), int(d["split_seed"]))] = d
    return out


def load_f1_series():
    s = {
        "mol_s": {k: v["f1"] for k, v in NS.read_with_shallow_existing().items()},
        "mol": {k: v["test_macro_f1"] for k, v in NS.read_new_runs("no_shallow").items()
                if not v.get("smoke_epochs")},
        "kg": {k: v["f1"] for k, v in NS.read_multiscale_existing().items()},
    }
    for arm in NEW_ARMS:
        s[arm] = {k: v["test_macro_f1"] for k, v in read_runs(arm).items()}
    return s
