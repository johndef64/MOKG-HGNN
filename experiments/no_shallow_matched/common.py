"""Shared helpers for the no-shallow matched experiment.

New code only: every function here REUSES the existing pipeline by import and never
modifies an existing file. The protocol is the one of scripts/collapse_mognntf.py
(paired feature-scarcity experiment, Sec. 4.5.3 of the thesis):

  * panel per (level, split): unified train-only variance FS (top 200 TF, 100 miRNA
    requested) + miRNA/TF that survive in the MOKG-HGNN graph, via the SAME function
    `_panel` used by the original experiment (imported, not copied);
  * config: the same overrides as collapse_mognntf.py, applied with its own `_set`;
  * training/evaluation: MOGNN-TF's ExperimentRunner, unchanged.

Importing this module changes the working directory to the repository root, because
the existing pipeline resolves its data paths relative to it.
"""

import copy
import csv
import glob
import hashlib
import json
import logging
import math
import os
import platform
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO / "src"), str(REPO / "benchmark" / "MOGNN-TF"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(REPO)

# --- experimental grid (identical to the original paired experiment) ---------------
LEVELS = [700, 500, 300, 150, 100, 50, 20]
SEEDS = [42, 43, 44, 45, 46]
MODEL_SEED = 2025

# --- inputs (read only) -----------------------------------------------------------
ORIGINAL_CONFIG = REPO / "benchmark" / "MOGNN-TF" / "configs" / "config_final.yml"
NO_SHALLOW_CONFIG = HERE / "configs" / "no_shallow.yaml"
EXISTING_ROOT = REPO / "results" / "feature_collapse"
EXISTING_MOGNNTF_SUMMARY = EXISTING_ROOT / "collapse_mognntf_summary.csv"
SPLITS_ROOT = REPO / "data" / "training" / "splits"
SPLIT_PREFIX = "common_trimmed_shuffle_index"

# --- outputs (all new, inside this folder) -----------------------------------------
OUT = HERE / "results"
PANELS = OUT / "panels"          # regenerated FS + survived panel per (level, split)
RUNS = OUT / "runs"              # one JSON per completed run
RUNS_RAW = OUT / "runs_raw"      # MOGNN-TF run folders (checkpoint, logs, confusion matrix)
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
SANITY_JSON = OUT / "sanity_check.json"

VARIANTS = {"no_shallow": NO_SHALLOW_CONFIG, "with_shallow": ORIGINAL_CONFIG}

# keys that legitimately differ between two runs of the same (variant, level, split):
#   paths.results_dir : output location
#   data.num_workers  : execution-only. The weighted sampler draws the batch indices in
#                       the main process with its own seeded generator, the DataLoader
#                       base seed is drawn in _BaseDataLoaderIter for any num_workers, and
#                       the dataset is deterministic, so batches are identical for 0 or N
#                       workers. On Windows (spawn) num_workers=1 re-launches a Python
#                       worker at every pass over a loader and makes epochs 5-50x slower.
VOLATILE_KEYS = {"paths.results_dir", "data.num_workers"}


# ----------------------------------------------------------------------------------
# hashing / config helpers
# ----------------------------------------------------------------------------------
def write_text_atomic(path, text):
    """Write via a temp file + os.replace, so a job killed mid-write (SLURM preemption)
    never leaves a truncated JSON that a resumed job would take as complete."""
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def sha256_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def config_diff(a, b, ignore=VOLATILE_KEYS):
    """{dotted_key: (a_value, b_value)} for every key whose value differs."""
    fa, fb = flatten(a), flatten(b)
    return {k: (fa.get(k), fb.get(k)) for k in sorted(set(fa) | set(fb))
            if k not in ignore and fa.get(k) != fb.get(k)}


def config_hash(cfg):
    """Hash of the full run config except the output path."""
    f = {k: v for k, v in flatten(cfg).items() if k not in VOLATILE_KEYS}
    return sha256_obj(f)


def load_yaml_config(path):
    from multiomics_gnn.config.loader import load_config
    return load_config(str(path))


# ----------------------------------------------------------------------------------
# splits
# ----------------------------------------------------------------------------------
def split_files(seed):
    d = SPLITS_ROOT / f"splits_seed_{seed}"
    return {part: d / f"{SPLIT_PREFIX}_{part}.tsv" for part in ("train", "val", "test")}


def split_hash(seed):
    h = hashlib.sha256()
    for part, p in split_files(seed).items():
        h.update(part.encode())
        h.update(Path(p).read_bytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------------
# panels (exactly the original procedure, reused by import)
# ----------------------------------------------------------------------------------
def panel_dir(level, seed):
    return PANELS / f"g{level}_seed{seed}"


_OMIC_CACHE = {}


def _memoize_read_omic():
    """select_features() re-reads the 750 MB expression matrix at every call. Memoize
    patient_features.read_omic inside THIS process only (no file is changed). It is a
    pure reader and select_features never mutates the returned frame (it only takes
    reindexed / iloc views and column names), so the panels are unaffected."""
    from multiomics_kg_hgnn.pancancer_prediction.preprocessing import patient_features as pf
    if getattr(pf.read_omic, "_memoized", False):
        return
    original = pf.read_omic

    def cached(path, sep="\t"):
        key = (os.path.abspath(str(path)), sep)
        if key not in _OMIC_CACHE:
            _OMIC_CACHE[key] = original(path, sep=sep)
        return _OMIC_CACHE[key]
    cached._memoized = True
    pf.read_omic = cached


def get_panel(level, seed, base_cfg, regenerate=False):
    """Ordered (genes, tf, mirna) for this (level, split).

    Regenerated with scripts/collapse_mognntf.py::_panel (same FS request sizes as the
    original: top_tf = data.num_tf, top_mirna = data.num_mirna of the base config) into
    a NEW folder, then cached as panel.json. Order matters: it is the node order of the
    MOGNN-TF supra-graph and therefore of its index-window max pooling.
    """
    from scripts.collapse_mognntf import _panel
    _memoize_read_omic()
    pdir = panel_dir(level, seed)
    pjson = pdir / "panel.json"
    if pjson.exists() and not regenerate:
        p = json.loads(pjson.read_text())
        return p["genes"], p["tf"], p["mirna"]
    fs_top_tf = int(base_cfg.get("data", {}).get("num_tf", 200))
    fs_top_mirna = int(base_cfg.get("data", {}).get("num_mirna", 100))
    split_dir = f"data/training/splits/splits_seed_{seed}"
    pdir.mkdir(parents=True, exist_ok=True)
    genes, tf, mirna = _panel(str(pdir), split_dir, level, fs_top_tf, fs_top_mirna)
    genes, tf, mirna = list(genes), list(tf), list(mirna)
    write_text_atomic(pjson, json.dumps({"level": level, "split_seed": seed, "genes": genes,
                                         "tf": tf, "mirna": mirna,
                                         "fs_request": {"top_tf": fs_top_tf, "top_mirna": fs_top_mirna}},
                                        indent=1))
    return genes, tf, mirna


def panel_hash(genes, tf, mirna):
    return sha256_obj({"genes": list(genes), "tf": list(tf), "mirna": list(mirna)})


# ----------------------------------------------------------------------------------
# run config (same overrides as scripts/collapse_mognntf.py lines 112-127)
# ----------------------------------------------------------------------------------
def build_run_config(base_cfg, level, seed, panel, results_dir, model_seed=MODEL_SEED,
                     smoke_epochs=None):
    from scripts.collapse_mognntf import _set
    genes, tf, mirna = panel
    cfg = copy.deepcopy(base_cfg)
    _set(cfg, "data.num_gene", int(level))
    _set(cfg, "data.num_tf", len(tf))
    _set(cfg, "data.num_mirna", len(mirna))
    _set(cfg, "data.gene_list", list(genes))
    _set(cfg, "data.tf_list", list(tf))
    _set(cfg, "data.mirna_keep", list(mirna))
    _set(cfg, "project.split_seed", int(seed))
    _set(cfg, "project.seed", int(model_seed))
    _set(cfg, "paths.results_dir", str(results_dir))
    if smoke_epochs:
        _set(cfg, "train.num_epochs", int(smoke_epochs))
    return cfg


# ----------------------------------------------------------------------------------
# existing results (read only)
# ----------------------------------------------------------------------------------
def existing_mognntf_run_configs():
    """{(level, seed): used_config dict} for the existing with-shallow runs available
    on THIS machine (results/ may be a partial transfer from the server)."""
    import yaml
    out = {}
    for uc in glob.glob(str(EXISTING_ROOT / "mognntf_g*" / "run_*" / "used_config.yaml")):
        cfg = yaml.safe_load(open(uc))
        key = (int(cfg["data"]["num_gene"]), int(cfg["project"]["split_seed"]))
        if key in out:
            raise RuntimeError(f"two existing with-shallow runs for {key}: ambiguous reference")
        out[key] = cfg
    return out


def read_with_shallow_existing():
    """{(level, seed): {f1, acc, val_f1, n_mirna, n_tf}} from the original summary CSV."""
    rows = {}
    with open(EXISTING_MOGNNTF_SUMMARY, newline="") as fh:
        for r in csv.DictReader(fh):
            key = (int(r["genes"]), int(r["split_seed"]))
            if key in rows:
                raise RuntimeError(f"duplicate row {key} in {EXISTING_MOGNNTF_SUMMARY}")
            rows[key] = {"f1": float(r["test_macro_f1"]), "acc": float(r["test_accuracy"]),
                         "val_f1": float(r["val_macro_f1"]),
                         "n_mirna": int(r["mirna"]), "n_tf": int(r["tf"])}
    return rows


def read_multiscale_existing():
    """{(level, seed): {f1, acc, num_nodes}} from the MOKG-HGNN collapse runs."""
    rows = {}
    for mj in glob.glob(str(EXISTING_ROOT / "mokghgnn_g*" / "*" / "metrics.json")):
        run = Path(mj).parent
        level = int(re.search(r"mokghgnn_g(\d+)", str(run)).group(1))
        cfg = json.loads((run / "config.json").read_text())
        seed = int(cfg["data"]["split_dir"].rstrip("/\\").split("_")[-1])
        if (level, seed) in rows:
            raise RuntimeError(f"two MOKG-HGNN runs for {(level, seed)}: ambiguous reference")
        m = json.loads(Path(mj).read_text())
        gi = json.loads((run / "graph_info.json").read_text())
        rows[(level, seed)] = {"f1": float(m["test_macro_f1"]), "acc": float(m["test_accuracy"]),
                               "num_nodes": gi["num_nodes"], "run_dir": str(run)}
    return rows


def read_new_runs(variant="no_shallow", runs_dir=RUNS):
    rows = {}
    for f in glob.glob(str(Path(runs_dir) / f"{variant}_g*_seed*.json")):
        d = json.loads(Path(f).read_text())
        rows[(int(d["level"]), int(d["split_seed"]))] = d
    return rows


# ----------------------------------------------------------------------------------
# metrics from a finished MOGNN-TF run folder
# ----------------------------------------------------------------------------------
def per_class_from_confusion(run_dir):
    """Per-class F1 and support from <run_dir>/confusion_matrix_cm.csv (test counts,
    rows = true, cols = predicted, plus a SUM column written by save_metrics)."""
    with open(Path(run_dir) / "confusion_matrix_cm.csv", newline="") as fh:
        rdr = list(csv.reader(fh))
    header = rdr[0][1:]
    pred_cols = [i for i, h in enumerate(header) if h.startswith("pred_")]
    names = [header[i][len("pred_"):] for i in pred_cols]
    M = [[int(float(row[1:][i])) for i in pred_cols] for row in rdr[1:]
         if row and row[0].startswith("true_")]
    n = len(names)
    f1, support, predicted = {}, {}, {}
    for c in range(n):
        tp = M[c][c]
        fn = sum(M[c]) - tp
        fp = sum(M[r][c] for r in range(n)) - tp
        support[names[c]] = tp + fn
        predicted[names[c]] = tp + fp
        den = 2 * tp + fp + fn
        f1[names[c]] = (2 * tp / den) if den else 0.0
    # sklearn macro-F1 averages over labels present in y_true or y_pred
    present = [k for k in names if support[k] + predicted[k] > 0]
    macro = sum(f1[k] for k in present) / len(present)
    return f1, support, macro


def environment():
    import torch
    return {"host": platform.node(), "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


def _close_file_handlers():
    """MOGNN-TF adds a FileHandler to the root logger at every run and never removes it,
    so a later run would also write into earlier experiment.log files. Close them after
    each run (logging only, no numerical effect)."""
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            root.removeHandler(h)
            h.close()


# ----------------------------------------------------------------------------------
# one training run through the unchanged MOGNN-TF pipeline
# ----------------------------------------------------------------------------------
_RUNNER = None


def _patch_rmtree_for_windows():
    """ExperimentRunner deletes <run_dir>/checkpoints with shutil.rmtree AFTER testing and
    BEFORE saving metrics. On Windows a transient lock on freshly written .pth files
    (antivirus / indexer) raises PermissionError and loses the run. Inside this process
    only, retry the deletion (clearing read-only flags) for up to ~30 s. Post-training
    cleanup: no effect on any number. On Linux the original call already succeeds."""
    import shutil
    import stat
    if getattr(shutil.rmtree, "_robust", False):
        return
    original = shutil.rmtree

    def _onerror(func, path, exc_info):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    def robust(path, *a, **kw):
        last = None
        for attempt in range(15):
            try:
                return original(path, onerror=_onerror) if not a and not kw else original(path, *a, **kw)
            except PermissionError as e:
                last = e
                time.sleep(2)
        raise last
    robust._robust = True
    shutil.rmtree = robust


def get_runner():
    """Load the omics once (as collapse_mognntf.py does) and reuse the runner."""
    global _RUNNER
    _patch_rmtree_for_windows()
    if _RUNNER is None:
        from multiomics_gnn.pancancer_prediction.experiments.experiment_runner import (
            ExperimentRunner, ExperimentDataLoader)
        base = load_yaml_config(ORIGINAL_CONFIG)
        exp, cnv, mirna = ExperimentDataLoader(config=base).load_raw_data()
        _RUNNER = ExperimentRunner(exp, cnv, mirna)
    return _RUNNER


def run_variant(variant, level, seed, raw_root=RUNS_RAW, smoke_epochs=None, num_workers=None):
    """Train + test one (variant, level, split). Returns the per-run record (dict).
    num_workers: optional execution-only override of data.num_workers (see VOLATILE_KEYS)."""
    base = load_yaml_config(VARIANTS[variant])
    panel = get_panel(level, seed, base)
    results_dir = Path(raw_root) / variant / f"g{level}_seed{seed}"
    cfg = build_run_config(base, level, seed, panel, results_dir, smoke_epochs=smoke_epochs)
    if num_workers is not None:
        cfg["data"]["num_workers"] = int(num_workers)
    runner = get_runner()

    before = set(glob.glob(str(results_dir / "run_*")))
    t0 = time.perf_counter()
    try:
        summary = runner.run_experiment(cfg, experiment_name="run")
    finally:
        _close_file_handlers()
    duration = time.perf_counter() - t0
    new = sorted(set(glob.glob(str(results_dir / "run_*"))) - before)
    if len(new) != 1:
        raise RuntimeError(f"expected exactly one new run folder in {results_dir}, got {new}")
    run_dir = Path(new[0])

    f1_pc, support_pc, macro_cm = per_class_from_confusion(run_dir)
    test_f1 = float(summary["test_f1_macro"])
    if abs(macro_cm - test_f1) > 1e-9:
        raise RuntimeError(f"macro-F1 from confusion matrix {macro_cm} != runner {test_f1}")
    genes, tf, mirna = panel
    return {
        "variant": variant, "level": int(level), "split_seed": int(seed),
        "model_seed": int(cfg["project"]["seed"]),
        "n_genes": len(genes), "n_tf": len(tf), "n_mirna": len(mirna),
        "test_macro_f1": test_f1, "test_accuracy": float(summary["test_accuracy"]),
        "val_macro_f1": float(summary["val_f1_macro"]),
        "best_epoch_0based": summary.get("best_epoch"),
        "best_epoch": (summary["best_epoch"] + 1) if summary.get("best_epoch") is not None else None,
        "per_class_f1": f1_pc, "per_class_support": support_pc,
        "config_hash": config_hash(cfg), "panel_hash": panel_hash(genes, tf, mirna),
        "split_hash": split_hash(seed), "model_parallel": bool(cfg["model"]["parallel"]),
        "smoke_epochs": smoke_epochs, "num_workers": int(cfg["data"]["num_workers"]),
        "duration_s": round(duration, 1),
        "run_dir": str(run_dir.relative_to(REPO)), "env": environment(),
    }


def mean_sd(xs):
    xs = list(xs)
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd
