"""Train + evaluate the heterogeneous model — cross-platform runner.

This is the Python port of train_and_eval.sh. All the logic that the shell
script emulated with heredocs / mktemp / arrays / `ls -1dt` lives here as plain
Python, so both bat_scripts\train_and_eval.bat (Windows) and a thin bash wrapper
can call it. Behaviour is identical to the original:

  * SPLIT seed varies (42, 43, ...): independent stratified partitions.
  * MODEL init seed is FIXED across splits (isolates split variance).
  * feature selection + template are REBUILT per split (leakage-free).
  * results per run -> results/<experiment>/<timestamp>/; at the end the mean
    +/- s.d. of every metric is aggregated from the per-run metrics.json.

Knobs come from env vars (same names as the .sh): CONFIG, ENV_NAME, MODEL_SEED,
START_SEED, TOP_GENES, TOP_TF, TOP_MIRNA, GO_MIN_SUPPORT, METAPATH, BACKBONE.
Multi-seed via the CLI:  python train_and_eval.py --runs 5
"""
import argparse
import json
import os
import shutil
import statistics as st
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def env(name, default):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


ENV_NAME = env("ENV_NAME", "gnn")
CONFIG = env("CONFIG", "configs/config_kg_hgnn.yml")
MODEL_SEED = int(env("MODEL_SEED", "2025"))       # fixed model init seed (MOGNN-TF style)
START_SEED = int(env("START_SEED", "42"))         # first split seed
TOP_GENES = env("TOP_GENES", "700")
TOP_TF = env("TOP_TF", "200")
TOP_MIRNA = env("TOP_MIRNA", "100")
GO_MIN_SUPPORT = env("GO_MIN_SUPPORT", "3")
METAPATH = env("METAPATH", "")                    # "--metapath" to add miRNA-miRNA / TF-TF
BACKBONE = env("BACKBONE", "")                    # hgt|hetero_sage|rgcn ("" -> config default)


def run(*args):
    """conda run --no-capture-output ... python -u <args> (streams live)."""
    cmd = ["conda", "run", "--no-capture-output", "-n", ENV_NAME, "python", "-u", *map(str, args)]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def newest_subdir(parent: Path) -> Path:
    subs = [p for p in parent.iterdir() if p.is_dir()]
    if not subs:
        sys.exit(f"no run sub-directories under {parent}")
    return max(subs, key=lambda p: p.stat().st_mtime)


def main():
    ap = argparse.ArgumentParser(description="Train + evaluate MOKG-HGNN (single or multi-seed).")
    ap.add_argument("--runs", "--run", type=int, default=1, dest="runs",
                    help="Number of split seeds (START_SEED .. START_SEED+runs-1).")
    args = ap.parse_args()
    runs = args.runs

    # so `python -m multiomics_kg_hgnn...` resolves even without an editable install
    os.environ["PYTHONPATH"] = "src" + (os.pathsep + os.environ["PYTHONPATH"]
                                        if os.environ.get("PYTHONPATH") else "")
    os.environ["PYTHONUNBUFFERED"] = "1"

    # results_dir + experiment_name from the YAML
    cfg0 = yaml.safe_load(open(REPO_ROOT / CONFIG))
    results_dir = cfg0.get("paths", {}).get("results_dir", "results")
    metapath_args = [METAPATH] if METAPATH else []

    run_dirs = []
    for i in range(runs):
        split_seed = START_SEED + i
        split_dir = f"data/training/splits/splits_seed_{split_seed}"
        fs_dir = f"data/training/feature_selection/splits_seed_{split_seed}"
        template = f"data/prior_knowledge/hetero/template_seed_{split_seed}.pt"
        print(f"\n########## run {i + 1}/{runs} | split seed {split_seed} | "
              f"model seed {MODEL_SEED} ##########", flush=True)

        # 1) split for this seed (idempotent)
        run("-m", "multiomics_kg_hgnn.pancancer_prediction.preprocessing.make_splits",
            "--seeds", split_seed)

        # 2) per-seed feature selection (variance on THIS split's train) + template
        run("-m", "multiomics_kg_hgnn.pancancer_prediction.preprocessing.feature_selection",
            "--split-dir", split_dir, "--top-genes", TOP_GENES, "--top-tf", TOP_TF,
            "--top-mirna", TOP_MIRNA, "--out-dir", fs_dir)
        run("scripts/preprocessing/priors/build_hetero_graph.py",
            "--gene-list", f"{fs_dir}/selected_genes.csv",
            "--tf-list", f"{fs_dir}/selected_tf.csv",
            "--mirna-list", f"{fs_dir}/selected_mirna.txt",
            "--go-min-support", GO_MIN_SUPPORT, *metapath_args,
            "--out-dir", "data/prior_knowledge/hetero", "--force")
        # keep a per-seed copy so parallel/rerun seeds don't clash
        shutil.copyfile(REPO_ROOT / "data/prior_knowledge/hetero/hetero_graph_template.pt",
                        REPO_ROOT / template)

        # 3) per-run config: fixed model seed, this split's dir + template, optional
        #    backbone override (backbones land in separate result folders).
        cfg = yaml.safe_load(open(REPO_ROOT / CONFIG))
        cfg["project"]["seed"] = MODEL_SEED
        cfg["data"]["split_dir"] = split_dir
        cfg["data"]["template_path"] = template
        if BACKBONE:
            cfg["model"]["backbone"] = BACKBONE
            cfg["project"]["experiment_name"] = f"{cfg['project']['experiment_name']}_{BACKBONE}"
        run_exp = cfg["project"]["experiment_name"]

        fd, run_cfg = tempfile.mkstemp(suffix=".yml")
        os.close(fd)
        yaml.safe_dump(cfg, open(run_cfg, "w"))

        # 4) train
        try:
            run("scripts/kg_hgnn/train.py", "--config", run_cfg)
        finally:
            os.unlink(run_cfg)

        # newest run dir for this (possibly backbone-suffixed) experiment
        run_dirs.append(newest_subdir(REPO_ROOT / results_dir / run_exp))

    # --- aggregate mean +/- s.d. across runs (from each run's metrics.json) ---
    print(f"\n########## AGGREGATE over {runs} run(s) ##########", flush=True)
    metrics = {}
    for d in run_dirs:
        m = json.load(open(Path(d) / "metrics.json"))
        for k, v in m.items():
            metrics.setdefault(k, []).append(float(v))
    print(f"runs: {len(run_dirs)}")
    for k, vals in metrics.items():
        sd = st.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {k:24s} {st.mean(vals):.4f} +/- {sd:.4f}   (n={len(vals)})")


if __name__ == "__main__":
    main()
