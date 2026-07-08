"""Feature-collapse study — MOKG-HGNN (heterogeneous). Cross-platform runner.

Python port of scripts/kg_hgnn/collapse_mokghgnn.sh. For each gene count in the
grid, rebuild the template with that many genes (per seed, leakage-free) and train
the hetero model. Results go under results/feature_collapse/mokghgnn_g<N>/<seed>/.

Hypothesis (proposta B): as genes shrink, the multi-scale KG scaffold
(pathway/GO/disease) keeps performance up while a molecular-only model collapses.

Knobs from env vars (same names as the .sh): CONFIG, ENV_NAME, BACKBONE,
GENE_GRID, SEEDS, MODEL_SEED, TOP_TF, TOP_MIRNA, GO_MIN_SUPPORT, METAPATH, OUT_ROOT.

    python collapse_mokghgnn.py
"""
import os
import shutil
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
CONFIG = env("CONFIG", "configs/config_kg_hgnn.yml")   # "best available" MOKG-HGNN config
BACKBONE = env("BACKBONE", "hgt")
GENE_GRID = env("GENE_GRID", "700 500 300 150 100 50 20").split()
SEEDS = env("SEEDS", "42 43 44 45 46").split()
MODEL_SEED = int(env("MODEL_SEED", "2025"))
TOP_TF = env("TOP_TF", "200")
TOP_MIRNA = env("TOP_MIRNA", "100")
GO_MIN_SUPPORT = env("GO_MIN_SUPPORT", "3")
METAPATH = env("METAPATH", "")
OUT_ROOT = env("OUT_ROOT", "results/feature_collapse")


def run(*args):
    cmd = ["conda", "run", "--no-capture-output", "-n", ENV_NAME, "python", "-u", *map(str, args)]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main():
    os.environ["PYTHONPATH"] = "src" + (os.pathsep + os.environ["PYTHONPATH"]
                                        if os.environ.get("PYTHONPATH") else "")
    os.environ["PYTHONUNBUFFERED"] = "1"
    metapath_args = [METAPATH] if METAPATH else []

    print("############################################################")
    print(f"# MOKG-HGNN feature-collapse | backbone={BACKBONE}")
    print(f"# genes: {' '.join(GENE_GRID)} | seeds: {' '.join(SEEDS)} | out: {OUT_ROOT}")
    print("############################################################")

    for g in GENE_GRID:
        for s in SEEDS:
            split_dir = f"data/training/splits/splits_seed_{s}"
            fs_dir = f"data/training/feature_selection/collapse_g{g}_seed{s}"
            template = f"data/prior_knowledge/hetero/collapse_g{g}_seed{s}.pt"
            exp = f"feature_collapse/mokghgnn_g{g}"
            print(f"\n===== MOKG-HGNN | genes={g} | split seed={s} =====", flush=True)

            # 1) split (idempotent)
            run("-m", "multiomics_kg_hgnn.pancancer_prediction.preprocessing.make_splits",
                "--seeds", s)

            # 2) feature selection with THIS gene count + template (per seed)
            run("-m", "multiomics_kg_hgnn.pancancer_prediction.preprocessing.feature_selection",
                "--split-dir", split_dir, "--top-genes", g, "--top-tf", TOP_TF,
                "--top-mirna", TOP_MIRNA, "--out-dir", fs_dir)
            run("scripts/preprocessing/priors/build_hetero_graph.py",
                "--gene-list", f"{fs_dir}/selected_genes.csv",
                "--tf-list", f"{fs_dir}/selected_tf.csv",
                "--mirna-list", f"{fs_dir}/selected_mirna.txt",
                "--go-min-support", GO_MIN_SUPPORT, *metapath_args,
                "--out-dir", "data/prior_knowledge/hetero", "--force")
            shutil.copyfile(REPO_ROOT / "data/prior_knowledge/hetero/hetero_graph_template.pt",
                            REPO_ROOT / template)

            # 3) per-run config. results_dir stays "results"; experiment_name carries
            #    the collapse sub-path so runs land under OUT_ROOT.
            cfg = yaml.safe_load(open(REPO_ROOT / CONFIG))
            cfg["project"]["seed"] = MODEL_SEED
            cfg["project"]["experiment_name"] = exp       # e.g. feature_collapse/mokghgnn_g300
            cfg["paths"]["results_dir"] = "results"
            cfg["data"]["split_dir"] = split_dir
            cfg["data"]["template_path"] = template
            cfg["model"]["backbone"] = BACKBONE

            fd, run_cfg = tempfile.mkstemp(suffix=".yml")
            os.close(fd)
            yaml.safe_dump(cfg, open(run_cfg, "w"))

            # 4) train
            try:
                run("scripts/kg_hgnn/train.py", "--config", run_cfg)
            finally:
                os.unlink(run_cfg)

    print(f"\n==> MOKG-HGNN feature-collapse done. Results under: {OUT_ROOT}/mokghgnn_g*/")


if __name__ == "__main__":
    main()
