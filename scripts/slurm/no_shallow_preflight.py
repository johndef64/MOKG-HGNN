"""Preflight for the no-shallow matched experiment on the iknos cluster.

Checks, in this order, and exits 1 at the first category that fails:
  1. the Python environment (the official `gnn` environment of the cluster) imports every
     module the experiment uses. optuna, yaml and statsmodels are imported by the
     MOGNN-TF runner at load time and are not in the package list the cluster guide
     declares for `gnn`: if one is missing, ask the administrator to add it to `gnn`
     (official environments cannot be modified by users);
  2. torch / torch_geometric versions match the ones the results were produced with
     (2.7.0 / 2.7.0: a mismatch is reported as an error);
  3. every input file is present under the repository root;
  4. with --gpu: CUDA is visible and the device name is printed.

  python scripts/slurm/no_shallow_preflight.py           # on the login node
  python scripts/slurm/no_shallow_preflight.py --gpu     # inside a GPU job
"""

import argparse
import glob
import importlib
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

MODULES = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "yaml", "optuna", "wandb",
           "statsmodels", "ijson", "torch", "torch_geometric"]
EXPECTED = {"torch": "2.7.0", "torch_geometric": "2.7.0"}

REQUIRED_FILES = [
    # omics and labels read by MOGNN-TF and by the unified feature selection
    "data/training/expression_data_pancan.tsv",
    "data/training/cnv_data_pancan.tsv",
    "data/training/mirna_data_top_100.tsv",
    "data/training/molecular_subtype.csv",
    "data/training/tf_nodes_all_in_vocab.csv",
    "data/training/adj_matrix_biogrid.npz",
    "data/training/standardized_mirna_mrna_edge_filtered_at_eight_with_top_100_mirna.npz",
    "data/training/tf_gene_adj_global.npz",
    # molecular priors used to decide which miRNA / TF survive in the graph
    "data/prior_knowledge/GGI/GGI_edges_undirected.csv",
    "data/prior_knowledge/miRNA/miRNA_gene_interactions.csv",
    "data/prior_knowledge/TF/TF_interactions.tsv",
    "scripts/preprocessing/priors/hgcn_mart_conversion_table.zip",   # NOT in git: copy it
    # existing series (read only) and the published table used as a cross-check
    "results/feature_collapse/collapse_mognntf_summary.csv",
    "supplementary/tables/S2_feature_collapse.csv",
    # the experiment itself (untracked in git: copy the folder)
    "experiments/no_shallow_matched/common.py",
    "experiments/no_shallow_matched/configs/no_shallow.yaml",
    "benchmark/MOGNN-TF/configs/config_final.yml",
]


def check_modules():
    bad = []
    for m in MODULES:
        try:
            mod = importlib.import_module(m)
            v = getattr(mod, "__version__", "?")
            want = EXPECTED.get(m)
            flag = "" if not want or v.split("+")[0] == want else f"   <-- expected {want}"
            if flag:
                bad.append(f"{m} {v} (expected {want})")
            print(f"  [ok] {m:16s} {v}{flag}")
        except Exception as e:  # noqa: BLE001
            bad.append(f"{m}: {type(e).__name__}: {e}")
            print(f"  [MISSING] {m:16s} {type(e).__name__}: {e}")
    # the real import chain of the experiment (catches anything not listed above)
    sys.path.insert(0, str(REPO / "experiments" / "no_shallow_matched"))
    cwd = os.getcwd()
    try:
        import common  # noqa: F401  (also adds src/ and benchmark/MOGNN-TF to sys.path)
        import model_no_shallow, sanity_check, run_no_shallow, compare, plot_comparison, report  # noqa: F401,E401
        from scripts.collapse_mognntf import _panel  # noqa: F401
        print("  [ok] experiment import chain (common, sanity_check, run_no_shallow, ...)")
    except Exception as e:  # noqa: BLE001
        bad.append(f"experiment import chain: {type(e).__name__}: {e}")
        print(f"  [FAIL] experiment import chain: {type(e).__name__}: {e}")
    finally:
        os.chdir(cwd)
    return bad


def check_files():
    missing = [f for f in REQUIRED_FILES if not (REPO / f).exists()]
    if not (REPO / "data/training/mirna_data_full.tsv").exists() and not (REPO / "data/omics/mirna.zip").exists():
        missing.append("data/training/mirna_data_full.tsv (or data/omics/mirna.zip to rebuild it)")
    for s in range(42, 47):
        for part in ("train", "val", "test"):
            f = f"data/training/splits/splits_seed_{s}/common_trimmed_shuffle_index_{part}.tsv"
            if not (REPO / f).exists():
                missing.append(f)
    n_ms = len(glob.glob(str(REPO / "results/feature_collapse/mokghgnn_g*/*/metrics.json")))
    if n_ms != 35:
        missing.append(f"results/feature_collapse/mokghgnn_g*/<ts>/metrics.json: found {n_ms}, need 35 "
                       f"(with config.json and graph_info.json next to each)")
    n_uc = len(glob.glob(str(REPO / "results/feature_collapse/mognntf_g*/run_*/used_config.yaml")))
    n_fs = len(glob.glob(str(REPO / "results/feature_collapse/_aligned_fs/g*_seed*/selected_genes.csv")))
    print(f"  [info] with-shallow run configs found: {n_uc}/35, original FS folders: {n_fs}/35 "
          f"(optional: what is missing is reported as 'unverifiable' by the sanity check)")
    for f in missing:
        print(f"  [MISSING] {f}")
    if not missing:
        print(f"  [ok] all required inputs present under {REPO}")
    return missing


def check_gpu():
    import torch
    if not torch.cuda.is_available():
        print("  [FAIL] CUDA not available inside the job")
        return ["cuda"]
    print(f"  [ok] GPU: {torch.cuda.get_device_name(0)} | CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES', '?')}")
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="Also require a visible CUDA device.")
    args = ap.parse_args()
    print(f"[preflight] repo: {REPO}")
    print("[preflight] python environment")
    bad = check_modules()
    print("[preflight] input files")
    bad += check_files()
    if args.gpu:
        print("[preflight] GPU")
        bad += check_gpu()
    if bad:
        print(f"\n[preflight] FAILED ({len(bad)} problem(s)). Missing Python packages: ask the "
              f"cluster administrator to add them to the 'gnn' environment. Missing files: see "
              f"scripts/slurm/README_no_shallow.md, section 1.")
        sys.exit(1)
    print("\n[preflight] OK")


if __name__ == "__main__":
    main()
