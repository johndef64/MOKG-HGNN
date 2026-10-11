"""Check that the active Python environment has every package the no-shallow
experiment needs. No GPU, no data, no repository files required.

From a Jupyter notebook (CPU only) with the `gnn` environment:
    %run scripts/slurm/no_shallow_check_packages.py
or from a terminal:
    source /home/apps/iknos-env.sh && iknos_activate gnn
    python scripts/slurm/no_shallow_check_packages.py

Each check uses the same import the experiment code uses (MOGNN-TF runner, unified
feature selection, graph-survival rule, plotting). If something is missing, ask the
cluster administrator to add it to `gnn`: official environments cannot be modified.
"""

import importlib
import platform
import sys

# (import statement as written in the code, where it is used)
CHECKS = [
    ("numpy", "everywhere"),
    ("pandas", "everywhere"),
    ("scipy.sparse", "MOGNN-TF supra-adjacency"),
    ("sklearn.model_selection", "split generator (sanity check)"),
    ("sklearn.metrics", "metrics"),
    ("sklearn.preprocessing", "min-max scaling"),
    ("matplotlib.pyplot", "MOGNN-TF curves, comparison figures"),
    ("yaml", "configs (pyyaml)"),
    ("optuna", "imported at load time by the MOGNN-TF runner"),
    ("wandb", "imported at load time by the MOGNN-TF runner (disabled)"),
    ("statsmodels.nonparametric.smoothers_lowess", "imported at load time by MOGNN-TF preprocessing"),
    ("ijson", "imported by the graph builder (survival rule)"),
    ("networkx", "imported inside MOGNN-TF preprocessing (community FS, not used here)"),
    ("torch", "training"),
    ("torch_geometric.nn", "ChebConv"),
    ("torch_geometric.loader", "DataLoader"),
    ("torch_geometric.data", "Data / Dataset"),
    ("torch_geometric.utils", "from_scipy_sparse_matrix"),
]
# versions the existing results were produced with: a different version is an error
EXPECTED = {"torch": "2.7.0", "torch_geometric": "2.7.0"}
SYMBOLS = [("torch_geometric.nn", "ChebConv"), ("torch_geometric.loader", "DataLoader"),
           ("statsmodels.nonparametric.smoothers_lowess", "lowess")]


def main():
    problems = []
    print(f"Python {platform.python_version()} ({sys.executable})")
    if sys.version_info[:2] != (3, 10):
        problems.append(f"Python {platform.python_version()} (expected 3.10)")

    for mod, why in CHECKS:
        try:
            importlib.import_module(mod)
            top = importlib.import_module(mod.split(".")[0])
            ver = getattr(top, "__version__", "")
            print(f"  ok       {mod:44s} {ver:14s} {why}")
        except Exception as e:  # noqa: BLE001
            problems.append(f"{mod}: {type(e).__name__}: {e}")
            print(f"  MISSING  {mod:44s} {'':14s} {why}  ->  {type(e).__name__}: {e}")

    for mod, sym in SYMBOLS:
        try:
            getattr(importlib.import_module(mod), sym)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{mod}.{sym}: {type(e).__name__}: {e}")

    for mod, want in EXPECTED.items():
        try:
            got = importlib.import_module(mod).__version__.split("+")[0]
            if got != want:
                problems.append(f"{mod} {got} (expected {want})")
        except Exception:  # noqa: BLE001
            pass  # already reported as missing

    try:
        import torch
        cuda = (f"yes, {torch.cuda.get_device_name(0)}" if torch.cuda.is_available()
                else "no (expected on a CPU-only session; the jobs request a GPU)")
        print(f"\nCUDA build: {torch.version.cuda} | GPU visible here: {cuda}")
    except Exception:  # noqa: BLE001
        pass

    if problems:
        print(f"\nRESULT: {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        print("Ask the cluster administrator to add the missing packages to the 'gnn' environment.")
    else:
        print("\nRESULT: OK, the environment has every package the experiment needs.")
    return not problems


if __name__ == "__main__":
    ok = main()
    # a non-zero exit code for terminals and jobs; inside Jupyter (%run) just print
    if not ok and "ipykernel" not in sys.modules:
        sys.exit(1)
