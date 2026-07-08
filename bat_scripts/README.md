# bat_scripts/ — Windows (.bat) port of the `.sh` launchers

Windows equivalents of the repo's bash launchers. Run them **from the repo root**,
e.g. `bat_scripts\tune.bat`. Each `.bat` `pushd`-es to the repo root itself, so it
also works if called from elsewhere. Same env-var knobs as the `.sh` versions.

Requires `conda` on `PATH` and the `gnn` env (`bat_scripts\setup_env.bat`).

## Script map

| Windows | Original | Notes |
|---|---|---|
| `setup_env.bat`          | `setup_env.sh`          | create env `gnn` + install repo |
| `prepare_data.bat`       | `prepare_data.sh`       | full preprocessing → `data\training\` |
| `make_graph.bat`         | `make_graph.sh`         | feature selection + hetero template |
| `train_and_eval.bat`     | `train_and_eval.sh`     | wrapper → `helpers\train_and_eval.py` |
| `tune.bat`               | `tune.sh`               | Optuna, one backbone |
| `tune_all.bat`           | `tune_all.sh`           | Optuna, all 3 backbones in sequence |
| `run_feature_collapse.bat` | `run_feature_collapse.sh` | both models over the gene grid |
| `collapse_mokghgnn.bat`  | `scripts/kg_hgnn/collapse_mokghgnn.sh` | wrapper → `helpers\collapse_mokghgnn.py` |

## Setting env-var knobs on Windows

`cmd.exe` has no inline `VAR=x cmd`. Use `set` first (note: no space before `&&`):

```bat
set N_TRIALS=20 && set TIMEOUT_HOURS=4 && bat_scripts\tune.bat
set SEEDS=42 43 44 && bat_scripts\prepare_data.bat
```

## Why two of them call a Python helper

`train_and_eval.sh` and `collapse_mokghgnn.sh` are **orchestrators**: they loop over
seeds/genes and, per iteration, run a fixed chain of Python commands (split →
feature selection → build graph → write a per-run config → train), then aggregate.

In bash that logic leaned on shell-only primitives with no `.bat` equivalent —
inline Python heredocs (to read YAML / average metrics), `mktemp`, arrays, and
`ls -1dt | head -1`. Re-implementing those in batch is fragile, so the orchestration
was moved into plain Python, where the rest of the pipeline already lives:

- `helpers\train_and_eval.py`  ← the body of `train_and_eval.sh`
- `helpers\collapse_mokghgnn.py` ← the body of `collapse_mokghgnn.sh`

The `.bat` is then a thin launcher (`conda run … python helper.py %*`). Behaviour —
the multi-seed protocol, per-seed leakage-free graph rebuild, and mean±s.d.
aggregation — is identical to the shell version. These two helpers are
cross-platform: a bash wrapper could call them too.

The other six `.bat` files are literal ports (no helper needed).

## Known differences from the `.sh`

- `setup_env.bat` drops the trailing `conda activate` (a no-op in the original: it
  activated only the dying sub-shell). Activate manually after: `conda activate gnn`.
- Paths use `\`; the underlying Python scripts accept either separator.
