#!/usr/bin/env bash
# Train + evaluate the heterogeneous model — no make, no sudo.
# Single run, or multi-seed (--runs N) following MOGNN-TF's protocol:
#   * SPLIT seed varies (42, 43, ...): independent stratified partitions.
#   * MODEL init seed is FIXED across splits (isolates split variance).
#   * feature selection + template are REBUILT per split (leakage-free), exactly
#     as MOGNN-TF recomputes variance_FS on each split's train fold.
# Results per run go to results/<experiment>/<timestamp>/; at the end the mean
# +/- s.d. of every metric is aggregated from the per-run metrics.json.
#
# Usage (from the repo root):
#   bash train_and_eval.sh                 # single run (seed 42)
#   bash train_and_eval.sh --runs 5        # seeds 42..46, per-seed graph, aggregate
#
# Knobs (env vars): CONFIG, MODEL_SEED, START_SEED, TOP_GENES, TOP_TF,
#                   TOP_MIRNA, GO_MIN_SUPPORT, METAPATH
set -euo pipefail

ENV_NAME="${ENV_NAME:-gnn}"
CONFIG="${CONFIG:-configs/config_kg_hgnn.yml}"
MODEL_SEED="${MODEL_SEED:-2025}"          # fixed model init seed (MOGNN-TF style)
START_SEED="${START_SEED:-42}"            # first split seed
TOP_GENES="${TOP_GENES:-700}"
TOP_TF="${TOP_TF:-200}"
TOP_MIRNA="${TOP_MIRNA:-100}"
GO_MIN_SUPPORT="${GO_MIN_SUPPORT:-3}"
METAPATH="${METAPATH:-}"                  # "--metapath" to add miRNA-miRNA / TF-TF
# extra flags forwarded verbatim to build_hetero_graph.py, e.g. to drop a scale
# for the ablation-driven "optimized" model:  GRAPH_FLAGS="--no-go"
GRAPH_FLAGS="${GRAPH_FLAGS:-}"
BACKBONE="${BACKBONE:-}"                  # hgt | hetero_sage | rgcn (empty -> config default)

RUNS=1
case "${1:-}" in
    --run|--runs) RUNS="${2:?--runs needs a number}" ;;
    "")           ;;
    *) echo "unknown argument: $1 (use --runs N)" >&2; exit 1 ;;
esac

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# --no-capture-output + python -u: stream the training log LIVE. Without it,
# `conda run` buffers the child's stdout and epochs appear only when the process
# ends, which looks like the run is "stuck".
run() { conda run --no-capture-output -n "$ENV_NAME" python -u "$@"; }

# read results_dir + experiment_name from the YAML (one small helper on disk,
# more portable than piping a heredoc through `conda run python -`)
HELPER="$(mktemp --suffix=.py)"
trap 'rm -f "$HELPER"' EXIT
cat > "$HELPER" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg.get("paths", {}).get("results_dir", "results"))
print(cfg["project"]["experiment_name"])
PY
_cfg_info="$(run "$HELPER" "$CONFIG")"
RESULTS_DIR="$(echo "$_cfg_info" | sed -n 1p)"
EXP_NAME="$(echo "$_cfg_info" | sed -n 2p)"

RUN_DIRS=()   # collect the run dir of each seed for aggregation

for i in $(seq 0 $((RUNS - 1))); do
    SPLIT_SEED=$((START_SEED + i))
    SPLIT_DIR="data/training/splits/splits_seed_${SPLIT_SEED}"
    FS_DIR="data/training/feature_selection/splits_seed_${SPLIT_SEED}"
    TEMPLATE="data/prior_knowledge/hetero/template_seed_${SPLIT_SEED}.pt"
    echo ""
    echo "########## run $((i + 1))/$RUNS | split seed $SPLIT_SEED | model seed $MODEL_SEED ##########"

    # 1) split for this seed (idempotent)
    run -m multiomics_kg_hgnn.pancancer_prediction.preprocessing.make_splits --seeds "$SPLIT_SEED"

    # 2) per-seed feature selection (variance on THIS split's train) + template
    run -m multiomics_kg_hgnn.pancancer_prediction.preprocessing.feature_selection \
        --split-dir "$SPLIT_DIR" \
        --top-genes "$TOP_GENES" --top-tf "$TOP_TF" --top-mirna "$TOP_MIRNA" \
        --out-dir "$FS_DIR"
    run scripts/preprocessing/priors/build_hetero_graph.py \
        --gene-list "$FS_DIR/selected_genes.csv" \
        --tf-list "$FS_DIR/selected_tf.csv" \
        --mirna-list "$FS_DIR/selected_mirna.txt" \
        --go-min-support "$GO_MIN_SUPPORT" $METAPATH $GRAPH_FLAGS \
        --out-dir data/prior_knowledge/hetero --force
    # keep a per-seed copy of the template so parallel/rerun seeds don't clash
    cp data/prior_knowledge/hetero/hetero_graph_template.pt "$TEMPLATE"

    # 3) per-run config: fixed model seed, this split's dir + template,
    #    optional backbone override. Prints the effective experiment_name so we
    #    collect the run dir from the right folder (backbones don't mix).
    RUN_CFG="$(mktemp --suffix=.yml)"
    MKCFG="$(mktemp --suffix=.py)"
    cat > "$MKCFG" <<'PY'
import sys, yaml
src, dst, mseed, split_dir, template, backbone = sys.argv[1:7]
cfg = yaml.safe_load(open(src))
cfg["project"]["seed"] = int(mseed)
cfg["data"]["split_dir"] = split_dir
cfg["data"]["template_path"] = template
if backbone:
    cfg["model"]["backbone"] = backbone
    # keep runs of different backbones in separate result folders
    cfg["project"]["experiment_name"] = f"{cfg['project']['experiment_name']}_{backbone}"
yaml.safe_dump(cfg, open(dst, "w"))
print(cfg["project"]["experiment_name"])   # effective experiment name
PY
    RUN_EXP="$(run "$MKCFG" "$CONFIG" "$RUN_CFG" "$MODEL_SEED" "$SPLIT_DIR" "$TEMPLATE" "$BACKBONE" | tail -1)"
    rm -f "$MKCFG"

    # 4) train
    run scripts/kg_hgnn/train.py --config "$RUN_CFG"
    rm -f "$RUN_CFG"

    # newest run dir for this (possibly backbone-suffixed) experiment
    RUN_DIRS+=("$(ls -1dt "$RESULTS_DIR/$RUN_EXP"/*/ | head -1)")
done

# --- aggregate mean +/- s.d. across runs (from each run's metrics.json) -------
echo ""
echo "########## AGGREGATE over $RUNS run(s) ##########"
AGG="$(mktemp --suffix=.py)"
cat > "$AGG" <<'PY'
import sys, json, os, statistics as st
dirs = sys.argv[1:]
metrics = {}
for d in dirs:
    m = json.load(open(os.path.join(d.rstrip("/\\"), "metrics.json")))
    for k, v in m.items():
        metrics.setdefault(k, []).append(float(v))
print(f"runs: {len(dirs)}")
for k, vals in metrics.items():
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    print(f"  {k:24s} {st.mean(vals):.4f} +/- {sd:.4f}   (n={len(vals)})")
PY
run "$AGG" "${RUN_DIRS[@]}"
rm -f "$AGG"
