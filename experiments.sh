#!/usr/bin/env bash
# Run one of the three thesis experiments on the FINAL trained model — no make, no sudo.
#
# ┌── QUICK REFERENCE ────────────────────────────────────────────────────────┐
# │  bash experiments.sh per_class            # per-subtype metrics (fast)      │
# │  bash experiments.sh explain              # GNNExplainer (fast)             │
# │  nohup bash experiments.sh collapse &     # feature collapse (LONG, detach) │
# │                                                                            │
# │  Pick the checkpoint (per_class/explain):                                  │
# │    bash experiments.sh explain --run <run_dir>   # exact run "results/best_model_full/20260711_103927" │
# │    MODEL=optimized bash experiments.sh explain    # best run of optimized  │
# │    (default: best test-F1 run of best_model_full)                          │
# │                                                                            │
# │  Explain knobs:  PER_CLASS=15  EPOCHS=50  TOPK=15                           │
# │  Collapse knobs: MODELS="mokghgnn mognntf"  BACKBONE=hetero_sage           │
# └────────────────────────────────────────────────────────────────────────────┘
#
#   per_class   per-subtype metrics from the trained best model (reuses model_best.pt)
#   explain     GNNExplainer over pathway/GO/disease per subtype (reuses model_best.pt)
#   collapse    feature-collapse study (RETRAINS across a shrinking gene grid)
#
# per_class and explain REUSE a checkpoint (fast, no retraining). collapse is a
# full experiment that retrains one model per gene level (it cannot reuse a
# checkpoint: each gene count is a different input size / graph — that's the point).
#
# Usage (from the repo root):
#   bash experiments.sh per_class
#   bash experiments.sh explain
#   bash experiments.sh collapse
#
# Which trained checkpoint to analyse (per_class/explain) — 3 ways, by priority:
#   1) explicit run dir (highest priority):
#        bash experiments.sh explain --run results/best_model_full/20260101_120000
#      (equivalently: RUN=<dir> bash experiments.sh explain)
#   2) automatic BEST run in a model folder (default): the run with the highest
#      test macro-F1 under results/best_model_<MODEL>/ — not merely the newest.
#        MODEL=optimized bash experiments.sh per_class
#   3) fallback: if no metrics.json exist, the newest run by mtime.
#
# Run collapse detached (it's long):
#   nohup bash experiments.sh collapse > collapse.log 2>&1 &
set -euo pipefail

ENV_NAME="${ENV_NAME:-gnn}"
run() { conda run --no-capture-output -n "$ENV_NAME" python -u "$@"; }

WHICH="${1:-}"
case "$WHICH" in
  per_class|explain|collapse) ;;
  "")  echo "usage: bash experiments.sh {per_class|explain|collapse} [--run <dir>]" >&2; exit 1 ;;
  *)   echo "unknown experiment: '$WHICH' (use per_class|explain|collapse)" >&2; exit 1 ;;
esac

# --- optional explicit run dir: `bash experiments.sh explain --run <dir>` -----
# Highest-priority way to pin EXACTLY which checkpoint to analyse.
RUN="${RUN:-}"                 # env RUN also works
if [ "${2:-}" = "--run" ]; then
  RUN="${3:?--run needs a run directory}"
fi

# --- locate the trained best model (for per_class / explain) -----------------
# Priority: explicit --run/RUN  >  best-test-F1 run in MODEL's folder  >  newest.
# MODEL=full|optimized picks which best-model folder to search.
# Best-model runs are saved per backbone: results/best_model_<MODEL>_<BACKBONE>.
# Fall back to the unsuffixed name for older runs.
MODEL="${MODEL:-full}"
EXP_BACKBONE="${BACKBONE:-hetero_sage}"
EXP_DIR="results/best_model_${MODEL}_${EXP_BACKBONE}"
[ -d "$EXP_DIR" ] || [ ! -d "results/best_model_${MODEL}" ] || EXP_DIR="results/best_model_${MODEL}"
[ -n "$RUN" ] && echo "[run] using explicit --run/RUN: $RUN"

# Pick the run with the HIGHEST test macro-F1 (the correct choice for explain:
# we want to interpret the BEST model, not the last one trained). Falls back to
# newest-by-mtime if no metrics.json are found. Reads metrics.json via python.
pick_best_run() {
  local base="$1"
  [ -d "$base" ] || { echo ""; return; }
  # NB: `conda run python - <args> <<HEREDOC` is BROKEN (stdin+args clash) and can
  # spiral into repeated shells ("shell level too high"). Write the helper to a
  # temp file and pass it as an argument instead — same pattern as the other
  # launchers in this repo.
  local helper; helper="$(mktemp --suffix=.py)"
  cat > "$helper" <<'PY'
import glob, json, os, sys
base = sys.argv[1]
best_dir, best_f1 = None, -1.0
newest_dir, newest_mt = None, -1.0
for run in glob.glob(os.path.join(base, "*")):
    ckpt = os.path.join(run, "model_best.pt")
    if not os.path.isfile(ckpt):
        continue
    mt = os.path.getmtime(run)
    if mt > newest_mt:
        newest_dir, newest_mt = run, mt
    mj = os.path.join(run, "metrics.json")
    if os.path.isfile(mj):
        try:
            f1 = float(json.load(open(mj)).get("test_macro_f1", -1))
        except Exception:
            f1 = -1
        if f1 > best_f1:
            best_dir, best_f1 = run, f1
chosen = best_dir or newest_dir or ""
print(chosen)
# report to stderr so it doesn't pollute the captured path
if chosen and best_dir:
    print(f"[pick] best test-F1={best_f1:.4f} -> {chosen}", file=sys.stderr)
elif chosen:
    print(f"[pick] no metrics.json; using newest -> {chosen}", file=sys.stderr)
PY
  conda run -n "$ENV_NAME" python "$helper" "$base"
  rm -f "$helper"
}

case "$WHICH" in
  # -------------------------------------------------------------------------
  per_class)
    # Aggregate across ALL seeds of the experiment and compare with MOGNN-TF.
    # Every run already writes its own per_class_metrics.csv at training time, so
    # there is nothing to recompute here: the experiment is the multi-seed
    # mean ± s.d. and the head-to-head against the published baseline.
    # (For OLD runs predating that, recompute first: eval_per_class.py --run <dir>.)
    echo "==> per-class aggregation | experiment: $EXP_DIR"
    [ -d "$EXP_DIR" ] || { echo "No such experiment folder: $EXP_DIR (train it first, or set MODEL=<name>)." >&2; exit 1; }

    # reference table (from the paper's LaTeX); build it once if missing
    REF="paper/table_per_class_mognn-tf.csv"
    if [ ! -f "$REF" ]; then
      echo "    [ref] $REF missing — converting from the LaTeX table"
      run scripts/kg_hgnn/tex_per_class_to_csv.py
    fi

    run scripts/kg_hgnn/per_class_aggregate.py --results "$EXP_DIR"
    echo "==> done. per_class_aggregate.csv + per_class_vs_mognntf.csv/.png in: $EXP_DIR"
    ;;

  # -------------------------------------------------------------------------
  explain)
    RUN="${RUN:-$(pick_best_run "$EXP_DIR")}"
    [ -n "$RUN" ] || { echo "No run with model_best.pt under $EXP_DIR/ (train it first, or pass --run <dir>)." >&2; exit 1; }
    PER_CLASS="${PER_CLASS:-15}"     # patients per subtype
    EPOCHS="${EPOCHS:-50}"           # GNNExplainer epochs per patient
    TOPK="${TOPK:-15}"               # top mechanisms per subtype in the table
    echo "==> explainability (GNNExplainer) | run: $RUN | per-class=$PER_CLASS epochs=$EPOCHS"
    echo "    (best on the FULL model: GO is present, so its role can be explained)"
    run scripts/kg_hgnn/explain.py --run "$RUN" \
        --per-class "$PER_CLASS" --epochs "$EPOCHS" --topk "$TOPK"
    echo "==> done. Interpretability tables/plots in: $RUN"
    ;;

  # -------------------------------------------------------------------------
  collapse)
    # RETRAINS across the gene grid. Points MOKG-HGNN at the tuned hetero_sage
    # config so the collapse curve matches the final model.
    MOKG_CONFIG="${MOKG_CONFIG:-configs/best_model_config.yml}"
    BACKBONE="${BACKBONE:-hetero_sage}"
    # BOTH curves by default: the experiment IS the comparison of the two
    # degradation slopes. MODELS="mokghgnn" runs only ours (e.g. to top up seeds).
    MODELS="${MODELS:-mokghgnn mognntf}"
    echo "==> feature-collapse | models=$MODELS | backbone=$BACKBONE | config=$MOKG_CONFIG"
    echo "    (this RETRAINS one model per gene level — long; run under nohup)"
    MODELS="$MODELS" BACKBONE="$BACKBONE" MOKG_CONFIG="$MOKG_CONFIG" \
      bash run_feature_collapse.sh
    echo "==> done. Collapse table + plot in: results/feature_collapse/"
    ;;
esac
