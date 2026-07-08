#!/usr/bin/env bash
# Ablation study (paper-style): one factor at a time, starting from the FULL
# configuration, so each variant isolates the contribution of one graph/model
# component. For MOKG-HGNN (heterogeneous).
#
# Variants (each run over SEEDS, model seed fixed):
#   full             baseline: metapath + disease + full readout + all omics
#   no_metapath      drop miRNA-miRNA / TF-TF co-target layers        (rebuild graph)
#   no_disease       drop the disease scale from the graph            (rebuild graph)
#   readout_mol      readout only from the molecular scale (gene)     (config only)
#   readout_pathway  readout = gene + pathway                          (config only)
#   no_cnv           drop the CNV omic channel                         (config only)
#   no_mirna         drop the miRNA omic features                      (config only)
#
# "rebuild graph" variants build their own template; "config only" variants reuse
# the FULL template (faster). Results under results/ablation/<variant>/<seed>/.
#
# Usage (from the repo root):
#   bash ablation.sh
#   SEEDS="42 43 44" VARIANTS="full no_metapath no_disease" bash ablation.sh
#   BACKBONE=rgcn bash ablation.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-gnn}"
CONFIG="${CONFIG:-configs/config_kg_hgnn.yml}"     # "best available" MOKG-HGNN config
BACKBONE="${BACKBONE:-hgt}"
SEEDS="${SEEDS:-42 43 44 45 46}"
MODEL_SEED="${MODEL_SEED:-2025}"
TOP_GENES="${TOP_GENES:-700}"
TOP_TF="${TOP_TF:-200}"
TOP_MIRNA="${TOP_MIRNA:-100}"
GO_MIN_SUPPORT="${GO_MIN_SUPPORT:-3}"
OUT_ROOT="${OUT_ROOT:-results/ablation}"
VARIANTS="${VARIANTS:-full no_metapath no_disease no_pathway no_go readout_mol readout_pathway no_cnv no_mirna}"

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
run() { conda run --no-capture-output -n "$ENV_NAME" python -u "$@"; }
mkdir -p "$OUT_ROOT"

# --- per-variant knobs -------------------------------------------------------
# graph build flags (empty = full graph with metapath + disease)
graph_flags() {
  case "$1" in
    no_metapath) echo "" ;;                                # drop metapath (keep all scales)
    no_disease)  echo "--metapath --no-disease" ;;         # drop disease scale
    no_pathway)  echo "--metapath --no-pathway" ;;         # drop pathway scale
    no_go)       echo "--metapath --no-go" ;;              # drop GO scale (+ GO->GO hierarchy)
    *)           echo "--metapath" ;;                      # full graph (all scales + metapath)
  esac
}
# does this variant need its own template? (else reuse the FULL one)
needs_rebuild() {
  case "$1" in no_metapath|no_disease|no_pathway|no_go) return 0 ;; *) return 1 ;; esac;
}
# config overrides (dotted key=value, space separated). empty = no override.
# For scale-removing variants also drop that scale from the readout (the model
# filters missing scales anyway, but keeping them consistent is cleaner).
config_overrides() {
  case "$1" in
    no_disease)      echo "model.readout_types=[gene,pathway,GO_term]" ;;
    no_pathway)      echo "model.readout_types=[gene,GO_term]" ;;
    no_go)           echo "model.readout_types=[gene,pathway]" ;;
    readout_mol)     echo "model.readout_types=[gene]" ;;
    readout_pathway) echo "model.readout_types=[gene,pathway]" ;;
    no_cnv)          echo "data.use_cnv=false" ;;
    no_mirna)        echo "data.use_mirna=false" ;;
    *)               echo "" ;;
  esac
}

# config generator: applies model seed, split, template, backbone, experiment
# name and any dotted overrides for the variant.
MKCFG="$(mktemp --suffix=.py)"
trap 'rm -f "$MKCFG"' EXIT
cat > "$MKCFG" <<'PY'
import sys, yaml
src, dst, mseed, split_dir, template, exp, backbone = sys.argv[1:8]
overrides = sys.argv[8:]  # "dotted.key=value"
cfg = yaml.safe_load(open(src))
cfg["project"]["seed"] = int(mseed)
cfg["project"]["experiment_name"] = exp
cfg["paths"]["results_dir"] = "results"
cfg["data"]["split_dir"] = split_dir
cfg["data"]["template_path"] = template
cfg["model"]["backbone"] = backbone
for ov in overrides:
    key, val = ov.split("=", 1)
    # parse the value as YAML so [gene] -> list, false -> bool, 16 -> int
    val = yaml.safe_load(val)
    d = cfg
    ks = key.split(".")
    for k in ks[:-1]:
        d = d.setdefault(k, {})
    d[ks[-1]] = val
yaml.safe_dump(cfg, open(dst, "w"))
PY

FULL_TEMPLATE="data/prior_knowledge/hetero/ablation_full_seed__SEED__.pt"

echo "############################################################"
echo "# ABLATION | backbone=$BACKBONE | seeds: $SEEDS"
echo "# variants: $VARIANTS | out: $OUT_ROOT"
echo "############################################################"

for V in $VARIANTS; do
  GF="$(graph_flags "$V")"
  OV="$(config_overrides "$V")"
  echo ""
  echo "===================== variant: $V ====================="
  echo "  graph_flags='$GF' | overrides='$OV' | rebuild=$(needs_rebuild "$V" && echo yes || echo no)"

  for S in $SEEDS; do
    SPLIT_DIR="data/training/splits/splits_seed_${S}"
    FS_DIR="data/training/feature_selection/splits_seed_${S}"
    FULL_T="${FULL_TEMPLATE/__SEED__/$S}"
    echo ""
    echo "----- $V | split seed $S -----"

    # 1) split + feature selection (shared across variants for this seed)
    run -m multiomics_kg_hgnn.pancancer_prediction.preprocessing.make_splits --seeds "$S"
    if [ ! -f "$FS_DIR/selected_genes.csv" ]; then
      run -m multiomics_kg_hgnn.pancancer_prediction.preprocessing.feature_selection \
          --split-dir "$SPLIT_DIR" \
          --top-genes "$TOP_GENES" --top-tf "$TOP_TF" --top-mirna "$TOP_MIRNA" \
          --out-dir "$FS_DIR"
    fi

    # 2) template: rebuild for graph-changing variants; else reuse the FULL one
    if needs_rebuild "$V"; then
      TEMPLATE="data/prior_knowledge/hetero/ablation_${V}_seed${S}.pt"
      run scripts/preprocessing/priors/build_hetero_graph.py \
          --gene-list "$FS_DIR/selected_genes.csv" --tf-list "$FS_DIR/selected_tf.csv" \
          --mirna-list "$FS_DIR/selected_mirna.txt" --go-min-support "$GO_MIN_SUPPORT" \
          $GF --out-dir data/prior_knowledge/hetero --force
      cp data/prior_knowledge/hetero/hetero_graph_template.pt "$TEMPLATE"
    else
      # build the FULL template once per seed, reuse it for all config-only variants
      if [ ! -f "$FULL_T" ]; then
        run scripts/preprocessing/priors/build_hetero_graph.py \
            --gene-list "$FS_DIR/selected_genes.csv" --tf-list "$FS_DIR/selected_tf.csv" \
            --mirna-list "$FS_DIR/selected_mirna.txt" --go-min-support "$GO_MIN_SUPPORT" \
            --metapath --out-dir data/prior_knowledge/hetero --force
        cp data/prior_knowledge/hetero/hetero_graph_template.pt "$FULL_T"
      fi
      TEMPLATE="$FULL_T"
    fi

    # 3) per-run config (experiment_name = ablation/<variant>) + train
    RUN_CFG="$(mktemp --suffix=.yml)"
    run "$MKCFG" "$CONFIG" "$RUN_CFG" "$MODEL_SEED" "$SPLIT_DIR" "$TEMPLATE" \
        "ablation/${V}" "$BACKBONE" $OV
    run scripts/kg_hgnn/train.py --config "$RUN_CFG"
    rm -f "$RUN_CFG"
  done
done

echo ""
echo "########## AGGREGATE ##########"
run scripts/kg_hgnn/ablation_aggregate.py --results "$OUT_ROOT"
echo ""
echo "==> ablation done. Table + plot in: $OUT_ROOT/"
