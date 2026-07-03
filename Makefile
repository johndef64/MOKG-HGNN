# MOKG-HGNN — pipeline orchestration (heterogeneous multi-scale model, Proposta B)
#
# This is the default `Makefile` for the NEW model. The MOGNN-TF pipeline lives
# in its own Makefile (run it with `make -f <that-file> <target>`).
#
# Usage:
#   make help
#   make data              # shared preprocessing (omics + priors)
#   make graph             # feature-selection -> build hetero template
#   make train             # single run (configs/config_kg_hgnn.yml)
#   make evaluate          # test the latest run (or pass CKPT=path/to/model_best.pt)
#   make all               # data -> graph -> train
#
# Runs inside the conda env `gnn`. Override the interpreter with e.g.
#   make train PY="python"

PY ?= conda run -n gnn python
export PYTHONPATH := src            # so `python -m multiomics_kg_hgnn...` resolves

# --- knobs (override on the command line: make ... SEED=43 TOP_GENES=900) -----
SEED ?= 42
SPLIT_DIR ?= data/training/splits/splits_seed_$(SEED)
FS_DIR ?= data/training/feature_selection/splits_seed_$(SEED)
TOP_GENES ?= 700
TOP_TF ?= 200
TOP_MIRNA ?= 100
GO_MIN_SUPPORT ?= 3
METAPATH ?=                          # set to "--metapath" to add miRNA-miRNA / TF-TF
CONFIG ?= configs/config_kg_hgnn.yml
SEEDS ?= 42                          # seeds for data-splits (e.g. SEEDS="42 43 44")

.PHONY: help install \
        data data-download data-omics data-priors data-wrap data-splits \
        graph feature-selection build-graph \
        train evaluate check \
        all clean

help:
	@echo "MOKG-HGNN pipeline"
	@echo "  install            - pip install -e ."
	@echo "  data               - FULL preprocessing: download -> omics -> priors -> wrap -> splits"
	@echo "  data-wrap          - build data/training/* (expression/cnv/labels/tf_nodes/variance)"
	@echo "  data-splits        - stratified train/val/test indices (SEEDS='42 43 ...')"
	@echo "  feature-selection  - unified variance FS (gene/TF/miRNA) on the train split -> $(FS_DIR)"
	@echo "  build-graph        - build the hetero template from the selected panels"
	@echo "  graph              - feature-selection + build-graph"
	@echo "  train              - single run ($(CONFIG))"
	@echo "  evaluate           - test the latest run (or pass CKPT=...)"
	@echo "  check              - end-to-end sanity check (data -> HeteroData -> HGT)"
	@echo "  all                - data + graph + train"
	@echo "  clean              - remove __pycache__ and stale outputs"
	@echo ""
	@echo "  knobs: SEED=$(SEED) SEEDS='$(SEEDS)' TOP_GENES=$(TOP_GENES) TOP_TF=$(TOP_TF) TOP_MIRNA=$(TOP_MIRNA) METAPATH='$(METAPATH)'"

install:
	$(PY) -m pip install -e .

# --------------------------------------------------------------------
# Data preprocessing (shared with MOGNN-TF). The PKT knowledge graph
# (data/prior_knowledge/PKT) is provided by the user, not downloaded here.
# Full chain: download -> omics -> priors -> wrap (writes data/training/*) ->
# splits. `data-wrap` and `data-splits` produce the training files that
# feature-selection and training consume.
# --------------------------------------------------------------------
data: data-download data-omics data-priors data-wrap data-splits

data-download:
	$(PY) scripts/download_pancan.py --out data/raw/tcga_pancan
	$(PY) scripts/preprocessing/omics/prepare_pancan.py \
	    --src data/raw/tcga_pancan \
	    --out data/raw/tcga_pancan/processed

data-omics:
	$(PY) scripts/preprocessing/omics/process_data_pancan.py

data-priors:
	$(PY) scripts/preprocessing/priors/get_raw_data.py
	$(PY) scripts/preprocessing/priors/refseq2gene.py
	$(PY) scripts/preprocessing/priors/load_interaction.py

# Produces data/training/*: expression/cnv/labels, tf_nodes_all_in_vocab.csv,
# expression_variance/fvalues, gene_nodes_filtered_for_tf.csv. (Also writes the
# top-100 miRNA panel, which the hetero pipeline bypasses via unified FS.)
data-wrap:
	$(PY) scripts/preprocessing/omics/data_wrapper.py

# Stratified train/val/test indices per seed (data/training/splits/splits_seed_*).
data-splits:
	$(PY) -m multiomics_kg_hgnn.pancancer_prediction.preprocessing.make_splits --seeds $(SEEDS)

# --------------------------------------------------------------------
# Heterogeneous graph construction (NEW — the MOKG-HGNN-specific steps)
# --------------------------------------------------------------------
graph: feature-selection build-graph

# Unified variance feature selection on the TRAIN split (leakage-free):
# gene / TF / miRNA selected together, same criterion, from the full sets.
feature-selection:
	$(PY) -m multiomics_kg_hgnn.pancancer_prediction.preprocessing.feature_selection \
	    --split-dir $(SPLIT_DIR) \
	    --top-genes $(TOP_GENES) --top-tf $(TOP_TF) --top-mirna $(TOP_MIRNA) \
	    --out-dir $(FS_DIR)

# Build the shared HeteroData topology from the selected panels.
# Add METAPATH="--metapath" to include the miRNA-miRNA / TF-TF co-target layers.
build-graph:
	$(PY) scripts/preprocessing/priors/build_hetero_graph.py \
	    --gene-list $(FS_DIR)/selected_genes.csv \
	    --tf-list $(FS_DIR)/selected_tf.csv \
	    --mirna-list $(FS_DIR)/selected_mirna.txt \
	    --go-min-support $(GO_MIN_SUPPORT) $(METAPATH) --force

# --------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------
train:
	$(PY) scripts/kg_hgnn/train.py --config $(CONFIG)

# Evaluate the latest run by default; override with CKPT=path/to/model_best.pt
evaluate:
	$(PY) scripts/kg_hgnn/evaluate.py --config $(CONFIG) $(if $(CKPT),--checkpoint $(CKPT),)

# Quick end-to-end sanity check (real data -> HeteroData -> HGT -> 27 classes)
check:
	$(PY) -m multiomics_kg_hgnn.pancancer_prediction.datasets.check_dataset

# --------------------------------------------------------------------
# Hyperparameter tuning (Optuna) — NOT YET IMPLEMENTED
# --------------------------------------------------------------------
# TODO: port scripts/run_optuna.py to the hetero runner + a config_kg_hgnn search space.
# optuna-smoke:
#	$(PY) scripts/kg_hgnn/run_optuna.py --config $(CONFIG) --n-trials 3 --timeout-hours 0.5
# optuna:
#	$(PY) scripts/kg_hgnn/run_optuna.py --config $(CONFIG) --n-trials 35 --timeout-hours 10

# --------------------------------------------------------------------
# Ablation studies — NOT YET IMPLEMENTED
# --------------------------------------------------------------------
# Intended axes (proposta sez. 6), each isolating one design choice:
#   depth     : molecular-only / +pathway / +pathway+GO / +disease  (--no-disease, template variants)
#   backbone  : hgt vs hetero_sage vs rgcn vs GCN-omogeneo          (model.backbone; sage/rgcn are TODO)
#   omics     : full / -CNV / -miRNA / -TF                          (data.use_cnv / use_mirna)
#   metapath  : with / without miRNA-miRNA & TF-TF                   (METAPATH="--metapath")
#   pathway   : Reactome vs KEGG vs MSigDB                          (KG source, TODO)
#   readout   : molecular / superiore / multi-scala                 (model.readout_types)
# TODO: a sweep runner (scripts/kg_hgnn/run_sweep.py) + sweep configs.
# ablation-depth:
#	$(PY) scripts/kg_hgnn/run_sweep.py --config $(CONFIG) --sweep configs/sweep_kg_depth.yml
# ablation-backbone:
#	$(PY) scripts/kg_hgnn/run_sweep.py --config $(CONFIG) --sweep configs/sweep_kg_backbone.yml
# ablation: ablation-depth ablation-backbone

# --------------------------------------------------------------------
# Analysis & statistical tests — NOT YET IMPLEMENTED
# --------------------------------------------------------------------
# TODO: aggregate per-seed test metrics + Friedman/Wilcoxon-Holm (proposta sez. 5).
# analysis:
#	$(PY) scripts/kg_hgnn/collect_results.py --base_dir results --out_dir analysis

all: data graph train

# --------------------------------------------------------------------
# Housekeeping
# --------------------------------------------------------------------
clean:
	@find . -type d -name __pycache__ -prune -exec rm -rf {} +
	@find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
	@rm -f pipeline.log
