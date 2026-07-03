# MOGNN-TF — pipeline orchestration
# Each target corresponds to a step in the pipeline described in scripts/README.md.
# Usage:
#   make install          # install dependencies (pip)
#   make data             # full preprocessing: download + omics + priors + training data
#   make train            # single run with default config (MOGNN-TF GCN, 700 genes, 200 TF)
#   make ablation         # sweep: gene count, sampler, TF, split ratio
#   make optuna           # hyperparameter search
#   make analysis         # aggregate results + statistical tests
#   make all              # data -> train -> analysis
#   make clean            # remove caches and intermediate outputs

PY ?= python

.PHONY: help install data data-download data-omics data-priors data-wrap \
        train train-final \
        optuna optuna-smoke \
        ablation ablation-baseline ablation-genes ablation-sampler ablation-tf ablation-split-ratio \
        ablation-smoke ablation-baseline-smoke ablation-genes-smoke ablation-sampler-smoke \
        ablation-tf-smoke ablation-split-ratio-smoke \
        analysis all clean

help:
	@echo "Targets:"
	@echo "  install                    - pip install -e ."
	@echo "  data                       - run full preprocessing pipeline"
	@echo ""
	@echo "  train                      - single run, starting baseline (configs/config.yml)"
	@echo "  train-final                - single run, paper final config (configs/config_final.yml)"
	@echo ""
	@echo "  optuna-smoke               - Optuna quick check (3 trials, 30 min)"
	@echo "  optuna                     - Optuna full study (35 trials, 10h, as in paper)"
	@echo ""
	@echo "  ablation-<name>            - reproduce paper Table <name> (full)"
	@echo "  ablation-<name>-smoke      - quick smoke version of the same ablation"
	@echo "  <name> in: baseline, genes, sampler, tf, split-ratio"
	@echo "  ablation                   - all 5 ablations (hundreds of runs!)"
	@echo "  ablation-smoke             - all 5 smoke ablations (~15 runs)"
	@echo ""
	@echo "  analysis                   - aggregate metrics + Friedman/Wilcoxon-Holm tests"
	@echo "  all                        - data + train-final + analysis"
	@echo "  clean                      - remove __pycache__ and stale logs"

install:
	$(PY) -m pip install -e .

# --------------------------------------------------------------------
# Data preprocessing pipeline
# --------------------------------------------------------------------
data: data-download data-omics data-priors data-wrap

data-download:
	$(PY) scripts/download_pancan.py --out data/raw/tcga_pancan
	$(PY) scripts/preprocessing/omics/prepare_pancan.py \
	    --src data/raw/tcga_pancan \
	    --out data/raw/tcga_pancan/processed

# Note: no --normalization here. The wrapper / training pipeline performs
# min-max scaling on the training fold only (to avoid data leakage), so
# applying it pan-cohort upstream is redundant and produces files with a
# "_normalized" suffix that data-wrap does not expect.
data-omics:
	$(PY) scripts/preprocessing/omics/process_data_pancan.py

data-priors:
	$(PY) scripts/preprocessing/priors/get_raw_data.py
	$(PY) scripts/preprocessing/priors/refseq2gene.py
	$(PY) scripts/preprocessing/priors/load_interaction.py

data-wrap:
	$(PY) scripts/preprocessing/omics/data_wrapper.py

# --------------------------------------------------------------------
# Single-run training
# --------------------------------------------------------------------
# `train`       : starting baseline (configs/config.yml, Laterza's pre-Optuna setup)
# `train-final` : paper final configuration (configs/config_final.yml, trial 18 + TF)
train:
	$(PY) scripts/train.py --config configs/config.yml

train-final:
	$(PY) scripts/train.py --config configs/config_final.yml

# --------------------------------------------------------------------
# Hyperparameter search (Optuna)
# --------------------------------------------------------------------
# Pipeline order:  optuna  →  update config_final.yml with best params  →  ablation
optuna-smoke:
	$(PY) scripts/run_optuna.py --config configs/config.yml \
	    --n-trials 3 --timeout-hours 0.5 --study-name optuna_smoke

optuna:
	$(PY) scripts/run_optuna.py --config configs/config.yml \
	    --n-trials 35 --timeout-hours 10 --study-name pancan_optuna_full

# --------------------------------------------------------------------
# Ablation studies (one target per paper table)
# --------------------------------------------------------------------
# Each *-smoke target uses 1-2 seeds and 10 epochs for a quick end-to-end
# sanity check. The non-smoke targets reproduce the full table reported in
# the paper (tens to hundreds of runs each). All use config_final.yml as
# the base configuration.

ablation-baseline:
	$(PY) scripts/run_sweep.py --config configs/config_final.yml --sweep configs/sweep_baseline.yml

ablation-baseline-smoke:
	$(PY) scripts/run_sweep.py --config configs/config_final.yml --sweep configs/sweep_baseline_smoke.yml

ablation-genes:
	$(PY) scripts/run_sweep.py --config configs/config_final.yml --sweep configs/sweep_genes.yml

ablation-genes-smoke:
	$(PY) scripts/run_sweep.py --config configs/config_final.yml --sweep configs/sweep_genes_smoke.yml

ablation-sampler:
	$(PY) scripts/run_sweep.py --config configs/config_final.yml --sweep configs/sweep_sampler.yml

ablation-sampler-smoke:
	$(PY) scripts/run_sweep.py --config configs/config_final.yml --sweep configs/sweep_sampler_smoke.yml

ablation-tf:
	$(PY) scripts/run_sweep.py --config configs/config_final.yml --sweep configs/sweep_tf.yml

ablation-tf-smoke:
	$(PY) scripts/run_sweep.py --config configs/config_final.yml --sweep configs/sweep_tf_smoke.yml

ablation-split-ratio:
	$(PY) scripts/run_sweep.py --config configs/config_final.yml --sweep configs/sweep_split_ratio.yml

ablation-split-ratio-smoke:
	$(PY) scripts/run_sweep.py --config configs/config_final.yml --sweep configs/sweep_split_ratio_smoke.yml

# Aggregate aliases
ablation: ablation-baseline ablation-genes ablation-sampler ablation-tf ablation-split-ratio
ablation-smoke: ablation-baseline-smoke ablation-genes-smoke ablation-sampler-smoke ablation-tf-smoke ablation-split-ratio-smoke

# --------------------------------------------------------------------
# Analysis & aggregation
# --------------------------------------------------------------------
# `collect_results.py` writes its CSVs into --out_dir.
# `analize_metrics.py` has no CLI: it reads runs_for_stats.csv and
# class_report_long.csv from the current working directory, so we cd into
# the analysis folder before invoking it.
analysis:
	$(PY) scripts/collect_results.py --base_dir . --dataset pancan --out_dir analysis
	cd analysis && $(PY) ../scripts/analysis/analize_metrics.py

all: data train-final analysis

# --------------------------------------------------------------------
# Housekeeping
# --------------------------------------------------------------------
clean:
	@find . -type d -name __pycache__ -prune -exec rm -rf {} +
	@find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
	@rm -f pipeline.log
