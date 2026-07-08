@echo off
setlocal
REM Train + evaluate the heterogeneous model (Windows launcher).
REM Thin wrapper: all logic lives in helpers\train_and_eval.py (the Python port of
REM the old train_and_eval.sh). Runs from the repo root.
REM
REM Usage (from the repo root):
REM   bat_scripts\train_and_eval.bat                 (single run, seed 42)
REM   bat_scripts\train_and_eval.bat --runs 5        (seeds 42..46, per-seed graph, aggregate)
REM
REM Knobs (env vars): CONFIG, ENV_NAME, MODEL_SEED, START_SEED, TOP_GENES, TOP_TF,
REM                   TOP_MIRNA, GO_MIN_SUPPORT, METAPATH, BACKBONE

pushd "%~dp0.."

if "%ENV_NAME%"=="" set "ENV_NAME=gnn"
set "PYTHONUNBUFFERED=1"

call conda run --no-capture-output -n "%ENV_NAME%" python -u "%~dp0helpers\train_and_eval.py" %*
set "RC=%errorlevel%"

popd
endlocal & exit /b %RC%
