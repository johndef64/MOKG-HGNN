@echo off
setlocal EnableDelayedExpansion
REM Build the heterogeneous backbone graph (Windows port of make_graph.sh).
REM Two steps: (1) unified variance feature selection on the train split,
REM            (2) build the HeteroData template from the selected panels.
REM
REM Usage (from the repo root, after setup_env.bat):
REM   bat_scripts\make_graph.bat
REM
REM Override any knob via env vars, e.g.:
REM   set SEED=43 && set TOP_GENES=900 && set METAPATH=--metapath && bat_scripts\make_graph.bat

pushd "%~dp0.."

if "%ENV_NAME%"=="" set "ENV_NAME=gnn"
if "%SEED%"=="" set "SEED=42"
if "%TOP_GENES%"=="" set "TOP_GENES=700"
if "%TOP_TF%"=="" set "TOP_TF=200"
if "%TOP_MIRNA%"=="" set "TOP_MIRNA=100"
if "%GO_MIN_SUPPORT%"=="" set "GO_MIN_SUPPORT=3"
REM set METAPATH to "--metapath" to add miRNA-miRNA / TF-TF (empty by default)
if not defined METAPATH set "METAPATH="
if "%SPLIT_DIR%"=="" set "SPLIT_DIR=data\training\splits\splits_seed_%SEED%"
if "%FS_DIR%"=="" set "FS_DIR=data\training\feature_selection\splits_seed_%SEED%"
if "%OUT_DIR%"=="" set "OUT_DIR=data\prior_knowledge\hetero"

if defined PYTHONPATH (set "PYTHONPATH=src;%PYTHONPATH%") else (set "PYTHONPATH=src")

echo ==^> feature selection ^(variance, train-only^) ^| seed=%SEED% genes=%TOP_GENES% tf=%TOP_TF% mirna=%TOP_MIRNA%
call conda run -n "%ENV_NAME%" python -m multiomics_kg_hgnn.pancancer_prediction.preprocessing.feature_selection --split-dir "%SPLIT_DIR%" --top-genes "%TOP_GENES%" --top-tf "%TOP_TF%" --top-mirna "%TOP_MIRNA%" --out-dir "%FS_DIR%" || goto :fail

echo ==^> build hetero template %METAPATH%
call conda run -n "%ENV_NAME%" python scripts\preprocessing\priors\build_hetero_graph.py --gene-list "%FS_DIR%\selected_genes.csv" --tf-list "%FS_DIR%\selected_tf.csv" --mirna-list "%FS_DIR%\selected_mirna.txt" --go-min-support "%GO_MIN_SUPPORT%" %METAPATH% --out-dir "%OUT_DIR%" --force || goto :fail

echo.
echo ==^> done. Template + node/edge CSVs in: %OUT_DIR%

popd
endlocal
exit /b 0

:fail
echo [ERROR] make_graph failed with exit code %errorlevel% 1>&2
popd
endlocal
exit /b 1
