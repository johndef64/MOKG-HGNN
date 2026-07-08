@echo off
setlocal
REM Feature-collapse study, MOKG-HGNN (heterogeneous) — Windows launcher.
REM Thin wrapper: logic lives in helpers\collapse_mokghgnn.py (port of the .sh).
REM
REM Usage (from the repo root):
REM   bat_scripts\collapse_mokghgnn.bat
REM   set SEEDS=42 43 44 45 46 && set GENE_GRID=700 500 300 150 100 50 20 && bat_scripts\collapse_mokghgnn.bat
REM   set BACKBONE=rgcn && set CONFIG=configs\config_kg_hgnn_best.yml && bat_scripts\collapse_mokghgnn.bat

pushd "%~dp0.."

if "%ENV_NAME%"=="" set "ENV_NAME=gnn"
set "PYTHONUNBUFFERED=1"

call conda run --no-capture-output -n "%ENV_NAME%" python -u "%~dp0helpers\collapse_mokghgnn.py" %*
set "RC=%errorlevel%"

popd
endlocal & exit /b %RC%
