@echo off
setlocal EnableDelayedExpansion
REM Feature-collapse experiment (thesis P1): "does the graph save performance?"
REM Windows port of run_feature_collapse.sh. Runs BOTH models across a decreasing
REM gene grid and collects results, so MOKG-HGNN's degradation slope can be
REM compared to the MOGNN-TF baseline. Results under results\feature_collapse\.
REM
REM Usage (from the repo root):
REM   bat_scripts\run_feature_collapse.bat
REM   set MODELS=mokghgnn && bat_scripts\run_feature_collapse.bat
REM   set GENE_GRID=700 300 100 50 && set SEEDS=42 43 44 && bat_scripts\run_feature_collapse.bat

pushd "%~dp0.."

if "%ENV_NAME%"=="" set "ENV_NAME=gnn"
if "%MODELS%"=="" set "MODELS=mognntf mokghgnn"
if "%GENE_GRID%"=="" set "GENE_GRID=700 500 300 150 100 50 20"
if "%SEEDS%"=="" set "SEEDS=42 43 44 45 46"
if "%MODEL_SEED%"=="" set "MODEL_SEED=2025"
if "%OUT_ROOT%"=="" set "OUT_ROOT=results\feature_collapse"
if "%MOGNNTF_CONFIG%"=="" set "MOGNNTF_CONFIG=configs\config_final.yml"
if "%MOKG_CONFIG%"=="" set "MOKG_CONFIG=configs\config_kg_hgnn.yml"
if "%BACKBONE%"=="" set "BACKBONE=hgt"

if defined PYTHONPATH (set "PYTHONPATH=src;%PYTHONPATH%") else (set "PYTHONPATH=src")
set "PYTHONUNBUFFERED=1"
if not exist "%OUT_ROOT%" mkdir "%OUT_ROOT%"

echo ############################################################
echo # FEATURE-COLLAPSE experiment
echo # models: %MODELS% ^| genes: %GENE_GRID% ^| seeds: %SEEDS%
echo # out: %OUT_ROOT%
echo ############################################################

for %%M in (%MODELS%) do (
    if /i "%%M"=="mognntf" (
        echo.
        echo ########## MOGNN-TF ^(homogeneous baseline^) ##########
        call conda run --no-capture-output -n "%ENV_NAME%" python -u scripts\collapse_mognntf.py --config "%MOGNNTF_CONFIG%" --gene-grid %GENE_GRID% --seeds %SEEDS% --model-seed "%MODEL_SEED%" --out-root "%OUT_ROOT%" || goto :fail
    ) else if /i "%%M"=="mokghgnn" (
        echo.
        echo ########## MOKG-HGNN ^(heterogeneous, backbone=%BACKBONE%^) ##########
        REM delegate to the MOKG-HGNN collapse launcher, passing knobs via env
        cmd /c "set GENE_GRID=%GENE_GRID%&& set SEEDS=%SEEDS%&& set MODEL_SEED=%MODEL_SEED%&& set CONFIG=%MOKG_CONFIG%&& set BACKBONE=%BACKBONE%&& set OUT_ROOT=%OUT_ROOT%&& call "%~dp0collapse_mokghgnn.bat"" || goto :fail
    ) else (
        echo unknown model: %%M ^(use mognntf ^| mokghgnn^) 1>&2
        goto :fail
    )
)

echo.
echo ########## AGGREGATE ##########
call conda run --no-capture-output -n "%ENV_NAME%" python -u scripts\kg_hgnn\collapse_aggregate.py --results "%OUT_ROOT%" || goto :fail

echo.
echo ==^> done. Comparison table + plot in: %OUT_ROOT%\

popd
endlocal
exit /b 0

:fail
echo [ERROR] run_feature_collapse failed with exit code %errorlevel% 1>&2
popd
endlocal
exit /b 1
