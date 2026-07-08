@echo off
setlocal EnableDelayedExpansion
REM Hyperparameter tuning (Optuna) for the heterogeneous model (Windows port of tune.sh).
REM Tunes model/training knobs on a FIXED template; objective = validation macro-F1.
REM Assumes setup_env.bat + make_graph.bat have run (template must exist).
REM
REM Usage (from the repo root):
REM   bat_scripts\tune.bat
REM   set N_TRIALS=20 && set TIMEOUT_HOURS=4 && bat_scripts\tune.bat

pushd "%~dp0.."

if "%ENV_NAME%"=="" set "ENV_NAME=gnn"
if "%CONFIG%"=="" set "CONFIG=configs\config_kg_hgnn.yml"
if "%BACKBONE%"=="" set "BACKBONE=hgt"
if "%N_TRIALS%"=="" set "N_TRIALS=35"
if "%TIMEOUT_HOURS%"=="" set "TIMEOUT_HOURS=10"
if "%TUNE_EPOCHS%"=="" set "TUNE_EPOCHS=60"
if "%TUNE_PATIENCE%"=="" set "TUNE_PATIENCE=12"
if "%STUDY_NAME%"=="" set "STUDY_NAME=kg_hgnn_optuna_%BACKBONE%"
if "%OUT_DIR%"=="" set "OUT_DIR=results\optuna"

if defined PYTHONPATH (set "PYTHONPATH=src;%PYTHONPATH%") else (set "PYTHONPATH=src")

set "TEMPLATE=data\prior_knowledge\hetero\hetero_graph_template.pt"
if not exist "%TEMPLATE%" (
    echo ERROR: template not found ^(%TEMPLATE%^). Run make_graph.bat first. 1>&2
    popd & endlocal & exit /b 1
)

echo ==^> Optuna tuning ^| backbone=%BACKBONE% ^| %N_TRIALS% trials ^| %TIMEOUT_HOURS%h timeout ^| epochs %TUNE_EPOCHS%
echo     objective: validation macro-F1 ^| template: fixed ^| study: %STUDY_NAME%
REM --no-capture-output + python -u: stream progress LIVE (conda run buffers otherwise).
set "PYTHONUNBUFFERED=1"
call conda run --no-capture-output -n "%ENV_NAME%" python -u scripts\kg_hgnn\run_optuna.py --config "%CONFIG%" --backbone "%BACKBONE%" --n-trials "%N_TRIALS%" --timeout-hours "%TIMEOUT_HOURS%" --tune-epochs "%TUNE_EPOCHS%" --tune-patience "%TUNE_PATIENCE%" --study-name "%STUDY_NAME%" --out-dir "%OUT_DIR%" || goto :fail

echo.
echo ==^> done. Best params in: %OUT_DIR%\%STUDY_NAME%\best.json
echo     full report:          %OUT_DIR%\%STUDY_NAME%\optuna_trials_report.csv
echo     next: copy best params into %CONFIG%, then: bat_scripts\train_and_eval.bat --runs 5

popd
endlocal
exit /b 0

:fail
echo [ERROR] tune failed with exit code %errorlevel% 1>&2
popd
endlocal
exit /b 1
