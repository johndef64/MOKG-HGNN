@echo off
setlocal EnableDelayedExpansion
REM Tune all three backbones (hgt, hetero_sage, rgcn) in sequence (port of tune_all.sh).
REM Simple fixed split of the total budget: 20h / 3 backbones ~= 6.5h each.
REM One separate Optuna study per backbone (best config each).
REM
REM Usage (from the repo root):
REM   bat_scripts\tune_all.bat

pushd "%~dp0.."

REM ~6.5h each x 3 = ~19.5h total (under 20h). Each study also stops at N_TRIALS.
if "%HOURS_EACH%"=="" set "HOURS_EACH=6.5"
if "%N_TRIALS%"=="" set "N_TRIALS=35"

for %%b in (hgt hetero_sage rgcn) do (
    echo.
    echo ===================== tuning backbone=%%b ^(%HOURS_EACH%h^) =====================
    REM run each study in its own cmd so per-backbone env vars don't leak between iterations
    cmd /c "set BACKBONE=%%b&& set TIMEOUT_HOURS=%HOURS_EACH%&& set N_TRIALS=%N_TRIALS%&& call "%~dp0tune.bat""
    if errorlevel 1 echo [tune_all] study '%%b' ended with a non-zero status; continuing.
)

echo.
echo ==^> done. Best params: results\optuna\kg_hgnn_optuna_{hgt,hetero_sage,rgcn}\best.json

popd
endlocal
exit /b 0
