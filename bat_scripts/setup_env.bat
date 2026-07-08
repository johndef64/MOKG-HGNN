@echo off
setlocal EnableDelayedExpansion
REM Create the conda env `gnn` and install this repo into it (Windows port of setup_env.sh).
REM conda installs everything under your home dir, so admin is never needed.
REM
REM Usage (from the repo root):
REM   bat_scripts\setup_env.bat
REM
REM Override the env name:  set ENV_NAME=myenv && bat_scripts\setup_env.bat

REM --- run from the repo root regardless of where this .bat is called from ---
pushd "%~dp0.."

if "%ENV_NAME%"=="" set "ENV_NAME=gnn"
if "%PY_VER%"=="" set "PY_VER=3.10"
REM CUDA build index for torch 2.7.0 (matches environment.yml). Override for a
REM different CUDA, or set TORCH_INDEX to empty and CPU wheels will be used.
if not defined TORCH_INDEX set "TORCH_INDEX=https://data.pyg.org/whl/torch-2.7.0+cu128.html"

for /f "delims=" %%i in ('where conda 2^>nul') do set "CONDA_PATH=%%i"
echo ==^> using conda: %CONDA_PATH%

REM 1) create the env if it does not exist yet
set "ENV_EXISTS="
for /f "usebackq tokens=1" %%i in (`conda env list`) do (
    if /i "%%i"=="%ENV_NAME%" set "ENV_EXISTS=1"
)
if defined ENV_EXISTS (
    echo ==^> env '%ENV_NAME%' already exists, reusing it
) else (
    echo ==^> creating env '%ENV_NAME%' ^(python %PY_VER%^)
    call conda create -y -n "%ENV_NAME%" "python=%PY_VER%" pip || goto :fail
)

REM 3) torch + PyG from the CUDA wheel index FIRST (they are not on plain PyPI)
if defined TORCH_INDEX (
    echo ==^> installing torch / torch_geometric ^(%TORCH_INDEX%^)
    call conda run -n "%ENV_NAME%" python -m pip install --find-links "%TORCH_INDEX%" torch==2.7.0 torch_geometric==2.7.0 || goto :fail
) else (
    echo ==^> installing torch / torch_geometric ^(CPU wheels^)
    call conda run -n "%ENV_NAME%" python -m pip install torch==2.7.0 torch_geometric==2.7.0 || goto :fail
)

REM 4) install the repo (editable) + the rest of the deps from pyproject
echo ==^> pip install -e . ^(repo + remaining deps^)
call conda run -n "%ENV_NAME%" python -m pip install -e . || goto :fail

REM 5) sanity check
echo ==^> verifying the install
call conda run -n "%ENV_NAME%" python -c "import torch, torch_geometric, multiomics_kg_hgnn; print('torch', torch.__version__, '| PyG', torch_geometric.__version__, '| cuda', torch.cuda.is_available())" || goto :fail

echo.
echo ==^> done. Activate with:  conda activate %ENV_NAME%
echo     then run training:     python scripts\kg_hgnn\train.py --config configs\config_kg_hgnn.yml

popd
endlocal
exit /b 0

:fail
echo [ERROR] setup failed with exit code %errorlevel% 1>&2
popd
endlocal
exit /b 1
