@echo off
setlocal EnableDelayedExpansion
REM Full data preprocessing for the hetero pipeline (Windows port of prepare_data.sh).
REM Produces everything under data\training\ that make_graph.bat and training need:
REM   download TCGA -> prepare/process omics -> priors -> data_wrapper -> splits.
REM
REM The PKT knowledge graph (data\prior_knowledge\PKT) is NOT downloaded here.
REM
REM Usage (from the repo root, after setup_env.bat):
REM   bat_scripts\prepare_data.bat
REM   set SEEDS=42 43 44 && bat_scripts\prepare_data.bat
REM   set SKIP_DOWNLOAD=1 && bat_scripts\prepare_data.bat   (raw already present)

pushd "%~dp0.."

if "%ENV_NAME%"=="" set "ENV_NAME=gnn"
if "%SEEDS%"=="" set "SEEDS=42"
if "%SKIP_DOWNLOAD%"=="" set "SKIP_DOWNLOAD=0"

if defined PYTHONPATH (set "PYTHONPATH=src;%PYTHONPATH%") else (set "PYTHONPATH=src")

if not "%SKIP_DOWNLOAD%"=="1" (
    echo ==^> [1/5] download TCGA pan-cancer + prepare ^(transpose/parquet^)
    call conda run -n "%ENV_NAME%" python scripts\download_pancan.py --out data\raw\tcga_pancan || goto :fail
    call conda run -n "%ENV_NAME%" python scripts\preprocessing\omics\prepare_pancan.py --src data\raw\tcga_pancan --out data\raw\tcga_pancan\processed || goto :fail
) else (
    echo ==^> [1/5] SKIP_DOWNLOAD=1 - using existing data\raw\tcga_pancan
)

echo ==^> [2/5] process omics ^(filter common samples, normalize^)
call conda run -n "%ENV_NAME%" python scripts\preprocessing\omics\process_data_pancan.py || goto :fail

echo ==^> [3/5] priors ^(BioGRID / miRDB / TFLink -^> GGI / miRNA / TF^)
call conda run -n "%ENV_NAME%" python scripts\preprocessing\priors\get_raw_data.py || goto :fail
call conda run -n "%ENV_NAME%" python scripts\preprocessing\priors\refseq2gene.py || goto :fail
call conda run -n "%ENV_NAME%" python scripts\preprocessing\priors\load_interaction.py || goto :fail

echo ==^> [4/5] data_wrapper -^> data\training\* ^(expression/cnv/labels/tf_nodes/variance^)
call conda run -n "%ENV_NAME%" python scripts\preprocessing\omics\data_wrapper.py || goto :fail

echo ==^> [5/5] stratified splits for seeds: %SEEDS%
call conda run -n "%ENV_NAME%" python -m multiomics_kg_hgnn.pancancer_prediction.preprocessing.make_splits --seeds %SEEDS% || goto :fail

echo.
echo ==^> done. data\training\ is populated. Next:
echo     bat_scripts\make_graph.bat ^&^& bat_scripts\train_and_eval.bat

popd
endlocal
exit /b 0

:fail
echo [ERROR] prepare_data failed with exit code %errorlevel% 1>&2
popd
endlocal
exit /b 1
