"""Processa i file scaricati: conversione in parquet, transpose nelle matrici dei dati omici, controllo dtypes.

Esempio:
  python prepare_pancan.py --src data/pancan_tcga/raw --out data/pancan_tcga/processed
"""
import os
import gc
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def build_dtypese(file) -> dict:
    header = pd.read_csv(file, nrows=0)
    cols = header.columns.tolist()
    dtypes = {}
    for c in cols:
        if c in ("sample", "icluster_cluster_assignment", "sample_id"):
            dtypes[c] = "string"
        elif c == "cohort":
            dtypes[c] = "category"
        else:
            dtypes[c] = "float32"
    return dtypes


def process_file(key, file_path, save_dir):
    print(f"Processing {file_path} (key={key})...")
    dtype = build_dtypese(file_path)
    try:
        if file_path.endswith('.gz'):
            df = pd.read_csv(file_path, sep='\t', compression='gzip', low_memory=False, dtype=dtype)
        else:
            df = pd.read_csv(file_path, sep='\t', low_memory=False, dtype=dtype)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # For omics data (expression, mirna, cnv, metylation) transpose
    if key in ['expression', 'mirna', 'cnv', 'metylation']:
        print(f"Transposing {file_path}...")
        df = df.set_index(df.columns[0]).T.reset_index().rename(columns={"index": "sample_id"})
    print(f"[{key}] Shape after transpose/cast: {df.shape}")
    # Convert numeric columns to float32 to save memory
    if df.columns.duplicated().any():
        print(f"Warning: Duplicate columns found in {file_path}: {df.columns[df.columns.duplicated()].tolist()}")
        print(f"Shape before removing duplicate columns: {df.shape}")
        df = df.loc[:, ~df.columns.duplicated()]
        print(f"Removed duplicate columns, new shape: {df.shape}")
    print(f"dtypes before conversion for {file_path}:")
    print(df.dtypes)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].astype('float32')

    # Remove duplicate columns if present
    print(f"dtypes after conversion for {file_path}:")
    print(df.dtypes)
    # Print memory usage
    print(df.info(memory_usage="deep"))

    parquet_file = os.path.basename(file_path).replace('.tsv.gz', '.parquet').replace('.tsv', '.parquet').replace('.gz', '.parquet')
    parquet_path = os.path.join(save_dir, parquet_file)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving {parquet_path}...")
    try:
        df.to_parquet(parquet_path, engine='pyarrow', compression='zstd')
    except Exception:
        # fallback to pyarrow write_table
        pq.write_table(pa.Table.from_pandas(df), parquet_path, compression='snappy')

    print(f"Saved {parquet_path} shape={df.shape}")
    del df
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='data/pancan_tcga/raw', help='Directory with downloaded raw files')
    parser.add_argument('--out', default='data/pancan_tcga/processed', help='Directory for parquet outputs')
    parser.add_argument('--keys', default=None, help='Comma-separated keys to process (clinical,expression,mirna,cnv)')
    args = parser.parse_args()

    keys = args.keys.split(',') if args.keys else None

    # map expected keys to filenames (if names differ, user can edit)
    expected = {
        'clinical': 'clinical.tsv.gz',
        'expression': 'expression.tsv.gz',
        'mirna': 'mirna.tsv.gz',
        'cnv': 'cnv.tsv.gz'
    }

    for key, fname in expected.items():
        if keys and key not in keys:
            continue
        fpath = os.path.join(args.src, fname)
        print(f"Checking file for key={key}: {fpath}")
        if os.path.exists(fpath):
            process_file(key, fpath, args.out)
        else:
            print(f"File not found, skipping: {fpath}")


if __name__ == '__main__':
    main()
