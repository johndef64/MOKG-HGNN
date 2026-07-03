import os
import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def get_common_samples(path_to_omics="omics"):
    """Find common sample IDs across all CSV files in omics folder."""
    csv_files = glob.glob(f"{path_to_omics}/*.csv")
    common_ids = None
    
    for file in csv_files:
        df = pd.read_csv(file)
        if 'sample_id' in df.columns:
            ids = set(df['sample_id'])
            common_ids = ids if common_ids is None else common_ids & ids
            print(f"{file}: {len(ids)} samples")
    
    print(f"Common: {len(common_ids) if common_ids else 0}")
    return common_ids

def filter_files_by_common_samples(input_folder, output_folder):

    """Filter CSV files to keep only common sample IDs."""
    print(f"Filtering files in {input_folder} by common samples...")
    print("=" * 20)
    common_ids = get_common_samples(input_folder)
    csv_files = glob.glob(f"{input_folder}/*.csv")
    print(csv_files)
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    for file in csv_files:
        df = pd.read_csv(file)
        if 'sample_id' in df.columns:
            filtered_df = df[df['sample_id'].isin(common_ids)]
            # Sort by sample_id to ensure consistent ordering
            filtered_df = filtered_df.sort_values(by='sample_id').reset_index(drop=True)
            output_path = file.replace(input_folder, output_folder)
            filtered_df.to_csv(output_path, index=False)
            print(f"Filtered {file} to {output_path}, kept {len(filtered_df)} samples")

def filter_by_common_samples_df(molecular_df, cnv_df, expression_df, mirna_df):
    """Filter molecular subtype dataframe and omics dataframe by common sample IDs."""
    print("Filtering molecular subtype and omics data by common samples...")
    print("=" * 20)

    common_ids = set(molecular_df['sample_id']) & set(cnv_df['sample_id']) & set(expression_df['sample_id']) & set(mirna_df['sample_id'])
    filtered_molecular = molecular_df[molecular_df['sample_id'].isin(common_ids)].reset_index(drop=True)
    filtered_cnv = cnv_df[cnv_df['sample_id'].isin(common_ids)].reset_index(drop=True)
    filtered_expression = expression_df[expression_df['sample_id'].isin(common_ids)].reset_index(drop=True)
    filtered_mirna = mirna_df[mirna_df['sample_id'].isin(common_ids)].reset_index(drop=True)
    print(f"Filtered molecular subtype data: {filtered_molecular.shape[0]} samples")
    print(f"Filtered CNV data: {filtered_cnv.shape[0]} samples")
    print(f"Filtered expression data: {filtered_expression.shape[0]} samples")
    print(f"Filtered miRNA data: {filtered_mirna.shape[0]} samples")
    return filtered_molecular, filtered_cnv, filtered_expression, filtered_mirna


def filter_common_columns(cnv, expression):
    """Filter CNV and expression data to keep only common columns."""
    print("Filtering CNV and expression data by common genes...")
    print("=" * 20)

    # find common columns
    common_columns = set(cnv.columns).intersection(set(expression.columns))
    print(f"Common genes: {len(common_columns) - 1}")  # subtract 1 for sample_id column
    # filter dataframes by common columns (convert set to list for pandas)
    common_columns_list = list(common_columns)
    cnv_filtered = cnv.loc[:, common_columns_list]
    expression_filtered = expression.loc[:, common_columns_list]
    
    print(f"Filtered CNV data: {cnv_filtered.shape[1] - 1} genes, {cnv_filtered.shape[0]} samples")
    print(f"Filtered expression data: {expression_filtered.shape[1] - 1} genes, {expression_filtered.shape[0]} samples")
    
    return cnv_filtered, expression_filtered

def normalize_sample_ids(sample_ids):
    """
    Normalize TCGA sample IDs to TCGA-XX-YYYY format.
    
    Args:
        sample_ids: List or Series of sample IDs
    
    Returns:
        list: Normalized sample IDs
    """
    # normlize to TCGA-XX-YYYY format
    normalized = []
    for sample_id in sample_ids:
        parts = str(sample_id).split("-")
        if len(parts) >= 3:
            # Take first 3 characters of the 4th part for ZZ format
            #zz_part = parts[3][:3] if len(parts[3]) >= 3 else parts[3]
            #normalized.append(f"{parts[0]}-{parts[1]}-{parts[2]}-{zz_part}")
            normalized.append(f"{parts[0]}-{parts[1]}-{parts[2]}")
        else:
            normalized.append(str(sample_id))
    return normalized

def check_nan(input_folder):
    """
    Check for NaN values in CSV files within the input folder.
    
    Args:
        input_folder (str): Path to the folder containing CSV files
    
    Returns:
        dict: Mapping of file names to count of NaN values
    """
    csv_files = glob.glob(f"{input_folder}/*.csv")
    nan_counts = {}
    
    for file in csv_files:
        df = pd.read_csv(file)
        total_nans = df.isna().sum().sum()
        nan_counts[os.path.basename(file)] = total_nans
        print(f"{file}: {total_nans} NaN values")
    
    return nan_counts

def check_duplicates(input_folder):
    """
    Check for duplicate sample IDs in CSV files within the input folder.
    
    Args:
        input_folder (str): Path to the folder containing CSV files
    
    Returns:
        dict: Mapping of file names to count of duplicate sample IDs
    """
    csv_files = glob.glob(f"{input_folder}/*.csv")
    duplicate_counts = {}
    
    for file in csv_files:
        df = pd.read_csv(file)
        if 'sample_id' in df.columns:
            duplicates = df['sample_id'].duplicated().sum()
            duplicate_counts[os.path.basename(file)] = duplicates
            print(f"{file}: {duplicates} duplicate sample IDs")
        else:
            duplicate_counts[os.path.basename(file)] = 0
            print(f"{file}: 'sample_id' column not found")
    
    return duplicate_counts

def normalize_omics_data(df, data_type):
    if data_type == "CNV":
        print(f"Normalizing {data_type} data using z-score normalization...")
        # Passa solo i dati numerici (senza sample_id)
        numeric_data = df.iloc[:, 1:]
        normalized_data = z_score_normalization(numeric_data)
        # Riassegna i dati normalizzati
        df.iloc[:, 1:] = normalized_data
        print(f"{data_type} data normalized.")
        return df
    else:
        print(f"Normalizing {data_type} data using min-max scaling...")
        # Passa solo i dati numerici (senza sample_id)
        numeric_data = df.iloc[:, 1:]
        normalized_data = min_max_normalize(numeric_data)
        # Riassegna i dati normalizzati
        df.iloc[:, 1:] = normalized_data
        print(f"{data_type} data normalized.")
        return df

def z_score_normalization(df):
    """Apply z-score normalization to the dataframe."""
    # Converti in numeric e gestisci i valori non numerici
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    
    # Sostituisci NaN con 0 (o il valore che preferisci)
    df_numeric = df_numeric.fillna(0)
    
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df_numeric)
    
    # Restituisci un DataFrame con gli stessi indici e colonne
    return pd.DataFrame(normalized_data, index=df.index, columns=df.columns)

def min_max_normalize(df):
    """Apply min-max normalization to the dataframe."""
    # Converti in numeric e gestisci i valori non numerici
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    
    # Sostituisci NaN con 0 (o il valore che preferisci)
    df_numeric = df_numeric.fillna(0)
    
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df_numeric)
    
    # Restituisci un DataFrame con gli stessi indici e colonne
    return pd.DataFrame(normalized_data, index=df.index, columns=df.columns)

def variance_processing(expression_file, output_file):
    """Calculate variance of gene expression and save to CSV."""
    print(f"Calculating variance for {expression_file}...")
    
    # Carica il file CSV
    expression_data = pd.read_csv(expression_file)
    
    # Rimuovi la colonna sample_id se presente
    if 'sample_id' in expression_data.columns:
        expression_data = expression_data.drop(columns=['sample_id'])
    
    # Converti in numeric per il calcolo della varianza
    expression_data = expression_data.apply(pd.to_numeric, errors='coerce')
    expression_data = expression_data.fillna(0)
    
    # Calcola la varianza per colonna (gene)
    expression_variance = expression_data.var(axis=0)
    expression_variance = expression_variance.sort_index()
    
    # Salva il risultato
    # Crea DataFrame con nomi di colonne corretti
    variance_df = pd.DataFrame({
        'gene': expression_variance.index,
        'variance': expression_variance.values
    })
    
    print(variance_df.head())
    variance_df.to_csv(output_file, index=False)
    print(f"Expression variance saved to '{output_file}'")