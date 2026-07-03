"""
Simple data processing script for TCGA BRCA data
"""

# Ensembl to Gene Symbol mapping from biomart, using the BioMart API
# check make_conversion_table.R to generate the conversion table

import pandas as pd
import os
#from boruta_feature_selection import feature_selection_boruta
from utils import filter_by_common_samples_df, filter_common_columns, normalize_sample_ids, filter_files_by_common_samples, normalize_omics_data, variance_processing
import argparse
# Select only the BRCA samples with 01 and not 11, 06, etc.
# 01 -> Primary Solid Tumor
# 11 -> Solid Tissue Normal
# 06 -> Metastatic
# https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/sample-type-codes?utm_source=chatgpt.com
# https://docs.gdc.cancer.gov/Encyclopedia/pages/TCGA_Barcode/#:~:text=TCGA%20barcodes%20were%20used%20to,metadata%20values%20for%20a%20sample.
# for the Sample ID has been choosen to keep only the vials with A>B>C.
# Preprocessing of the data
# 1) Get the BRCA subtype from the clinical data filtering and excluding normal-like samples
# 2) Process omics data (miRNA, CNV, FPKM) filtering by BRCA patients only and sorting by sample_id
# 3) Convert Ensembl IDs to gene symbols in FPKM data using conversion table
# the conversion keep only the genes that have entrez id for the intercompatibility with Biogrid GGI database
# for example TP53, BRCA1, EGFR, etc.
# 4) Generate not normalized and normalized (z-score) versions of the omics data
# 5) Save processed data to CSV files

def process_clinical_data(file_path, output_dir="omics",filter_TCGA=None):
    """
    Load and process BRCA subtype data.
    
    Args:
        file_path (str): Path to clinical data CSV file
        output_dir (str): Directory to save processed data
    
    Returns:
        pd.DataFrame: Processed clinical data with sample_id and BRCA_Subtype columns
    """
    if file_path.endswith('.parquet'):
        clinical_df = pd.read_parquet(file_path)
    else:
        clinical_df = pd.read_csv(file_path)
    # print original shape
    print(f"\nProcessing clinical data from: {file_path}")
    print(f"Original clinical data shape: {clinical_df.shape[0]} samples x {clinical_df.shape[1]} columns")
    print(clinical_df.head())
    
    # Map column names
    #clinical_df = clinical_df.rename(columns={
    #    'pan.samplesID': 'sample_id',
    #    'Subtype_Selected': 'Molecular_Subtype'
    #})
    # Map column names
    clinical_df = clinical_df.rename(columns={
        'sample': 'sample_id',
        'icluster_cluster_assignment': 'Molecular_Subtype'
    })


    
    # Select relevant columns
    clinical_df = clinical_df[['sample_id', 'Molecular_Subtype']]
    print(f"Clinical data shape: {clinical_df.shape[0]} patients x {clinical_df.shape[1]} columns")
    # Normalize sample IDs
    #clinical_df['sample_id'] = normalize_sample_ids(clinical_df['sample_id'])
    print(f"Clinical data shape: {clinical_df.shape[0]} patients x {clinical_df.shape[1]} columns")
    # Filter for BRCA subtypes only
    if filter_TCGA != None:
        clinical_df = clinical_df[clinical_df['Molecular_Subtype'].str.contains(filter_TCGA, na=False)]
        clinical_df['Molecular_Subtype'] = clinical_df['Molecular_Subtype'].str.replace(filter_TCGA + '.', '', regex=False)
        print(f"Filtered clinical data to {len(clinical_df)} samples with specified TCGA filter: {filter_TCGA}")

    print(f"Number of samples:\n{clinical_df['Molecular_Subtype'].value_counts()}")
    # Optional: Filter out normal-like samples
    print(f"Clinical data shape: {clinical_df.shape[0]} patients x {clinical_df.shape[1]} columns")
    mask = clinical_df['Molecular_Subtype'].str.contains('Normal', na=False)
    clinical_df = clinical_df[~mask]
    print(f"Clinical data shape: {clinical_df.shape[0]} patients x {clinical_df.shape[1]} columns")
    # Optional: Filter out the samples that are not primary solid tumor (01) and belong to the A vial
    #clinical_df = clinical_df[clinical_df['sample_id'].str.contains('-01A')]
    #print(f"Clinical data shape: {clinical_df.shape[0]} patients x {clinical_df.shape[1]} columns")
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

     # Print shape information
    print(f"Clinical data shape: {clinical_df.shape[0]} patients x {clinical_df.shape[1]} columns")
    print(f"BRCA subtypes: {clinical_df['Molecular_Subtype'].value_counts().to_dict()}")

    
    # Save processed clinical data
    clinical_output_file = os.path.join(output_dir, "clinical.csv")
    clinical_df.to_csv(clinical_output_file, index=False)
    print(f"Saved clinical data to: {clinical_output_file}")
    
    return clinical_df

def process_omics_data(file_path, brca_subtype_df, data_type="omics", output_dir="omics", ensembl_to_symbol_dict=None, normalize=False):
    """
    Load and process omics data (miRNA, CNV, etc.), filtering by BRCA patients.
    
    Args:
        file_path (str): Path to data CSV file
        brca_subtype_df (pd.DataFrame): Clinical data with sample IDs
        data_type (str): Type of data for logging
        output_dir (str): Directory to save processed data
        ensembl_to_symbol_dict (dict): Mapping from Ensembl IDs to gene symbols

    Returns:
        pd.DataFrame: Filtered omics data
    """

    # print brca_subtype_df shape
    print(f"\Molecular subtype data shape: {brca_subtype_df.shape[0]} samples x {brca_subtype_df.shape[1]} columns")

    print(f"\nProcessing {data_type} data from: {file_path}")
    # adding parquet support
    if file_path.endswith('.parquet'):
        data = pd.read_parquet(file_path)
    else:
        data = pd.read_csv(file_path)
    print(f"Original {data_type} data shape: {data.shape[0]} samples x {data.shape[1]} features")
    print(data.head())
    
    if 'sampleID' in data.columns:
        data = data.drop(columns=['sampleID'])
    
    # Normalize sample IDs if sample_id column exists
    if 'sample_id' in data.columns:
        #data['sample_id'] = normalize_sample_ids(data['sample_id'])
        
        # Filter for BRCA patients
        original_samples = len(data)
        data = data[data['sample_id'].isin(brca_subtype_df['sample_id'])]
        print(f"Filtered to BRCA patients: {original_samples} -> {len(data)} samples")
    
    # Get PAR_Y columns and remove them first
    par_y_cols = [col for col in data.columns if 'PAR_Y' in col]
    if par_y_cols:
        print(f"Removing {len(par_y_cols)} PAR_Y columns")
        data = data.drop(columns=par_y_cols)

    duplicate_columns = data.columns[data.columns.duplicated()].unique()
    print(f"Duplicate columns in {data_type} data before renaming: {len(duplicate_columns)} duplicates found")
    # check how many columns have | in their name
    pipe_columns = [col for col in data.columns if '|' in col]
    print(f"Columns with '|' in their name in {data_type} data: {len(pipe_columns)} columns found")

    # Process column names (remove version numbers from Ensembl IDs)
    data.columns = [col.split('.')[0] for col in data.columns]

    print(f"Processed column names (removed version numbers)")

    duplicate_columns = data.columns[data.columns.duplicated()].unique()
    print(f"Duplicate columns in {data_type} data after renaming: {len(duplicate_columns)} duplicates found")

    # Convert Ensembl IDs to gene symbols if mapping is provided
    #if ensembl_to_symbol_dict is not None:
    #    converted_cols = []
    #    unconverted_cols = []
    #    
    #    for col in data.columns:
    #        if isinstance(col, str) and col.startswith('ENSG'):
    #            if col in ensembl_to_symbol_dict:
    #                converted_cols.append(col)
    #            else:
    #                unconverted_cols.append(col)
    #    
        # Apply the mapping and keep only columns that can be converted in the data
    #    if converted_cols:
    #        data = data.rename(columns=ensembl_to_symbol_dict)
    #        print(f"Converted {len(converted_cols)} Ensembl IDs to gene symbols in {data_type} data")
    #    if unconverted_cols:
    #        print(f"Could not convert {len(unconverted_cols)} Ensembl IDs in {data_type} data")
    #        # Keep only columns that were converted
    #        data = data.drop(columns=unconverted_cols)

    # Drop columns that are all NaN
    print(f"Number of column with all Nan values before dropping: {data.isna().all().sum()}")
    original_cols = data.shape[1]
    data = data.dropna(axis=1, how='all')
    dropped_cols = original_cols - data.shape[1]
    if dropped_cols > 0:
        print(f"Dropped {dropped_cols} columns with all NaN values")

    # check colunm of cnv_df with a large number of nan
    column_nan_counts = data.isna().sum()
    high_nan_columns = column_nan_counts[column_nan_counts > (0.3 * len(data))].index.tolist()
    print(f"Columns with more than 30% NaNs in {data_type} data: {len(high_nan_columns)}")

    data = data.drop(columns=high_nan_columns)
    print(f"{data_type} number of NaN: {data.isna().sum().sum()} after dropping high NaN columns")
    # data imputation missing values.
    data.fillna(0, inplace=True)
    print(f"{data_type} number of NaN: {data.isna().sum().sum()} after filling NaN values with -1")
    
     # Apply normalization if requested
    if normalize:
        print(f"Applying normalization to {data_type} data...")
        # Apply normalization using utils function
        normalize_omics_data(data, data_type)
        print(f"Applied normalization to {data_type} data")
    # sort columns alphabetically except sample_id
    cols = data.columns.tolist()
    cols.remove('sample_id')
    cols = sorted(cols)
    data = data[['sample_id'] + cols]
    print(f"Sorted {data_type} data columns alphabetically (except sample_id)")

    # Print final shape information
    print(f"Final {data_type} data shape: {data.shape[0]} samples x {data.shape[1]} features")
    # check duplicates in sample_id
    duplicate_sample_ids = data['sample_id'][data['sample_id'].duplicated()].unique()
    print(f"Duplicate sample IDs in {data_type} data: {len(duplicate_sample_ids)} duplicates found")
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed data to CSV
    suffix = "_normalized" if normalize else ""
    output_file = os.path.join(output_dir, f"{data_type.lower()}{suffix}.csv")
    #data.to_csv(output_file, index=False)
    #print(f"Saved {data_type} data to: {output_file}")
    print(f"First 5 rows of processed {data_type} data:")
    print(data.head())
    
    return data

def load_conversion_table(file_path):
    """
    Load gene conversion table from CSV file.
    
    Args:
        file_path (str): Path to conversion table CSV file
    """
    conversion_table = pd.read_csv(file_path)
    print(conversion_table)
    return conversion_table

def load_conversion_table_new(file_path):
    """
    Load gene conversion table from CSV file (new version).
    
    Args:
        file_path (str): Path to conversion table CSV file
    """
    conversion_table = pd.read_csv(file_path, sep="\t", compression='zip')
    print(conversion_table)
    # renaming columns
    conversion_table = conversion_table.rename(columns={'Ensembl gene ID': 'ensembl_gene_id',
                                                        'Approved symbol': 'symbol'})
    return conversion_table

def get_dict():
    """
    Create Ensembl to Gene Symbol mapping dictionary from conversion table.
    """
    conversion_table = load_conversion_table_new("hgcn_mart_conversion_table.zip")
    # Create a mapping dictionary from the conversion table
    # conversion_table = load_conversion_table("gene_id_conversion_table_final.csv")
    ensembl_to_symbol_dict = pd.Series(conversion_table.symbol.values,index=conversion_table.ensembl_gene_id).to_dict()
    print(f"Loaded {len(ensembl_to_symbol_dict)} Ensembl to Gene Symbol mappings.")
    return ensembl_to_symbol_dict

def ascat2gistic(file_path, output_file):
    """
    Convert ASCAT CNV data to GISTIC format.
    Creates a table with sample_id as first column and genes as subsequent columns.
    
    Args:
        file_path (str): Path to ASCAT CNV data CSV file
        output_file (str): Path to save GISTIC formatted CSV file
    """
    ascat_df = pd.read_csv(file_path, index_col=0)
    print(f"Original ascat data shape: {ascat_df.shape[0]} samples x {ascat_df.shape[1]} features")
    
    # Convert to int, handling any non-numeric values
    ascat_df = ascat_df.astype(int)

    print("Converting ASCAT values to GISTIC format...")
    # Mapping ASCAT3 -> GISTIC values
    value_mapping = {
        0: -2,  # Homozygous deletion
        1: -1,  # Hemizygous deletion  
        2:  0,  # Neutral/Normal
        3:  1,  # Low-level gain
        4:  2   # High-level amplification
    }
    
    # Apply the mapping with 2 as default for values >=5
    cnv_gistic_df = ascat_df.map(lambda x: value_mapping.get(x, 2) if pd.notnull(x) else x)
    
    # Reset index to make sample_id a column
    cnv_gistic_df = cnv_gistic_df.reset_index()
    
    # Ensure the first column is named 'sample_id'
    if cnv_gistic_df.columns[0] != 'sample_id':
        cnv_gistic_df = cnv_gistic_df.rename(columns={cnv_gistic_df.columns[0]: 'sample_id'})

    # Save to CSV
    cnv_gistic_df.to_csv(output_file, index=False)
    print(f"Saved GISTIC formatted CNV data to: {output_file}")
    print(f"Final shape: {cnv_gistic_df.shape[0]} samples x {cnv_gistic_df.shape[1]} features (including sample_id)")
    
    return cnv_gistic_df

def main():
    """
    Main function to process TCGA BRCA data.
    """
     # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process TCGA pan-cancer data with optional normalization and variance calculation')
    parser.add_argument('--normalization', action='store_true',
                       help='Apply z-score normalization to omics data')
    parser.add_argument('--variance_calc', action='store_true',
                       help='Calculate variance for gene expression data')
    parser.add_argument('--input-dir', default='data/raw/tcga_pancan/processed',
                       help='Directory containing the parquet files produced by prepare_pancan.py')
    parser.add_argument('--output-dir', default='data/omics',
                       help='Output directory for processed files')

    args = parser.parse_args()

    root_path = args.input_dir
    # File paths
    clinical_file = f"{root_path}/clinical.parquet"
    #mirna_file = "data/raw/TCGA-BRCA/processed/TCGA-BRCA.miRNA.csv"
    #cnv_file = "data/raw/TCGA-BRCA/processed/TCGA-BRCA.gene-level_ascat3.csv"
    #fpkm_file = "data/raw/TCGA-BRCA/processed/TCGA-BRCA.star_fpkm.csv"
    #metylation_file = "data/raw/TCGA-BRCA/processed/TCGA-BRCA.methylation27.csv"
    #annotation_file = "data/raw/annotations/gencode.v36.annotation.gtf.gene.probemap"
    
    # PANCANCER FILES
    mirna_file = f"{root_path}/mirna.parquet"
    cnv_file = f"{root_path}/cnv.parquet"
    fpkm_file = f"{root_path}/expression.parquet"

    annotation_file_metylation = "data/raw/annotations/HM27.hg38.manifest.gencode.v36.probeMap"
    ascat_file = "data/omics/cnv.csv"
    output_dir = args.output_dir

    normalize = args.normalization
    calc_variance = args.variance_calc

    print("=" * 50)
    print("TCGA BRCA Data Processing Pipeline")
    print("=" * 50)
    #--------------------------------------------------------
    # Process clinical data
    #--------------------------------------------------------
    print("\n1. Processing clinical data...")
    molecular_subtype_df = process_clinical_data(clinical_file, output_dir, filter_TCGA=None)
    print(f"Processed clinical data shape: {molecular_subtype_df.shape[0]} samples x {molecular_subtype_df.shape[1]} columns")
    #--------------------------------------------------------
    # LOAD CONVERISION TABLE
    # -------------------------------------------------------
    #conversion_table = pd.read_csv(gene_conversion_table)
    # load conversion dictionary
    # ensembl_to_symbol_dict = get_dict()
    # print(f"Loaded {len(ensembl_to_symbol_dict)} Ensembl to Gene Symbol mappings from conversion dictionary.")
    #print ("Loaded conversion table.")
    
    
    # --------------------------------------------------------
    # LOAD ANNOTATION FILE
    # --------------------------------------------------------
    print("\n2. Loading annotation file for gene conversion...")
    #annotation_df = pd.read_csv(annotation_file, sep="\t")
    #ensembl_to_symbol_dict = pd.Series(annotation_df.gene.values,index=annotation_df.id).to_dict()
    #print(f"Loaded {len(ensembl_to_symbol_dict)} Ensembl to Gene Symbol mappings from annotation file.")
    #print("First 5 elements:")
    #print(dict(list(ensembl_to_symbol_dict.items())[:5]))
    
     # LOAD METHYLATION ANNOTATION FILE

    #methylation_annotation_df = pd.read_csv(annotation_file_metylation, sep="\t")
    #met2symbol_dict = pd.Series(methylation_annotation_df.gene.values,index=methylation_annotation_df["#id"]).to_dict()
    #print(f"Loaded {len(met2symbol_dict)} Methylation probe to Gene Symbol mappings from annotation file.")

    #--------------------------------------------------------
    # Process omics data
    #--------------------------------------------------------
    print("\n3. Processing omics data...")
    ensembl_to_symbol_dict  = {}
    cnv_data = process_omics_data(cnv_file, molecular_subtype_df, "CNV", output_dir, ensembl_to_symbol_dict, normalize=normalize)

    mirna_data = process_omics_data(mirna_file, molecular_subtype_df, "miRNA", output_dir, normalize=normalize)

    fpkm_data = process_omics_data(fpkm_file, molecular_subtype_df, "FPKM", output_dir, ensembl_to_symbol_dict, normalize=normalize)


    print("\n" + "=" * 50)
    print("PROCESSING SUMMARY")
    print("=" * 50)
    print(f"Clinical data: {molecular_subtype_df.shape}")
    print(f"miRNA data: {mirna_data.shape}")
    print(f"CNV data: {cnv_data.shape}")
    print(f"FPKM data: {fpkm_data.shape}")

    print(f"\nAll processed files saved to: {output_dir}/")
    print("Data processing completed successfully!")

    # Filter files by common samples
    print("\n4. Filtering files by common samples...")
    filtered_molecular, filtered_cnv, filtered_expression, filtered_mirna = filter_by_common_samples_df(molecular_subtype_df, cnv_data, fpkm_data, mirna_data)
    # filter common columns between cnv and expression data
    print("\n5. Filtering CNV and expression data by common genes...")
    #cnv_data = pd.read_csv(os.path.join(output_dir, f"cnv{'_normalized' if normalize else ''}.csv"))
    #fpkm_data = pd.read_csv(os.path.join(output_dir, f"fpkm{'_normalized' if normalize else ''}.csv")) 
    cnv_data_filtered, fpkm_data_filtered = filter_common_columns(filtered_cnv, filtered_expression)
    print(f"Filtered CNV and expression, kept {cnv_data_filtered.shape[1]} and {fpkm_data_filtered.shape[1]} common genes.")
    # save filtered data
    import os

    cnv_name = f"cnv{'_normalized' if normalize else ''}.csv"
    fpkm_name = f"fpkm{'_normalized' if normalize else ''}.csv"
    clin_name = "clinical.csv"
    mirna_name = f"mirna{'_normalized' if normalize else ''}.csv"

    # 1. CNV
    cnv_data_filtered.to_csv(
        os.path.join(output_dir, cnv_name.replace('.csv', '.zip')), # Il file su disco sarà .zip
        index=False,
        compression={'method': 'zip', 'archive_name': cnv_name}     # Il file dentro sarà .csv
    )

    # 2. FPKM
    fpkm_data_filtered.to_csv(
        os.path.join(output_dir, fpkm_name.replace('.csv', '.zip')),
        index=False,
        compression={'method': 'zip', 'archive_name': fpkm_name}
    )

    # 3. Clinical
    print("Saving filtered clinical data...")
    # print shape
    print(f"Filtered clinical data shape: {filtered_molecular.shape[0]} samples x {filtered_molecular.shape[1]} columns")
    filtered_molecular.to_csv(
        os.path.join(output_dir, clin_name.replace('.csv', '.zip')),
        index=False,
        compression={'method': 'zip', 'archive_name': clin_name}
    )

    # 4. MiRNA
    filtered_mirna.to_csv(
        os.path.join(output_dir, mirna_name.replace('.csv', '.zip')),
        index=False,
        compression={'method': 'zip', 'archive_name': mirna_name}
    )
    #cnv_data_filtered.to_csv(os.path.join(output_dir, f"cnv{'_normalized' if normalize else ''}.csv"), index=False, compression='zip')
    #fpkm_data_filtered.to_csv(os.path.join(output_dir, f"fpkm{'_normalized' if normalize else ''}.csv"), index=False, compression='zip')
    #filtered_molecular.to_csv(os.path.join(output_dir, f"clinical.csv"), index=False, compression='zip')
    #filtered_mirna.to_csv(os.path.join(output_dir, f"mirna{'_normalized' if normalize else ''}.csv"), index=False, compression='zip')
    print(f"Filtered data saved to: {output_dir}/")
    print("generate gist files for cnv data...")
    suffix = "_normalized" if normalize else ""
    #cnv_file_path = os.path.join(output_dir, f"cnv{suffix}.csv")
    #ascat2gistic(cnv_file_path, os.path.join(output_dir, f"cnv_gistic{suffix}.csv"))

    # Calculate variance if requested
    if calc_variance:
        print("\n6. Calculating variance for gene expression data...")
        fpkm_file_processed = os.path.join(output_dir, f"fpkm{suffix}.csv")
        variance_output = os.path.join(output_dir, f"expression_variance{suffix}.csv")
        
        variance_processing(fpkm_file_processed, variance_output)
        print(f"Variance calculation completed. Results saved to: {variance_output}")

    
    #return brca_subtype_df, mirna_data, cnv_data, fpkm_data

if __name__ == "__main__":
    main()



