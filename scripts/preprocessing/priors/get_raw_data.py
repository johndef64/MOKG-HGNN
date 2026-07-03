# TCGA Data Download Script - Specific Datasets
# This script downloads specific datasets:
# 1. miRNA_HiSeq_gene from TCGA Pancancer 
# -https://xenabrowser.net/datapages/?dataset=TCGA.PANCAN.sampleMap%2FmiRNA_HiSeq_gene&host=https%3A%2F%2Ftcga.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443
# 2. Clinical data from Pancancer  
# 
# 3. TCGA-BRCA.star_fpkm.tsv from GDC
# -https://xenabrowser.net/datapages/?dataset=TCGA-BRCA.star_fpkm.tsv&host=https%3A%2F%2Fgdc.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443
# 4. Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes from TCGA BRCA
# -https://xenabrowser.net/datapages/?dataset=TCGA.BRCA.sampleMap%2FGistic2_CopyNumber_Gistic2_all_thresholded.by_genes&host=https%3A%2F%2Ftcga.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443
# 5. (Refactor) TCGA-BRCA.gene-level_ascat2.tsv from GDC
# https://xenabrowser.net/datapages/?dataset=TCGA-BRCA.gene-level_ascat2.tsv&host=https%3A%2F%2Fgdc.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443
# 6. Methylation data from GDC-TCGA-BRCA
# - https://xenabrowser.net/datapages/?dataset=TCGA-BRCA.methylation450.tsv&host=https%3A%2F%2Fgdc.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443

import os
import requests
import pandas as pd
import zipfile
import io

cohorts = [
    #'TCGA-PANCAN',
    'TCGA-BRCA'
    #'TCGA-LUAD'
]
data_types = [
    'mirna',
    'protein',
    'clinical',
    'star_fpkm',
    'gene-level_ascat3',
    #'methylation450',
    'methylation27'
]

annotations = [
    'gencode.v36.annotation.gtf.gene.probemap',
    'HM27.hg38.manifest.gencode.v36.probeMap',
    'HM450.hg38.manifest.gencode.v36.probeMap'
]



def download_files(cohorts, data_types, download_path, save_dir):   
    for cohort in cohorts:
        save_dir_cohort = os.path.join(save_dir, cohort)
        os.makedirs(save_dir_cohort, exist_ok=True)
        for data_type in data_types:
            file_name = f"{cohort}.{data_type}.tsv.gz"
            url = download_path + file_name
            print(f"Downloading: {file_name} from {url}")
            response = requests.get(url)
            if response.status_code == 200:
                with open(os.path.join(save_dir_cohort, file_name), 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded: {file_name}")
            else:
                print(f"Failed to download: {file_name}, Status code: {response.status_code}")

def download_annotation_files(annotations, download_path, save_dir_annotations):
    for annot in annotations:
        url = download_path + annot
        response = requests.get(url)
        if response.status_code == 200:
            with open(os.path.join(save_dir_annotations, annot), 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {annot}")
        else:
            print(f"Failed to download: {annot}, Status code: {response.status_code}")

def process_data_files(cohort, save_dir):
    save_dir_cohort = os.path.join(save_dir, cohort)
    print(f"\nProcessing data files in: {save_dir_cohort}")
    save_dir_processed = 'processed'
    for files in os.listdir(save_dir_cohort):
        print(f"Reading file: {files}")
        df = pd.read_csv(os.path.join(save_dir_cohort, files), sep='\t')
        print(f"Preview of {files}:")
        print(df.shape)
        #print(df.head())
        # transpose matrix, first column sample_ids, and other columns features
        df = df.set_index(df.columns[0]).transpose()
        df = df.reset_index().rename(columns={'index': 'sample_id'})
        #save processed file in cohort processed directory, the processed folder should be inside as raw/cohort/processed
        save_dir_processed_cohort = os.path.join(save_dir_cohort, save_dir_processed)
        print(f"Saving processed file to: {save_dir_processed_cohort}")
        os.makedirs(save_dir_processed_cohort, exist_ok=True)
        # save files as csv but with same name
        files = files.replace('.tsv.gz', '.csv').replace('.tsv', '.csv')
        df.to_csv(os.path.join(save_dir_processed_cohort, files), index=False)   

def download_clinical_data(save_dir):
    print("\n" + "=" * 60)
    print("DOWNLOADING CLINICAL DATA FROM PANCANCER")
    print("=" * 60)
    url = "https://api.gdc.cancer.gov/data/0f31b768-7f67-4fc4-abc3-06ac5bd90bf0"
    clinical_file = os.path.join(save_dir, "clinical_data.csv")
    try:
        print(f"\nDownloading from GDC API: {url}")
        clinical_df = pd.read_csv(url, sep="\t")
        print("\nFirst 5 rows and 5 columns:")
        print(clinical_df.iloc[:5, :5])
        print(f"\nSaving to: {clinical_file}")
        
        # Save clinical data as CSV
        clinical_df.to_csv(clinical_file, index=False)    
        print("Clinical data saved successfully!")
    except Exception as e:
        print(f"Error : {e}")
        clinical_df = None

def get_biogrid_data(save_dir):
    # check if file already exists
    if os.path.exists(os.path.join(save_dir, "GGI.zip")):
        print("BioGRID data already exists, skipping download.")
        return
    biogrid_data_zip = "https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-5.0.250/BIOGRID-ORGANISM-5.0.250.tab3.zip"
    print("Downloading BioGRID data...")
    response = requests.get(biogrid_data_zip)
    if response.status_code == 200:
        # extract homo sapiens BIOGRID-ORGANISM-Homo_sapiens-5.0.250.tab3.txt
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            print("Extracting BioGRID data...")
            z.extract("BIOGRID-ORGANISM-Homo_sapiens-5.0.250.tab3.txt", save_dir)
        # rename extracted file
        os.rename(os.path.join(save_dir, "BIOGRID-ORGANISM-Homo_sapiens-5.0.250.tab3.txt"), os.path.join(save_dir, "homo_sapiens.txt"))
        # zip extracted file
        with zipfile.ZipFile(os.path.join(save_dir, "GGI.zip"), "w") as z:
            z.write(os.path.join(save_dir, "homo_sapiens.txt"), arcname="homo_sapiens.txt")
        os.remove(os.path.join(save_dir, "homo_sapiens.txt"))

def get_data(link, output_dir, file_name):
    # check if exist
    if os.path.exists(os.path.join(output_dir, file_name)):
        print(f"{file_name} already exists, skipping download.")
        return
    print(f"Downloading {file_name}...")
    response = requests.get(link)
    if response.status_code == 200:
        with open(os.path.join(output_dir, file_name), "wb") as f:
            f.write(response.content)
    else:
        print(f"Failed to download {file_name}.")


def main():
    download_path = 'https://gdc-hub.s3.us-east-1.amazonaws.com/download/'
    save_dir = './data/raw/'
    ggi_save_dir = './data/prior_knowledge/GGI'
    mirna_save_dir = './data/prior_knowledge/miRNA'
    tf_save_dir = './data/prior_knowledge/TF'
    os.makedirs(ggi_save_dir, exist_ok=True)
    os.makedirs(mirna_save_dir, exist_ok=True)
    os.makedirs(tf_save_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    save_dir_annotations = os.path.join(save_dir, 'annotations')
    os.makedirs(save_dir_annotations, exist_ok=True)

    print("Downloading and processing data...")
    # download_files(cohorts, data_types, download_path, save_dir)
    # download_annotation_files(annotations, download_path, save_dir_annotations)
    # for cohort in cohorts:
    #     print(f"\nProcessing data files for cohort: {cohort}")
    #     process_data_files(cohort, save_dir)
        
    # download_clinical_data(save_dir)
    # print("\nDownloading prior knowledge data...")
    get_biogrid_data(ggi_save_dir)
    mirdb_data_zip = "https://mirdb.org/download/miRDB_v6.0_prediction_result.txt.gz"
    get_data(mirdb_data_zip, mirna_save_dir, "miRDB.gz")
    tf_link = "https://cdn.netbiol.org/tflink/download_files/TFLink_Homo_sapiens_interactions_SS_simpleFormat_v1.0.tsv"
    get_data(tf_link, tf_save_dir, "TFLink_simple.tsv")

if __name__ == "__main__":
    main()
