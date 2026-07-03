"""Scarica i file TCGA pancan richiesti nella cartella raw.

Esempio:
  python download_pancan.py: Scarica tutti i file nella cartella data/pancan_tcga/raw default
  python download_pancan.py --out data/pancan_tcga/raw --keys expression,mirna
"""
import os
import argparse
import gdown

base = "https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/"
base_cnv = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/"

files = {
    "clinical": "TCGA_PanCan33_iCluster_k28_tumor.gz",
    "expression": "EB%2B%2BAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz",
    "mirna": "pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.xena.gz",
    #"metylation": "jhu-usc.edu_PANCAN_HumanMethylation450.betaValue_whitelisted.tsv.synapse_download_5096262.xena.gz",
    "cnv": "TCGA.PANCAN.sampleMap%2FGistic2_CopyNumber_Gistic2_all_data_by_genes.gz"
}


def download(out_dir: str, keys=None):
    os.makedirs(out_dir, exist_ok=True)
    print("Starting downloads using gdown...")
    for key, file_name in files.items():
        if keys and key not in keys:
            continue
        output_name = key + ".tsv.gz" if file_name.endswith('.gz') else key + ".tsv"
        output_path = os.path.join(out_dir, output_name)

        if key == 'cnv':
            url = base_cnv + file_name
        else:
            url = base + file_name

        if os.path.exists(output_path):
            print(f"Skip download, exists: {output_path}")
            continue

        print(f"Downloading {file_name} -> {output_path} ...")
        try:
            gdown.download(url, output_path, quiet=True)
            print(f"Downloaded {output_name}")
        except Exception as e:
            print(f"Error downloading {file_name}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/pancan_tcga/raw', help='Output directory for raw downloads')
    parser.add_argument('--keys', default=None, help='Comma-separated list of keys to download (clinical,expression,mirna,cnv)')
    print(f"Working directory: {os.getcwd()}")
    args = parser.parse_args()
    keys = args.keys.split(',') if args.keys else None
    download(args.out, keys)


if __name__ == '__main__':
    main()
