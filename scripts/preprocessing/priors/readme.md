
# TCGA BRCA Multi-Omics Dataset Processing Pipeline

## Overview
This repository implements a comprehensive data preprocessing pipeline for The Cancer Genome Atlas Breast Invasive Carcinoma (TCGA-BRCA) multi-omics datasets. The pipeline integrates gene expression, copy number variation, microRNA expression, and clinical phenotype data to generate analysis-ready datasets for cancer subtype classification research.

## Methodology

### Step 1: Raw Data Acquisition
```bash
python get_raw_data.py
```
Retrieves TCGA-BRCA datasets from the Xena Browser repository using the xenaPython API. The module implements automated downloading protocols for four distinct molecular profiling modalities as referenced in the [Data Sources](#data-sources) section.
#### Output
```text
raw/
├─ annotations/
│  ├─ gencode.v36.annotation.gtf.gene.probemap
│  ├─ HM27.hg38.manifest.gencode.v36.probeMap
│  └─ HM450.hg38.manifest.gencode.v36.probeMap
├─ TCGA-BRCA/
│  └─ processed/
│     ├─ TCGA-BRCA.clinical.csv
│     ├─ TCGA-BRCA.gene-level_ascat3.csv
│     ├─ TCGA-BRCA.methylation27.csv
│     ├─ TCGA-BRCA.mirna.csv
│     ├─ TCGA-BRCA.protein.csv
│     ├─ TCGA-BRCA.star_fpkm.csv
│     ├─ TCGA-BRCA.clinical.tsv.gz
│     ├─ TCGA-BRCA.gene-level_ascat3.tsv.gz
│     ├─ TCGA-BRCA.methylation27.tsv.gz
│     ├─ TCGA-BRCA.mirna.tsv.gz
│     ├─ TCGA-BRCA.protein.tsv.gz
│     └─ TCGA-BRCA.star_fpkm.tsv.gz
├─ clinical_data.csv
└─ TCGA-BRCA.all_file_raw
```

### Step 2: Gene Identifier Mapping Construction
```bash
hgnc_mart_conversion_table.zip, conversion_table_ensembl_gencode_hgnc.tsv
```
This step generates mapping dictionaries for gene identifier conversion. In the proeprocessing steps there are two main tables, hgnc_mart_conversion_table (all genes annotated with Ensembl and Entrez IDs) and conversion_table_ensembl_gencode_hgnc (all probes from annotation file with gencode and official symbol). To ensure compatibility with BioGRID and other downstream analyses, genes are filtered to retain only unique entries based on Entrez IDs, resolving data consistency problems.

### Step 3: Multi-Omics Data Integration and Preprocessing
```bash
python process_data.py --normalization --variance_calc
```
Implements systematic data preprocessing protocols including sample identifier standardization, feature filtering, and data harmonization.

**Processing Workflow:**
- Sample identifier normalization (`pan.samplesID` → `sample_id`, `sampleID` → `sample_id`) with the format TCGA-YYYY-XXX-ZZZ
- BRCA subtype patient cohort selection
- Ensembl ID to gene symbol conversion for improved interpretability
- PAR_Y chromosomal region feature removal
- Missing value
- (Optional) Normalization "--normalization"
- Filtering datasets keeping common samples across the omics
- ASCAT3 -> GISTIC2 conversion for interoperability
- (Optional) variance calculation "--variance_calc"
#### Detailed operations
- Clinical data: filtering to exclude normal-like samples; standardization of TCGA IDs to the TCGA-XXXX-YYY-ZZZ format.
- CNV: ASCAT3 format; filtering samples to keep only those present in the clinical data; ID standardization; NaN cleaning (removing columns with all NaNs and columns with >30% NaNs); ENSEMBL → HGNC symbol conversion.
- Expression data: FPKM format; filtering samples to keep only those present in the clinical data; ID standardization; NaN cleaning (removing columns with all NaNs and columns with >30% NaNs); ENSEMBL → HGNC symbol conversion.
- miRNA data: filtering samples to keep only those present in the clinical data; ID standardization; NaN cleaning (removing columns with all NaNs and columns with >30% NaNs).
- Methylation (to be implemented): probe conversion using the annotation file (probes → GENCODE symbol → HGNC symbol); NaN cleaning (removing columns with all NaNs and columns with >30% NaNs).
- (Optional) Conversion of CNV data from ASCAT3 to GISTIC2.
- (Optional) Data normalization:
  - CNV → Z-score
  - Expression → Min–Max
  - miRNA → Min–Max
  - Methylation → Min–Max

**Output:**
- `omics/clinical.csv` - standardized clinical annotations
- `omics/fpkm.csv` - processed gene expression matrix FPKM
- `omics/cnv.csv` - processed copy number data ASCAT3
- `omics/mirna.csv` - processed microRNA expression data
- `omics/cnv_gistic.csv` - processed copy number data GISTIC2
**Optional**
- `omics/fpkm_normalized.csv` - processed gene expression matrix
- `omics/cnv_normalized.csv` - processed copy number data
- `omics/mirna_normalized.csv` - processed microRNA expression data
- `omics/expression_variance_normalized.csv` - sorted variance per single gene of expression data
- `omics/cnv_gistic_normalized.csv` - processed copy number data


### Step 4: Statistical Feature Selection (Optional)
```bash
python boruta_feature_selection.py
```
Applies the Boruta wrapper feature selection algorithm to identify statistically significant molecular features for cancer subtype discrimination. This approach utilizes Random Forest importance rankings compared against shadow features to determine relevance.
After the process a z-score is applied.

Another optional method is variance analysis, which selects the top N features with the highest variance across samples. Low-variance features contribute little to discrimination, as they show minimal change between conditions. By ranking features by variance and retaining only the most variable ones, the dataset becomes more informative and less noisy, improving downstream learning efficiency. This approach is model-agnostic, fast to compute, and serves as an effective preliminary filtering step before more advanced selection methods.

**Output:**
- `datasets/labels.csv` - Labels
- `datasets/fpkm.csv` - processed gene expression matrix
- `datasets/cnv.csv` - processed copy number data
- `datasets/mirna.csv` - processed microRNA expression data

**Methodological Approach:**
- Random Forest-based feature importance assessment
- Statistical significance testing against permuted shadow features
- Dimensionality reduction while preserving predictive information content
## Dependencies and Requirements

The pipeline requires the following Python packages for execution:

```bash
pip install requests pandas numpy glob sklearn boruta
```

### Computational Requirements
- Python 3.7+
- Internet connectivity for API-based gene annotation services
- Approximately 6GB storage space for intermediate and final datasets

## Data Preprocessing Specifications

### Sample Identifier Standardization
The pipeline implements systematic normalization of TCGA barcode identifiers to ensure cross-platform compatibility. Primary solid tumor samples (sample type code: 01) with vial identifier 'A' are prioritized for analysis consistency.

### Gene Annotation Harmonization  
Ensembl gene identifiers are systematically converted to Entrez annotation gene symbols (used in NCBI) using the [HGNC Mart platform](https://biomart.genenames.org/martform/#!/default/HGNC?datasets=hgnc_gene_mart_2025_11_04&attributes=hgnc_gene__hgnc_gene_id_1010%2Chgnc_gene__status_1010%2Chgnc_gene__approved_symbol_1010%2Chgnc_gene__approved_name_1010%2Chgnc_gene__ensembl_gene__ensembl_gene_id_104%2Chgnc_gene__ncbi_gene__gene_id_1026%2Chgnc_gene__locus_type_1010&bool_list%5B%5D=hgnc_gene__has_ncbi_gene_1010&bool_list%5B%5D=only&bool_list%5B%5D=Filter+by+genes...+with+NCBI+gene+ID), facilitating biological interpretation and cross-study comparability. This conversion enables researchers to map genome-wide data generated from various platforms (such as RNA-Seq, microarrays, or CRISPR screens) to standardized and widely recognized gene nomenclature. The use of HGNC (HUGO Gene Nomenclature Committee) ensures that gene names and symbols are consistent, up-to-date, and unambiguous. By aligning Ensembl IDs to Entrez and HGNC symbols, it becomes easier to integrate datasets from different sources.


### Nan values  
For the CNV datas columns with nan values > 30% has been dropped, and remains 2340 null values. To ensure a consistent matrix structure across all types of genetic variations, genes not affected by a specific variation type are assigned a placeholder value of -1. This padding maintains dimensional integrity while distinguishing true absence of data from wild-type (0) or mutated (1) states.

# Prior Knowledge Construction Pipeline

## Step 1: Prior Knowledge Data Download
```bash
python get_raw_data.py
```
Automates the download of biological prior knowledge databases required for molecular network construction.

**Downloaded Databases:**
- **BioGRID**: Human protein-protein interaction database
- **miRDB**: miRNA-gene targeting prediction database

#### Output Structure
```text
data/
├─ prior_knowledge/
│  ├─ GGI/
│  │  └─ GGI.zip                    # BioGRID interactions (compressed)
│  └─ miRNA/
│     └─ miRDB.gz                   # miRNA-target predictions
```

## Step 2: Gene Identifier Conversion
```bash
python refseq2gene.py
```
Converts RefSeq identifiers from the miRDB database to standardized gene symbols using the MyGene API.

**Processing Steps:**
1. Filters miRDB data for human miRNAs (prefix "hsa-")
2. Extracts unique RefSeq IDs (format "NM_XXXXXX")
3. Uses MyGene to map RefSeq → gene symbol + Entrez ID + Ensembl ID
4. Handles unfound IDs (86 unmappable RefSeq IDs)

#### Output
- `refseq2gene_mappings.tsv` - RefSeq-to-gene conversion table

## Step 3: Biological Network Construction
```bash
python load_interaction.py
```
Processes and integrates prior knowledge databases to construct multi-layer biological networks.

### Gene-Gene Interaction Processing

**Preprocessing Steps:**
1. **Data Loading**: Loads BioGRID interactions from compressed ZIP file
2. **Quality Control**: 
   - Removes interactions with invalid Entrez IDs ("-")
   - Converts Entrez IDs to Int64 numeric format
3. **Gene Mapping**: 
   - Integrates with HGNC conversion table to obtain standardized symbols
   - Filters to keep only genes present in conversion table
4. **Network Preprocessing**:
   - Removes self-loops (genes interacting with themselves)
   - Creates undirected graph by normalizing pairs (min, max)
   - Eliminates duplicate interactions
5. **Adjacency Matrix Generation**:
   - Creates gene→numeric index mapping
   - Constructs sparse adjacency matrix (COO format)
   - Adds self-loops for GNN compatibility
   - Saves in `.npz` format for efficiency

#### Gene-Gene Network Output
```text
data/prior_knowledge/GGI/
├─ GGI_interactions_processed.csv      # Clean interactions with mapping
├─ GGI_edges_undirected.csv           # Undirected edge list
├─ GGI_nodes_list.csv                 # Sorted unique gene list
└─ GGI_adjacency_sparse.npz           # Sparse adjacency matrix
```

### miRNA-Gene Interaction Processing

**Preprocessing Steps:**
1. **Data Integration**:
   - Merges miRDB database with RefSeq→gene mappings
   - Filters by confidence score ≥ 80
   - Removes duplicates and invalid interactions
2. **Gene Symbol Standardization**:
   - Applies conversion to approved HGNC symbols
   - Filters for genes present in conversion table
3. **Network Validation**:
   - Verifies dimensional consistency between datasets
   - Eliminates redundant interactions

#### miRNA-Gene Network Output
```text
data/prior_knowledge/miRNA/
├─ miRNA_gene_interactions.csv         # Validated miRNA→gene interactions
├─ miRNA_nodes_list.csv               # Unique miRNA list
├─ metapath_mirna_mirna.csv           # Indirect miRNA-miRNA connections
└─ miRNA_adjacency_sparse.npz         # miRNA-miRNA adjacency matrix
```

### miRNA-miRNA Metapath Computation

**Algorithm:**
1. **Gene-Based Grouping**: Groups miRNAs by shared target genes
2. **Metapath Generation**: 
   - For each gene, identifies all miRNAs that regulate it
   - Creates connections between all miRNA pairs sharing ≥1 target
3. **Graph Processing**:
   - Removes self-loops (miRNA→same miRNA)
   - Eliminates exact and directional duplicates
   - Constructs sparse miRNA-miRNA adjacency matrix

**Biological Rationale:**
miRNAs that regulate the same genes tend to have correlated biological functions and may be co-regulated or part of the same molecular pathways.


## Step 4: Heterogeneous Multi-scale Backbone (Proposta B)
```bash
conda run -n gnn python build_hetero_graph.py \
    --gene-list data/training/gene_nodes_filtered_for_tf.csv \
    --go-min-support 3
```
Fuses the single-scale molecular layer above (gene↔gene, miRNA→gene, TF→gene,
HGNC-symbol-keyed) with the superior scales pulled from the **PKT knowledge
graph** (PheKnowLator / OWL-NETS, Entrez-keyed) into a PyG `HeteroData` template.
The gene is the join anchor: symbols are mapped to Entrez via the HGNC conversion
table so the `gene` node vocabulary stays 1:1 with the MOGNN-TF omics matrices.
See `docs/piano-mapping-PKT-heterodata.md` for the measured KG schema and routing.

**Node types:** `gene, miRNA, TF, pathway (Reactome R-HSA), GO_term, disease (MONDO)`.
**Relations:** `interacts` (BioGRID), `targets` (miRDB), `regulates` (TFLink),
`member_of` (gene→pathway), `annotated_with` (gene→GO, via the gene→protein→GO
bridge), `is_a` (GO→GO hierarchy), `associated_with` (gene→disease).
`ToUndirected()` adds the typed reverse relations for message passing.

The template is **topology only** (no per-patient features — those are injected
downstream, one `HeteroData` per patient). It feeds `to_hetero`, `HeteroConv`,
`RGCNConv`, `HGTConv` and `HANConv` (see `docs/task_e_reti_da_usare.md`).

Useful flags for the hierarchical-depth ablation (proposta sez. 6):
`--no-disease`, `--go-min-support N` (anti GO blow-up), `--directed`
(keep GO hierarchy flowing only upward), `--gene-list PATH`.

**Output (`data/prior_knowledge/hetero/`):**
- `hetero_graph_template.pt` — the PyG `HeteroData` backbone
- `node_gene.csv`, `node_miRNA.csv`, `node_TF.csv`, `node_pathway.csv`,
  `node_GO_term.csv`, `node_disease.csv` — id↔index vocabularies per node type

Validate with:
```bash
conda run -n gnn python check_hetero_graph.py   # forward pass through each net
```

# Data Sources
# Raw data
All data are available on [GDC Hub on Xena brwoser](https://xenabrowser.net/datapages/?cohort=GDC%20TCGA%20Breast%20Cancer%20(BRCA)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443)

## CDG BioPortal
https://bioportal.bioontology.org/ontologies/CDG

## Xena Browser:
https://xenabrowser.net/datapages/
- https://ucsc-xena.gitbook.io/project/public-data-we-host/tcga

## Multimodal-GNN-for-Cancer-Subtype-Clasification dataset:
https://uconn-my.sharepoint.com/personal/bingjun_li_uconn_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fbingjun%5Fli%5Fuconn%5Fedu%2FDocuments%2FPublic%20Data%20Host%2FMultimodal%2DGNN%2Dfor%2DCancer%2DSubtype%2DClasification&ga=1

## Prior knowledge
![testo alternativo](../benchmark/images/graph_mo.png)
### BioGRID
Gene-Gene interaction network:
https://downloads.thebiogrid.org/File/BioGRID/Release-Archive/BIOGRID-5.0.250/BIOGRID-ORGANISM-5.0.250.tab3.zip

### miRDB
miRNA-Target interaction network:
https://mirdb.org/download.html