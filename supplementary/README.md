# Supplementary material

Supplementary material accompanying the MOKG-HGNN manuscript (under review), released
here so that the tables it reports are available in machine-readable form.

## Contents

| File | Description |
|---|---|
| `supplementary.pdf` | Compiled supplementary document (Tables S1–S3, Fig. S1) |
| `supplementary.tex` | LaTeX source |
| `images/semantic_graph.png` | Fig. S1 |
| `tables/` | Machine-readable version of each table |

## Tables

| File | Corresponds to | Content |
|---|---|---|
| `tables/S1_per_class.csv` | Table S1 | Per-class test performance across seeds, side by side with the molecular baseline |
| `tables/S2_feature_collapse.csv` | Table S2 | Full feature-collapse grid, both models at every gene level |
| `tables/S2_feature_collapse_robustness.csv` | Table S2 | Degradation summary derived from the grid |
| `tables/S3_top_pathway_by_subtype.csv` | Table S3 | Top pathway mechanisms per subtype, with Reactome identifiers, names and distinctive scores |
| `tables/S3_top_GO_term_by_subtype.csv` | Table S3 | Top Gene Ontology mechanisms per subtype |
| `tables/graph_template_manifest.csv` | Fig. S1 | Node and edge inventory of the graph template |

The CSV files carry the full lists; the compiled tables show a truncated view for width.

## Subtype labels

Subtypes are the integrated iCluster classes. The label space runs C1–C28 with **C24
(LAML) absent**, giving 27 classes with a gap at C24. Column names use the real iCluster
values, not renumbered indices.

## Rebuilding

Compile the document with:

```bash
pdflatex supplementary.tex
```

The tables are produced by the aggregation scripts in the main repository; see the
Experiments section of the top-level [README](../README.md) for the commands that
regenerate each one.
