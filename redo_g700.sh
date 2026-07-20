#!/usr/bin/env bash
# TEMP: re-run ONLY the g700 level of the MOGNN-TF collapse, feature-ALIGNED.
# The old g700 runs (17 Jul) used 100 miRNA / 200 TF instead of the survived
# 63/98, so they mix protocols. This deletes them, drops their summary rows,
# reruns g700 aligned (--resume skips 500..20), and re-aggregates.
set -euo pipefail

ROOT="results/feature_collapse"
SUMMARY="$ROOT/collapse_mognntf_summary.csv"

echo "1) delete old g700 runs"
rm -rf "$ROOT/mognntf_g700"
rm -f "$ROOT/collapse_mognntf_summary (1).csv"   # stray duplicate

echo "2) drop g700 rows from the summary (keep header + other levels)"
grep -v '^mognn-tf,700,' "$SUMMARY" > "$SUMMARY.tmp" && mv "$SUMMARY.tmp" "$SUMMARY"

echo "3) rerun ONLY g700, aligned (resume skips the levels already done)"
conda run --no-capture-output -n gnn python -u scripts/collapse_mognntf.py \
    --gene-grid 700 --seeds 42 43 44 45 46 \
    --out-root "$ROOT" --resume

echo "4) re-aggregate both curves"
conda run -n gnn python scripts/kg_hgnn/collapse_aggregate.py --results "$ROOT"

echo
echo "DONE. Check that g700 now shows mirna=63 tf=98 in $SUMMARY"
grep '^mognn-tf,700,' "$SUMMARY" || true
