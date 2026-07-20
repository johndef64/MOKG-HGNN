"""Extract id -> name tables for pathways (R-HSA) and GO terms from the PKT KG.

The heterogeneous graph carries only ids (R-HSA-.., GO_..); the human-readable
names live in the source KG (data/prior_knowledge/PKT/nodes.zip, `label` field).
That JSON is ~62MB and slow to parse, so this extracts the names ONCE into small
CSVs that explain.py / annotate_explanations.py then read directly.

These outputs are DERIVED and gitignored (data/ is in .gitignore) — regenerate
them with this script after cloning or on the server:

    conda run -n gnn python scripts/kg_hgnn/extract_pkt_names.py

Outputs (id, label, description):
    data/prior_knowledge/PKT/pathway_names.csv   (class_code R-HSA)
    data/prior_knowledge/PKT/go_names.csv        (class_code GO)
"""

import csv
import io
import json
import os
import zipfile

PKT_DIR = "data/prior_knowledge/PKT"
NODES_ZIP = os.path.join(PKT_DIR, "nodes.zip")
# class_code in the KG -> output filename
SCALES = {"R-HSA": "pathway_names.csv", "GO": "go_names.csv"}


def main():
    if not os.path.isfile(NODES_ZIP):
        raise SystemExit(f"source KG not found: {NODES_ZIP} (needed to regenerate names)")

    with zipfile.ZipFile(NODES_ZIP) as z:
        name = z.namelist()[0]
        nodes = json.load(io.TextIOWrapper(z.open(name), encoding="utf-8", errors="replace"))

    buckets = {cc: [] for cc in SCALES}
    for n in nodes:
        cc = n.get("class_code", "")
        if cc in buckets:
            buckets[cc].append({
                "id": n.get("_key", ""),
                "label": (n.get("label") or "").strip(),
                "description": (n.get("description") or "").strip().replace("\n", " "),
            })

    for cc, fname in SCALES.items():
        out = os.path.join(PKT_DIR, fname)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["id", "label", "description"])
            w.writeheader(); w.writerows(buckets[cc])
        print(f"[saved] {out}  ({len(buckets[cc])} rows)")


if __name__ == "__main__":
    main()
