"""Download the PKT property-graph knowledge graph (nodes + edges) into
data/prior_knowledge/PKT/.

PKT is the OWL-NETS/PheKnowLator property graph that supplies the SUPERIOR scales
(pathway / GO / disease) of the heterogeneous graph. It is hosted on Hugging Face
(KG-TransomicNet dataset) and is NOT bundled in the repo (data/ is gitignored, and
the files are ~290MB total). build_hetero_graph.py and the explainability name
extraction both read these two zips.

    conda run -n gnn python scripts/download_pkt.py
    conda run -n gnn python scripts/download_pkt.py --out data/prior_knowledge/PKT
"""

import argparse
import os

import requests

HF_BASE = "https://huggingface.co/datasets/johndef64/KG-TransomicNet/resolve/main/PKT"
FILES = {
    "nodes.zip": f"{HF_BASE}/nodes.zip?download=true",
    "edges.zip": f"{HF_BASE}/edges.zip?download=true",
}


def download(out_dir, force=False):
    os.makedirs(out_dir, exist_ok=True)
    for name, url in FILES.items():
        dst = os.path.join(out_dir, name)
        if os.path.exists(dst) and not force:
            print(f"[skip] exists: {dst} ({os.path.getsize(dst) / 1e6:.0f} MB)")
            continue
        print(f"[download] {name} <- {url}")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            with open(dst, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {done / 1e6:6.0f} / {total / 1e6:.0f} MB", end="", flush=True)
            print(f"\r[saved] {dst} ({os.path.getsize(dst) / 1e6:.0f} MB)          ")


def main():
    ap = argparse.ArgumentParser(description="Download the PKT knowledge graph from Hugging Face.")
    ap.add_argument("--out", default="data/prior_knowledge/PKT", help="Output dir for nodes.zip / edges.zip")
    ap.add_argument("--force", action="store_true", help="Re-download even if the files exist.")
    args = ap.parse_args()
    download(args.out, force=args.force)
    print("\n==> PKT ready. Next: bash make_graph.sh (builds the hetero template).")


if __name__ == "__main__":
    main()
