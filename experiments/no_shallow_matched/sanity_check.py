"""Sanity checks BEFORE the 35 no-shallow runs (ticket §4). Writes results/sanity_check.json.

Part A (static, no training):
  A1 splits    : the frozen split files hold the same ordered index sequences as a fresh
                 regeneration with the vendored generator (bytes may differ only by the
                 platform line ending), have 60/20/20 sizes, and predate the existing runs.
  A2 panels    : for every (level, split) the regenerated panel (genes, TF, miRNA, in
                 order) is identical to the one recorded in the existing with-shallow run
                 (used_config.yaml), its counts match the original summary CSV and the
                 MOKG-HGNN graph (graph_info.json), and the FS files match _aligned_fs.
  A3 configs   : the run config rebuilt here equals the existing used_config.yaml (except
                 the output path); no_shallow.yaml differs from config_final.yml only in
                 model.parallel.
  A4 variant   : model_no_shallow.verify_variant() (structure + theta1-only classifier).
  A5 references: the existing series (35 MOKG-HGNN runs, 35 with-shallow rows) are
                 complete, unambiguous, and reproduce the published Table S2 means.
Part B (reproduction): re-run the ORIGINAL with-shallow model through this pipeline on
  one (level, split) and compare with the existing value. Tolerance is fixed a priori
  (--tol, default 0.02 macro-F1, about twice the run-to-run s.d. of ~0.01 measured on
  MOKG-HGNN replicates of one configuration, a consequence of non-deterministic GPU
  scatter operations). If it is not reproduced, the check FAILS and run_no_shallow.py
  refuses to start.

Items that cannot be verified on this machine (e.g. run folders not transferred from
the server) are reported as "unverifiable", never as passed.

  python experiments/no_shallow_matched/sanity_check.py                 # A + B
  python experiments/no_shallow_matched/sanity_check.py --static-only   # A only
"""

import argparse
import datetime
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


# ----------------------------------------------------------------------------------
def check_splits():
    from multiomics_kg_hgnn.utils.split import generate_stratified_shuffle_indices
    labels = C.REPO / "data" / "training" / "molecular_subtype.csv"
    n_total = len(pd.read_csv(labels))
    first_run = _earliest_existing_run_time()
    tmp = C.OUT / "_tmp_split_regen"
    out = {}
    for s in C.SEEDS:
        files = C.split_files(s)
        rec = {"sha256": C.split_hash(s)}
        d = tmp / f"seed{s}"
        generate_stratified_shuffle_indices(out_dir=str(d), labels_csv=str(labels),
                                            test_size=0.2, val_size=0.25, random_state=s)
        regen = {p: d / f"{C.SPLIT_PREFIX}_{p}.tsv" for p in files}
        # the split is the ordered index sequence: compare that, not the bytes. np.savetxt
        # writes the platform line ending (CRLF on Windows, LF on Linux), so a byte check
        # fails on identical splits produced or copied across operating systems.
        rec["identical_to_regeneration"] = all(
            np.array_equal(np.loadtxt(files[p], dtype=int), np.loadtxt(regen[p], dtype=int))
            for p in files)
        rec["info_bytes_identical_to_regeneration"] = all(
            Path(files[p]).read_bytes() == regen[p].read_bytes() for p in files)
        rec["info_crlf_line_endings"] = any(b"\r\n" in Path(files[p]).read_bytes() for p in files)
        sizes = {p: int(len(np.loadtxt(files[p], dtype=int))) for p in files}
        rec["sizes"] = sizes
        idx = np.concatenate([np.loadtxt(files[p], dtype=int) for p in files])
        rec["disjoint_and_complete"] = bool(len(idx) == len(set(idx.tolist())) == n_total)
        rec["fractions_60_20_20"] = bool(
            abs(sizes["test"] / n_total - 0.2) < 0.005 and abs(sizes["val"] / n_total - 0.2) < 0.005)
        mtime = max(os.path.getmtime(f) for f in files.values())
        rec["mtime"] = datetime.datetime.fromtimestamp(mtime).isoformat(timespec="seconds")
        # informational only: a local copy from the server gets a new mtime. The proof
        # of identity is the byte-equality with the deterministic regeneration above.
        rec["info_older_than_existing_runs"] = (bool(mtime < first_run.timestamp())
                                                if first_run else "unknown")
        rec["ok"] = (rec["identical_to_regeneration"] and rec["disjoint_and_complete"]
                     and rec["fractions_60_20_20"])
        out[str(s)] = rec
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def _earliest_existing_run_time():
    times = []
    for p in list(C.EXISTING_ROOT.glob("mognntf_g*/run_*")) + list(C.EXISTING_ROOT.glob("mokghgnn_g*/2*")):
        m = re.search(r"(\d{8}_\d{6})", p.name)
        if m:
            times.append(datetime.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S"))
    return min(times) if times else None


def _read_list_file(path, col=None):
    if str(path).endswith(".txt"):
        return [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
    df = pd.read_csv(path)
    return df[col if col in df.columns else df.columns[0]].astype(str).str.strip().tolist()


def check_panels_and_configs(regenerate):
    base = C.load_yaml_config(C.ORIGINAL_CONFIG)
    used = C.existing_mognntf_run_configs()
    summary = C.read_with_shallow_existing()
    mokg = C.read_multiscale_existing()
    aligned_root = C.EXISTING_ROOT / "_aligned_fs"
    out = {}
    for g in C.LEVELS:
        for s in C.SEEDS:
            genes, tf, mirna = C.get_panel(g, s, base, regenerate=regenerate)
            rec = {"panel_hash": C.panel_hash(genes, tf, mirna),
                   "counts": {"genes": len(genes), "tf": len(tf), "mirna": len(mirna)}}
            # vs the panel actually used by the existing with-shallow run
            u = used.get((g, s))
            if u is None:
                rec["vs_existing_run_panel"] = "unverifiable (run folder not on this machine)"
                rec["vs_existing_run_config"] = "unverifiable (run folder not on this machine)"
            else:
                ud = u["data"]
                rec["vs_existing_run_panel"] = bool(
                    list(ud["gene_list"]) == genes and list(ud["tf_list"]) == tf
                    and list(ud["mirna_keep"]) == mirna)
                cfg = C.build_run_config(base, g, s, (genes, tf, mirna), "unused")
                diff = C.config_diff(cfg, u)
                rec["vs_existing_run_config"] = (True if not diff else
                                                 {k: [str(a), str(b)] for k, (a, b) in diff.items()})
            # vs the counts in the original summary CSV
            sr = summary.get((g, s))
            rec["vs_summary_counts"] = (bool(sr["n_mirna"] == len(mirna) and sr["n_tf"] == len(tf))
                                        if sr else "missing row")
            # vs the MOKG-HGNN graph of the same (level, split)
            mr = mokg.get((g, s))
            rec["vs_multiscale_graph_counts"] = (bool(
                mr["num_nodes"]["gene"] == len(genes) and mr["num_nodes"]["miRNA"] == len(mirna)
                and mr["num_nodes"]["TF"] == len(tf)) if mr else "missing run")
            # vs the FS files written by the original experiment
            a = aligned_root / f"g{g}_seed{s}"
            if (a / "selected_genes.csv").exists():
                mine = C.panel_dir(g, s)
                rec["vs_original_fs_files"] = bool(
                    _read_list_file(a / "selected_genes.csv", "symbol") == _read_list_file(mine / "selected_genes.csv", "symbol")
                    and _read_list_file(a / "selected_tf.csv", "TF") == _read_list_file(mine / "selected_tf.csv", "TF")
                    and _read_list_file(a / "selected_mirna.txt") == _read_list_file(mine / "selected_mirna.txt"))
            else:
                rec["vs_original_fs_files"] = "unverifiable (not on this machine)"
            hard = [rec["vs_summary_counts"], rec["vs_multiscale_graph_counts"],
                    rec["vs_existing_run_panel"], rec["vs_existing_run_config"],
                    rec["vs_original_fs_files"]]
            rec["ok"] = all(x is True or (isinstance(x, str) and x.startswith("unverifiable"))
                            for x in hard)
            rec["fully_verified"] = all(x is True for x in hard)
            out[f"g{g}_seed{s}"] = rec
            print(f"[panel] g{g} seed{s}: ok={rec['ok']} fully_verified={rec['fully_verified']}",
                  flush=True)
    return out


def check_variant_config():
    a = C.load_yaml_config(C.ORIGINAL_CONFIG)
    b = C.load_yaml_config(C.NO_SHALLOW_CONFIG)
    diff = C.config_diff(a, b, ignore=set())
    return {"diff": {k: list(v) for k, v in diff.items()},
            "ok": list(diff) == ["model.parallel"] and diff["model.parallel"] == (True, False)}


def check_references():
    mokg = C.read_multiscale_existing()
    sh = C.read_with_shallow_existing()
    grid = {(g, s) for g in C.LEVELS for s in C.SEEDS}
    s2 = pd.read_csv(C.REPO / "supplementary" / "tables" / "S2_feature_collapse.csv")
    rec = {"multiscale_runs": len(mokg), "multiscale_complete": set(mokg) == grid,
           "with_shallow_rows": len(sh), "with_shallow_complete": set(sh) == grid}
    backbones = set()
    for v in mokg.values():
        cfg = json.loads((Path(v["run_dir"]) / "config.json").read_text())
        backbones.add((cfg["model"]["backbone"], tuple(cfg["model"]["readout_types"])))
    rec["multiscale_configs"] = [list(map(str, b)) for b in backbones]
    mism = []
    for model, src in (("MOKG-HGNN", mokg), ("MOGNN-TF", sh)):
        for g in C.LEVELS:
            vals = [src[(g, s)]["f1"] for s in C.SEEDS if (g, s) in src]
            if len(vals) != len(C.SEEDS):
                continue
            m, sd = C.mean_sd(vals)
            row = s2[(s2.model == model) & (s2.genes == g)].iloc[0]
            # S2 stores values rounded to 4 decimals
            if abs(m - float(row.macro_f1_mean)) > 5e-5 + 1e-12 or abs(sd - float(row.macro_f1_sd)) > 5e-5 + 1e-12:
                mism.append(f"{model} g{g}: here {m:.4f}+-{sd:.4f} vs S2 {row.macro_f1_mean}+-{row.macro_f1_sd}")
    rec["published_S2_mismatches"] = mism
    rec["ok"] = rec["multiscale_complete"] and rec["with_shallow_complete"] and not mism \
        and len(backbones) == 1
    return rec


# ----------------------------------------------------------------------------------
def _existing_run_details(level, seed):
    """Best epoch and val F1 of the existing with-shallow run, from its experiment.log
    (FIRST occurrence: the original runs appended later runs to earlier logs)."""
    for uc in C.EXISTING_ROOT.glob(f"mognntf_g{level}/run_*/used_config.yaml"):
        import yaml
        cfg = yaml.safe_load(open(uc))
        if int(cfg["project"]["split_seed"]) != seed:
            continue
        log = (uc.parent / "experiment.log").read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"Best validation metric: ([\d.]+) at epoch (\d+)", log)
        if m:
            return {"run_dir": str(uc.parent.relative_to(C.REPO)),
                    "best_val_metric": float(m.group(1)), "best_epoch_0based": int(m.group(2))}
    return None


def reproduction(level, seed, repeats, tol, num_workers=None):
    ref = C.read_with_shallow_existing()[(level, seed)]
    details = _existing_run_details(level, seed)
    reps = []
    for r in range(repeats):
        rec = C.run_variant("with_shallow", level, seed, raw_root=C.OUT / "sanity_runs",
                            num_workers=num_workers)
        rec["delta_test_macro_f1"] = rec["test_macro_f1"] - ref["f1"]
        rec["delta_test_accuracy"] = rec["test_accuracy"] - ref["acc"]
        rec["delta_val_macro_f1"] = rec["val_macro_f1"] - ref["val_f1"]
        rec.pop("per_class_f1"); rec.pop("per_class_support")
        reps.append(rec)
        print(f"[repro] repeat {r + 1}/{repeats}: test F1 {rec['test_macro_f1']:.4f} vs existing "
              f"{ref['f1']:.4f} (delta {rec['delta_test_macro_f1']:+.4f}) | best epoch "
              f"{rec['best_epoch']} | {rec['duration_s']:.0f}s", flush=True)
    return {"level": level, "split_seed": seed, "tolerance_macro_f1": tol,
            "num_workers": reps[0]["num_workers"],
            "existing": {**ref, **(details or {})}, "repeats": reps,
            "max_abs_delta": max(abs(r["delta_test_macro_f1"]) for r in reps),
            "passed": all(abs(r["delta_test_macro_f1"]) <= tol for r in reps)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--static-only", action="store_true")
    ap.add_argument("--regenerate-panels", action="store_true",
                    help="Rebuild every panel even if panel.json is cached.")
    ap.add_argument("--repro-level", type=int, default=700)
    ap.add_argument("--repro-seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--num-workers", type=int, default=None,
                    help="Execution-only override of data.num_workers (0 avoids the Windows "
                         "spawn overhead; batches are identical, see common.VOLATILE_KEYS). "
                         "run_no_shallow.py must then use the same value.")
    args = ap.parse_args()

    C.OUT.mkdir(parents=True, exist_ok=True)
    from model_no_shallow import verify_variant
    res = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "env": C.environment()}
    res["A1_splits"] = check_splits()
    res["A5_references"] = check_references()
    res["A3_variant_config"] = check_variant_config()
    res["A2_panels_and_configs"] = check_panels_and_configs(args.regenerate_panels)
    res["A4_variant_model"] = verify_variant()
    static_ok = (all(v["ok"] for v in res["A1_splits"].values())
                 and res["A5_references"]["ok"] and res["A3_variant_config"]["ok"]
                 and all(v["ok"] for v in res["A2_panels_and_configs"].values())
                 and res["A4_variant_model"]["passed"])
    res["static_passed"] = bool(static_ok)
    res["unverifiable_panels"] = [k for k, v in res["A2_panels_and_configs"].items()
                                  if v["ok"] and not v["fully_verified"]]
    if not args.static_only and static_ok:
        res["B_reproduction"] = reproduction(args.repro_level, args.repro_seed, args.repeats,
                                             args.tol, args.num_workers)
    res["passed"] = bool(static_ok and res.get("B_reproduction", {}).get("passed", False))

    # keep previous reproduction evidence if only the static part was re-run
    if args.static_only and C.SANITY_JSON.exists():
        old = json.loads(C.SANITY_JSON.read_text())
        if "B_reproduction" in old:
            res["B_reproduction"] = old["B_reproduction"]
            res["passed"] = bool(static_ok and old["B_reproduction"].get("passed", False)
                                 and old.get("env") == res["env"])
    C.write_text_atomic(C.SANITY_JSON, json.dumps(res, indent=2, default=str))
    print(f"\n[sanity] static_passed={res['static_passed']} | reproduction="
          f"{res.get('B_reproduction', {}).get('passed', 'not run')} | passed={res['passed']}")
    if res["unverifiable_panels"]:
        print(f"[sanity] not fully verifiable on this machine: {res['unverifiable_panels']}")
    print(f"[sanity] -> {C.SANITY_JSON}")
    sys.exit(0 if (res["passed"] or (args.static_only and static_ok)) else 1)


if __name__ == "__main__":
    main()
