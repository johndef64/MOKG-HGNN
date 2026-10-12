"""Run the no-shallow MOGNN-TF over the paired feature-scarcity grid (ticket §4).

7 levels x 5 splits = 35 runs. Each run reuses, by import, the original data loading,
panel construction (scripts/collapse_mognntf.py::_panel) and MOGNN-TF training /
evaluation loop (ExperimentRunner); the only change is configs/mognntf_no_shallow.yaml
(model.parallel: false). No re-tuning.

Writes one JSON per run into results/runs/no_shallow_g<L>_seed<S>.json (level, split,
macro-F1, accuracy, per-class F1, selected epoch, config / panel / split hashes,
duration, environment); MOGNN-TF's own run folders go to results/runs_raw/.

Guard: refuses to start unless results/sanity_check.json says passed=true AND was
produced with the same torch version and num_workers. Re-run sanity_check.py on every new
machine. A different GPU model from the sanity one (e.g. on a cluster with several GPU
types) is allowed but printed; each run JSON records its GPU and the report lists them.

  python scripts/no_shallow/run_no_shallow.py                # all 35, resumable
  python scripts/no_shallow/run_no_shallow.py --levels 700 --seeds 42
  python scripts/no_shallow/run_no_shallow.py --smoke        # 3 epochs, separate folder
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def _check_gate(num_workers):
    if not C.SANITY_JSON.exists():
        sys.exit(f"[gate] {C.SANITY_JSON} missing: run sanity_check.py first.")
    s = json.loads(C.SANITY_JSON.read_text())
    if not s.get("passed"):
        sys.exit("[gate] sanity check NOT passed: fix the reported problem before running "
                 "(ticket §4: stop and report if the original value is not reproduced).")
    env_here, env_there = C.environment(), s.get("env", {})
    if env_here["torch"] != env_there.get("torch"):
        sys.exit(f"[gate] sanity check was run with torch {env_there.get('torch')}, this "
                 f"environment has {env_here['torch']}: re-run sanity_check.py here first.")
    if env_here["cuda_device"] != env_there.get("cuda_device"):
        print(f"[gate] NOTE: sanity check ran on '{env_there.get('cuda_device')}', this run is "
              f"on '{env_here['cuda_device']}' (recorded in the run JSON, listed in the report).",
              flush=True)
    nw_sanity = s["B_reproduction"].get("num_workers")
    base_nw = int(C.load_yaml_config(C.NO_SHALLOW_CONFIG)["data"]["num_workers"])
    nw_here = base_nw if num_workers is None else num_workers
    if nw_sanity != nw_here:
        sys.exit(f"[gate] reproduction used num_workers={nw_sanity}, this call would use "
                 f"{nw_here}: pass --num-workers {nw_sanity} (or redo the sanity check).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=C.LEVELS)
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    ap.add_argument("--smoke", action="store_true",
                    help="3 epochs into results/smoke/ (pipeline test only, never compared).")
    ap.add_argument("--no-resume", action="store_true", help="Redo runs whose JSON exists.")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="Execution-only override of data.num_workers; must match the value "
                         "used in the sanity reproduction.")
    args = ap.parse_args()

    smoke_epochs = 3 if args.smoke else None
    runs_dir = C.OUT / "smoke" / "runs" if args.smoke else C.RUNS
    raw_dir = C.OUT / "smoke" / "runs_raw" if args.smoke else C.RUNS_RAW
    if not args.smoke:
        _check_gate(args.num_workers)
    runs_dir.mkdir(parents=True, exist_ok=True)

    todo = [(g, s) for g in args.levels for s in args.seeds]
    print(f"[no-shallow] {len(todo)} run(s) | smoke={args.smoke} | -> {runs_dir}", flush=True)
    t_all = time.perf_counter()
    for i, (g, s) in enumerate(todo, 1):
        out = runs_dir / f"no_shallow_g{g}_seed{s}.json"
        if out.exists() and not args.no_resume:
            print(f"[no-shallow] {i}/{len(todo)} g{g} seed{s}: done already, skip", flush=True)
            continue
        print(f"\n[no-shallow] {i}/{len(todo)} === level {g} | split seed {s} ===", flush=True)
        rec = C.run_variant("no_shallow", g, s, raw_root=raw_dir, smoke_epochs=smoke_epochs,
                            num_workers=args.num_workers)
        if rec["model_parallel"]:
            raise RuntimeError("model.parallel is True: wrong config loaded")
        C.write_text_atomic(out, json.dumps(rec, indent=2))
        print(f"[no-shallow] g{g} seed{s}: test macro-F1 {rec['test_macro_f1']:.4f} | acc "
              f"{rec['test_accuracy']:.4f} | best epoch {rec['best_epoch']} | "
              f"{rec['duration_s']:.0f}s", flush=True)
    print(f"\n[no-shallow] finished in {(time.perf_counter() - t_all) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
