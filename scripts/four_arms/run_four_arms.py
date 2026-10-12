"""Run the new arms of the version-3 factorial experiment (plan §1, §7).

kg_s (MOKG-HGNN + shallow), sh_kg and sh_tf (shallow only) over 7 levels x 5 splits; one
JSON per run in results/four_arms/runs/<arm>_g<L>_seed<S>.json (written atomically, so the
run is resumable: an existing JSON is skipped). Run folders go to results/four_arms/runs_raw/.

Gate: refuses to start unless results/four_arms/sanity_check.json says passed=true with the
kg reproduction done, under the same torch version (--smoke skips the gate and writes to
results/four_arms/smoke/, never read by the analysis).

  python scripts/four_arms/run_four_arms.py                          # all arms, all 35
  python scripts/four_arms/run_four_arms.py --arms kg_s --levels 700 --seeds 42
  python scripts/four_arms/run_four_arms.py --smoke 2 --levels 20 --seeds 42
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402


def check_gate():
    if not core.SANITY_JSON.exists():
        sys.exit(f"[gate] {core.SANITY_JSON} missing: run sanity_check.py --repro 1 first.")
    s = json.loads(core.SANITY_JSON.read_text())
    if not (s.get("passed") and s.get("reproduction_done")):
        sys.exit("[gate] sanity check not passed or reproduction not done: see sanity_check.json.")
    here = core.NS.environment()
    if here["torch"] != s["env"]["torch"]:
        sys.exit(f"[gate] sanity ran with torch {s['env']['torch']}, here {here['torch']}: redo it.")
    if here["cuda_device"] != s["env"]["cuda_device"]:
        print(f"[gate] NOTE: sanity on '{s['env']['cuda_device']}', this run on "
              f"'{here['cuda_device']}' (recorded in each run JSON).", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(core.NEW_ARMS), choices=core.NEW_ARMS)
    ap.add_argument("--levels", type=int, nargs="+", default=core.LEVELS)
    ap.add_argument("--seeds", type=int, nargs="+", default=core.SEEDS)
    ap.add_argument("--smoke", type=int, default=0, metavar="EPOCHS",
                    help="pipeline test with this many epochs, into results/four_arms/smoke/")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="MOGNN-TF DataLoader workers (execution only; 0 on Windows)")
    a = ap.parse_args()

    if a.smoke:
        runs_dir, raw = core.SMOKE / "runs", core.SMOKE / "runs_raw"
    else:
        check_gate()
        runs_dir, raw = core.RUNS, core.RUNS_RAW

    todo = [(arm, g, s) for g in a.levels for s in a.seeds for arm in a.arms
            if a.smoke or not core.run_json(arm, g, s, runs_dir).exists()]
    print(f"[four_arms] {len(todo)} runs to do: arms {a.arms} | levels {a.levels} | splits {a.seeds}",
          flush=True)
    for i, (arm, g, s) in enumerate(todo, 1):
        t0 = time.perf_counter()
        print(f"\n===== [{i}/{len(todo)}] {arm} | {g} genes | split {s} =====", flush=True)
        rec = core.run_arm(arm, g, s, raw_root=raw, smoke_epochs=a.smoke or None,
                           num_workers=a.num_workers)
        core.write_json(core.run_json(arm, g, s, runs_dir), rec)
        print(f"[four_arms] {arm} g{g} s{s}: macro-F1 {rec['test_macro_f1']:.4f} "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
