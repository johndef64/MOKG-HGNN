"""Per-class F1 of the six arms at the full panel (700 genes), paired by split.

Extends scripts/no_shallow/per_class_three_arms.py (imported: mol_s, mol, kg with its
support checks) with the new arms from their run JSONs; classes are aligned by position
in the encoder order and the test support is checked to be identical in every arm.
Win / tie / loss counts use the threshold of v2 (0.015).

Writes results/four_arms/tables/per_class_four_arms.{csv,md}.

  python scripts/four_arms/per_class_four_arms.py [--level 700]
"""

import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
from compare import _write  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "ns_per_class", core.REPO / "scripts" / "no_shallow" / "per_class_three_arms.py")
ns_pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ns_pc)

TIE = 0.015
OLD = {"MS": "kg", "WS": "mol_s", "NS": "mol"}


def mean(v):
    return sum(v) / len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=700)
    g = ap.parse_args().level

    names, seeds, per_old, support = ns_pc.load(g)
    per = {OLD[k]: v for k, v in per_old.items()}
    for arm in core.NEW_ARMS:
        runs = core.read_runs(arm)
        if not all((g, s) in runs for s in seeds):
            print(f"[per-class] {arm}: not all splits {seeds} available, skipped")
            continue
        per[arm] = {n: [] for n in names}
        for s in seeds:
            pc = runs[(g, s)]["per_class"]
            if len(pc) != len(names):
                sys.exit(f"[per-class] {arm} split {s}: {len(pc)} classes, expected {len(names)}")
            for i, n in enumerate(names):
                if pc[i][2] != support[n]:
                    sys.exit(f"[per-class] support mismatch {arm} split {s} class {n}")
                per[arm][n].append(pc[i][1])

    def diff(a, b, n):
        return mean(per[a][n]) - mean(per[b][n]) if a in per and b in per else None

    rows = []
    for n in names:
        r = {"class": n, "support": support[n]}
        r.update({f"F1_{a}": f"{mean(per[a][n]):.3f}" for a in core.ARMS if a in per})
        for key, a, b in (("G_S", "kg_s", "mol_s"), ("G_0", "kg", "mol"),
                          ("S_KG", "kg_s", "kg"), ("S_MOL", "mol_s", "mol")):
            d = diff(a, b, n)
            r[key] = "" if d is None else f"{d:+.3f}"
        if "kg_s" in per:
            r["kg_s>mol_s splits"] = f"{sum(x > y for x, y in zip(per['kg_s'][n], per['mol_s'][n]))}/{len(seeds)}"
        rows.append(r)

    counts = []
    for key, a, b in (("G_S", "kg_s", "mol_s"), ("G_0", "kg", "mol"), ("kg - mol_s", "kg", "mol_s")):
        ds = [diff(a, b, n) for n in names]
        if None in ds:
            continue
        counts.append(f"{key}: {sum(d > TIE for d in ds)} win / {sum(abs(d) <= TIE for d in ds)} "
                      f"tie / {sum(d < -TIE for d in ds)} loss")
    note = (f"{g} genes, splits {seeds}; mean per-class test F1. Counts (threshold {TIE}): "
            + "; ".join(counts))
    _write("per_class_four_arms", rows, f"Per-class F1, six arms, {g} genes", note)
    print(note)


if __name__ == "__main__":
    main()
