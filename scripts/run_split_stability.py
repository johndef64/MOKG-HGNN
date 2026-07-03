"""TODO 5 — Esperimento 1: stabilità sotto split multipli (validità interna).

Riallena la configurazione finale del paper (``configs/config_final.yml``) su
N partizioni train/val/test stratificate INDIPENDENTI, variando solo lo split
(``project.split_seed``) e tenendo FISSO il seed del modello
(``project.seed``).  La varianza osservata sul macro-F1 cross-split è quindi
attribuibile esclusivamente alla scelta della partizione, non all'inizializzazione
dei pesi.

Scopo (cfr. ``docs/TODO_v2_comune_CiBM_JBHI.md``, TODO 5): mostrare che il
risultato non dipende dallo split, rispondendo in anticipo all'obiezione di
validità n.1 dei reviewer.  Rischio ~nullo: gli 11 seed già riportati nel
paper hanno s.d. bassa (0.7918 ± 0.0156); questo esperimento la conferma su
partizioni indipendenti.

Differenza rispetto a ``run_sweep.py``: i dati (~1.7 GB) vengono caricati una
sola volta e riusati per tutti gli split, e a fine corsa viene prodotto un
report aggregato (media ± s.d. ± IC95% di macro-F1 di validation e test) sia
in CSV sia in Markdown, pronto per il manoscritto.

Resumable: ogni split già completato (cartella ``split_stability_seedNN_*``
contenente ``metrics.txt``) viene riusato anziché riallenato.  Puoi quindi
interrompere e rilanciare lo stesso comando senza perdere lavoro: riparte solo
dagli split mancanti.  Un rilancio "a freddo" (tutti gli split già su disco)
si limita a ricostruire il report aggregato.  Usa ``--force`` per riallenare
comunque.

Uso:
    conda run -n gnn python scripts/run_split_stability.py
    conda run -n gnn python scripts/run_split_stability.py --split-seeds 42 43 44 45 46
    conda run -n gnn python scripts/run_split_stability.py --model-seed 2025 --smoke
    conda run -n gnn python scripts/run_split_stability.py --force   # riallena tutto

Lo schema degli split prodotti è identico a quello del sweep equivalente
``configs/sweep_split_stability.yml`` (mantenuto per parità con la pipeline make).
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multiomics_gnn.config.loader import load_config
from multiomics_gnn.pancancer_prediction.experiments.experiment_runner import (
    ExperimentRunner,
    ExperimentDataLoader,
)

# split_seed=42 è quello riportato nel paper -> incluso come ancora/sanity-check.
DEFAULT_SPLIT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_MODEL_SEED = 2025


def _set_nested(d: dict, dotted_key: str, value):
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def _parse_metrics_txt(path: Path) -> dict:
    """Estrae le metriche globali da un metrics.txt prodotto dal runner.

    Il blocco '=== OVERALL METRICS ===' contiene righe 'nome: valore'
    (accuracy, f1 (micro), f1_macro, f1_weighted).
    """
    out = {}
    if not path.exists():
        return out
    in_overall = False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if s.startswith("=== OVERALL METRICS"):
            in_overall = True
            continue
        if in_overall:
            if s.startswith("===") or s == "":
                break
            if ":" in s:
                k, v = s.split(":", 1)
                try:
                    out[k.strip()] = float(v.strip())
                except ValueError:
                    pass
    return out


def _summary_from_dir(run_dir: Path) -> dict | None:
    """Ricostruisce il dizionario summary (chiavi test_*/val_*) da una run su disco.

    Restituisce None se la run non è completa (manca metrics.txt = test eval).
    """
    test = _parse_metrics_txt(run_dir / "metrics.txt")
    if not test:
        return None
    val = _parse_metrics_txt(run_dir / "val_metrics.txt")

    def g(d, *keys):
        for k in keys:
            if k in d:
                return d[k]
        return None

    summary = {
        "test_accuracy": g(test, "accuracy"),
        "test_f1_macro": g(test, "f1_macro"),
        "test_f1_weighted": g(test, "f1_weighted"),
        "val_accuracy": g(val, "accuracy"),
        "val_f1_macro": g(val, "f1_macro"),
        "val_f1_weighted": g(val, "f1_weighted"),
        # val_auc_roc non è in val_metrics.txt -> resta assente in modalità resume
    }
    return {k: v for k, v in summary.items() if v is not None}


def _find_completed_run(results_dir: Path, split_seed: int) -> Path | None:
    """Trova la run completata più recente per uno split_seed, se esiste."""
    candidates = sorted(
        results_dir.glob(f"split_stability_seed{split_seed}_*"),
        key=lambda p: p.name,
        reverse=True,
    )
    for c in candidates:
        if (c / "metrics.txt").exists():
            return c
    return None


def _ci95(std: float, n: int) -> float:
    """Half-width of the 95% CI for the mean (normal approx, 1.96·sd/√n)."""
    if n <= 1:
        return float("nan")
    return 1.96 * std / np.sqrt(n)


def aggregate(rows: list[dict], metrics: list[str]) -> pd.DataFrame:
    """Mean / std / 95% CI half-width over the per-split metric values."""
    n = len(rows)
    out = []
    for m in metrics:
        vals = np.array([r[m] for r in rows if r.get(m) is not None], dtype=float)
        if len(vals) == 0:
            continue
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out.append(
            {
                "metric": m,
                "n_splits": len(vals),
                "mean": float(vals.mean()),
                "std": std,
                "ci95_halfwidth": _ci95(std, len(vals)),
                "min": float(vals.min()),
                "max": float(vals.max()),
            }
        )
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "config_final.yml"),
        help="Base config (default: the paper final config).",
    )
    ap.add_argument(
        "--split-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SPLIT_SEEDS,
        help="Split seeds = independent stratified partitions (default: 42 43 44 45 46).",
    )
    ap.add_argument(
        "--model-seed",
        type=int,
        default=DEFAULT_MODEL_SEED,
        help="Fixed model init seed, held constant across splits (default: 2025).",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Quick check: overrides num_epochs to 3 to validate the pipeline end-to-end.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Retrain every split even if a completed run already exists on disk.",
    )
    args = ap.parse_args()

    import copy

    base_config = load_config(args.config)
    results_dir = Path(base_config["paths"]["results_dir"])
    print(f"[split-stability] base config: {args.config}")
    print(f"[split-stability] split seeds: {args.split_seeds}")
    print(f"[split-stability] model seed (fixed): {args.model_seed}")
    if args.smoke:
        print("[split-stability] SMOKE mode: num_epochs -> 3")

    # --- resume: stabilisci quali split mancano davvero ----------------------
    reuse = {}  # split_seed -> (run_dir, summary) già su disco
    todo = []
    for split_seed in args.split_seeds:
        if not args.force:
            done = _find_completed_run(results_dir, split_seed)
            if done is not None:
                summary = _summary_from_dir(done)
                if summary:
                    reuse[split_seed] = (done, summary)
                    print(f"[split-stability] REUSE split {split_seed} "
                          f"<- {done.name} (test_f1_macro={summary.get('test_f1_macro')})")
                    continue
        todo.append(split_seed)

    # Carica i dati (~1.7 GB) UNA volta sola, e SOLO se c'è almeno uno split da fare.
    runner = None
    if todo:
        print(f"[split-stability] splits to train: {todo}")
        print("[split-stability] loading omics data once...")
        loader = ExperimentDataLoader(config=base_config)
        expression_data, cnv_data, mirna_data = loader.load_raw_data()
        runner = ExperimentRunner(expression_data, cnv_data, mirna_data)
        print("[split-stability] data loaded; starting runs.")
    else:
        print("[split-stability] all splits already on disk; rebuilding report only.")

    rows = []
    for split_seed in args.split_seeds:
        if split_seed in reuse:
            _, summary = reuse[split_seed]
        else:
            cfg = copy.deepcopy(base_config)
            _set_nested(cfg, "project.split_seed", int(split_seed))
            _set_nested(cfg, "project.seed", int(args.model_seed))
            if args.smoke:
                _set_nested(cfg, "train.num_epochs", 3)

            name = f"split_stability_seed{split_seed}"
            print(f"\n[split-stability] === RUN {name} "
                  f"(split_seed={split_seed}, model_seed={args.model_seed}) ===")
            summary = runner.run_experiment(cfg, experiment_name=name)
            print(f"[split-stability] {name}: "
                  f"test_f1_macro={summary.get('test_f1_macro')}, "
                  f"val_f1_macro={summary.get('val_f1_macro')}")

        row = {"split_seed": split_seed, "model_seed": args.model_seed}
        row.update(summary)
        rows.append(row)

    # ---- raccolta + aggregazione -------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(base_config["paths"]["results_dir"]) / f"split_stability_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_split = pd.DataFrame(rows)
    per_split_path = out_dir / "per_split_metrics.csv"
    per_split.to_csv(per_split_path, index=False)

    metrics = [
        "test_f1_macro",
        "test_f1_weighted",
        "test_accuracy",
        "val_f1_macro",
        "val_f1_weighted",
        "val_accuracy",
        "val_auc_roc_ovr_macro",
    ]
    metrics = [m for m in metrics if m in per_split.columns]
    agg = aggregate(rows, metrics)
    agg_path = out_dir / "aggregate_metrics.csv"
    agg.to_csv(agg_path, index=False)

    # ---- report Markdown pronto per il paper -------------------------------
    md = ["# TODO 5 — Esperimento 1: stabilità sotto split multipli", ""]
    md.append(f"- Config base: `{Path(args.config).name}`"
              + ("  **(SMOKE: 3 epoche — non per il paper)**" if args.smoke else ""))
    md.append(f"- Seed modello (fisso): `{args.model_seed}`")
    md.append(f"- Split seeds (partizioni indipendenti): "
              f"`{', '.join(map(str, args.split_seeds))}`")
    md.append(f"- N split: {len(rows)}")
    md.append("")
    md.append("## Aggregato (media ± s.d., IC95%)")
    md.append("")
    md.append("| metrica | n | media | s.d. | ±IC95% | min | max |")
    md.append("|---|---|---|---|---|---|---|")
    for _, r in agg.iterrows():
        md.append(
            f"| {r['metric']} | {int(r['n_splits'])} | {r['mean']:.4f} | "
            f"{r['std']:.4f} | {r['ci95_halfwidth']:.4f} | "
            f"{r['min']:.4f} | {r['max']:.4f} |"
        )
    md.append("")
    md.append("## Per split")
    md.append("")
    cols = ["split_seed"] + metrics
    cols = [c for c in cols if c in per_split.columns]
    md.append("| " + " | ".join(cols) + " |")
    md.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in per_split.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        md.append("| " + " | ".join(cells) + " |")
    md.append("")

    # frase pronta per la Discussion (test macro-F1)
    f1row = agg[agg["metric"] == "test_f1_macro"]
    if not f1row.empty:
        r = f1row.iloc[0]
        md.append("## Frase per la Discussion")
        md.append("")
        md.append(
            f"> Across {int(r['n_splits'])} independent stratified train/val/test "
            f"partitions (fixed model seed), test macro-F1 was "
            f"{r['mean']:.4f} ± {r['std']:.4f} "
            f"(95% CI ± {r['ci95_halfwidth']:.4f}; range "
            f"{r['min']:.4f}–{r['max']:.4f}), confirming the result is not an "
            f"artefact of the particular data split."
        )
        md.append("")

    md_path = out_dir / "REPORT.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    # snapshot del comando per replicabilità
    (out_dir / "command.txt").write_text(
        "python scripts/run_split_stability.py "
        + " ".join(sys.argv[1:]) + "\n",
        encoding="utf-8",
    )

    print(f"\n[split-stability] DONE.")
    print(f"[split-stability] per-split : {per_split_path}")
    print(f"[split-stability] aggregate : {agg_path}")
    print(f"[split-stability] report    : {md_path}")
    if not f1row.empty:
        r = f1row.iloc[0]
        print(f"[split-stability] >>> test macro-F1 = "
              f"{r['mean']:.4f} ± {r['std']:.4f} (n={int(r['n_splits'])})")


if __name__ == "__main__":
    main()
