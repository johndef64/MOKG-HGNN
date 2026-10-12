"""Assemble results/REPORT_no_shallow.md (ticket §7) from the artefacts of the other
scripts: preliminary_check.md, sanity_check.json, tables/*.md, interpretation.json and
figures/*. Run after compare.py and plot_comparison.py.

The "Lettura" section applies the criteria of ticket §8 as fixed in compare.py BEFORE
the runs; it is a proposed reading to be validated by the author, not a conclusion.

  python scripts/no_shallow/report.py
"""

import csv
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

CAPTIONS = {
    "fig_scarcity_three_curves": (
        "Test macro-F1 (mean ± s.d. over five stratified splits) as the requested gene panel "
        "shrinks from 700 to 20, for the multi-scale model and for the molecular model with and "
        "without its shallow branch, all on the identical surviving panel at every level. Shaded "
        "bands mark the interval between consecutive panel sizes where a pair of curves crosses; "
        "markers are slightly offset horizontally for legibility."),
    "fig_gap_vs_panel": (
        "Macro-F1 difference from the multi-scale model as a function of panel size, for the "
        "molecular model with and without the shallow branch. Faint points are individual "
        "splits (paired by split), lines join the means."),
    "fig_shallow_effect": (
        "Effect of removing the shallow branch from the molecular model: paired macro-F1 "
        "difference (without minus with) for each of the five splits at every panel size, with "
        "the mean shown as a horizontal segment."),
    "fig_spread_vs_panel": (
        "Standard deviation of test macro-F1 across the five splits at each panel size, for the "
        "three models."),
}

CONFOUNDERS = """\
Il confronto rende entrambi i modelli privi di una via che aggiri il grafo, ma non li rende
identici a meno del grafo. Restano, dichiarati e non risolti qui:

1. **Encoder e operatore**: ChebConv (K=5, 2 layer, hidden 6) contro HeteroSAGE (3 layer, hidden 256).
2. **Readout**: il molecolare fa max pooling su finestre di 16 nodi consecutivi nell'ordine degli
   indici e poi appiattisce, quindi conserva la posizione (l'identità) del nodo; il multi-scala fa
   mean pooling per tipo, invariante all'ordine, e l'identità del gene arriva solo dalla topologia.
   Anche senza ramo shallow, il molecolare resta un classificatore sensibile all'identità dei nodi.
   La larghezza del readout molecolare scala con il pannello (floor(N/16)·6 valori, poi un quarto:
   circa 80 unità a 700 geni, 58 a 500, 36 a 300, 19 a 150, 13 a 100, 6 a 50, 1-3 a 20), mentre
   quella del multi-scala è fissa (4 scale × 256). Ai livelli bassi il declino senza shallow
   riflette anche questa capacità ridotta, non solo il grafo.
3. **Grafo molecolare**: il molecolare include gli archi co-target miRNA-miRNA, TF-TF e TF-miRNA
   (config_final: `mirna_mirna`, `tf_tf`, `tf_mirna` attivi); il multi-scala appaiato non ha
   metapath.
4. **Scaling**: min-max (molecolare) contro z-score (multi-scala), entrambi stimati sul solo train.
5. **Sbilanciamento**: sampler pesato gamma 0.98 senza pesi nella loss (molecolare) contro sampler
   pesato gamma 0.99 più cross-entropy pesata (multi-scala).
6. **Nessun re-tuning**: la variante senza shallow usa gli iperparametri scelti con lo shallow;
   questo può svantaggiarla leggermente.
7. **Decoder mantenuto**: il decoder di ricostruzione (lambda = 1) legge theta1, cioè la
   rappresentazione del grafo, quindi non è un bypass; resta come regolarizzatore.
8. **Penalità L2**: nel codice è `0.2 * mu * sum(theta^2)` con mu = 5e-4 su tutti i parametri (non
   solo sull'encoder come descritto nel paper di MOGNN-TF), in aggiunta al weight decay AdamW 1e-4.
9. **Inizializzazione**: il seed del modello è fisso (2025), ma togliere due layer cambia la sequenza
   di numeri casuali consumata, quindi il classificatore della variante parte da pesi diversi:
   effetto inevitabile, di natura casuale.
10. **Hardware**: le serie esistenti sono state calcolate sul server precedente; la serie nuova
    gira sulle GPU del cluster. Il sanity check certifica la riproduzione del modello con shallow
    sulla GPU su cui è stato eseguito (vedi sopra); i run finiti su un altro modello di GPU sono
    elencati nella sezione del sanity check. Un cambio di modello di GPU, a parità di software,
    produce differenze numeriche dello stesso tipo del non determinismo (punto 11), non
    verificate per singolo modello.
11. **Rumore run-to-run**: con operazioni scatter non deterministiche su GPU, repliche dello stesso
    setup differiscono di circa 0.01 in macro-F1; differenze di quest'ordine non sono interpretabili.
"""


def _read(p):
    p = Path(p)
    return p.read_text(encoding="utf-8") if p.exists() else f"_(manca: {p.relative_to(C.REPO)})_\n"


def _strip_title(md):
    lines = md.splitlines()
    return "\n".join(lines[1:]).strip() if lines and lines[0].startswith("# ") else md


def _sanity_section():
    if not C.SANITY_JSON.exists():
        return "_sanity_check.json assente._\n"
    s = json.loads(C.SANITY_JSON.read_text())
    out = [f"- Macchina: `{s['env']['cuda_device']}` ({s['env']['host']}), torch {s['env']['torch']}",
           f"- Controlli statici superati: **{s['static_passed']}**"]
    out.append("- Split: " + ", ".join(f"seed {k} {'ok' if v['ok'] else 'FAIL'}"
                                      for k, v in s["A1_splits"].items()))
    p = s["A2_panels_and_configs"]
    out.append(f"- Pannelli: {sum(v['ok'] for v in p.values())}/{len(p)} coerenti, "
               f"{sum(v['fully_verified'] for v in p.values())}/{len(p)} verificati su tutte le fonti")
    if s.get("unverifiable_panels"):
        out.append(f"  - non verificabili su questa macchina (cartelle non trasferite): "
                   f"{', '.join(s['unverifiable_panels'])}")
    out.append(f"- Config variante: differisce dall'originale solo in `model.parallel`: "
               f"**{s['A3_variant_config']['ok']}**")
    out.append(f"- Modello variante (struttura e classifier su theta1): "
               f"**{s['A4_variant_model']['passed']}**")
    r = s["A5_references"]
    out.append(f"- Serie esistenti complete e coerenti con la Tab. S2 pubblicata: **{r['ok']}**")
    b = s.get("B_reproduction")
    if b:
        reps = ", ".join(f"{x['test_macro_f1']:.4f} (delta {x['delta_test_macro_f1']:+.4f})" for x in b["repeats"])
        out.append(f"- Riproduzione del modello con shallow, livello {b['level']}, split {b['split_seed']}: "
                   f"esistente {b['existing']['f1']:.4f}, ripetuto {reps}; tolleranza a priori "
                   f"{b['tolerance_macro_f1']}: **{'superata' if b['passed'] else 'NON superata'}**")
    out.append(f"- Esito complessivo: **{s['passed']}**")
    gpus = {}
    for (g, seed), d in sorted(C.read_new_runs().items()):
        gpus.setdefault(d.get("env", {}).get("cuda_device", "?"), []).append(f"g{g}/s{seed}")
    if gpus:
        out.append("- GPU dei run senza shallow: " + "; ".join(
            f"`{k}` {len(v)} run" + ("" if k == s["env"]["cuda_device"] else f" ({', '.join(v)})")
            for k, v in gpus.items()))
    return "\n".join(out) + "\n"


_EPOCH_RE = re.compile(r"Epoch (\d+) - Train \(with dropout\) Loss: (\S+),.*?Train \(eval mode\) "
                       r"Loss: (\S+),.*?Val Loss: (\S+),")


def _loss_trace(run_dir):
    """(n epochs, first / min train-eval loss, min val loss, n non-finite) from experiment.log."""
    log = C.resolve_run_dir(run_dir) / "experiment.log"
    if not log.exists():
        return None
    rows = [(int(m[1]), float(m[3]), float(m[4]))
            for m in _EPOCH_RE.finditer(log.read_text(encoding="utf-8", errors="ignore"))]
    if not rows:
        return None
    tr = [r[1] for r in rows]
    va = [r[2] for r in rows]
    bad = sum(not math.isfinite(x) for x in tr + va)
    fin = [x for x in tr if math.isfinite(x)]
    return {"epochs": len(rows), "first": tr[0], "min_train": min(fin) if fin else float("nan"),
            "min_val": min(x for x in va if math.isfinite(x)) if fin else float("nan"), "nonfinite": bad}


def _sensitivity():
    p = C.TABLES / "sensitivity.json"
    if not p.exists():
        return "_sensitivity.json assente: eseguire compare.py sulle serie complete._\n"
    s = json.loads(p.read_text())
    runs = C.read_new_runs()
    degen = {(d["genes"], d["split_seed"]) for d in s["degenerate_runs"]}
    out = [f"Regola **definita dopo aver visto i risultati** (2026-10-12): è degenere un run il cui "
           f"F1 di test è maggiore di zero per al più una classe, cioè un predittore costante. "
           f"Run degeneri: {', '.join(f'{g} geni split {sd}' for g, sd in sorted(degen)) or 'nessuno'}. "
           f"L'analisi primaria (sopra, tutti i 35 run) resta il riferimento; questa sezione verifica "
           f"solo se la lettura §8 dipende da quei run."]
    if degen:
        levels = sorted({g for g, _ in degen})
        ln_k = math.log(27)
        topo = {}
        tp = C.TABLES / "graph_topology.csv"
        if tp.exists():
            with open(tp, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    topo[(int(r["genes"]), int(r["split_seed"]))] = int(r["ms_n_molecular"])
        out += ["", f"Diagnosi dai log di training (loss di cross-entropy; predizione uniforme su 27 "
                    f"classi = ln 27 = {ln_k:.3f}) e dalla larghezza del readout del ramo grafo di "
                    f"MOGNN-TF (max-pool su finestre di 16 nodi, 6 canali, poi `linear1` a un quarto: "
                    f"floor(N/16)·6 // 4 unità; N da `graph_topology.py`):", "",
                "| geni | split | degenere | nodi N | unità dopo linear1 | epoche | loss train iniziale | "
                "loss train minima | loss val minima | valori non finiti |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for g in levels:
            for sd in C.SEEDS:
                d = runs.get((g, sd))
                t = _loss_trace(d["run_dir"]) if d else None
                n = topo.get((g, sd))
                units = "" if n is None else str((n // 16) * 6 // 4)
                if t:
                    out.append(f"| {g} | {sd} | {'sì' if (g, sd) in degen else 'no'} | {n or ''} | {units} | "
                               f"{t['epochs']} | {t['first']:.3f} | {t['min_train']:.3f} | "
                               f"{t['min_val']:.3f} | {t['nonfinite']} |")
        out += ["", "Nei run degeneri la loss non scende mai in modo apprezzabile sotto ln 27, senza "
                    "valori non finiti: non è un guasto numerico ma un mancato avvio "
                    "dell'apprendimento. Senza ramo shallow il classificatore riceve solo theta1, e "
                    "theta1 passa per un readout la cui larghezza scala con il numero di nodi. I run "
                    "degeneri sono esattamente quelli con N < 32: una sola finestra di pooling, "
                    "compressa in una sola unità ReLU prima del classificatore; se quell'unità resta "
                    "spenta il classificatore riceve una costante. I run che imparano hanno 3 unità. "
                    "La corrispondenza (2 su 2) indica il meccanismo; non è stata verificata "
                    "ispezionando le attivazioni. Il valore a 20 geni senza shallow riflette quindi "
                    "anche il collo di bottiglia del readout di MOGNN-TF, non solo il grafo."]

    def line(name, it):
        g, sl, c = it.get("gap_700"), it.get("slope"), it.get("crossing")
        parts = []
        if g:
            parts.append(f"divario a 700 {g['G_WS']:+.4f} → {g['G_NS']:+.4f} "
                         f"(riduzione {100 * g['relative_reduction']:.0f}%, **{g['verdict']}**)")
        if sl:
            parts.append(f"perdita per dimezzamento {sl['NS']:.4f} senza shallow contro {sl['WS']:.4f} "
                         f"con shallow ({'**più in fretta**' if sl['NS_degrades_faster'] else 'non più in fretta'})")
        if c:
            parts.append("incrocio con il multi-scala " + (", ".join(
                f"{x['between'][0]}-{x['between'][1]} (circa {x['approx_genes']:.0f})"
                for x in c["NS_vs_MS"]) or "nessuno"))
        return f"- **{name}**: " + "; ".join(parts) + "."

    out += ["", line("Senza i run degeneri (medie)", s["excl_degenerate_means"]),
            line("Mediane per livello (tutti i run)", s["medians"])]
    for name in ("sensitivity_by_panel", "sensitivity_summary"):
        out += ["", f"### {name}", "", _strip_title(_read(C.TABLES / f"{name}.md"))]
    return "\n".join(out) + "\n"


def _reading():
    p = C.TABLES / "interpretation.json"
    if not p.exists():
        return "_interpretation.json assente: eseguire compare.py._\n"
    it = json.loads(p.read_text())
    out = []
    g = it.get("gap_700")
    if g:
        out.append(f"- **Divario a 700 geni**: con shallow {g['G_WS']:+.4f}, senza shallow {g['G_NS']:+.4f} "
                   f"(variazione {g['change']:+.4f}, riduzione relativa {100 * g['relative_reduction']:.0f}%). "
                   f"Criterio §8: il divario **{g['verdict']}**.")
    s = it.get("slope")
    if s:
        how = ("più in fretta" if s["NS_degrades_faster"] else
               "più lentamente" if s["NS_degrades_slower"] else "con pendenza simile (entro 0.005)")
        out.append(f"- **Pendenza**: perdita per dimezzamento {s['NS']:.4f} senza shallow contro "
                   f"{s['WS']:.4f} con shallow: il molecolare senza shallow degrada {how}.")
    c = it.get("crossing")
    if c:
        def f(cr):
            return ", ".join(f"{x['between'][0]}-{x['between'][1]} (circa {x['approx_genes']:.0f})"
                             for x in cr) or "nessuno"
        out.append(f"- **Incrocio con il multi-scala**: con shallow {f(c['WS_vs_MS'])}; senza shallow "
                   f"{f(c['NS_vs_MS'])}. Incrocio sopra i 50 geni: **{c['NS_crossing_above_50']}**.")
    out.append("\nCriteri fissati in `compare.py::interpretation` prima dei run (2026-10-11): "
               "riduzione 'molto' se >= 50% e > 0.02; 'simile' se |variazione| <= 0.02; "
               "pendenze diverse se differiscono di oltre 0.005. Con 5 split le differenze sono "
               "effect size, non test (Wilcoxon appaiato minimo p = 0.0625). Lettura proposta, "
               "da validare dall'autore alla luce dei confondenti sotto.")
    return "\n".join(out) + "\n"


def _degenerate_note():
    p = C.TABLES / "sensitivity.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text())["degenerate_runs"]
    if not d:
        return ""
    by = {}
    for r in d:
        by.setdefault(r["genes"], []).append(str(r["split_seed"]))
    where = "; ".join(f"at {g} genes, splits {', '.join(s)}" for g, s in sorted(by.items(), reverse=True))
    return (f" Without the shallow branch, {len(d)} of the 35 runs never started learning and "
            f"predict a single class ({where}); they are included here and drive the no-shallow "
            f"value and spread at that panel size (see the post hoc sensitivity analysis).")


def main():
    note = _degenerate_note()
    figs = "\n".join(
        f"### {name}\n\n![{name}](figures/{name}.png)\n\nPDF: `figures/{name}.pdf`\n\n"
        f"*Bozza di didascalia.* {cap}{note}\n" for name, cap in CAPTIONS.items())
    md = f"""# Report: confronto appaiato senza ramo shallow nel modello molecolare

## Scopo e motivazione

Nell'esperimento appaiato di scarsità di feature (tesi Sez. 4.5.3) il modello molecolare
MOGNN-TF ha due rami: il grafo produce theta1 e un MLP shallow, che non vede il grafo, produce
theta2; il classificatore legge theta1 concatenato a theta2. Il multi-scala non ha una via
analoga. Per attribuire la differenza al grafo (e alla sua profondità semantica) e non alla
scorciatoia sugli omici, qui si rieseguono i soli 35 run del molecolare **senza** ramo shallow,
a pannelli, split, iperparametri e protocollo invariati, e si confrontano con le due serie già
ottenute (multi-scala e molecolare con shallow), che non vengono rieseguite.

## File

Letti (sola lettura):
- `results/feature_collapse/mokghgnn_g*/<ts>/{{metrics,config,graph_info}}.json`: serie multi-scala (35 run)
- `results/feature_collapse/collapse_mognntf_summary.csv`: serie molecolare con shallow (35 righe)
- `results/feature_collapse/mognntf_g*/run_*/used_config.yaml`: pannelli e config dei run con shallow
- `results/feature_collapse/_aligned_fs/`: file di feature selection dell'esperimento originale
- `data/training/splits/splits_seed_4[2-6]/`: split congelati
- `benchmark/MOGNN-TF/configs/config_final.yml`, `scripts/collapse_mognntf.py` (riusati per import)
- `supplementary/tables/S2_feature_collapse.csv`: tabella pubblicata, per il controllo di coerenza

Prodotti (tutti in `results/no_shallow/`):
`sanity_check.json`, `panels/`, `runs/` (un JSON per run), `runs_raw/`, `tables/`, `figures/`,
questo report.

## Verifica preliminare (§0bis)

Premessa confermata: il multi-scala non ha alcuna via verso il classificatore che aggiri il
backbone relazionale. Dettaglio con file e righe in
[scripts/no_shallow/preliminary_check.md](../../scripts/no_shallow/preliminary_check.md).

## Sanity check (§4)

{_sanity_section()}
## Tabelle

### comparison_by_panel

{_strip_title(_read(C.TABLES / "comparison_by_panel.md"))}

### summary_metrics

{_strip_title(_read(C.TABLES / "summary_metrics.md"))}

### paired_differences_summary

{_strip_title(_read(C.TABLES / "paired_differences_summary.md"))}

La tabella per split è in `tables/paired_differences.csv`.

## Figure

{figs}
## Lettura dei risultati (§8)

{_reading()}
## Analisi di sensibilità (post hoc)

{_sensitivity()}
## Confondenti residui

{CONFOUNDERS}"""
    out = C.OUT / "REPORT_no_shallow.md"
    out.write_text(md, encoding="utf-8")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
