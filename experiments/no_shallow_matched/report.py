"""Assemble results/REPORT_no_shallow.md (ticket §7) from the artefacts of the other
scripts: preliminary_check.md, sanity_check.json, tables/*.md, interpretation.json and
figures/*. Run after compare.py and plot_comparison.py.

The "Lettura" section applies the criteria of ticket §8 as fixed in compare.py BEFORE
the runs; it is a proposed reading to be validated by the author, not a conclusion.

  python experiments/no_shallow_matched/report.py
"""

import json
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
10. **Hardware**: le serie esistenti sono state calcolate sul server; se la serie nuova gira su
    un'altra GPU, la differenza rientra nel rumore solo nella misura stabilita dal sanity check
    (riproduzione del modello con shallow sulla stessa macchina dei 35 run).
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


def main():
    figs = "\n".join(
        f"### {name}\n\n![{name}](figures/{name}.png)\n\nPDF: `figures/{name}.pdf`\n\n"
        f"*Bozza di didascalia.* {cap}\n" for name, cap in CAPTIONS.items())
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

Prodotti (tutti in `experiments/no_shallow_matched/results/`): `preliminary_check.md`,
`sanity_check.json`, `panels/`, `runs/` (un JSON per run), `runs_raw/`, `tables/`, `figures/`,
questo report.

## Verifica preliminare (§0bis)

Premessa confermata: il multi-scala non ha alcuna via verso il classificatore che aggiri il
backbone relazionale. Dettaglio con file e righe in [preliminary_check.md](preliminary_check.md).

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
## Confondenti residui

{CONFOUNDERS}"""
    out = C.OUT / "REPORT_no_shallow.md"
    out.write_text(md, encoding="utf-8")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
