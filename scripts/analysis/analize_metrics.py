#%%
# Importa librerie
import pandas as pd
# Force non-interactive backend BEFORE importing pyplot so plt.show() is a no-op
# when this script is invoked from the Makefile / a non-GUI context.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Configurazione visualizzazioni
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

#%%
# Carica i dati
df = pd.read_csv("runs_for_stats.csv")

print(f"Totale run caricati: {len(df)}")
print(f"Colonne disponibili: {list(df.columns)}")

# `group_from_name` is produced by collect_results.py only when the run
# folder matches "<group>_seed<NNNN>_<date>_<time>". Newer sweep names
# (e.g. ablation_genes_smoke_001_..., optuna_trial_0_...) do not match
# that pattern, so the column may be absent here. Fall back to run_name
# (with the trailing _YYYYMMDD_HHMMSS stripped) so downstream groupby
# operations still work.
if "group_from_name" not in df.columns:
    df["group_from_name"] = df["run_name"].astype(str).str.replace(
        r"_\d{8}_\d{6}$", "", regex=True
    )
    print("Note: 'group_from_name' was missing — derived it from run_name.")

#%%
# Panoramica dati
print("\n=== PRIME RIGHE ===")
print(df.head())

print("\n=== INFO DATASET ===")
print(df.info())

#%%
# Statistiche descrittive per le metriche principali
print("\n=== STATISTICHE TEST SET ===")
metrics = ['test_accuracy', 'test_f1_macro', 'test_f1_weighted']
print(df[metrics].describe())

print("\n=== STATISTICHE VALIDATION SET ===")
val_metrics = ['val_accuracy', 'val_f1_macro', 'val_f1_weighted']
print(df[val_metrics].describe())

#%%
# Conta esperimenti per cartella
print("\n=== ESPERIMENTI PER CARTELLA ===")
count_by_folder = df['results_root'].value_counts()
print(count_by_folder)

#%%
# Conta esperimenti per gruppo
if 'group_from_name' in df.columns:
    print("\n=== ESPERIMENTI PER GRUPPO ===")
    count_by_group = df.groupby(['results_root', 'group_from_name']).size().reset_index(name='count')
    print(count_by_group.to_string())

#%%
# Media metriche per gruppo (Test Set)
if 'group_from_name' in df.columns:
    print("\n=== MEDIA F1-MACRO (TEST) PER GRUPPO ===")
    avg_by_group = df.groupby('group_from_name')['test_f1_macro'].agg(['mean', 'std', 'count']).round(4)
    avg_by_group = avg_by_group.sort_values('mean', ascending=False)
    print(avg_by_group)

#%%
# Visualizza distribuzione F1-macro per cartella
if 'results_root' in df.columns and 'test_f1_macro' in df.columns:
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df, x='results_root', y='test_f1_macro')
    plt.title('Distribuzione F1-Macro (Test) per Cartella')
    plt.xticks(rotation=45)
    plt.ylabel('F1-Macro')
    plt.tight_layout()
    plt.savefig('f1_macro_by_folder.png', dpi=150)
    plt.show()

#%%
# Confronto Val vs Test
if 'val_f1_macro' in df.columns and 'test_f1_macro' in df.columns:
    plt.figure(figsize=(8, 8))
    plt.scatter(df['val_f1_macro'], df['test_f1_macro'], alpha=0.5)
    plt.plot([0.6, 0.9], [0.6, 0.9], 'r--', label='y=x')
    plt.xlabel('Validation F1-Macro')
    plt.ylabel('Test F1-Macro')
    plt.title('Validation vs Test F1-Macro')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('val_vs_test_f1.png', dpi=150)
    plt.show()

#%%
# Top 10 migliori run (per test_f1_macro)
print("\n=== TOP 10 RUN (F1-MACRO TEST) ===")
top10 = df.nlargest(10, 'test_f1_macro')[['run_name', 'test_f1_macro', 'test_accuracy', 'group_from_name']]
print(top10.to_string())

#%%
#=========================
# Analisi Sampler: confronto tra RandomSampler e WeightedRandomSampler
# T3_random vs T3_alpha_* vs T3_nulite_g*
#=========================
# ragruppa solo questi esperimenti in un nuovo dataframe
sampler_df = df[df['group_from_name'].isin([
    'T3_random',
    'T3_alpha_1', 'T3_alpha_2', 'T3_alpha_3',
    'T3_gamma_090', 'T3_gamma_095', 'T3_gamma_098', 'T3_gamma_099',
])]
print("\n=== ANALISI SAMPLER: T3_RANDOM vs T3_ALPHA_* vs T3_NULITE_G* ===")
print(f"Totale run per analisi sampler: {len(sampler_df)}")
# %%
# per ogni sampler calcola media e std delle metriche di validation e test macro f1 e weighted f1
sampler_metrics = sampler_df.groupby('group_from_name')[['val_f1_macro', 'test_f1_macro', 'val_f1_weighted', 'test_f1_weighted']].agg(['mean', 'std', 'count']).round(4)
sampler_metrics.columns = ['_'.join(col) for col in sampler_metrics.columns]
print(sampler_metrics.sort_values('test_f1_macro_mean', ascending=False))
# %%
# %%
import numpy as np
from scipy.stats import t

t3 = df[df["group_from_name"].astype(str).str.startswith("T3_")].copy()

def ci95(x):
    x = pd.to_numeric(x, errors="coerce").dropna().to_numpy()
    n = len(x)
    if n < 2: return (np.nan, np.nan)
    m, s = x.mean(), x.std(ddof=1)
    h = t.ppf(0.975, n-1) * s / np.sqrt(n)
    return (m-h, m+h)

sum_t3 = (t3.groupby("group_from_name")
          .agg(n=("split_id","nunique"),
               val_macro=("val_f1_macro","mean"),
               val_weight=("val_f1_weighted","mean"))
          .reset_index())

sum_t3["val_macro_ci95"]  = sum_t3["group_from_name"].map(lambda g: ci95(t3.loc[t3.group_from_name==g,"val_f1_macro"]))
sum_t3["val_weight_ci95"] = sum_t3["group_from_name"].map(lambda g: ci95(t3.loc[t3.group_from_name==g,"val_f1_weighted"]))
print(sum_t3.sort_values("val_macro", ascending=False))
#%%
# plot degli intervalli di confidenza per val_f1_macro
plt.figure(figsize=(10, 6))
for _, row in sum_t3.iterrows():
    plt.errorbar(row["group_from_name"], row["val_macro"], 
                 yerr=[[row["val_macro"] - row["val_macro_ci95"][0]], 
                       [row["val_macro_ci95"][1] - row["val_macro"]]], 
                 fmt='o', capsize=5)
plt.title("Intervallo di Confidenza 95% per Val F1-Macro (T3)")
plt.xlabel("Gruppo")
plt.ylabel("Val F1-Macro")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ci95_val_f1_macro_t3.png', dpi=150)
plt.show()

# %%
# test t per confronto tra T3_random e T3_alpha_0.9 su val_f1_macro
# %%
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

t3 = df[df["group_from_name"].astype(str).str.startswith("T3_")].copy()
BASE = "T3_random"

# If no T3_* runs are present in the current results set, skip the
# Wilcoxon-vs-random analysis entirely (it was specific to Laterza's
# original sampler ablation). Avoids crashing on smoke/full runs that
# don't use the T3_ naming convention.
_skip_t3_block = t3.empty or BASE not in t3["group_from_name"].unique()
if _skip_t3_block:
    print("\n[skip] No T3_* runs found — Wilcoxon-vs-random block skipped.")

def wilcoxon_vs_random(metric):
    wide = t3.pivot_table(index="split_id", columns="group_from_name", values=metric, aggfunc="mean")

    if BASE not in wide.columns:
        raise ValueError(f"Baseline {BASE} non presente. Colonne: {list(wide.columns)}")

    rows = []
    for g in [c for c in wide.columns if c != BASE]:
        pair = wide[[BASE, g]].dropna()
        d = (pair[g] - pair[BASE]).to_numpy()
        n = len(d)
        if n < 2:
            rows.append([g, n, np.nan, np.nan, np.nan, np.nan, np.nan])
            continue

        # Wilcoxon signed-rank (paired)
        stat, p = wilcoxon(d, zero_method="wilcox", alternative="two-sided")

        rows.append([
            g, n,
            float(d.mean()),
            float(np.median(d)),
            int((d > 0).sum()),
            int((d < 0).sum()),
            float(p),
        ])

    res = pd.DataFrame(
        rows,
        columns=["group", "n", "mean_diff", "median_diff", "wins", "losses", "p_wilcoxon"]
    ).sort_values(["p_wilcoxon", "mean_diff"], ascending=[True, False])

    return res

if not _skip_t3_block:
    print("\n=== Wilcoxon vs T3_random (VAL macro-F1) ===")
    print(wilcoxon_vs_random("val_f1_macro").round(6).to_string(index=False))

    print("\n=== Wilcoxon vs T3_random (VAL weighted-F1) ===")
    print(wilcoxon_vs_random("val_f1_weighted").round(6).to_string(index=False))
    # %%
    #salva i risultati in un file CSV
    sum_t3.to_csv("summary_t3_wilcoxon.csv", index=False)
# %%
# metriche per classe (specifico per le run T3_*)
# %%
import os
import pandas as pd

if _skip_t3_block:
    print("[skip] T3 per-class analysis skipped (no T3_* runs).")
    raise SystemExit(0)

os.makedirs("analysis", exist_ok=True)

cls = pd.read_csv("class_report_long.csv")

# Same fallback as above: derive group_from_name from run_name if absent.
if "group_from_name" not in cls.columns:
    cls["group_from_name"] = cls["run_name"].astype(str).str.replace(
        r"_\d{8}_\d{6}$", "", regex=True
    )

t3c = cls[
    (cls["split"] == "val") &
    (cls["group_from_name"].astype(str).str.startswith("T3_"))
].copy()

t3c = t3c[t3c["class_label"].astype(str).str.fullmatch(r"\d+")].copy()
t3c["class_label"] = t3c["class_label"].astype(int)

tbl = (
    t3c.groupby(["group_from_name", "class_label"], as_index=False)
       .agg(
           n_splits=("split_id", "nunique"),
           support_mean=("support", "mean"),
           support_std=("support", "std"),
           precision_mean=("precision", "mean"),
           precision_std=("precision", "std"),
           recall_mean=("recall", "mean"),
           recall_std=("recall", "std"),
           f1_mean=("f1", "mean"),
           f1_std=("f1", "std"),
       )
)

tbl = tbl.sort_values(["class_label", "group_from_name"])
round_cols = [
    "support_mean","support_std",
    "precision_mean","precision_std",
    "recall_mean","recall_std",
    "f1_mean","f1_std"
]
tbl[round_cols] = tbl[round_cols].round(4)

print("\n=== T3 VALIDATION - per classe (Precision/Recall/F1) mean±std ===")
print(tbl.to_string(index=False))

out_csv = "analysis/T3_val_per_class_PRF_mean_std.csv"
tbl.to_csv(out_csv, index=False)
print(f"\nSaved -> {out_csv}")
# %%
# in markdown, mostra la tabella con i risultati per classe (precision/recall/f1) mean±std
print("\n=== T3 VALIDATION - per classe (Precision/Recall/F1) mean±std ===")
for class_label in sorted(tbl["class_label"].unique()):
    print(f"\n**Classe {class_label}**")
    subset = tbl[tbl["class_label"] == class_label]
    for _, row in subset.iterrows():
        print(f"- {row['group_from_name']}: Precision={row['precision_mean']}±{row['precision_std']}, "
              f"Recall={row['recall_mean']}±{row['recall_std']}, F1={row['f1_mean']}±{row['f1_std']}")
# salva in md
md_out = "analysis/T3_val_per_class_PRF_mean_std.md"
with open(md_out, "w") as f:
    f.write("# T3 VALIDATION - per classe (Precision/Recall/F1) mean&plusmn;std\n")
    for class_label in sorted(tbl["class_label"].unique()):
        f.write(f"\n## Classe {class_label}\n\n")
        subset = tbl[tbl["class_label"] == class_label]
        
        # Intestazione tabella
        f.write("| Gruppo | Precision | Recall | F1-Score |\n")
        f.write("|--------|-----------|--------|----------|\n")
        
        # Righe tabella
        for _, row in subset.iterrows():
            f.write(f"| {row['group_from_name']} | "
                    f"{row['precision_mean']:.4f}&plusmn;{row['precision_std']:.4f} | "
                    f"{row['recall_mean']:.4f}&plusmn;{row['recall_std']:.4f} | "
                    f"{row['f1_mean']:.4f}&plusmn;{row['f1_std']:.4f} |\n")
print(f"\nSaved -> {md_out}")

# %%
