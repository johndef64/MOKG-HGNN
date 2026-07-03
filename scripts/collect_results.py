from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml


# -----------------------
# Parsers
# -----------------------
OVERALL_HEADER = "=== OVERALL METRICS ==="
REPORT_HEADER = "=== CLASSIFICATION REPORT (sklearn) ==="


def safe_read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def safe_read_yaml(p: Path) -> Dict[str, Any]:
    try:
        obj = yaml.safe_load(p.read_text(encoding="utf-8", errors="ignore"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def flatten_dict(d: Dict[str, Any], parent: str = "", sep: str = "_") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        k = str(k)
        key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, key, sep))
        elif isinstance(v, list):
            out[key] = json.dumps(v, ensure_ascii=False)
        else:
            out[key] = v
    return out


def parse_run_name(run_name: str) -> Dict[str, Any]:
    """
    Esempio: T3_alpha_a1_seed2020_20260227_175043
    -> group_from_name = T3_alpha_a1
       seed_from_name = 2020
       timestamp_date/time
    """
    out: Dict[str, Any] = {}
    m = re.match(r"^(?P<group>.+)_seed(?P<seed>\d+?)_(?P<date>\d{8})_(?P<time>\d{6})$", run_name)
    if m:
        out["group_from_name"] = m.group("group")
        out["seed_from_name"] = int(m.group("seed"))
        out["timestamp_date"] = m.group("date")
        out["timestamp_time"] = m.group("time")
    else:
        m2 = re.search(r"seed(\d+)", run_name, flags=re.IGNORECASE)
        if m2:
            out["seed_from_name"] = int(m2.group(1))
    return out


def parse_overall_metrics(text: str) -> Dict[str, float]:
    """
    Estrae i 4 numeri in sezione OVERALL METRICS:
      accuracy, f1 (micro), f1_macro, f1_weighted
    """
    out: Dict[str, float] = {}
    if OVERALL_HEADER not in text:
        return out

    after = text.split(OVERALL_HEADER, 1)[1]
    # prendiamo fino alla prossima sezione
    chunk = after.split("===", 1)[0]

    for line in chunk.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip().lower()
        v = v.strip()
        try:
            val = float(v)
        except Exception:
            continue

        if k == "accuracy":
            out["accuracy"] = val
        elif k.startswith("f1") and "micro" in k:
            out["f1_micro"] = val
        elif k == "f1_macro":
            out["f1_macro"] = val
        elif k == "f1_weighted":
            out["f1_weighted"] = val

    return out


def parse_classification_report(text: str) -> List[Dict[str, Any]]:
    """
    Estrae le righe della tabella sotto:
      === CLASSIFICATION REPORT (sklearn) ===
    Ritorna lista di dict: class_label, precision, recall, f1, support
    Include anche 'macro avg' e 'weighted avg' se presenti.
    """
    if REPORT_HEADER not in text:
        return []

    after = text.split(REPORT_HEADER, 1)[1]
    lines = after.splitlines()

    rows: List[Dict[str, Any]] = []
    in_table = False

    for line in lines:
        line = line.rstrip()
        if not line.strip():
            if in_table:
                break
            continue

        # salta header/separator
        if set(line.strip()) <= {"-"}:
            continue
        if line.strip().lower().startswith("class") and "precision" in line.lower():
            in_table = True
            continue
        if not in_table:
            continue

        # riga tabella: label + 4 colonne numeriche
        parts = line.split()
        if len(parts) < 5:
            continue
        # ultime 4 sono precision recall f1 support
        try:
            precision = float(parts[-4])
            recall = float(parts[-3])
            f1 = float(parts[-2])
            support = int(float(parts[-1]))
        except Exception:
            continue

        label = " ".join(parts[:-4])
        rows.append(
            {
                "class_label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    return rows


# -----------------------
# Collector
# -----------------------
def find_config_file(run_dir: Path) -> Optional[Path]:
    for name in ("used_config.yaml", "used_config.yml", "config.yaml", "config.yml"):
        p = run_dir / name
        if p.exists():
            return p
    return None


def collect_one_run(run_dir: Path, results_root: str, dataset: str) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
      - run_summary_row (wide): overall metrics + flattened config
      - class_rows (long): per-class report for val/test (if present)
    """
    test_p = run_dir / "metrics.txt"
    val_p = run_dir / "val_metrics.txt"
    if not (test_p.exists() or val_p.exists()):
        return None, []

    run_name = run_dir.name
    meta = {
        "results_root": results_root,
        "dataset": dataset,
        "run_name": run_name,
        "run_dir": str(run_dir),
    }
    meta.update(parse_run_name(run_name))

    # config
    cfg_p = find_config_file(run_dir)
    meta["config_file"] = str(cfg_p) if cfg_p else None
    if cfg_p:
        cfg = flatten_dict(safe_read_yaml(cfg_p))
        for k, v in cfg.items():
            meta[f"config_{k}"] = v

    # split_id for paired stats:
    # prefer config_project_split_seed (se esiste), altrimenti seed del nome
    split_id = None
    if "config_project_split_seed" in meta and pd.notna(meta["config_project_split_seed"]):
        split_id = meta["config_project_split_seed"]
    elif "seed_from_name" in meta and pd.notna(meta["seed_from_name"]):
        split_id = meta["seed_from_name"]
    else:
        split_id = run_name
    meta["split_id"] = split_id

    # parse metrics
    summary = dict(meta)
    class_rows: List[Dict[str, Any]] = []

    if val_p.exists():
        t = safe_read_text(val_p)
        ov = parse_overall_metrics(t)
        for k, v in ov.items():
            summary[f"val_{k}"] = v
        rep = parse_classification_report(t)
        for r in rep:
            class_rows.append({**meta, "split": "val", **r})

    if test_p.exists():
        t = safe_read_text(test_p)
        ov = parse_overall_metrics(t)
        for k, v in ov.items():
            summary[f"test_{k}"] = v
        rep = parse_classification_report(t)
        for r in rep:
            class_rows.append({**meta, "split": "test", **r})

    return summary, class_rows


def scan_all(base_dir: Path, dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: List[Dict[str, Any]] = []
    class_rows: List[Dict[str, Any]] = []

    for results_root in sorted(base_dir.glob("results*")):
        ds_dir = results_root / dataset
        if not ds_dir.exists():
            continue

        # run dirs = parent di metrics.txt/val_metrics.txt
        run_dirs = set()
        for fname in ("metrics.txt", "val_metrics.txt"):
            for p in ds_dir.rglob(fname):
                if p.is_file():
                    run_dirs.add(p.parent)

        for rd in sorted(run_dirs):
            run_row, cls = collect_one_run(rd, results_root.name, dataset)
            if run_row:
                run_rows.append(run_row)
            if cls:
                class_rows.extend(cls)

    df_runs = pd.DataFrame(run_rows)
    df_cls = pd.DataFrame(class_rows)
    return df_runs, df_cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default=".", help="Root repo (contiene results*/)")
    ap.add_argument("--dataset", type=str, default="pancan", help="Nome dataset folder (es. pancan)")
    ap.add_argument("--out_dir", type=str, default="./analysis", help="Output folder")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df_runs, df_cls = scan_all(base_dir, args.dataset)

    if df_runs.empty:
        print("Nessuna run trovata. Controlla la struttura: results*/<dataset>/**/(metrics.txt|val_metrics.txt)")
        return

    # 1) Raw wide
    raw_csv = out_dir / "all_runs_raw.csv"
    df_runs.to_csv(raw_csv, index=False)
    print(f"Saved {raw_csv} (n={len(df_runs)})")

    # 2) “Ready for stats” subset
    wanted = [
        "results_root", "dataset", "group_from_name", "run_name", "run_dir",
        "split_id", "seed_from_name", "timestamp_date", "timestamp_time",
        "val_f1_macro", "val_f1_weighted", "val_accuracy",
        "test_f1_macro", "test_f1_weighted", "test_accuracy",
        "config_sampler_mode", "config_sampler_gamma", "config_sampler_strategy",
        "config_data_omic_mode", "config_data_num_gene", "config_data_num_mirna", "config_data_num_tf",
        "config_model_name", "config_model_jumping_knowledge", "config_model_jk_mode",
        "config_project_split_seed", "config_project_seed",
    ]
    df_stats = df_runs[[c for c in wanted if c in df_runs.columns]].copy()
    stats_csv = out_dir / "runs_for_stats.csv"
    df_stats.to_csv(stats_csv, index=False)
    print(f"Saved {stats_csv}")

    # 3) Long per-class report (se ti serve per analisi classe-per-classe)
    if not df_cls.empty:
        cls_csv = out_dir / "class_report_long.csv"
        df_cls.to_csv(cls_csv, index=False)
        print(f"Saved {cls_csv} (n={len(df_cls)})")

    # 4) Sanity: quanti split per “config” (dal nome), utile per verificare che siano 16
    if "group_from_name" in df_runs.columns:
        sanity = (
            df_runs.groupby(["results_root", "group_from_name"])["split_id"]
            .nunique()
            .reset_index(name="n_splits")
            .sort_values(["results_root", "n_splits"], ascending=[True, False])
        )
        sanity_csv = out_dir / "sanity_splits_per_group.csv"
        sanity.to_csv(sanity_csv, index=False)
        print(f"Saved {sanity_csv}")


if __name__ == "__main__":
    main()