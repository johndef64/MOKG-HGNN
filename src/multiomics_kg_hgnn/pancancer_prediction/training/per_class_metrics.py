"""Per-class metrics (precision / recall / F1 + confusion matrix) for the
27-class task. Mirrors what MOGNN-TF saves via classification_report, so the
tables are directly comparable.

Used both by the runner (every run saves per-class metrics) and by the
standalone eval_per_class.py (recompute from a saved checkpoint, no retraining).
"""

import csv
import json
import os

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def compute_per_class(y_true, y_pred, num_classes=None, class_names=None):
    """Return a dict: {'per_class': [...], 'macro_f1':.., 'weighted_f1':.., 'accuracy':..}.
    Each per_class entry: {class, support, precision, recall, f1}."""
    labels = list(range(num_classes)) if num_classes else sorted(set(map(int, y_true)))
    rep = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    per_class = []
    for i in labels:
        r = rep.get(str(i), {})
        per_class.append({
            "class": class_names[i] if class_names else f"C{i + 1}",
            "class_idx": i,
            "support": int(r.get("support", 0)),
            "precision": float(r.get("precision", 0.0)),
            "recall": float(r.get("recall", 0.0)),
            "f1": float(r.get("f1-score", 0.0)),
        })
    return {
        "per_class": per_class,
        "accuracy": float(rep.get("accuracy", 0.0)),
        "macro_f1": float(rep["macro avg"]["f1-score"]),
        "weighted_f1": float(rep["weighted avg"]["f1-score"]),
    }


def save_per_class(run_dir, y_true, y_pred, num_classes=None, class_names=None):
    """Write per_class_metrics.csv, per_class_metrics.json and confusion_matrix.csv
    into run_dir. Returns the computed dict."""
    res = compute_per_class(y_true, y_pred, num_classes, class_names)

    with open(os.path.join(run_dir, "per_class_metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["class", "class_idx", "support",
                                           "precision", "recall", "f1"])
        w.writeheader()
        w.writerows(res["per_class"])

    with open(os.path.join(run_dir, "per_class_metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2)

    labels = list(range(num_classes)) if num_classes else sorted(set(map(int, y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.savetxt(os.path.join(run_dir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
    return res
