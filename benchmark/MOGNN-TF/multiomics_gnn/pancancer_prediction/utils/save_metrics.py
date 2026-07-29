import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score

def save_curves(history, filepath):
    curves_df = pd.DataFrame({
    "epoch": np.arange(1, len(history["train_loss"]) + 1),
    "lr": history["lr"],
    "train_loss": history["train_loss"],
    "val_loss": history["val_loss"],
    "train_acc": history["train_acc"],
    "val_acc": history["val_acc"],
    "train_f1": history["train_f1"],
    "val_f1": history["val_f1"]
    })
    curves_df.to_csv(os.path.join(filepath, "learning_curves.csv"), index=False)
    # save plots
    plt.figure()
    plt.plot(curves_df["epoch"], curves_df["train_loss"], label="Train Loss")
    plt.plot(curves_df["epoch"], curves_df["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve - Loss")
    plt.legend()
    plt.savefig(os.path.join(filepath, "learning_curve_loss.png"))
    print("Saved learning curves to:", os.path.join(filepath, "learning_curves.csv"))
    plt.close()
    #print(test(test_loader,nclass))
    # F1 score
    plt.figure()
    plt.plot(curves_df["epoch"], curves_df["train_f1"], label="Train F1 Score")
    plt.plot(curves_df["epoch"], curves_df["val_f1"], label="Validation F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Learning Curve - F1 Score")
    plt.legend()
    plt.savefig(os.path.join(filepath, "learning_curve_f1.png"))
    plt.close()
    print("Saved learning curve F1 plot to:", os.path.join(filepath, "learning_curve_f1.png"))
    

def get_all_metrics(true_labels, pred_labels, num_classes):
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    # metriche globali
    acc = accuracy_score(true_labels, pred_labels)
    f1_micro = f1_score(true_labels, pred_labels, average="micro")
    f1_macro = f1_score(true_labels, pred_labels, average="macro")
    f1_weighted = f1_score(true_labels, pred_labels, average="weighted")

    global_metrics = {
        "accuracy": acc,
        "f1_micro": f1_micro,          # F1 micro (sull'insieme)
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }

    # metriche per classe
    report = classification_report(
        true_labels,
        pred_labels,
        labels=range(num_classes),
        output_dict=True,
        zero_division=0,
    )
    
    confusion_mat = confusion_matrix(
        true_labels,
        pred_labels,
        labels=range(num_classes),
    )

    # metriche per classe: precision, recall + "accuratezza per classe"
    per_class_metrics = {}

    # accuratezza per classe = TP / (TP + FN) = elemento diagonale / somma riga
    row_sums = confusion_mat.sum(axis=1).astype(float)
    row_sums[row_sums == 0] = 1.0  # per sicurezza, evitare divisioni per zero

    for cls_idx in range(num_classes):
        key = str(cls_idx)
        cls_stats = report.get(key, {})

        precision = cls_stats.get("precision", 0.0)
        recall = cls_stats.get("recall", 0.0)

        # "accuracy" della singola classe (in pratica coincide con la recall di quella classe)
        class_accuracy = confusion_mat[cls_idx, cls_idx] / row_sums[cls_idx]

        per_class_metrics[cls_idx] = {
            "precision": precision,
            "recall": recall,
            "accuracy": class_accuracy,
        }

    return global_metrics, per_class_metrics, confusion_mat, report

def save_confusion_matrix_pretty(confusion_mat, filepath):
    """
    Salva una confusion matrix in stile 'paper':
      - normalizzata per riga (valori in [0,1]) annotati nelle celle
      - etichette delle classi prese da `class_names`
      - colonna finale con la somma (conteggi) per classe ("SUM")
      - CSV + PNG

    Parameters
    ----------
    confusion_mat : np.ndarray
        Matrice di confusione (n_class x n_class) con i conteggi.
    filepath : str
        Path del file di metrics txt: lo uso come base per cm.csv e cm.png.
    class_names : list[str]
        Lista dei nomi delle classi (stessa lunghezza di confusion_mat).
        Esempio: ["C1","C2","C3",...,"C23","C25","C26","C27","C28"].
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    base, _ = os.path.splitext(filepath)
    csv_path = base + "_cm.csv"
    png_path = base + "_cm.png"
    class_names = [
        "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
        "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20",
        "C21", "C22", "C23", "C25", "C26", "C27", "C28"
    ]
    class_names = list(class_names)
    num_classes = len(class_names)

    # ------------------  CSV (con colonna SUM)  ------------------
    row_sums = confusion_mat.sum(axis=1)
    cm_df = pd.DataFrame(
        confusion_mat,
        index=[f"true_{c}" for c in class_names],
        columns=[f"pred_{c}" for c in class_names],
    )
    cm_df["SUM"] = row_sums
    cm_df.to_csv(csv_path)

    # ------------------  immagine tipo quella che hai mandato  ------------------
    confusion_mat = confusion_mat.astype(float)
    row_sums_safe = row_sums.copy().astype(float)
    row_sums_safe[row_sums_safe == 0] = 1.0
    cm_norm = confusion_mat / row_sums_safe[:, None]

    fig, ax = plt.subplots(figsize=(num_classes * 0.5, num_classes * 0.5))

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Ground-truth")

    # spazio per la colonna SUM
    ax.set_xlim(-0.5, num_classes + 0.5)

    # valori nelle celle
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_norm[i, j]
            text = "0" if val == 0 else f"{val:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=6)

    # colonna SUM a destra (conteggi grezzi)
    ax.text(num_classes, -0.8, "SUM", ha="center", va="center", fontsize=8)
    for i in range(num_classes):
        ax.text(num_classes, i, str(int(row_sums[i])), ha="center", va="center", fontsize=7)

    # griglia tipo tabella
    ax.set_xticks(np.arange(-0.5, num_classes + 0.5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_classes + 0.5, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    return csv_path, png_path

def save_metrics_pretty_txt(global_metrics,
                            per_class_metrics,
                            clf_report,
                            filepath,
                            num_classes,
                            class_names=None):
    """
    Scrive un file di testo leggibile con:
      - metriche globali
      - tabella per classe (precision, recall, accuracy)
      - matrice di confusione tabellare
    """
    import os
    import numpy as np

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    with open(filepath, "w") as f:
        # ---------- overall ----------
        f.write("=== OVERALL METRICS ===\n")
        f.write(f"{'accuracy':>15}: {global_metrics['accuracy']:.4f}\n")
        f.write(f"{'f1 (micro)':>15}: {global_metrics['f1_micro']:.4f}\n")
        f.write(f"{'f1_macro':>15}: {global_metrics['f1_macro']:.4f}\n")
        f.write(f"{'f1_weighted':>15}: {global_metrics['f1_weighted']:.4f}\n\n")

        # ---------- per-class ----------
        f.write("=== PER-CLASS METRICS ===\n")
        f.write(f"{'class':>6} {'precision':>10} {'recall':>10} {'accuracy':>10}\n")
        f.write("-" * 40 + "\n")
        for cls_idx in range(num_classes):
            m = per_class_metrics.get(cls_idx, {})
            f.write(
                f"{class_names[cls_idx]:>6} "
                f"{m.get('precision', 0.0):10.4f} "
                f"{m.get('recall', 0.0):10.4f} "
                f"{m.get('accuracy', 0.0):10.4f}\n"
            )

        f.write("\n")
        # ---------- CLASSIFICATION REPORT ----------
        f.write("=== CLASSIFICATION REPORT (sklearn) ===\n")

        # Se è una stringa, la scrivo così com'è
        if isinstance(clf_report, str):
            f.write(clf_report)
        else:
            # Presumo sia un dict (output_dict=True)
            # Stampo almeno le righe per ciascuna classe
            f.write(f"{'class':>6} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}\n")
            f.write("-" * 60 + "\n")
            for cls_idx in range(num_classes):
                stats = clf_report.get(str(cls_idx), {})
                f.write(
                    f"{class_names[cls_idx]:>6} "
                    f"{stats.get('precision', 0.0):10.4f} "
                    f"{stats.get('recall', 0.0):10.4f} "
                    f"{stats.get('f1-score', 0.0):10.4f} "
                    f"{stats.get('support', 0):10.0f}\n"
                )

            # opzionale: macro avg / weighted avg se presenti
            for key in ["macro avg", "weighted avg"]:
                if key in clf_report:
                    stats = clf_report[key]
                    f.write(
                        f"{key:>12} "
                        f"{stats.get('precision', 0.0):10.4f} "
                        f"{stats.get('recall', 0.0):10.4f} "
                        f"{stats.get('f1-score', 0.0):10.4f} "
                        f"{stats.get('support', 0):10.0f}\n"
                    )


