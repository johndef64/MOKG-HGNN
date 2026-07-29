#%%
import pandas as pd
import matplotlib.pyplot as plt

num_classes = 4  # adatta al tuo caso

for c in range(num_classes):
    df = pd.read_csv(f"roc_curve_class_{c}.csv")
    plt.figure()
    plt.plot(df["fpr"], df["tpr"], label=f"Classe {c}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve - Classe {c}")
    plt.legend()
    plt.savefig(f"roc_curve_class_{c}.png", dpi=300, bbox_inches="tight")
    plt.close()
#%%