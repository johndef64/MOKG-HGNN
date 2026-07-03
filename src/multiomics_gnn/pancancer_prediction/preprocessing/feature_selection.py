import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import norm

# ============================================================
# 0) LOAD DATA
# ============================================================
#expression_data = pd.read_csv("traning_data/expression_data_pancan.tsv", sep="\t", index_col=0)
#
# ============================================================
# 1) READ LABELS (iCluster) + CHECK COUNTS
# ============================================================
def select_features(expression_data, k_total=700, pool_min_per_class=200, extra_max=300, m_final_per_class=5):
    labels_raw = expression_data["icluster_cluster_assignment"].astype(int).values

    unique_labels, label_counts = np.unique(labels_raw, return_counts=True)
    print("Label counts per original class:", dict(zip(unique_labels, label_counts)))
    #
    # ============================================================
    # 2) MAKE LABELS CONTIGUOUS (0..C-1)
    #    Why? Because iCluster labels may have "holes" (e.g., missing 24).
    #    We need contiguous indices to safely index arrays like k_per_class[c].
    # ============================================================
    classes = np.sort(np.unique(labels_raw))                 # e.g. [1,2,3,...,23,25,26,27,28]
    mapping = {lab: i for i, lab in enumerate(classes)}      # original -> compact
    y = np.array([mapping[v] for v in labels_raw], dtype=int)

    counts = np.bincount(y)

    print("\nCounts per compact class (0..C-1):", counts)
    print("Total samples:", counts.sum())
    print("Number of classes (present):", len(counts))
    print("Original labels present:", classes)
    #
    # ============================================================
    # 3) BUILD X (GENE MATRIX) AND REMOVE NON-GENE COLUMNS
    # ============================================================
    # IMPORTANT: keep only gene columns. Adjust the drop list if your columns differ.
    X_df = expression_data.drop(columns=["sample", "icluster_cluster_assignment"], errors="ignore")

    # ============================================================
    # 4) VARIANCE FILTERING
    #    Remove genes with variance <= 0 (constant genes).
    #    This never removes truly informative genes (constants cannot help classification).
    # ============================================================
    variance = X_df.var(ddof=0)  # ddof=0 = population variance (fine for filtering)
    genes_to_keep = variance[variance > 0].index
    X_df = X_df[genes_to_keep]

    print("\nGenes after variance>0 filter:", X_df.shape[1])

    # Convert to numpy for speed
    X = X_df.values
    gene_names = X_df.columns.to_numpy()
    
    C = len(counts)
    G = X_df.shape[1]
    #
    # ============================================================
    # 5) ALLOCATE NUMBER OF FEATURES PER CLASS
    # ============================================================
    # esempio extra proporzionale a 1/sqrt(n), scalato tra 0..extra_max
    w = 1.0 / np.sqrt(counts)
    w_scaled = (w - w.min()) / (w.max() - w.min() + 1e-12)
    pool_k_per_class = pool_min_per_class + np.round(extra_max * w_scaled).astype(int)
    print("\nNumber of features to select per class (pool):", pool_k_per_class)
    # adjust to have exactly k_total genes in the pool
    # generate a pool of genes per class minimum 200 gnenes + extra based on class weights
    # ensure at least pool_min_per_class per class
    #
    # ============================================================
    # 7) OVR FEATURE SELECTION (F-SCORE) TO BUILD THE POOL
    #    For each class c, select pool_k_per_class[c] genes with highest F-score
    # ============================================================
    from sklearn.feature_selection import f_classif

    per_class_selected = {}      # c -> indices of selected genes (pool for class c)
    per_class_scores_full = {}   # c -> full scores array (length G)

    for c in range(C):
        y_bin = (y == c).astype(int)

        # compute F-score for all genes (univariate)
        scores, pvals = f_classif(X, y_bin)

        # handle NaN / inf safely
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(float).max, neginf=-np.inf)

        per_class_scores_full[c] = scores

        k_c = int(pool_k_per_class[c])

        # top-k for this class
        top_idx = np.argpartition(scores, -k_c)[-k_c:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        per_class_selected[c] = top_idx

        # short print (do NOT dump hundreds of genes)
        print(f"\nClass original {classes[c]} (compact {c}) | n={counts[c]} -> pool select k_c={k_c}")
        for j in top_idx[:5]:
            print(f"  {gene_names[j]} | F={scores[j]:.2f}")

    # ============================================================
    # 8) BUILD GLOBAL POOL METRICS (support + best_score)
    #    support[g] = in how many classes gene g appears in the class pool
    #    best_score[g] = best (max) F-score gene g achieved across classes (when selected)
    # ============================================================
    support = np.zeros(G, dtype=int)
    best_score = np.full(G, -np.inf)

    for c in range(C):
        idx = per_class_selected[c]
        support[idx] += 1
        best_score[idx] = np.maximum(best_score[idx], per_class_scores_full[c][idx])

    # pool genes are those with support > 0
    pool_mask = support > 0
    pool_size = int(pool_mask.sum())

    print("\n============================================================")
    print("POOL SUMMARY")
    print("============================================================")
    print("Pool size (unique genes across classes):", pool_size)
    print("Support distribution on pool genes (min/median/max):",
        support[pool_mask].min(), int(np.median(support[pool_mask])), support[pool_mask].max())
    # ============================================================
    # 9) GLOBAL RANKING -> KEEP ONLY k_total GENES FOR THE FINAL SET
    #    Ranking rule:
    #      1) higher support first
    #      2) if tie, higher best_score
    # ============================================================

    rank_idx = np.lexsort((support, best_score))[::-1]
    rank_idx = rank_idx[support[rank_idx] > 0]   # ignore never-selected genes

    print("X shape:", X.shape)                 # (N, G)
    print("len(gene_names):", len(gene_names))
    print("support len:", len(support))
    print("rank_idx max:", rank_idx.max(), "rank_idx min:", rank_idx.min())

    # final selection
    final_k = k_total
    final_idx = rank_idx[:final_k]
    final_genes = gene_names[final_idx].tolist()

    print("\n============================================================")
    print("FINAL GENE SET")
    print("============================================================")
    print("Final genes selected:", len(final_genes))
    print("Support stats (min/median/max):",
        support[final_idx].min(), int(np.median(support[final_idx])), support[final_idx].max())

    print("\nTop 20 final genes:")
    for i in range(min(20, len(final_idx))):
        gi = final_idx[i]
        print(f"{i+1:02d}. {gene_names[gi]} | support={support[gi]} | best_F={best_score[gi]:.2f}")
    # ============================================================
    # 10) OPTIONAL: ENSURE MINIMUM REPRESENTATION PER CLASS IN FINAL SET
    #     (Simple safeguard)
    #     If you want, guarantee at least m_final_per_class genes from each class pool
    #     inside the final set. This helps very rare classes not be pushed out.
    # ============================================================

    if m_final_per_class > 0:
        final_set = set(final_idx.tolist())

        # First force-in top m_final_per_class for each class
        forced = []
        for c in range(C):
            top_c = per_class_selected[c][:m_final_per_class]
            forced.extend(top_c.tolist())

        forced = list(dict.fromkeys(forced))  # unique, preserve order
        forced_set = set(forced)

        # Build a new final list:
        # 1) all forced genes first
        # 2) then fill with ranked genes until reaching final_k
        new_final = []
        for gi in forced:
            if gi not in new_final:
                new_final.append(gi)

        for gi in rank_idx:
            if len(new_final) >= final_k:
                break
            if gi not in forced_set and gi not in new_final:
                new_final.append(int(gi))

        final_idx = np.array(new_final[:final_k], dtype=int)
        final_genes = gene_names[final_idx].tolist()

        print("\n[After enforcing minimum per class]")
        print("Final genes selected:", len(final_genes))
        print("Support stats (min/median/max):",
            support[final_idx].min(), int(np.median(support[final_idx])), support[final_idx].max())
    # ============================================================
    # 11) SAVE OUTPUTS
    # ============================================================
    pd.Series(final_genes).to_csv("selected_genes_final.csv", index=False, header=False)

    out = pd.DataFrame({
        "gene": gene_names[final_idx],
        "support": support[final_idx],
        "best_fscore": best_score[final_idx]
    })
    out.to_csv("selected_genes_final_with_scores.csv", index=False)

    print("\nSaved:")
    print(" - selected_genes_final.csv")
    print(" - selected_genes_final_with_scores.csv")
    # ============================================================
    # 12) GET INDICES OF SELECTED GENES IN ORIGINAL DATA
    # ===========================================================
    selected_feature_indices = [expression_data.columns.get_loc(feat) for feat in final_genes]
    print("\nSelected feature indices in original data:", selected_feature_indices)
    return final_genes, selected_feature_indices

def ebpp_to_positive(df: pd.DataFrame) -> np.ndarray:
    # x = log2(norm_value + 1)  ->  y = 2^x = norm_value + 1  > 0
    # np.exp2 è più veloce di np.power(2, x)
    x = df.to_numpy(dtype=np.float32, copy=False)
    y = np.exp2(x)  # always > 0 for finite x
    # gestisci NaN/inf
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y

def gini_chunked(Y: np.ndarray, chunk_size: int = 512, eps: float = 1e-12) -> np.ndarray:
    """
    Y: shape (n_samples, n_genes), >=0
    returns gini per gene: shape (n_genes,)
    """
    n, p = Y.shape
    out = np.empty(p, dtype=np.float64)

    for j0 in range(0, p, chunk_size):
        j1 = min(j0 + chunk_size, p)
        A = Y[:, j0:j1].astype(np.float64, copy=False)

        # somma per gene
        s = A.sum(axis=0)
        # evita divisioni strane
        s_safe = np.where(s > eps, s, 1.0)

        # sort per colonna
        A_sorted = np.sort(A, axis=0)
        cum = np.cumsum(A_sorted, axis=0)

        # formula: G = 1 - 2*sum(cum)/(n*sumx) + 1/n
        g = 1.0 - (2.0 * cum.sum(axis=0)) / (n * s_safe) + (1.0 / n)
        g = np.where(s > eps, g, 0.0)

        out[j0:j1] = g

    return out

def gini_bidirectional_from_ebpp(X: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    # X = log2(norm_value+1) shape (n_samples, n_genes)
    Y_up = np.exp2(X)           # norm_value + 1
    Y_dn = np.exp2(-X)          # emphasizes negative tail in X

    Y_up = np.nan_to_num(Y_up, nan=0.0, posinf=0.0, neginf=0.0)
    Y_dn = np.nan_to_num(Y_dn, nan=0.0, posinf=0.0, neginf=0.0)

    g_up = gini_chunked(Y_up, chunk_size=chunk_size)
    g_dn = gini_chunked(Y_dn, chunk_size=chunk_size)

    return np.maximum(g_up, g_dn)

def giniclust2_loess_normalize(
    gini_raw: pd.Series,
    max_expr_log2: pd.Series,
    frac: float = 0.3,
    it: int = 1,
    outlier_q: float = 0.75,
    eps: float = 1e-12,
    use_abs_resid_for_outliers: bool = True,
):
    """
    Implements the 2-step LOESS normalization described in GiniClust (used by GiniClust2):
    1) LOESS fit Gini ~ log2(max_expr)
    2) remove outliers (residuals above 75th percentile)
    3) refit LOESS
    4) normalized_gini = gini_raw - fitted_trend

    Then, for RNA-seq: p-values via normal approximation and select p < 1e-4 (one-sided, high tail).
    """

    # Align and clean
    df = pd.concat([gini_raw.rename("gini"), max_expr_log2.rename("t")], axis=1).dropna()
    g = df["gini"].to_numpy(dtype=np.float64)
    t = df["t"].to_numpy(dtype=np.float64)

    # ---- Step 1: initial LOESS ----
    fit1 = lowess(endog=g, exog=t, frac=frac, it=it, return_sorted=False)
    resid1 = g - fit1

    # ---- Outlier removal (paper: residues above 75th percentile) ----
    if use_abs_resid_for_outliers:
        r = np.abs(resid1)
    else:
        r = resid1

    thr = np.quantile(r, outlier_q)
    keep = r <= thr

    # ---- Step 2: refit LOESS on non-outliers ----
    t_keep = t[keep]
    g_keep = g[keep]

    # lowess does not natively "predict" on all x; easiest is:
    # get sorted curve from keep-points, then interpolate to all t.
    curve = lowess(endog=g_keep, exog=t_keep, frac=frac, it=it, return_sorted=True)
    t_curve = curve[:, 0]
    g_curve = curve[:, 1]

    # Ensure monotonic x for interpolation (should already be sorted)
    # Predict trend for all points:
    fit2 = np.interp(t, t_curve, g_curve)

    # ---- Normalized Gini (detrended) ----
    g_norm = g - fit2

    # ---- p-values (normal approximation; one-sided for "high" normalized Gini) ----
    # Estimate sigma from (non-outlier) normalized residuals:
    sigma = np.std(g_norm[keep], ddof=1)
    sigma = max(sigma, eps)

    z = g_norm / sigma
    p = 1.0 - norm.cdf(z)  # high-tail

    # Return as Series aligned to input index
    out = pd.DataFrame(index=df.index)
    out["gini_raw"] = g
    out["max_expr_log2"] = t
    out["gini_trend"] = fit2
    out["gini_norm"] = g_norm
    out["z"] = z
    out["p"] = p
    out["is_outlier_step1"] = ~keep
    return out

def ebpp_to_positive(X_log2: np.ndarray) -> np.ndarray:
    # y = 2^x = norm_value + 1  (strictly >0 for finite x)
    Y = np.exp2(X_log2.astype(np.float64, copy=False))
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.clip(Y, 0.0, None)
    return Y

def ebpp_to_positive_down(X_log2: np.ndarray) -> np.ndarray:
    # y_down = 2^{-x} (emphasizes negative tail of x)
    Y = np.exp2((-X_log2).astype(np.float64, copy=False))
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.clip(Y, 0.0, None)
    return Y

def gini_feature_selection(X: pd.DataFrame, top_k: int = 700) -> list:
    # ============================================================
    # PIPELINE: compute (bi)Gini + LOESS-normalized z/p + select genes
    # ============================================================
    # Convert to numpy once
    X = X.to_numpy(dtype=np.float32, copy=False)  # (n_samples, n_genes)
    genes = X.columns.to_numpy()

    # Optional fast prefilter: remove almost-constant genes (robust MAD on x)
    # (helps speed and avoids junk genes)
    mad = np.median(np.abs(X - np.median(X, axis=0, keepdims=True)), axis=0)
    keep_mad = mad > 1e-6
    Xf = X[:, keep_mad]
    genes_f = genes[keep_mad]
    print(f"After MAD filter: {Xf.shape[1]} genes kept out of {X.shape[1]}")

    # Positive transforms
    Y_up = ebpp_to_positive(Xf)       # 2^x
    Y_dn = ebpp_to_positive_down(Xf)  # 2^{-x}

    # Raw Gini (chunked, fast)
    gini_up = gini_chunked(Y_up, chunk_size=512)
    gini_dn = gini_chunked(Y_dn, chunk_size=512)

    # Bidirectional raw Gini score
    gini_bi = np.maximum(gini_up, gini_dn)

    # For LOESS covariate they use log(max expression). Here:
    # max_expr_log2 for up-branch equals max(X) (since log2(max(2^x)) = max(x))
    t_up = np.max(Xf.astype(np.float64), axis=0)
    # for down-branch, log2(max(2^{-x})) = max(-x) = -min(x)
    t_dn = np.max((-Xf).astype(np.float64), axis=0)

    # LOESS normalize up and down separately (cleanest)
    df_up = giniclust2_loess_normalize(
        gini_raw=pd.Series(gini_up, index=genes_f),
        max_expr_log2=pd.Series(t_up, index=genes_f),
        frac=0.3, it=1, outlier_q=0.75, use_abs_resid_for_outliers=True
    )

    df_dn = giniclust2_loess_normalize(
        gini_raw=pd.Series(gini_dn, index=genes_f),
        max_expr_log2=pd.Series(t_dn, index=genes_f),
        frac=0.3, it=1, outlier_q=0.75, use_abs_resid_for_outliers=True
    )

    # Bidirectional normalized score: max of z-scores (or min of p-values)
    z_bi = np.maximum(df_up["z"].values, df_dn["z"].values)
    p_bi = np.minimum(df_up["p"].values, df_dn["p"].values)

    df_bi = pd.DataFrame(index=genes_f)
    df_bi["gini_up"] = gini_up
    df_bi["gini_dn"] = gini_dn
    df_bi["gini_bi"] = gini_bi
    df_bi["z_bi"] = z_bi
    df_bi["p_bi"] = p_bi

    # Selection options:
    # (A) GiniClust-style significance threshold
    p_cut = 1e-4
    high_gini_genes = df_bi.index[df_bi["p_bi"] < p_cut].tolist()
    print(f"High-Gini genes (p < {p_cut}): {len(high_gini_genes)}")

    # (B) Or fixed top-K by z_bi (recommended for bulk, deterministic size)
    topK_genes = df_bi.sort_values("z_bi", ascending=False).head(top_k).index.tolist()
    print(f"Top {top_k} genes by LOESS-normalized bidirectional Gini (z_bi): {len(topK_genes)}")
    return topK_genes
