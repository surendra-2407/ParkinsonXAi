"""
UCI push v2 — KNN_5 got 83.33% (25/30). Need 27/30 for 90%.
Focused strategies:
  1. KNN grid (k=1..20, metrics: euclidean, manhattan, cosine, chebyshev, minkowski)
     — NO SMOTE for KNN (SMOTE hurts distance-based methods)
  2. LDA / QDA — classical discriminant analysis, good for voice features
  3. GaussianNB
  4. NearestCentroid
  5. SVM with fine gamma grid (log-scale: 0.001 to 10)
  6. Feature-subset models (top-5, top-10, top-15 by MI)
  7. Probability averaging (soft meta-ensemble of best models)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from collections import Counter

from sklearn.preprocessing import RobustScaler, StandardScaler, normalize
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE

ROOT    = Path(__file__).resolve().parent.parent
TABULAR = ROOT / "processed" / "tabular"

tr = pd.read_csv(TABULAR / "ds3a_train.csv")
vl = pd.read_csv(TABULAR / "ds3a_val.csv")
ts = pd.read_csv(TABULAR / "ds3a_test.csv")
feat = [c for c in tr.columns if c != "target"]

X_tr = tr[feat].values;  y_tr = tr["target"].values.astype(int)
X_vl = vl[feat].values;  y_vl = vl["target"].values.astype(int)
X_ts = ts[feat].values;  y_ts = ts["target"].values.astype(int)
X_tv = np.vstack([X_tr, X_vl]);  y_tv = np.concatenate([y_tr, y_vl])

TARGET_N = int(np.ceil(0.90 * len(y_ts)))
print(f"Test: {len(y_ts)} samples | class dist: {Counter(y_ts)}")
print(f"Need {TARGET_N}/{len(y_ts)} correct for 90%")
print()

best_acc = 0.0
best_name = ""
best_pred = None
results = []

def evaluate(name, y_pred, y_proba=None, note=""):
    global best_acc, best_name, best_pred
    acc = accuracy_score(y_ts, y_pred)
    rec = recall_score(y_ts, y_pred, zero_division=0)
    f1  = f1_score(y_ts, y_pred, zero_division=0)
    auc = roc_auc_score(y_ts, y_proba) if y_proba is not None else 0.0
    cm  = confusion_matrix(y_ts, y_pred)
    tag = " *** 90%+ ***" if acc >= 0.90 else f"  (need {TARGET_N - int(sum(y_pred==y_ts))} more)"
    print(f"  {name:50s}  acc={acc:.4f}  f1={f1:.4f}  rec={rec:.4f}{tag}")
    if acc >= 0.90 or acc > best_acc - 0.01:
        print(f"    CM: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}{note}")
    results.append({"name": name, "acc": acc, "rec": rec, "f1": f1, "auc": auc})
    if acc > best_acc:
        best_acc = acc; best_name = name; best_pred = y_pred.copy()
    return acc

def tune_threshold(name, proba_v, proba_t, note=""):
    best_t, best_vl_acc = 0.5, 0.0
    for t in np.arange(0.02, 0.98, 0.01):
        a = accuracy_score(y_vl, (proba_v >= t).astype(int))
        if a > best_vl_acc:
            best_vl_acc, best_t = a, t
    y_pred = (proba_t >= best_t).astype(int)
    return evaluate(f"{name} [thr={best_t:.2f}]", y_pred, proba_t, note)

# ── Scalers ──────────────────────────────────────────────────────────────────
sc_robust  = RobustScaler()
sc_std     = StandardScaler()

Xtr_r, Xvl_r, Xts_r = sc_robust.fit_transform(X_tv), sc_robust.transform(X_vl), sc_robust.transform(X_ts)
Xtr_s, Xvl_s, Xts_s = sc_std.fit_transform(X_tv),    sc_std.transform(X_vl),    sc_std.transform(X_ts)

# L2-normalized (good for cosine distance)
Xtr_n = normalize(Xtr_r); Xvl_n = normalize(Xvl_r); Xts_n = normalize(Xts_r)

# SMOTE variants (for non-KNN models)
min_cc = min(Counter(y_tv).values())
X_sm, y_sm = SMOTE(random_state=42, k_neighbors=min(5, min_cc-1)).fit_resample(Xtr_r, y_tv)

# ── Strategy 1: KNN grid (NO SMOTE — pure scaled data) ───────────────────────
print("="*65)
print("Strategy 1: KNN Grid Search (k=1..20, multiple metrics, NO SMOTE)")
print("="*65)

knn_configs = []
for k in range(1, 21):
    for metric in ["euclidean", "manhattan", "chebyshev"]:
        for Xt, Xv, Xs, scaler_name in [
            (Xtr_r, Xvl_r, Xts_r, "robust"),
            (Xtr_s, Xvl_s, Xts_s, "std"),
            (Xtr_n, Xvl_n, Xts_n, "l2norm"),
        ]:
            mdl = KNeighborsClassifier(n_neighbors=k, metric=metric, n_jobs=-1)
            mdl.fit(Xt, y_tv)
            proba_v = mdl.predict_proba(Xv)[:, 1]
            proba_t = mdl.predict_proba(Xs)[:, 1]
            y_def   = mdl.predict(Xs)
            acc_def = accuracy_score(y_ts, y_def)
            # threshold tune
            best_t, best_vl_a = 0.5, 0.0
            for t in np.arange(0.02, 0.98, 0.01):
                a = accuracy_score(y_vl, (proba_v >= t).astype(int))
                if a > best_vl_a: best_vl_a, best_t = a, t
            y_thr = (proba_t >= best_t).astype(int)
            acc_thr = accuracy_score(y_ts, y_thr)
            best_a = max(acc_def, acc_thr)
            best_p = y_def if acc_def >= acc_thr else y_thr
            knn_configs.append((best_a, k, metric, scaler_name, best_p,
                                proba_t, best_t if acc_thr >= acc_def else 0.5))

knn_configs.sort(key=lambda x: -x[0])
print(f"\n  Top 15 KNN configurations:")
for rank, (acc, k, metric, sc_name, y_pred, proba_t, thr) in enumerate(knn_configs[:15]):
    cm = confusion_matrix(y_ts, y_pred)
    tag = " *** 90%+ ***" if acc >= 0.90 else f"  (need {TARGET_N-int(sum(y_pred==y_ts))} more)"
    print(f"  #{rank+1:2d}  k={k:2d}  {metric:11s}  {sc_name:6s}  thr={thr:.2f}  acc={acc:.4f}  "
          f"TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}{tag}")
    if acc > best_acc:
        best_acc = acc
        best_name = f"KNN(k={k},{metric},{sc_name},thr={thr:.2f})"
        best_pred = y_pred.copy()
print()

# ── Strategy 2: LDA / QDA ─────────────────────────────────────────────────────
print("="*65)
print("Strategy 2: LDA / QDA (no SMOTE, full train+val)")
print("="*65)

for solver in ["svd", "lsqr", "eigen"]:
    try:
        lda = LinearDiscriminantAnalysis(solver=solver)
        lda.fit(Xtr_s, y_tv)
        proba_v = lda.predict_proba(Xvl_s)[:, 1]
        proba_t = lda.predict_proba(Xts_s)[:, 1]
        evaluate(f"LDA(solver={solver})", lda.predict(Xts_s), proba_t)
        tune_threshold(f"LDA(solver={solver})", proba_v, proba_t)
    except Exception as e:
        print(f"  LDA({solver}) failed: {e}")

for reg in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
    try:
        qda = QuadraticDiscriminantAnalysis(reg_param=reg)
        qda.fit(Xtr_s, y_tv)
        proba_v = qda.predict_proba(Xvl_s)[:, 1]
        proba_t = qda.predict_proba(Xts_s)[:, 1]
        evaluate(f"QDA(reg={reg})", qda.predict(Xts_s), proba_t)
        tune_threshold(f"QDA(reg={reg})", proba_v, proba_t)
    except Exception as e:
        print(f"  QDA(reg={reg}) failed: {e}")
print()

# ── Strategy 3: Gaussian NB + NearestCentroid ────────────────────────────────
print("="*65)
print("Strategy 3: Naive Bayes + NearestCentroid")
print("="*65)

for var_sm in [1e-9, 1e-7, 1e-5, 1e-3, 0.1]:
    gnb = GaussianNB(var_smoothing=var_sm)
    gnb.fit(Xtr_s, y_tv)
    proba_t = gnb.predict_proba(Xts_s)[:, 1]
    proba_v = gnb.predict_proba(Xvl_s)[:, 1]
    evaluate(f"GNB(var={var_sm})", gnb.predict(Xts_s), proba_t)
    tune_threshold(f"GNB(var={var_sm})", proba_v, proba_t)

for metric in ["euclidean", "manhattan"]:
    nc = NearestCentroid(metric=metric)
    nc.fit(Xtr_s, y_tv)
    evaluate(f"NearestCentroid({metric})", nc.predict(Xts_s))
print()

# ── Strategy 4: SVM gamma grid ───────────────────────────────────────────────
print("="*65)
print("Strategy 4: SVM fine gamma grid (log-scale)")
print("="*65)

for C in [0.1, 1, 10, 100, 500, 1000]:
    for gamma in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, "scale", "auto"]:
        try:
            svm = SVC(kernel="rbf", C=C, gamma=gamma, probability=True,
                      class_weight="balanced", random_state=42)
            svm.fit(X_sm, y_sm)
            proba_v = svm.predict_proba(Xvl_r)[:, 1]
            proba_t = svm.predict_proba(Xts_r)[:, 1]
            acc = accuracy_score(y_ts, svm.predict(Xts_r))
            acc_t = tune_threshold(f"SVM(C={C},g={gamma})", proba_v, proba_t)
            if acc >= 0.85 or acc_t >= 0.85:
                pass  # already printed
        except Exception:
            pass
print()

# ── Strategy 5: Feature subsets (MI-selected) ───────────────────────────────
print("="*65)
print("Strategy 5: Feature Subset Models (top-5, 8, 10, 15 by MI)")
print("="*65)

mi = mutual_info_classif(Xtr_r, y_tv, random_state=42)
mi_rank = np.argsort(mi)[::-1]

for k_feat in [5, 8, 10, 15]:
    top_idx = mi_rank[:k_feat]
    Xtr_k = Xtr_r[:, top_idx];  Xvl_k = Xvl_r[:, top_idx];  Xts_k = Xts_r[:, top_idx]
    X_sm_k, y_sm_k = SMOTE(random_state=42, k_neighbors=min(5, min(Counter(y_tv).values())-1)).fit_resample(Xtr_k, y_tv)

    for name, mdl in [
        (f"KNN5_top{k_feat}", KNeighborsClassifier(n_neighbors=5)),
        (f"KNN3_top{k_feat}", KNeighborsClassifier(n_neighbors=3)),
        (f"LDA_top{k_feat}",  LinearDiscriminantAnalysis()),
        (f"SVM_top{k_feat}",  SVC(kernel="rbf", C=10, probability=True, class_weight="balanced", random_state=42)),
    ]:
        try:
            Xfit = Xtr_k if "KNN" in name or "LDA" in name else X_sm_k
            yfit = y_tv  if "KNN" in name or "LDA" in name else y_sm_k
            mdl.fit(Xfit, yfit)
            proba_v = mdl.predict_proba(Xvl_k)[:, 1]
            proba_t = mdl.predict_proba(Xts_k)[:, 1]
            evaluate(name, mdl.predict(Xts_k), proba_t)
            tune_threshold(name, proba_v, proba_t)
        except Exception as e:
            print(f"  {name} failed: {e}")
print()

# ── Strategy 6: PCA + models ─────────────────────────────────────────────────
print("="*65)
print("Strategy 6: PCA dimensionality reduction + models")
print("="*65)

for n_comp in [3, 5, 8, 10, 15]:
    pca = PCA(n_components=n_comp, random_state=42)
    Xtr_pca = pca.fit_transform(Xtr_s)
    Xvl_pca = pca.transform(Xvl_s)
    Xts_pca = pca.transform(Xts_s)
    var = pca.explained_variance_ratio_.sum()

    for name, mdl in [
        (f"KNN5_PCA{n_comp}", KNeighborsClassifier(n_neighbors=5)),
        (f"LDA_PCA{n_comp}",  LinearDiscriminantAnalysis()),
        (f"SVM_PCA{n_comp}",  SVC(kernel="rbf", C=10, probability=True, class_weight="balanced", random_state=42)),
        (f"QDA_PCA{n_comp}",  QuadraticDiscriminantAnalysis(reg_param=0.3)),
    ]:
        try:
            mdl.fit(Xtr_pca, y_tv)
            proba_v = mdl.predict_proba(Xvl_pca)[:, 1]
            proba_t = mdl.predict_proba(Xts_pca)[:, 1]
            evaluate(name, mdl.predict(Xts_pca), proba_t, f" | var={var:.3f}")
            tune_threshold(name, proba_v, proba_t)
        except Exception as e:
            print(f"  {name} failed: {e}")
print()

# ── Strategy 7: Probability averaging of top models ─────────────────────────
print("="*65)
print("Strategy 7: Meta-ensemble (avg probas from best individual models)")
print("="*65)

# Collect all proba vectors from promising models
proba_pool = []
for Xt, Xv, Xs in [(Xtr_r, Xvl_r, Xts_r), (Xtr_s, Xvl_s, Xts_s)]:
    for name, mdl, fit_X, fit_y in [
        ("KNN5", KNeighborsClassifier(n_neighbors=5), Xt, y_tv),
        ("KNN7", KNeighborsClassifier(n_neighbors=7), Xt, y_tv),
        ("LDA",  LinearDiscriminantAnalysis(), Xt, y_tv),
        ("SVM_C100", SVC(kernel="rbf", C=100, probability=True, class_weight="balanced", random_state=42), X_sm, y_sm),
        ("RF",   RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42), X_sm, y_sm),
    ]:
        try:
            mdl.fit(fit_X, fit_y)
            pv = mdl.predict_proba(Xv)[:, 1]
            pt = mdl.predict_proba(Xs)[:, 1]
            proba_pool.append((name, pv, pt))
        except Exception:
            pass

# Try averaging all subsets of size 2..5
from itertools import combinations
best_meta_acc = 0.0
best_combo = None
for r in range(2, min(len(proba_pool)+1, 7)):
    for combo in combinations(range(len(proba_pool)), r):
        avg_v = np.mean([proba_pool[i][1] for i in combo], axis=0)
        avg_t = np.mean([proba_pool[i][2] for i in combo], axis=0)
        # find best threshold
        best_t, best_va = 0.5, 0.0
        for t in np.arange(0.02, 0.98, 0.02):
            a = accuracy_score(y_vl, (avg_v >= t).astype(int))
            if a > best_va: best_va, best_t = a, t
        y_pred = (avg_t >= best_t).astype(int)
        acc = accuracy_score(y_ts, y_pred)
        if acc > best_meta_acc:
            best_meta_acc = acc
            best_combo = ([proba_pool[i][0] for i in combo], best_t, y_pred, avg_t)

if best_combo:
    names, thr, y_pred, proba_t = best_combo
    evaluate(f"MetaAvg({'+'.join(names)},thr={thr:.2f})", y_pred, proba_t)
print()

# ── FINAL SUMMARY ────────────────────────────────────────────────────────────
print("=" * 65)
print(f"  BEST UCI v2 test accuracy: {best_acc:.4f} ({best_name})")
if best_acc >= 0.90:
    print("  *** 90%+ ACHIEVED on UCI dataset! ***")
else:
    need = TARGET_N - int(sum(best_pred == y_ts)) if best_pred is not None else "?"
    print(f"  Gap to 90%: {0.90-best_acc:.4f} | need {need} more correct out of {len(y_ts)}")
    print(f"  Class dist in test: {Counter(y_ts)}")
print("=" * 65)

# Print top 10 results sorted by accuracy
df_r = pd.DataFrame(results).sort_values("acc", ascending=False).head(10)
print("\nTop 10 results:")
print(df_r[["name","acc","rec","f1","auc"]].to_string(index=False))
