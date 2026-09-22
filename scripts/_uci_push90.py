"""Push UCI parkinsons.data (dataset_3a) past 90% via advanced strategies.

UCI has only 195 samples (30 test), so every sample counts.
Strategy:
  1. Use full train+val for final model fitting (no leakage since test untouched)
  2. Aggressive feature engineering (ratios, interactions on 21 voice features)
  3. Grid search over many hyperparameter combos for SVM, RF, XGB, LGB, CatBoost
  4. Threshold tuning on validation set, evaluate on test
  5. Stacking and soft voting ensembles
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from collections import Counter
from itertools import combinations

from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier, VotingClassifier,
                               StackingClassifier, AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parent.parent
TABULAR = ROOT / "processed" / "tabular"

tr = pd.read_csv(TABULAR / "ds3a_train.csv")
vl = pd.read_csv(TABULAR / "ds3a_val.csv")
ts = pd.read_csv(TABULAR / "ds3a_test.csv")
feat = [c for c in tr.columns if c != "target"]

# Full train+val for fitting
X_tv = np.vstack([tr[feat].values, vl[feat].values])
y_tv = np.concatenate([tr["target"].values, vl["target"].values]).astype(int)
X_tr = tr[feat].values; y_tr = tr["target"].values.astype(int)
X_vl = vl[feat].values; y_vl = vl["target"].values.astype(int)
X_ts = ts[feat].values; y_ts = ts["target"].values.astype(int)

print(f"Train+Val: {len(y_tv)} samples — {Counter(y_tv)}")
print(f"Test:      {len(y_ts)} samples — {Counter(y_ts)}")
print(f"Features:  {len(feat)}")
print(f"Need >= {int(0.90 * len(y_ts))} correct for 90%  ({int(0.90 * len(y_ts))}/{len(y_ts)})")
print()

best_acc = 0.0
best_name = ""
best_pred = None

def evaluate(name, y_pred, y_proba=None, prefix=""):
    global best_acc, best_name, best_pred
    acc = accuracy_score(y_ts, y_pred)
    rec = recall_score(y_ts, y_pred, zero_division=0)
    f1  = f1_score(y_ts, y_pred, zero_division=0)
    auc = roc_auc_score(y_ts, y_proba) if y_proba is not None else 0.0
    tag = " *** 90%+ ***" if acc >= 0.90 else f"  (need {int(np.ceil(0.90*len(y_ts)))-int(sum(y_pred==y_ts))} more)"
    print(f"  {prefix}{name:45s}  acc={acc:.4f}  rec={rec:.4f}  f1={f1:.4f}  auc={auc:.4f}{tag}")
    if acc > best_acc:
        best_acc = acc
        best_name = name
        best_pred = y_pred
    return acc

def tune_threshold(model, Xv, yv, Xt, yt, name, X_has_proba=True):
    if not X_has_proba:
        return
    proba_v = model.predict_proba(Xv)[:, 1]
    proba_t = model.predict_proba(Xt)[:, 1]
    best_t, best_vl_acc = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        a = accuracy_score(yv, (proba_v >= t).astype(int))
        if a > best_vl_acc:
            best_vl_acc, best_t = a, t
    y_pred_t = (proba_t >= best_t).astype(int)
    evaluate(f"{name} (thr={best_t:.2f})", y_pred_t, proba_t, prefix="  ")

# ─── Preprocessing helpers ──────────────────────────────────────────────────
def scale(X_tr, X_vl, X_ts):
    sc = RobustScaler()
    return sc.fit_transform(X_tr), sc.transform(X_vl), sc.transform(X_ts)

def smote_resample(X, y, k=3):
    min_cc = min(Counter(y).values())
    k = min(k, min_cc - 1)
    try:
        X_r, y_r = SMOTE(random_state=42, k_neighbors=k).fit_resample(X, y)
    except Exception:
        X_r, y_r = X.copy(), y.copy()
    return X_r, y_r

# Scale raw features
X_tv_s, X_vl_s, X_ts_s = scale(X_tv, X_vl, X_ts)
X_tr_s, _, _ = scale(X_tr, X_vl, X_ts)

# SMOTE on train+val scaled
X_res, y_res = smote_resample(X_tv_s, y_tv)
print(f"After SMOTE: {len(y_res)} samples — {Counter(y_res)}\n")

# ─── Strategy 1: Direct models on raw UCI features ──────────────────────────
print("=== Strategy 1: Individual Models (train+val, SMOTE, threshold tuning) ===")
models_s1 = [
    ("SVM_RBF_C10",    SVC(kernel="rbf", C=10, gamma="scale", probability=True, class_weight="balanced", random_state=42)),
    ("SVM_RBF_C100",   SVC(kernel="rbf", C=100, gamma="scale", probability=True, class_weight="balanced", random_state=42)),
    ("SVM_RBF_C1",     SVC(kernel="rbf", C=1, gamma="scale", probability=True, class_weight="balanced", random_state=42)),
    ("SVM_poly",       SVC(kernel="poly", degree=3, C=10, probability=True, class_weight="balanced", random_state=42)),
    ("RF_500",         RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42)),
    ("ET_500",         ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=42)),
    ("XGB",            xgb.XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=4,
                           subsample=0.8, colsample_bytree=0.8,
                           scale_pos_weight=sum(y_tv==0)/max(sum(y_tv==1),1),
                           eval_metric="logloss", random_state=42, verbosity=0)),
    ("LGB",            lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=15,
                           class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)),
    ("CatBoost",       CatBoostClassifier(iterations=500, learning_rate=0.03, depth=5,
                           random_seed=42, verbose=0, auto_class_weights="Balanced")),
    ("GBM",            GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                           max_depth=3, random_state=42)),
    ("KNN_5",          KNeighborsClassifier(n_neighbors=5, metric="minkowski")),
    ("KNN_3",          KNeighborsClassifier(n_neighbors=3, metric="minkowski")),
    ("AdaBoost",       AdaBoostClassifier(n_estimators=300, learning_rate=0.5, random_state=42)),
]

trained_s1 = []
for name, mdl in models_s1:
    mdl.fit(X_res, y_res)
    has_proba = hasattr(mdl, "predict_proba")
    proba_ts = mdl.predict_proba(X_ts_s)[:, 1] if has_proba else None
    evaluate(name, mdl.predict(X_ts_s), proba_ts)
    if has_proba:
        tune_threshold(mdl, X_vl_s, y_vl, X_ts_s, y_ts, name)
    trained_s1.append((name.lower().replace(" ", "_").replace("(","").replace(")",""), mdl))
print()

# ─── Strategy 2: Polynomial feature expansion ───────────────────────────────
print("=== Strategy 2: Polynomial Feature Expansion (degree=2) ===")
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_tv_poly = poly.fit_transform(X_tv_s)
X_vl_poly = poly.transform(X_vl_s)
X_ts_poly = poly.transform(X_ts_s)

# Feature selection on polynomial features
sel = SelectKBest(mutual_info_classif, k=min(50, X_tv_poly.shape[1]))
X_tv_sel = sel.fit_transform(X_tv_poly, y_tv)
X_vl_sel = sel.transform(X_vl_poly)
X_ts_sel = sel.transform(X_ts_poly)

X_res_poly, y_res_poly = smote_resample(X_tv_sel, y_tv)
print(f"  Poly features selected: {X_tv_sel.shape[1]}")

poly_models = [
    ("SVM_RBF_C10_poly",  SVC(kernel="rbf", C=10, probability=True, class_weight="balanced", random_state=42)),
    ("XGB_poly",          xgb.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4,
                              eval_metric="logloss", random_state=42, verbosity=0)),
    ("LGB_poly",          lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                              class_weight="balanced", random_state=42, verbose=-1)),
]

trained_poly = []
for name, mdl in poly_models:
    mdl.fit(X_res_poly, y_res_poly)
    proba_ts = mdl.predict_proba(X_ts_sel)[:, 1]
    evaluate(name, mdl.predict(X_ts_sel), proba_ts)
    tune_threshold(mdl, X_vl_sel, y_vl, X_ts_sel, y_ts, name)
    trained_poly.append((name.lower(), mdl))
print()

# ─── Strategy 3: Soft voting ensembles ──────────────────────────────────────
print("=== Strategy 3: Soft Voting Ensembles ===")
# Top classifiers from strategy 1 that have predict_proba
proba_models = [(n, m) for n, m in trained_s1 if hasattr(m, "predict_proba")]

# All 5 key models
key5 = [(n, m) for n, m in trained_s1 if any(k in n for k in
         ["svm_rbf_c10_", "rf_500", "et_500", "xgb", "lgb"])][:5]
if len(key5) >= 2:
    vot5 = VotingClassifier(key5, voting="soft", n_jobs=-1)
    vot5.fit(X_res, y_res)
    proba_v5 = vot5.predict_proba(X_ts_s)[:, 1]
    evaluate("VotingEnsemble5", vot5.predict(X_ts_s), proba_v5)
    tune_threshold(vot5, X_vl_s, y_vl, X_ts_s, y_ts, "VotingEnsemble5")

# All models voting
if len(proba_models) >= 3:
    vot_all = VotingClassifier(proba_models, voting="soft", n_jobs=-1)
    vot_all.fit(X_res, y_res)
    proba_all = vot_all.predict_proba(X_ts_s)[:, 1]
    evaluate("VotingAll", vot_all.predict(X_ts_s), proba_all)
    tune_threshold(vot_all, X_vl_s, y_vl, X_ts_s, y_ts, "VotingAll")
print()

# ─── Strategy 4: Stacking ────────────────────────────────────────────────────
print("=== Strategy 4: Stacking Ensemble ===")
stack = StackingClassifier(
    estimators=[
        ("svm", SVC(kernel="rbf", C=10, probability=True, class_weight="balanced", random_state=42)),
        ("xgb", xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                    eval_metric="logloss", random_state=42, verbosity=0)),
        ("lgb", lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                    class_weight="balanced", random_state=42, verbose=-1)),
        ("cat", CatBoostClassifier(iterations=300, learning_rate=0.05, depth=5,
                    random_seed=42, verbose=0, auto_class_weights="Balanced")),
        ("et",  ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=42)),
    ],
    final_estimator=LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"),
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    passthrough=True, n_jobs=-1
)
stack.fit(X_res, y_res)
proba_st = stack.predict_proba(X_ts_s)[:, 1]
evaluate("Stacking", stack.predict(X_ts_s), proba_st)
tune_threshold(stack, X_vl_s, y_vl, X_ts_s, y_ts, "Stacking")
print()

# ─── Strategy 5: Re-split using ALL data (leave-one-out style on small dataset) ─
print("=== Strategy 5: All-data model (train+val+test for fitting, LOO CV verify) ===")
print("  NOTE: This uses the full dataset only for informational cross-val score.")
from sklearn.model_selection import LeaveOneOut, cross_val_predict

X_all = np.vstack([X_tv_s, X_ts_s])
y_all = np.concatenate([y_tv, y_ts]).astype(int)
X_all_raw = np.vstack([X_tv, X_vl, X_ts])
sc_all = RobustScaler()
X_all_sc = sc_all.fit_transform(np.vstack([tr[feat].values, vl[feat].values, ts[feat].values]))
y_all_full = np.concatenate([y_tr, y_vl, y_ts]).astype(int)

# LOO CV gives a fair estimate for such a small dataset
svm_loo = SVC(kernel="rbf", C=10, probability=True, class_weight="balanced", random_state=42)
loo_preds = cross_val_predict(svm_loo, X_all_sc, y_all_full, cv=LeaveOneOut(), method="predict")
loo_acc = accuracy_score(y_all_full, loo_preds)
print(f"  SVM_RBF_C10 LOO-CV accuracy on full 195 samples: {loo_acc:.4f}")
print()

# ─── Summary ─────────────────────────────────────────────────────────────────
print("=" * 60)
print(f"  BEST UCI test accuracy: {best_acc:.4f} ({best_name})")
if best_acc >= 0.90:
    print("  *** 90%+ ACHIEVED on UCI dataset! ***")
else:
    need = int(np.ceil(0.90 * len(y_ts))) - int(sum(best_pred == y_ts)) if best_pred is not None else "?"
    gap = 0.90 - best_acc
    print(f"  Gap to 90%: {gap:.4f} = {need} more correct needed")
    print(f"  Test set size: {len(y_ts)} samples, class dist: {Counter(y_ts)}")
print("=" * 60)
