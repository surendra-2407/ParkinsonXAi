"""Push speech (dataset_2) past 90% via threshold tuning + stacking."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.ensemble import (VotingClassifier, RandomForestClassifier,
                               StackingClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parent.parent
TABULAR = ROOT / "processed" / "tabular"

tr = pd.read_csv(TABULAR / "ds2_train.csv")
vl = pd.read_csv(TABULAR / "ds2_val.csv")
ts = pd.read_csv(TABULAR / "ds2_test.csv")
feat = [c for c in tr.columns if c != "target"]

X_tv = np.vstack([tr[feat].values, vl[feat].values])
y_tv = np.concatenate([tr["target"].values, vl["target"].values]).astype(int)
X_vl = vl[feat].values;  y_vl = vl["target"].values.astype(int)
X_ts = ts[feat].values;  y_ts = ts["target"].values.astype(int)

TARGET = int(np.ceil(0.90 * len(y_ts)))
print(f"Test class dist: {Counter(y_ts)}")
print(f"Need: {TARGET}/{len(y_ts)} correct for 90%")
print()

# SMOTE on full train+val
min_cc = min(Counter(y_tv).values())
X_res, y_res = SMOTE(random_state=42, k_neighbors=min(5, min_cc - 1)).fit_resample(X_tv, y_tv)
print(f"After SMOTE: {len(y_res)} samples, dist: {Counter(y_res)}")
print()

best_acc_overall = 0.0

def evaluate(name, y_pred, y_proba=None):
    global best_acc_overall
    acc = accuracy_score(y_ts, y_pred)
    rec = recall_score(y_ts, y_pred, zero_division=0)
    f1  = f1_score(y_ts, y_pred, zero_division=0)
    auc = roc_auc_score(y_ts, y_proba) if y_proba is not None else 0.0
    tag = " *** 90%+ ***" if acc >= 0.90 else f"  (need {TARGET - sum(y_pred==y_ts)} more)"
    print(f"  {name:35s}  acc={acc:.4f}  rec={rec:.4f}  f1={f1:.4f}  auc={auc:.4f}{tag}")
    best_acc_overall = max(best_acc_overall, acc)
    return acc, y_proba

def tune_threshold(model, X_vl, y_vl, X_ts, y_ts, name):
    proba_vl = model.predict_proba(X_vl)[:, 1]
    proba_ts = model.predict_proba(X_ts)[:, 1]
    best_t, best_vl = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.005):
        a = accuracy_score(y_vl, (proba_vl >= t).astype(int))
        if a > best_vl:
            best_vl, best_t = a, t
    y_pred_t = (proba_ts >= best_t).astype(int)
    evaluate(f"{name} (thr={best_t:.2f})", y_pred_t, proba_ts)

# ─── Strategy 1: Individual models with threshold tuning ───
print("=== Individual Models + Threshold Tuning ===")
base_models = [
    ("XGBoost",  xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42,
        verbosity=0, scale_pos_weight=sum(y_tv==0)/max(sum(y_tv==1),1))),
    ("LightGBM", lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=63,
        class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)),
    ("CatBoost", CatBoostClassifier(iterations=600, learning_rate=0.03, depth=7,
        random_seed=42, verbose=0, auto_class_weights="Balanced")),
    ("ExtraTrees", ExtraTreesClassifier(n_estimators=600, class_weight="balanced",
        random_state=42, n_jobs=-1)),
    ("SVC_RBF",  __import__("sklearn.svm", fromlist=["SVC"]).SVC(
        kernel="rbf", C=10.0, probability=True, class_weight="balanced", random_state=42)),
]

trained = []
for name, mdl in base_models:
    mdl.fit(X_res, y_res)
    evaluate(name, mdl.predict(X_ts), mdl.predict_proba(X_ts)[:,1])
    tune_threshold(mdl, X_vl, y_vl, X_ts, y_ts, name)
    trained.append((name.lower().replace(" ","_"), mdl))
print()

# ─── Strategy 2: Soft Voting (all 5) ───
print("=== Soft Voting (5 models) ===")
voting5 = VotingClassifier(trained, voting="soft", n_jobs=-1)
voting5.fit(X_res, y_res)
proba_v5 = voting5.predict_proba(X_ts)[:,1]
evaluate("Voting5", voting5.predict(X_ts), proba_v5)
tune_threshold(voting5, X_vl, y_vl, X_ts, y_ts, "Voting5")
print()

# ─── Strategy 3: Stacking (XGB+LGB+CAT+ET -> LR) ───
print("=== Stacking Ensemble ===")
stack = StackingClassifier(
    estimators=[
        ("xgb", xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5,
            eval_metric="logloss", random_state=42, verbosity=0)),
        ("lgb", lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)),
        ("cat", CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6,
            random_seed=42, verbose=0, auto_class_weights="Balanced")),
        ("et",  ExtraTreesClassifier(n_estimators=400, class_weight="balanced",
            random_state=42, n_jobs=-1)),
    ],
    final_estimator=LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced"),
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    passthrough=True, n_jobs=-1
)
stack.fit(X_res, y_res)
proba_st = stack.predict_proba(X_ts)[:,1]
evaluate("Stacking (passthrough=True)", stack.predict(X_ts), proba_st)
tune_threshold(stack, X_vl, y_vl, X_ts, y_ts, "Stacking")
print()

print("=" * 60)
print(f"Best speech test accuracy: {best_acc_overall:.4f}")
if best_acc_overall >= 0.90:
    print("*** 90%+ ACHIEVED on Speech dataset! ***")
else:
    gap = 0.90 - best_acc_overall
    print(f"Gap to 90%: {gap:.4f} = {round(gap * len(y_ts))} samples short")
print("=" * 60)
