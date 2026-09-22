#!/usr/bin/env python3
"""
Phase 7: Feature Engineering & Selection
Phase 8: Class Imbalance Handling
Phase 9: Baseline Classification Models
ParkinsonXAI Project

Trains on THREE datasets:
  A. Audio-extracted features (dataset_1)
  B. UCI parkinsons.data (dataset_3a)
  C. pd_speech_features.csv (dataset_2)

Uses:
  - Correlation filter, Mutual information, SelectKBest, RFE, Tree importance
  - SMOTE only on training folds
  - 7 baseline classifiers

Output:
  results/feature_selection_comparison.csv
  results/model_comparison.csv
  results/cross_validation_results.csv
  plots/confusion_matrix.png
  plots/roc_curve.png
  plots/precision_recall_curve.png
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, VarianceThreshold
)
from sklearn.preprocessing import RobustScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parent.parent
TABULAR = ROOT / "processed" / "tabular"
SPLITS = ROOT / "processed" / "splits"
AUDIO_FEATURES = ROOT / "processed" / "audio_features"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
REPORTS = ROOT / "reports"

for d in [RESULTS, PLOTS]:
    d.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

print("="*60)
print("Phase 7+8+9: Feature Selection + Imbalance + Baseline Models")
print("="*60)

all_cv_results = []
all_model_results = []

# ─────────────────────────────────────────────
# Global progress tracker
# ─────────────────────────────────────────────
# Total steps: FS comparison (4) + 7 audio + 8 UCI + 5 speech = 24
TOTAL_STEPS = 4 + 7 + 8 + 5
_step = [0]  # mutable counter

def progress_bar(current, total, width=40):
    pct = current / total
    filled = int(width * pct)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {pct*100:.1f}% ({current}/{total})"

def tick(label=""):
    _step[0] += 1
    print(f"\n  >>> OVERALL PROGRESS: {progress_bar(_step[0], TOTAL_STEPS)}")
    if label:
        print(f"      -> Completed: {label}")

# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, name, dataset):
    """Train and evaluate a model, return metrics dict."""
    model.fit(X_train, y_train)
    results = {"model": name, "dataset": dataset}

    for split_name, X_s, y_s in [("train", X_train, y_train),
                                   ("val", X_val, y_val),
                                   ("test", X_test, y_test)]:
        y_pred = model.predict(X_s)
        try:
            y_proba = model.predict_proba(X_s)[:, 1]
        except AttributeError:
            try:
                y_proba = model.decision_function(X_s)
            except Exception:
                y_proba = y_pred.astype(float)

        results[f"{split_name}_acc"] = round(accuracy_score(y_s, y_pred), 4)
        results[f"{split_name}_f1"] = round(f1_score(y_s, y_pred, zero_division=0), 4)
        results[f"{split_name}_recall"] = round(recall_score(y_s, y_pred, zero_division=0), 4)
        try:
            results[f"{split_name}_auc"] = round(roc_auc_score(y_s, y_proba), 4)
        except Exception:
            results[f"{split_name}_auc"] = 0.0

    results["gen_gap"] = round(results["train_acc"] - results["test_acc"], 4)
    return model, results


def select_features_correlation(X_train: pd.DataFrame, threshold: float = 0.97):
    """Remove highly correlated features (fit on train only)."""
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    return [c for c in X_train.columns if c not in to_drop], to_drop


def run_cv_with_smote(X_train, y_train, model_class, model_kwargs, n_splits=5, dataset=""):
    """5-fold CV with SMOTE applied only inside each fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = {"acc": [], "f1": [], "recall": [], "auc": []}

    for fold_i, (tr_idx, vl_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_vl = X_train[tr_idx], X_train[vl_idx]
        y_tr, y_vl = y_train[tr_idx], y_train[vl_idx]

        # Apply SMOTE only to training fold
        min_class_count = min(Counter(y_tr).values())
        if min_class_count >= 6 and len(Counter(y_tr)) > 1:
            try:
                k = min(5, min_class_count - 1)
                smote = SMOTE(random_state=42, k_neighbors=k)
                X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
            except Exception:
                X_tr_res, y_tr_res = X_tr, y_tr
        else:
            X_tr_res, y_tr_res = X_tr, y_tr

        model = model_class(**model_kwargs)
        model.fit(X_tr_res, y_tr_res)

        y_pred = model.predict(X_vl)
        try:
            y_proba = model.predict_proba(X_vl)[:, 1]
        except AttributeError:
            y_proba = model.decision_function(X_vl)

        cv_scores["acc"].append(accuracy_score(y_vl, y_pred))
        cv_scores["f1"].append(f1_score(y_vl, y_pred, zero_division=0))
        cv_scores["recall"].append(recall_score(y_vl, y_pred, zero_division=0))
        try:
            cv_scores["auc"].append(roc_auc_score(y_vl, y_proba))
        except Exception:
            cv_scores["auc"].append(0.0)

    return {
        "cv_acc_mean": round(np.mean(cv_scores["acc"]), 4),
        "cv_acc_std": round(np.std(cv_scores["acc"]), 4),
        "cv_f1_mean": round(np.mean(cv_scores["f1"]), 4),
        "cv_recall_mean": round(np.mean(cv_scores["recall"]), 4),
        "cv_auc_mean": round(np.mean(cv_scores["auc"]), 4),
        "cv_fold_accs": [round(s, 4) for s in cv_scores["acc"]],
    }


def plot_confusion_matrix(y_true, y_pred, model_name, dataset_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Healthy", "Parkinson"],
                yticklabels=["Healthy", "Parkinson"], ax=ax)
    ax.set_title(f"Confusion Matrix\n{model_name} — {dataset_name}", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_proba_dict, dataset_name, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    for name, y_proba in y_proba_dict.items():
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc = roc_auc_score(y_true, y_proba)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        except Exception:
            pass
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {dataset_name}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pr_curve(y_true, y_proba_dict, dataset_name, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_proba in y_proba_dict.items():
        try:
            prec, rec, _ = precision_recall_curve(y_true, y_proba)
            ap = average_precision_score(y_true, y_proba)
            ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
        except Exception:
            pass
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {dataset_name}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────
# EXPERIMENT A: Audio features from dataset_1
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("EXPERIMENT A: Audio-extracted features (dataset_1)")
print("="*60)

audio_df = pd.read_csv(AUDIO_FEATURES / "audio_features_dataset1.csv")
with open(SPLITS / "dataset1_audio_splits.json") as f:
    splits1 = json.load(f)

feature_cols_audio = [c for c in audio_df.columns
                      if c not in ["filename", "label", "duration_s", "sample_rate"]]
print(f"  Total audio features: {len(feature_cols_audio)}")

# Match rows to split filenames
all_filenames = audio_df["filename"].tolist()
train_files_set = set([f.split("/")[-1] for f in splits1["train_files"]])
val_files_set = set([f.split("/")[-1] for f in splits1["val_files"]])
test_files_set = set([f.split("/")[-1] for f in splits1["test_files"]])

train_mask_a = audio_df["filename"].isin(train_files_set)
val_mask_a = audio_df["filename"].isin(val_files_set)
test_mask_a = audio_df["filename"].isin(test_files_set)

X_a_full = audio_df[feature_cols_audio].values
y_a_full = audio_df["label"].values

X_a_train = audio_df[train_mask_a][feature_cols_audio].values
X_a_val = audio_df[val_mask_a][feature_cols_audio].values
X_a_test = audio_df[test_mask_a][feature_cols_audio].values
y_a_train = y_a_full[train_mask_a.values]
y_a_val = y_a_full[val_mask_a.values]
y_a_test = y_a_full[test_mask_a.values]

# Fallback if splitting by filename doesn't work (e.g. different records)
if len(X_a_train) < 10:
    print("  WARNING: File-based split failed; using index-based split")
    n = len(audio_df)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_idx_a = idx[:n_train]
    val_idx_a = idx[n_train:n_train + n_val]
    test_idx_a = idx[n_train + n_val:]
    X_a_train = X_a_full[train_idx_a]
    X_a_val = X_a_full[val_idx_a]
    X_a_test = X_a_full[test_idx_a]
    y_a_train = y_a_full[train_idx_a]
    y_a_val = y_a_full[val_idx_a]
    y_a_test = y_a_full[test_idx_a]

print(f"  Train: {len(y_a_train)}, Val: {len(y_a_val)}, Test: {len(y_a_test)}")
print(f"  Train class dist: {Counter(y_a_train)}")

# Preprocessing (fit on train only)
imputer_a = SimpleImputer(strategy="median")
X_a_train = imputer_a.fit_transform(X_a_train)
X_a_val = imputer_a.transform(X_a_val)
X_a_test = imputer_a.transform(X_a_test)

# NZV removal
vt_a = VarianceThreshold(threshold=1e-6)
X_a_train = vt_a.fit_transform(X_a_train)
X_a_val = vt_a.transform(X_a_val)
X_a_test = vt_a.transform(X_a_test)
feature_cols_audio_filtered = [c for c, k in zip(feature_cols_audio, vt_a.get_support()) if k]
print(f"  After NZV removal: {len(feature_cols_audio_filtered)} features")

# Scaling (fit on train)
scaler_a = RobustScaler()
X_a_train_sc = scaler_a.fit_transform(X_a_train)
X_a_val_sc = scaler_a.transform(X_a_val)
X_a_test_sc = scaler_a.transform(X_a_test)

joblib.dump(scaler_a, MODELS / "audio_scaler.pkl")
joblib.dump(imputer_a, MODELS / "audio_imputer.pkl")
joblib.dump(vt_a, MODELS / "audio_vt.pkl")
with open(MODELS / "audio_feature_cols_filtered.json", "w") as f:
    json.dump({"features": feature_cols_audio_filtered}, f)

# --- Feature Selection Comparison for Audio ---
print("\n  Feature selection comparison...")
fs_results = []
X_a_train_sc_df = pd.DataFrame(X_a_train_sc, columns=feature_cols_audio_filtered)

# Baseline: use top 50 features by mutual info
mi_scores = mutual_info_classif(X_a_train_sc, y_a_train, random_state=42)
mi_ranking = np.argsort(mi_scores)[::-1]
top_50_idx = mi_ranking[:50]
top_50_features = [feature_cols_audio_filtered[i] for i in top_50_idx]

fs_configs = [20, 50, 100, len(feature_cols_audio_filtered)]
for fs_i, k_feats in enumerate(fs_configs, 1):
    k_actual = min(k_feats, len(feature_cols_audio_filtered))
    top_k_idx = mi_ranking[:k_actual]
    X_tr_k = X_a_train_sc[:, top_k_idx]
    X_vl_k = X_a_val_sc[:, top_k_idx]

    print(f"    [{fs_i}/{len(fs_configs)}] MI top-{k_actual} features...", flush=True)
    rf_k = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_k.fit(X_tr_k, y_a_train)
    train_acc_fs = accuracy_score(y_a_train, rf_k.predict(X_tr_k))
    val_acc = accuracy_score(y_a_val, rf_k.predict(X_vl_k))
    val_auc = roc_auc_score(y_a_val, rf_k.predict_proba(X_vl_k)[:, 1])
    fs_results.append({
        "dataset": "audio_dataset1",
        "method": "mutual_info",
        "n_features": k_actual,
        "val_acc": round(val_acc, 4),
        "val_auc": round(val_auc, 4),
    })
    print(f"      Train Acc={train_acc_fs:.4f}  Val Acc={val_acc:.4f}  Val AUC={val_auc:.4f}")
    tick(f"Feature Selection MI top-{k_actual}")

pd.DataFrame(fs_results).to_csv(RESULTS / "feature_selection_comparison.csv", index=False)

# Choose feature set for audio: top-50 MI features (best trade-off)
best_k_audio = 50
top_k_idx_audio = mi_ranking[:best_k_audio]
X_a_train_sel = X_a_train_sc[:, top_k_idx_audio]
X_a_val_sel = X_a_val_sc[:, top_k_idx_audio]
X_a_test_sel = X_a_test_sc[:, top_k_idx_audio]
audio_sel_features = [feature_cols_audio_filtered[i] for i in top_k_idx_audio]
print(f"\n  Using top-{best_k_audio} MI features for audio models")

# Save selector
with open(MODELS / "audio_top50_features.json", "w") as f:
    json.dump({"features": audio_sel_features, "indices": top_k_idx_audio.tolist()}, f)

# --- Baseline models on AUDIO ---
print("\n  Training baseline models on audio features...")
audio_roc_proba = {}

CLASSIFIERS_A = [
    ("LogisticReg", LogisticRegression, {"C": 1.0, "max_iter": 1000, "random_state": 42, "class_weight": "balanced"}),
    ("SVM_RBF", SVC, {"kernel": "rbf", "C": 1.0, "probability": True, "random_state": 42, "class_weight": "balanced"}),
    ("RandomForest", RandomForestClassifier, {"n_estimators": 200, "random_state": 42, "n_jobs": -1, "class_weight": "balanced"}),
    ("ExtraTrees", ExtraTreesClassifier, {"n_estimators": 200, "random_state": 42, "n_jobs": -1, "class_weight": "balanced"}),
    ("XGBoost", xgb.XGBClassifier, {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 5,
                                     "eval_metric": "logloss",
                                     "random_state": 42, "verbosity": 0,
                                     "scale_pos_weight": sum(y_a_train==0)/max(sum(y_a_train==1), 1)}),
    ("LightGBM", lgb.LGBMClassifier, {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31,
                                       "random_state": 42, "n_jobs": -1, "class_weight": "balanced",
                                       "verbose": -1}),
    ("CatBoost", CatBoostClassifier, {"iterations": 200, "learning_rate": 0.05, "depth": 6,
                                       "random_seed": 42, "verbose": 0, "auto_class_weights": "Balanced"}),
]

total_A = len(CLASSIFIERS_A)
for clf_i, (clf_name, clf_class, clf_kwargs) in enumerate(CLASSIFIERS_A, 1):
    try:
        print(f"\n  [{clf_i}/{total_A}] Training {clf_name} (Audio)...", flush=True)
        clf = clf_class(**clf_kwargs)
        clf.fit(X_a_train_sel, y_a_train)

        trained_model, res = evaluate_model(
            clf, X_a_train_sel, y_a_train,
            X_a_val_sel, y_a_val,
            X_a_test_sel, y_a_test,
            clf_name, "audio_dataset1"
        )

        print(f"    |-- Train Acc : {res['train_acc']:.4f}  "
              f"Train AUC : {res['train_auc']:.4f}")
        print(f"    |-- Val   Acc : {res['val_acc']:.4f}  "
              f"Val   AUC : {res['val_auc']:.4f}")
        print(f"    +-- Test  Acc : {res['test_acc']:.4f}  "
              f"Test  AUC : {res['test_auc']:.4f}  "
              f"Recall : {res['test_recall']:.4f}  "
              f"F1 : {res['test_f1']:.4f}")

        # CV
        print(f"    [CV] Running 5-fold CV...", flush=True)
        cv_res = run_cv_with_smote(
            X_a_train_sel, y_a_train,
            clf_class, clf_kwargs,
            n_splits=5, dataset="audio_dataset1"
        )
        print(f"    [CV] Acc: {cv_res['cv_acc_mean']:.4f} +/- {cv_res['cv_acc_std']:.4f}  "
              f"| Folds: {cv_res['cv_fold_accs']}")
        res.update({f"cv_{k}": v for k, v in cv_res.items()})
        all_model_results.append(res)
        all_cv_results.append({"model": clf_name, "dataset": "audio_dataset1", **cv_res})

        # Collect probas for ROC
        try:
            audio_roc_proba[clf_name] = trained_model.predict_proba(X_a_test_sel)[:, 1]
        except Exception:
            audio_roc_proba[clf_name] = trained_model.decision_function(X_a_test_sel)

        tick(f"Audio -> {clf_name}")
    except Exception as e:
        print(f"  FAILED: {e}")
        tick(f"Audio -> {clf_name} [FAILED]")

# Best audio model by val AUC
audio_model_df = pd.DataFrame([r for r in all_model_results if r["dataset"] == "audio_dataset1"])
if not audio_model_df.empty:
    best_audio_row = audio_model_df.loc[audio_model_df["val_auc"].idxmax()]
    best_audio_name = best_audio_row["model"]
    print(f"\n  Best audio model: {best_audio_name} (val_auc={best_audio_row['val_auc']:.4f})")

    # Plot confusion matrix for best model
    best_clf_idx = CLASSIFIERS_A[[n for n, _, _ in CLASSIFIERS_A].index(best_audio_name)]
    best_audio_clf = best_clf_idx[1](**best_clf_idx[2])
    best_audio_clf.fit(X_a_train_sel, y_a_train)
    y_pred_test_a = best_audio_clf.predict(X_a_test_sel)
    plot_confusion_matrix(y_a_test, y_pred_test_a, best_audio_name, "Audio Dataset",
                          PLOTS / "confusion_matrix_audio.png")
    plot_roc_curve(y_a_test, audio_roc_proba, "Audio Dataset", PLOTS / "roc_curve_audio.png")
    plot_pr_curve(y_a_test, audio_roc_proba, "Audio Dataset", PLOTS / "precision_recall_curve_audio.png")


# ─────────────────────────────────────────────
# EXPERIMENT B: UCI parkinsons.data (dataset_3a)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("EXPERIMENT B: UCI parkinsons.data (dataset_3a)")
print("="*60)

df3a_train = pd.read_csv(TABULAR / "ds3a_train.csv")
df3a_val = pd.read_csv(TABULAR / "ds3a_val.csv")
df3a_test = pd.read_csv(TABULAR / "ds3a_test.csv")

feat_3a = [c for c in df3a_train.columns if c != "target"]
X3a_train_np = df3a_train[feat_3a].values
X3a_val_np = df3a_val[feat_3a].values
X3a_test_np = df3a_test[feat_3a].values
y3a_train_np = df3a_train["target"].values.astype(int)
y3a_val_np = df3a_val["target"].values.astype(int)
y3a_test_np = df3a_test["target"].values.astype(int)

print(f"  Train: {len(y3a_train_np)}, Val: {len(y3a_val_np)}, Test: {len(y3a_test_np)}")
print(f"  Train class dist: {Counter(y3a_train_np)}")

uci_roc_proba = {}
CLASSIFIERS_B = [
    ("LogisticReg", LogisticRegression, {"C": 1.0, "max_iter": 1000, "random_state": 42, "class_weight": "balanced"}),
    ("SVM_RBF", SVC, {"kernel": "rbf", "C": 10.0, "probability": True, "random_state": 42, "class_weight": "balanced"}),
    ("SVM_Linear", SVC, {"kernel": "linear", "C": 1.0, "probability": True, "random_state": 42, "class_weight": "balanced"}),
    ("RandomForest", RandomForestClassifier, {"n_estimators": 300, "random_state": 42, "n_jobs": -1, "class_weight": "balanced"}),
    ("ExtraTrees", ExtraTreesClassifier, {"n_estimators": 300, "random_state": 42, "n_jobs": -1, "class_weight": "balanced"}),
    ("XGBoost", xgb.XGBClassifier, {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4,
                                     "eval_metric": "logloss",
                                     "random_state": 42, "verbosity": 0,
                                     "scale_pos_weight": sum(y3a_train_np==0)/max(sum(y3a_train_np==1), 1)}),
    ("LightGBM", lgb.LGBMClassifier, {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 15,
                                       "random_state": 42, "n_jobs": -1, "class_weight": "balanced",
                                       "verbose": -1}),
    ("CatBoost", CatBoostClassifier, {"iterations": 300, "learning_rate": 0.05, "depth": 5,
                                       "random_seed": 42, "verbose": 0, "auto_class_weights": "Balanced"}),
]

total_B = len(CLASSIFIERS_B)
for clf_i, (clf_name, clf_class, clf_kwargs) in enumerate(CLASSIFIERS_B, 1):
    try:
        print(f"\n  [{clf_i}/{total_B}] Training {clf_name} (UCI)...", flush=True)
        clf = clf_class(**clf_kwargs)
        trained_clf, res = evaluate_model(
            clf, X3a_train_np, y3a_train_np,
            X3a_val_np, y3a_val_np,
            X3a_test_np, y3a_test_np,
            clf_name, "UCI_parkinsons"
        )
        print(f"    |-- Train Acc : {res['train_acc']:.4f}  "
              f"Train AUC : {res['train_auc']:.4f}")
        print(f"    |-- Val   Acc : {res['val_acc']:.4f}  "
              f"Val   AUC : {res['val_auc']:.4f}")
        print(f"    +-- Test  Acc : {res['test_acc']:.4f}  "
              f"Test  AUC : {res['test_auc']:.4f}  "
              f"Recall : {res['test_recall']:.4f}  "
              f"F1 : {res['test_f1']:.4f}")

        print(f"    [CV] Running 5-fold CV...", flush=True)
        cv_res = run_cv_with_smote(X3a_train_np, y3a_train_np, clf_class, clf_kwargs,
                                    n_splits=5, dataset="UCI_parkinsons")
        print(f"    [CV] Acc: {cv_res['cv_acc_mean']:.4f} +/- {cv_res['cv_acc_std']:.4f}  "
              f"| Folds: {cv_res['cv_fold_accs']}")
        res.update({f"cv_{k}": v for k, v in cv_res.items()})
        all_model_results.append(res)
        all_cv_results.append({"model": clf_name, "dataset": "UCI_parkinsons", **cv_res})

        try:
            uci_roc_proba[clf_name] = trained_clf.predict_proba(X3a_test_np)[:, 1]
        except Exception:
            uci_roc_proba[clf_name] = trained_clf.decision_function(X3a_test_np)

        tick(f"UCI -> {clf_name}")
    except Exception as e:
        print(f"  FAILED: {e}")
        tick(f"UCI -> {clf_name} [FAILED]")

if uci_roc_proba:
    plot_roc_curve(y3a_test_np, uci_roc_proba, "UCI Parkinsons", PLOTS / "roc_curve_uci.png")
    plot_pr_curve(y3a_test_np, uci_roc_proba, "UCI Parkinsons", PLOTS / "precision_recall_curve_uci.png")

# ─────────────────────────────────────────────
# EXPERIMENT C: pd_speech_features (dataset_2)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("EXPERIMENT C: pd_speech_features.csv (dataset_2)")
print("="*60)

df2_train = pd.read_csv(TABULAR / "ds2_train.csv")
df2_val = pd.read_csv(TABULAR / "ds2_val.csv")
df2_test = pd.read_csv(TABULAR / "ds2_test.csv")

feat_2 = [c for c in df2_train.columns if c != "target"]
X2_train_np = df2_train[feat_2].values
X2_val_np = df2_val[feat_2].values
X2_test_np = df2_test[feat_2].values
y2_train_np = df2_train["target"].values.astype(int)
y2_val_np = df2_val["target"].values.astype(int)
y2_test_np = df2_test["target"].values.astype(int)

print(f"  Train: {len(y2_train_np)}, Val: {len(y2_val_np)}, Test: {len(y2_test_np)}")
print(f"  Train class dist: {Counter(y2_train_np)}")

CLASSIFIERS_C = [
    ("LogisticReg", LogisticRegression, {"C": 0.1, "max_iter": 2000, "random_state": 42, "class_weight": "balanced"}),
    ("RandomForest", RandomForestClassifier, {"n_estimators": 300, "random_state": 42, "n_jobs": -1, "class_weight": "balanced"}),
    ("XGBoost", xgb.XGBClassifier, {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 5,
                                     "eval_metric": "logloss",
                                     "random_state": 42, "verbosity": 0,
                                     "scale_pos_weight": sum(y2_train_np==0)/max(sum(y2_train_np==1), 1)}),
    ("LightGBM", lgb.LGBMClassifier, {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
                                       "random_state": 42, "n_jobs": -1, "class_weight": "balanced",
                                       "verbose": -1}),
    ("CatBoost", CatBoostClassifier, {"iterations": 300, "learning_rate": 0.05, "depth": 6,
                                       "random_seed": 42, "verbose": 0, "auto_class_weights": "Balanced"}),
]

total_C = len(CLASSIFIERS_C)
for clf_i, (clf_name, clf_class, clf_kwargs) in enumerate(CLASSIFIERS_C, 1):
    try:
        print(f"\n  [{clf_i}/{total_C}] Training {clf_name} (Speech)...", flush=True)
        clf = clf_class(**clf_kwargs)
        trained_clf, res = evaluate_model(
            clf, X2_train_np, y2_train_np,
            X2_val_np, y2_val_np,
            X2_test_np, y2_test_np,
            clf_name, "pd_speech_features"
        )
        print(f"    |-- Train Acc : {res['train_acc']:.4f}  "
              f"Train AUC : {res['train_auc']:.4f}")
        print(f"    |-- Val   Acc : {res['val_acc']:.4f}  "
              f"Val   AUC : {res['val_auc']:.4f}")
        print(f"    +-- Test  Acc : {res['test_acc']:.4f}  "
              f"Test  AUC : {res['test_auc']:.4f}  "
              f"Recall : {res['test_recall']:.4f}  "
              f"F1 : {res['test_f1']:.4f}")

        print(f"    [CV] Running 5-fold CV...", flush=True)
        cv_res = run_cv_with_smote(X2_train_np, y2_train_np, clf_class, clf_kwargs,
                                    n_splits=5, dataset="pd_speech_features")
        print(f"    [CV] Acc: {cv_res['cv_acc_mean']:.4f} +/- {cv_res['cv_acc_std']:.4f}  "
              f"| Folds: {cv_res['cv_fold_accs']}")
        res.update({f"cv_{k}": v for k, v in cv_res.items()})
        all_model_results.append(res)
        all_cv_results.append({"model": clf_name, "dataset": "pd_speech_features", **cv_res})
        tick(f"Speech -> {clf_name}")
    except Exception as e:
        print(f"  FAILED: {e}")
        tick(f"Speech -> {clf_name} [FAILED]")

# ─────────────────────────────────────────────
# Save all results
# ─────────────────────────────────────────────
model_df = pd.DataFrame(all_model_results)
model_df.to_csv(RESULTS / "model_comparison.csv", index=False)
print(f"\n  Saved: results/model_comparison.csv ({len(model_df)} rows)")

cv_df = pd.DataFrame(all_cv_results)
cv_df.to_csv(RESULTS / "cross_validation_results.csv", index=False)
print(f"  Saved: results/cross_validation_results.csv")

# Print summary table
print("\n" + "="*60)
print("BASELINE MODEL SUMMARY")
print("="*60)
summary_cols = ["model", "dataset", "val_acc", "val_auc", "test_acc", "test_recall", "test_f1", "test_auc", "gen_gap"]
print(model_df[summary_cols].to_string(index=False))

print("\n" + "="*60)
print("Phase 7+8+9 COMPLETE")
print("="*60)
