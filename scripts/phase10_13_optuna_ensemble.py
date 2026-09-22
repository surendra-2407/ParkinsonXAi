#!/usr/bin/env python3
"""
Phase 10: Hyperparameter Optimization with Optuna
Phase 11: Cross-Validation Reporting
Phase 12: Overfitting Analysis
Phase 13: Ensemble Learning
ParkinsonXAI Project

Optimizes the strongest models across all three experiments.
Produces tuned final models.

Outputs:
  models/detection_best_model.pkl
  models/detection_feature_selector.pkl
  models/detection_label_encoder.pkl
  results/hyperparameter_results.csv
  results/final_test_results.csv
  plots/training_vs_validation.png
  plots/learning_curve.png
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               VotingClassifier, StackingClassifier)
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
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

np.random.seed(42)

print("="*60)
print("Phase 10+11+12+13: Optuna + CV + Overfitting + Ensemble")
print("="*60)

# ---------------------------------------------
# Load preprocessed datasets
# ---------------------------------------------

# Audio dataset
audio_df = pd.read_csv(AUDIO_FEATURES / "audio_features_dataset1.csv")
with open(SPLITS / "dataset1_audio_splits.json") as f:
    splits1 = json.load(f)

feature_cols_audio = [c for c in audio_df.columns
                      if c not in ["filename", "label", "duration_s", "sample_rate"]]
y_audio = audio_df["label"].values
train_files_set = set([f.split("/")[-1] for f in splits1["train_files"]])
val_files_set = set([f.split("/")[-1] for f in splits1["val_files"]])
test_files_set = set([f.split("/")[-1] for f in splits1["test_files"]])

train_m = audio_df["filename"].isin(train_files_set)
val_m = audio_df["filename"].isin(val_files_set)
test_m = audio_df["filename"].isin(test_files_set)

if train_m.sum() < 10:
    n = len(audio_df)
    idx = np.arange(n); np.random.shuffle(idx)
    n_train = int(0.70 * n); n_val = int(0.15 * n)
    train_m = pd.Series([False]*n)
    val_m = pd.Series([False]*n)
    test_m = pd.Series([False]*n)
    train_m.iloc[idx[:n_train]] = True
    val_m.iloc[idx[n_train:n_train+n_val]] = True
    test_m.iloc[idx[n_train+n_val:]] = True

X_a_tr_raw = audio_df[train_m][feature_cols_audio].values
X_a_vl_raw = audio_df[val_m][feature_cols_audio].values
X_a_ts_raw = audio_df[test_m][feature_cols_audio].values
y_a_tr = y_audio[train_m.values]; y_a_vl = y_audio[val_m.values]; y_a_ts = y_audio[test_m.values]

imp_a = SimpleImputer(strategy="median"); X_a_tr_raw = imp_a.fit_transform(X_a_tr_raw)
X_a_vl_raw = imp_a.transform(X_a_vl_raw); X_a_ts_raw = imp_a.transform(X_a_ts_raw)
vt_a = VarianceThreshold(1e-6); X_a_tr_raw = vt_a.fit_transform(X_a_tr_raw)
X_a_vl_raw = vt_a.transform(X_a_vl_raw); X_a_ts_raw = vt_a.transform(X_a_ts_raw)
scl_a = RobustScaler(); X_a_tr = scl_a.fit_transform(X_a_tr_raw)
X_a_vl = scl_a.transform(X_a_vl_raw); X_a_ts = scl_a.transform(X_a_ts_raw)
feat_a_filtered = [c for c, k in zip(feature_cols_audio, vt_a.get_support()) if k]

# Top-50 MI features for audio
mi_scores_a = mutual_info_classif(X_a_tr, y_a_tr, random_state=42)
top50_a = np.argsort(mi_scores_a)[::-1][:50]
X_a_tr_sel = X_a_tr[:, top50_a]; X_a_vl_sel = X_a_vl[:, top50_a]; X_a_ts_sel = X_a_ts[:, top50_a]

# UCI dataset
df3a_train = pd.read_csv(TABULAR / "ds3a_train.csv")
df3a_val = pd.read_csv(TABULAR / "ds3a_val.csv")
df3a_test = pd.read_csv(TABULAR / "ds3a_test.csv")
feat_3a = [c for c in df3a_train.columns if c != "target"]
X_u_tr = df3a_train[feat_3a].values; y_u_tr = df3a_train["target"].values.astype(int)
X_u_vl = df3a_val[feat_3a].values; y_u_vl = df3a_val["target"].values.astype(int)
X_u_ts = df3a_test[feat_3a].values; y_u_ts = df3a_test["target"].values.astype(int)

print(f"  Audio: train={len(y_a_tr)}, val={len(y_a_vl)}, test={len(y_a_ts)}")
print(f"  UCI:   train={len(y_u_tr)}, val={len(y_u_vl)}, test={len(y_u_ts)}")

# ---------------------------------------------
# Optuna optimization
# ---------------------------------------------
N_TRIALS = 30   # Reduced for speed; timeout per study is the hard cap
STUDY_TIMEOUT = 120  # seconds per Optuna study

def cv_score_with_smote(X, y, model, n_splits=3):  # 3-fold for speed
    """Return mean CV AUC using StratifiedKFold with optional SMOTE."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for tr_idx, vl_idx in skf.split(X, y):
        X_tr, X_vl = X[tr_idx], X[vl_idx]
        y_tr, y_vl = y[tr_idx], y[vl_idx]
        min_cc = min(Counter(y_tr).values())
        if min_cc >= 6:
            try:
                k = min(5, min_cc - 1)
                X_tr, y_tr = SMOTE(random_state=42, k_neighbors=k).fit_resample(X_tr, y_tr)
            except Exception:
                pass
        model.fit(X_tr, y_tr)
        try:
            scores.append(roc_auc_score(y_vl, model.predict_proba(X_vl)[:, 1]))
        except Exception:
            scores.append(0.0)
    return np.mean(scores)

hp_results = []

# -- XGBoost on AUDIO --
print("\n[Optuna] XGBoost on Audio features...")
def objective_xgb_audio(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        "eval_metric": "logloss",
        "random_state": 42, "verbosity": 0,
        "scale_pos_weight": sum(y_a_tr==0)/max(sum(y_a_tr==1), 1),
    }
    return cv_score_with_smote(X_a_tr_sel, y_a_tr, xgb.XGBClassifier(**params))

study_xgb_a = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_xgb_a.optimize(objective_xgb_audio, n_trials=N_TRIALS, timeout=STUDY_TIMEOUT, show_progress_bar=False)
best_xgb_a = study_xgb_a.best_params
best_xgb_a.update({"eval_metric": "logloss", "random_state": 42,
                   "verbosity": 0, "scale_pos_weight": sum(y_a_tr==0)/max(sum(y_a_tr==1), 1)})
print(f"  Best CV AUC: {study_xgb_a.best_value:.4f}")
hp_results.append({"model": "XGBoost_Audio", "best_cv_auc": study_xgb_a.best_value,
                   **study_xgb_a.best_params})

# -- LightGBM on AUDIO --
print("[Optuna] LightGBM on Audio features...")
def objective_lgb_audio(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 100),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        "class_weight": "balanced", "random_state": 42, "n_jobs": -1, "verbose": -1,
    }
    return cv_score_with_smote(X_a_tr_sel, y_a_tr, lgb.LGBMClassifier(**params))

study_lgb_a = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_lgb_a.optimize(objective_lgb_audio, n_trials=N_TRIALS, timeout=STUDY_TIMEOUT, show_progress_bar=False)
best_lgb_a = study_lgb_a.best_params
best_lgb_a.update({"class_weight": "balanced", "random_state": 42, "n_jobs": -1, "verbose": -1})
print(f"  Best CV AUC: {study_lgb_a.best_value:.4f}")
hp_results.append({"model": "LightGBM_Audio", "best_cv_auc": study_lgb_a.best_value,
                   **study_lgb_a.best_params})

# -- RandomForest on AUDIO --
print("[Optuna] RandomForest on Audio features...")
def objective_rf_audio(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_float("max_features", 0.2, 1.0),
        "class_weight": "balanced", "random_state": 42, "n_jobs": -1,
    }
    return cv_score_with_smote(X_a_tr_sel, y_a_tr, RandomForestClassifier(**params))

study_rf_a = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_rf_a.optimize(objective_rf_audio, n_trials=N_TRIALS, timeout=STUDY_TIMEOUT, show_progress_bar=False)
best_rf_a = study_rf_a.best_params
best_rf_a.update({"class_weight": "balanced", "random_state": 42, "n_jobs": -1})
print(f"  Best CV AUC: {study_rf_a.best_value:.4f}")
hp_results.append({"model": "RF_Audio", "best_cv_auc": study_rf_a.best_value, **study_rf_a.best_params})

# -- XGBoost on UCI --
print("[Optuna] XGBoost on UCI parkinsons.data...")
def objective_xgb_uci(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "eval_metric": "logloss", "random_state": 42, "verbosity": 0,
        "scale_pos_weight": sum(y_u_tr==0)/max(sum(y_u_tr==1), 1),
    }
    return cv_score_with_smote(X_u_tr, y_u_tr, xgb.XGBClassifier(**params))

study_xgb_u = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_xgb_u.optimize(objective_xgb_uci, n_trials=N_TRIALS, timeout=STUDY_TIMEOUT, show_progress_bar=False)
best_xgb_u = study_xgb_u.best_params
best_xgb_u.update({"eval_metric": "logloss", "random_state": 42,
                   "verbosity": 0, "scale_pos_weight": sum(y_u_tr==0)/max(sum(y_u_tr==1), 1)})
print(f"  Best CV AUC: {study_xgb_u.best_value:.4f}")
hp_results.append({"model": "XGBoost_UCI", "best_cv_auc": study_xgb_u.best_value,
                   **study_xgb_u.best_params})

# -- SVM on UCI --
print("[Optuna] SVM on UCI parkinsons.data...")
def objective_svm_uci(trial):
    kernel = trial.suggest_categorical("kernel", ["rbf", "linear"])
    C = trial.suggest_float("C", 0.01, 100, log=True)
    gamma = trial.suggest_float("gamma", 1e-5, 10, log=True) if kernel == "rbf" else "scale"
    model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True,
                random_state=42, class_weight="balanced")
    return cv_score_with_smote(X_u_tr, y_u_tr, model)

study_svm_u = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_svm_u.optimize(objective_svm_uci, n_trials=N_TRIALS, timeout=STUDY_TIMEOUT, show_progress_bar=False)
best_svm_u = study_svm_u.best_params
best_svm_u.update({"probability": True, "random_state": 42, "class_weight": "balanced"})
print(f"  Best CV AUC: {study_svm_u.best_value:.4f}")
hp_results.append({"model": "SVM_UCI", "best_cv_auc": study_svm_u.best_value, **study_svm_u.best_params})

# ── Speech dataset (dataset_2) ──
df2_train = pd.read_csv(TABULAR / "ds2_train.csv")
df2_val   = pd.read_csv(TABULAR / "ds2_val.csv")
df2_test  = pd.read_csv(TABULAR / "ds2_test.csv")
feat_2 = [c for c in df2_train.columns if c != "target"]
X_s_tr = df2_train[feat_2].values;  y_s_tr = df2_train["target"].values.astype(int)
X_s_vl = df2_val[feat_2].values;    y_s_vl = df2_val["target"].values.astype(int)
X_s_ts = df2_test[feat_2].values;   y_s_ts = df2_test["target"].values.astype(int)
print(f"  Speech: train={len(y_s_tr)}, val={len(y_s_vl)}, test={len(y_s_ts)}")

print("[Optuna] XGBoost on Speech features...")
def objective_xgb_speech(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma":             trial.suggest_float("gamma", 0, 5),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        "eval_metric": "logloss", "random_state": 42, "verbosity": 0,
        "scale_pos_weight": sum(y_s_tr==0)/max(sum(y_s_tr==1), 1),
    }
    return cv_score_with_smote(X_s_tr, y_s_tr, xgb.XGBClassifier(**params))

study_xgb_s = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_xgb_s.optimize(objective_xgb_speech, n_trials=N_TRIALS, timeout=STUDY_TIMEOUT, show_progress_bar=False)
best_xgb_s = study_xgb_s.best_params
best_xgb_s.update({"eval_metric": "logloss", "random_state": 42, "verbosity": 0,
                   "scale_pos_weight": sum(y_s_tr==0)/max(sum(y_s_tr==1), 1)})
print(f"  Best CV AUC: {study_xgb_s.best_value:.4f}")
hp_results.append({"model": "XGBoost_Speech", "best_cv_auc": study_xgb_s.best_value, **study_xgb_s.best_params})

print("[Optuna] LightGBM on Speech features...")
def objective_lgb_speech(trial):
    params = {
        "n_estimators":       trial.suggest_int("n_estimators", 100, 600),
        "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves":         trial.suggest_int("num_leaves", 15, 120),
        "max_depth":          trial.suggest_int("max_depth", 3, 10),
        "min_child_samples":  trial.suggest_int("min_child_samples", 5, 50),
        "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha":          trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
        "reg_lambda":         trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        "class_weight": "balanced", "random_state": 42, "n_jobs": -1, "verbose": -1,
    }
    return cv_score_with_smote(X_s_tr, y_s_tr, lgb.LGBMClassifier(**params))

study_lgb_s = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_lgb_s.optimize(objective_lgb_speech, n_trials=N_TRIALS, timeout=STUDY_TIMEOUT, show_progress_bar=False)
best_lgb_s = study_lgb_s.best_params
best_lgb_s.update({"class_weight": "balanced", "random_state": 42, "n_jobs": -1, "verbose": -1})
print(f"  Best CV AUC: {study_lgb_s.best_value:.4f}")
hp_results.append({"model": "LightGBM_Speech", "best_cv_auc": study_lgb_s.best_value, **study_lgb_s.best_params})

print("[Optuna] CatBoost on Speech features...")
def objective_cat_speech(trial):
    params = {
        "iterations":    trial.suggest_int("iterations", 100, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth":         trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
        "random_seed": 42, "verbose": 0, "auto_class_weights": "Balanced",
    }
    return cv_score_with_smote(X_s_tr, y_s_tr, CatBoostClassifier(**params))

study_cat_s = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_cat_s.optimize(objective_cat_speech, n_trials=N_TRIALS, timeout=STUDY_TIMEOUT, show_progress_bar=False)
best_cat_s = study_cat_s.best_params
best_cat_s.update({"random_seed": 42, "verbose": 0, "auto_class_weights": "Balanced"})
print(f"  Best CV AUC: {study_cat_s.best_value:.4f}")
hp_results.append({"model": "CatBoost_Speech", "best_cv_auc": study_cat_s.best_value, **study_cat_s.best_params})

# UCI: combine train+val for better generalization on tiny dataset
X_u_trval = np.vstack([X_u_tr, X_u_vl])
y_u_trval = np.concatenate([y_u_tr, y_u_vl])
print(f"  UCI train+val combined: {len(y_u_trval)} samples for final model")

pd.DataFrame(hp_results).to_csv(RESULTS / "hyperparameter_results.csv", index=False)
print("\n  Saved: results/hyperparameter_results.csv")

# ---------------------------------------------
# Train tuned models and evaluate
# ---------------------------------------------
print("\n" + "="*60)
print("Training Tuned Models")
print("="*60)

final_results = []

def eval_tuned(model, X_tr, y_tr, X_vl, y_vl, X_ts, y_ts, name, dataset):
    model.fit(X_tr, y_tr)
    res = {"model": name, "dataset": dataset}
    for split, X_s, y_s in [("train", X_tr, y_tr), ("val", X_vl, y_vl), ("test", X_ts, y_ts)]:
        y_pred = model.predict(X_s)
        try:
            y_proba = model.predict_proba(X_s)[:, 1]
        except Exception:
            y_proba = model.decision_function(X_s)
        res[f"{split}_acc"] = round(accuracy_score(y_s, y_pred), 4)
        res[f"{split}_f1"] = round(f1_score(y_s, y_pred, zero_division=0), 4)
        res[f"{split}_recall"] = round(recall_score(y_s, y_pred, zero_division=0), 4)
        try:
            res[f"{split}_auc"] = round(roc_auc_score(y_s, y_proba), 4)
        except Exception:
            res[f"{split}_auc"] = 0.0
    res["gen_gap"] = round(res["train_acc"] - res["test_acc"], 4)
    return model, res

# Tuned models — AUDIO
print("\n[Audio] Tuned XGBoost:")
xgb_a_tuned = xgb.XGBClassifier(**best_xgb_a)
xgb_a_tuned, r = eval_tuned(xgb_a_tuned, X_a_tr_sel, y_a_tr, X_a_vl_sel, y_a_vl, X_a_ts_sel, y_a_ts,
                              "XGBoost_Tuned", "audio_dataset1")
final_results.append(r)
print(f"  test_acc={r['test_acc']:.4f}, recall={r['test_recall']:.4f}, auc={r['test_auc']:.4f}, gap={r['gen_gap']:.4f}")

print("[Audio] Tuned LightGBM:")
lgb_a_tuned = lgb.LGBMClassifier(**best_lgb_a)
lgb_a_tuned, r = eval_tuned(lgb_a_tuned, X_a_tr_sel, y_a_tr, X_a_vl_sel, y_a_vl, X_a_ts_sel, y_a_ts,
                              "LightGBM_Tuned", "audio_dataset1")
final_results.append(r)
print(f"  test_acc={r['test_acc']:.4f}, recall={r['test_recall']:.4f}, auc={r['test_auc']:.4f}, gap={r['gen_gap']:.4f}")

print("[Audio] Tuned RandomForest:")
rf_a_tuned = RandomForestClassifier(**best_rf_a)
rf_a_tuned, r = eval_tuned(rf_a_tuned, X_a_tr_sel, y_a_tr, X_a_vl_sel, y_a_vl, X_a_ts_sel, y_a_ts,
                             "RF_Tuned", "audio_dataset1")
final_results.append(r)
print(f"  test_acc={r['test_acc']:.4f}, recall={r['test_recall']:.4f}, auc={r['test_auc']:.4f}, gap={r['gen_gap']:.4f}")

# Tuned models — UCI  (train on combined train+val for better test generalization)
print("\n[UCI] Tuned XGBoost (train+val combined):")
xgb_u_tuned = xgb.XGBClassifier(**best_xgb_u)
xgb_u_tuned, r = eval_tuned(xgb_u_tuned, X_u_trval, y_u_trval, X_u_vl, y_u_vl, X_u_ts, y_u_ts,
                              "XGBoost_Tuned", "UCI_parkinsons")
final_results.append(r)
print(f"  test_acc={r['test_acc']:.4f}, recall={r['test_recall']:.4f}, auc={r['test_auc']:.4f}, gap={r['gen_gap']:.4f}")

print("[UCI] Tuned SVM (train+val combined):")
svm_u_tuned = SVC(**best_svm_u)
svm_u_tuned, r = eval_tuned(svm_u_tuned, X_u_trval, y_u_trval, X_u_vl, y_u_vl, X_u_ts, y_u_ts,
                              "SVM_Tuned", "UCI_parkinsons")
final_results.append(r)
print(f"  test_acc={r['test_acc']:.4f}, recall={r['test_recall']:.4f}, auc={r['test_auc']:.4f}, gap={r['gen_gap']:.4f}")

# Tuned models — Speech (dataset_2)
print("\n[Speech] Tuned XGBoost:")
xgb_s_tuned = xgb.XGBClassifier(**best_xgb_s)
xgb_s_tuned, r = eval_tuned(xgb_s_tuned, X_s_tr, y_s_tr, X_s_vl, y_s_vl, X_s_ts, y_s_ts,
                              "XGBoost_Tuned", "pd_speech_features")
final_results.append(r)
print(f"  test_acc={r['test_acc']:.4f}, recall={r['test_recall']:.4f}, auc={r['test_auc']:.4f}, gap={r['gen_gap']:.4f}")

print("[Speech] Tuned LightGBM:")
lgb_s_tuned = lgb.LGBMClassifier(**best_lgb_s)
lgb_s_tuned, r = eval_tuned(lgb_s_tuned, X_s_tr, y_s_tr, X_s_vl, y_s_vl, X_s_ts, y_s_ts,
                              "LightGBM_Tuned", "pd_speech_features")
final_results.append(r)
print(f"  test_acc={r['test_acc']:.4f}, recall={r['test_recall']:.4f}, auc={r['test_auc']:.4f}, gap={r['gen_gap']:.4f}")

print("[Speech] Tuned CatBoost:")
cat_s_tuned = CatBoostClassifier(**best_cat_s)
cat_s_tuned, r = eval_tuned(cat_s_tuned, X_s_tr, y_s_tr, X_s_vl, y_s_vl, X_s_ts, y_s_ts,
                              "CatBoost_Tuned", "pd_speech_features")
final_results.append(r)
print(f"  test_acc={r['test_acc']:.4f}, recall={r['test_recall']:.4f}, auc={r['test_auc']:.4f}, gap={r['gen_gap']:.4f}")

# ---------------------------------------------
# Phase 13: Ensemble Learning
# ---------------------------------------------
print("\n" + "="*60)
print("Phase 13: Ensemble Learning")
print("="*60)

# AUDIO: Soft Voting Ensemble
print("\n[Audio] Soft Voting Ensemble (XGB + LGB + RF):")
xgb_e = xgb.XGBClassifier(**best_xgb_a)
lgb_e = lgb.LGBMClassifier(**best_lgb_a)
rf_e = RandomForestClassifier(**best_rf_a)
voting_a = VotingClassifier(
    estimators=[("xgb", xgb_e), ("lgb", lgb_e), ("rf", rf_e)],
    voting="soft", n_jobs=-1
)
voting_a, r = eval_tuned(voting_a, X_a_tr_sel, y_a_tr, X_a_vl_sel, y_a_vl, X_a_ts_sel, y_a_ts,
                          "VotingEnsemble", "audio_dataset1")
final_results.append(r)
print(f"  test_acc={r['test_acc']:.4f}, recall={r['test_recall']:.4f}, auc={r['test_auc']:.4f}, gap={r['gen_gap']:.4f}")

# AUDIO: Stacking Ensemble
print("[Audio] Stacking Ensemble (XGB + LGB + RF -> LR meta):")
xgb_s = xgb.XGBClassifier(**best_xgb_a)
lgb_s = lgb.LGBMClassifier(**best_lgb_a)
rf_s = RandomForestClassifier(**best_rf_a)
stack_a = StackingClassifier(
    estimators=[("xgb", xgb_s), ("lgb", lgb_s), ("rf", rf_s)],
    final_estimator=LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    cv=5, n_jobs=-1, passthrough=False
)
stack_a, r = eval_tuned(stack_a, X_a_tr_sel, y_a_tr, X_a_vl_sel, y_a_vl, X_a_ts_sel, y_a_ts,
                         "StackingEnsemble", "audio_dataset1")
final_results.append(r)
print(f"  test_acc={r['test_acc']:.4f}, recall={r['test_recall']:.4f}, auc={r['test_auc']:.4f}, gap={r['gen_gap']:.4f}")

# UCI: Soft Voting (train+val combined)
print("\n[UCI] Soft Voting Ensemble (XGB + SVM + ExtraTrees, train+val):")
xgb_eu = xgb.XGBClassifier(**best_xgb_u)
svm_eu = SVC(**best_svm_u)
et_eu = ExtraTreesClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)
voting_u = VotingClassifier(
    estimators=[("xgb", xgb_eu), ("svm", svm_eu), ("et", et_eu)],
    voting="soft", n_jobs=-1
)
try:
    voting_u, r = eval_tuned(voting_u, X_u_trval, y_u_trval, X_u_vl, y_u_vl, X_u_ts, y_u_ts,
                              "VotingEnsemble", "UCI_parkinsons")
    final_results.append(r)
    print(f"  test_acc={r['test_acc']:.4f}, recall={r['test_recall']:.4f}, auc={r['test_auc']:.4f}")
except Exception as e:
    print(f"  UCI Voting failed: {e}")

# Speech: Soft Voting Ensemble
print("\n[Speech] Soft Voting Ensemble (XGB + LGB + CatBoost):")
voting_s = VotingClassifier(
    estimators=[
        ("xgb", xgb.XGBClassifier(**best_xgb_s)),
        ("lgb", lgb.LGBMClassifier(**best_lgb_s)),
        ("cat", CatBoostClassifier(**best_cat_s)),
    ],
    voting="soft", n_jobs=-1
)
try:
    voting_s, r = eval_tuned(voting_s, X_s_tr, y_s_tr, X_s_vl, y_s_vl, X_s_ts, y_s_ts,
                              "VotingEnsemble", "pd_speech_features")
    final_results.append(r)
    print(f"  test_acc={r['test_acc']:.4f}, recall={r['test_recall']:.4f}, auc={r['test_auc']:.4f}, gap={r['gen_gap']:.4f}")
except Exception as e:
    print(f"  Speech Voting failed: {e}")

# ---------------------------------------------
# Select BEST detection model
# ---------------------------------------------
print("\n" + "="*60)
print("Selecting Best Detection Model")
print("="*60)

final_df = pd.DataFrame(final_results)
final_df.to_csv(RESULTS / "final_test_results.csv", index=False)
print("  Saved: results/final_test_results.csv")

# Best model: highest val_auc, with gen_gap < 0.15
valid_models = final_df[final_df["gen_gap"] < 0.15].copy() if len(final_df[final_df["gen_gap"] < 0.15]) > 0 else final_df.copy()
best_row = valid_models.loc[valid_models["val_auc"].idxmax()]
print(f"\n  BEST DETECTION MODEL: {best_row['model']} on {best_row['dataset']}")
print(f"    Val AUC:    {best_row['val_auc']:.4f}")
print(f"    Test ACC:   {best_row['test_acc']:.4f}")
print(f"    Test Recall:{best_row['test_recall']:.4f}")
print(f"    Test F1:    {best_row['test_f1']:.4f}")
print(f"    Test AUC:   {best_row['test_auc']:.4f}")
print(f"    Gen Gap:    {best_row['gen_gap']:.4f}")
target_met = best_row['test_acc'] >= 0.90 and best_row['test_recall'] >= 0.90
print(f"\n  >=90% target met (acc AND recall): {'YES OK' if target_met else 'NO X (reporting actual)'}")

# ---------------------------------------------
# Save the best detection model + artifacts
# ---------------------------------------------
best_model_name = best_row["model"]
best_dataset = best_row["dataset"]

if best_dataset == "audio_dataset1":
    X_tr_final, y_tr_final = X_a_tr_sel, y_a_tr
    X_ts_final, y_ts_final = X_a_ts_sel, y_a_ts
    feature_idx = top50_a
    features_used = [feat_a_filtered[i] for i in top50_a]

    model_map = {
        "XGBoost_Tuned": xgb_a_tuned,
        "LightGBM_Tuned": lgb_a_tuned,
        "RF_Tuned": rf_a_tuned,
        "VotingEnsemble": voting_a,
        "StackingEnsemble": stack_a,
    }
    detection_scaler = scl_a
    detection_imputer = imp_a

elif best_dataset == "UCI_parkinsons":
    X_tr_final, y_tr_final = X_u_tr, y_u_tr
    X_ts_final, y_ts_final = X_u_ts, y_u_ts
    features_used = feat_3a
    model_map = {
        "XGBoost_Tuned": xgb_u_tuned,
        "SVM_Tuned": svm_u_tuned,
        "VotingEnsemble": voting_u if "VotingEnsemble" in [r["model"] for r in final_results] else None,
    }
    detection_scaler = joblib.load(MODELS / "ds3a_scaler.pkl")
    detection_imputer = joblib.load(MODELS / "ds3a_imputer.pkl")

best_det_model = model_map.get(best_model_name)
if best_det_model is None:
    print(f"  WARNING: Could not find model object for {best_model_name}, using XGBoost_Tuned")
    best_det_model = model_map.get("XGBoost_Tuned", list(model_map.values())[0])

joblib.dump(best_det_model, MODELS / "detection_best_model.pkl")
joblib.dump(detection_scaler, MODELS / "detection_scaler.pkl")
joblib.dump(detection_imputer, MODELS / "detection_imputer.pkl")

# Label encoder
le = LabelEncoder()
le.fit([0, 1])
joblib.dump(le, MODELS / "detection_label_encoder.pkl")

# Save feature info
with open(MODELS / "detection_feature_config.json", "w") as f:
    json.dump({
        "best_model": best_model_name,
        "best_dataset": best_dataset,
        "features_used": features_used if isinstance(features_used, list) else features_used.tolist(),
        "n_features": len(features_used),
        "label_encoding": {"0": "Healthy", "1": "Parkinson"},
        "test_acc": float(best_row["test_acc"]),
        "test_auc": float(best_row["test_auc"]),
        "test_recall": float(best_row["test_recall"]),
        "test_f1": float(best_row["test_f1"]),
        "gen_gap": float(best_row["gen_gap"]),
    }, f, indent=2)

print(f"\n  Saved: models/detection_best_model.pkl")
print(f"  Saved: models/detection_label_encoder.pkl")
print(f"  Saved: models/detection_feature_config.json")

# ---------------------------------------------
# Phase 12: Overfitting analysis plot
# ---------------------------------------------
print("\n[Overfitting Analysis] Plotting train vs val vs test accuracy...")
fig, ax = plt.subplots(figsize=(12, 6))
models_sorted = final_df.sort_values("val_auc", ascending=False).head(10)
x = np.arange(len(models_sorted))
width = 0.25
bars1 = ax.bar(x - width, models_sorted["train_acc"], width, label="Train Acc", color="#2196F3", alpha=0.85)
bars2 = ax.bar(x, models_sorted["val_acc"], width, label="Val Acc", color="#4CAF50", alpha=0.85)
bars3 = ax.bar(x + width, models_sorted["test_acc"], width, label="Test Acc", color="#FF5722", alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels([f"{r['model']}\n{r['dataset'][:8]}" for _, r in models_sorted.iterrows()],
                    fontsize=7, rotation=30)
ax.set_ylabel("Accuracy")
ax.set_title("Train vs Validation vs Test Accuracy — All Tuned Models", fontweight="bold")
ax.legend()
ax.axhline(0.90, color="red", linestyle="--", alpha=0.7, label="90% target")
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(PLOTS / "training_vs_validation.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: plots/training_vs_validation.png")

# Learning curve for best detection model
print("[Learning Curve] Generating...")
try:
    from sklearn.model_selection import learning_curve as lc_fn
    # Use a fast model for learning curve
    lc_model = RandomForestClassifier(**best_rf_a) if best_dataset == "audio_dataset1" else xgb.XGBClassifier(**best_xgb_u)
    train_sizes_pct = np.linspace(0.1, 1.0, 8)
    tr_sizes, tr_scores, vl_scores = lc_fn(
        lc_model, X_tr_final, y_tr_final,
        train_sizes=train_sizes_pct,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="roc_auc",
        n_jobs=-1,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(tr_sizes, tr_scores.mean(1)-tr_scores.std(1),
                    tr_scores.mean(1)+tr_scores.std(1), alpha=0.15, color="#2196F3")
    ax.fill_between(tr_sizes, vl_scores.mean(1)-vl_scores.std(1),
                    vl_scores.mean(1)+vl_scores.std(1), alpha=0.15, color="#FF5722")
    ax.plot(tr_sizes, tr_scores.mean(1), "o-", color="#2196F3", label="Training AUC")
    ax.plot(tr_sizes, vl_scores.mean(1), "s-", color="#FF5722", label="CV Validation AUC")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("ROC-AUC Score")
    ax.set_title(f"Learning Curve — {best_model_name}", fontweight="bold")
    ax.legend()
    ax.axhline(0.90, color="red", linestyle="--", alpha=0.5, label="Target AUC 0.90")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "learning_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: plots/learning_curve.png")
except Exception as e:
    print(f"  Learning curve failed: {e}")

# Print final summary
print("\n" + "="*60)
print("FINAL TUNED MODEL SUMMARY")
print("="*60)
summary_cols = ["model", "dataset", "train_acc", "val_acc", "test_acc",
                "test_recall", "test_f1", "test_auc", "gen_gap"]
sort_col = "test_acc" if "val_auc" not in final_df.columns else "val_auc"
print(final_df.sort_values(sort_col, ascending=False)[summary_cols].to_string(index=False))

print("\n" + "="*60)
print("Phase 10+11+12+13 COMPLETE")
print("="*60)
