#!/usr/bin/env python3
"""
Phase 15: UPDRS Severity Regression
ParkinsonXAI Project

Trains 8 regressors on parkinsons_updrs.data with subject-wise GroupKFold.
Uses Optuna for top models.

Outputs:
  models/severity_best_model.pkl
  models/severity_scaler.pkl
  models/severity_feature_selector.pkl
  results/severity_results.csv
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from collections import Counter

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parent.parent
TABULAR = ROOT / "processed" / "tabular"
SPLITS = ROOT / "processed" / "splits"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

np.random.seed(42)

print("="*60)
print("Phase 15: UPDRS Severity Regression")
print("="*60)

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
df3b_train = pd.read_csv(TABULAR / "ds3b_train.csv")
df3b_val = pd.read_csv(TABULAR / "ds3b_val.csv")
df3b_test = pd.read_csv(TABULAR / "ds3b_test.csv")

feat_3b = [c for c in df3b_train.columns if c not in ["motor_UPDRS", "total_UPDRS"]]
X_tr = df3b_train[feat_3b].values
X_vl = df3b_val[feat_3b].values
X_ts = df3b_test[feat_3b].values

ym_tr = df3b_train["motor_UPDRS"].values
ym_vl = df3b_val["motor_UPDRS"].values
ym_ts = df3b_test["motor_UPDRS"].values
yt_tr = df3b_train["total_UPDRS"].values
yt_vl = df3b_val["total_UPDRS"].values
yt_ts = df3b_test["total_UPDRS"].values

print(f"  Train: {len(X_tr)}, Val: {len(X_vl)}, Test: {len(X_ts)}")
print(f"  motor_UPDRS train: mean={ym_tr.mean():.2f}, std={ym_tr.std():.2f}")
print(f"  total_UPDRS train: mean={yt_tr.mean():.2f}, std={yt_tr.std():.2f}")

# Load subject IDs for GroupKFold
with open(SPLITS / "dataset3b_updrs_splits.json") as f:
    splits3b = json.load(f)

df3b_full = pd.read_csv(TABULAR / "parkinsons_updrs.csv")
train_row_idx = splits3b["train_row_indices"]
groups_train = df3b_full.iloc[train_row_idx]["subject#"].values

# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────

def eval_regressor(model, X_tr, y_tr, X_vl, y_vl, X_ts, y_ts, name, target):
    model.fit(X_tr, y_tr)
    results = {"model": name, "target": target}
    for split, X_s, y_s in [("train", X_tr, y_tr), ("val", X_vl, y_vl), ("test", X_ts, y_ts)]:
        y_pred = model.predict(X_s)
        results[f"{split}_mae"] = round(mean_absolute_error(y_s, y_pred), 4)
        results[f"{split}_rmse"] = round(np.sqrt(mean_squared_error(y_s, y_pred)), 4)
        results[f"{split}_r2"] = round(r2_score(y_s, y_pred), 4)
    return model, results


def cv_groupkfold_regression(X, y, groups, model_fn, n_splits=5):
    """GroupKFold CV for regression — subject-aware."""
    gkf = GroupKFold(n_splits=n_splits)
    maes, rmses, r2s = [], [], []
    for tr_idx, vl_idx in gkf.split(X, y, groups=groups):
        X_tr_f, X_vl_f = X[tr_idx], X[vl_idx]
        y_tr_f, y_vl_f = y[tr_idx], y[vl_idx]
        m = model_fn()
        m.fit(X_tr_f, y_tr_f)
        y_pred = m.predict(X_vl_f)
        maes.append(mean_absolute_error(y_vl_f, y_pred))
        rmses.append(np.sqrt(mean_squared_error(y_vl_f, y_pred)))
        r2s.append(r2_score(y_vl_f, y_pred))
    return {
        "cv_mae_mean": round(np.mean(maes), 4),
        "cv_mae_std": round(np.std(maes), 4),
        "cv_rmse_mean": round(np.mean(rmses), 4),
        "cv_r2_mean": round(np.mean(r2s), 4),
    }


all_severity_results = []

# ─────────────────────────────────────────────
# Train baseline regressors
# ─────────────────────────────────────────────
REGRESSORS = [
    ("LinearRegression", LinearRegression, {}),
    ("Ridge", Ridge, {"alpha": 1.0}),
    ("RandomForest", RandomForestRegressor, {"n_estimators": 200, "random_state": 42, "n_jobs": -1}),
    ("ExtraTrees", ExtraTreesRegressor, {"n_estimators": 200, "random_state": 42, "n_jobs": -1}),
    ("SVR", SVR, {"kernel": "rbf", "C": 10.0, "epsilon": 0.5}),
    ("XGBoost", xgb.XGBRegressor, {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 5,
                                    "random_state": 42, "verbosity": 0}),
    ("LightGBM", lgb.LGBMRegressor, {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31,
                                      "random_state": 42, "n_jobs": -1, "verbose": -1}),
    ("CatBoost", CatBoostRegressor, {"iterations": 200, "learning_rate": 0.05, "depth": 6,
                                      "random_seed": 42, "verbose": 0}),
]

print("\n--- Training on motor_UPDRS ---")
for reg_name, reg_class, reg_kwargs in REGRESSORS:
    try:
        print(f"  {reg_name}...", end=" ", flush=True)
        reg = reg_class(**reg_kwargs)
        trained_reg, res = eval_regressor(reg, X_tr, ym_tr, X_vl, ym_vl, X_ts, ym_ts,
                                           reg_name, "motor_UPDRS")
        cv_res = cv_groupkfold_regression(X_tr, ym_tr, groups_train,
                                           lambda: reg_class(**reg_kwargs), n_splits=5)
        res.update(cv_res)
        all_severity_results.append(res)
        print(f"test_mae={res['test_mae']:.3f}, test_rmse={res['test_rmse']:.3f}, "
              f"test_r2={res['test_r2']:.4f}")
    except Exception as e:
        print(f"FAILED: {e}")

print("\n--- Training on total_UPDRS ---")
for reg_name, reg_class, reg_kwargs in REGRESSORS:
    try:
        print(f"  {reg_name}...", end=" ", flush=True)
        reg = reg_class(**reg_kwargs)
        trained_reg, res = eval_regressor(reg, X_tr, yt_tr, X_vl, yt_vl, X_ts, yt_ts,
                                           reg_name, "total_UPDRS")
        cv_res = cv_groupkfold_regression(X_tr, yt_tr, groups_train,
                                           lambda: reg_class(**reg_kwargs), n_splits=5)
        res.update(cv_res)
        all_severity_results.append(res)
        print(f"test_mae={res['test_mae']:.3f}, test_rmse={res['test_rmse']:.3f}, "
              f"test_r2={res['test_r2']:.4f}")
    except Exception as e:
        print(f"FAILED: {e}")

# ─────────────────────────────────────────────
# Optuna tuning for top 3 regressors
# ─────────────────────────────────────────────
print("\n[Optuna] Tuning top regression models for motor_UPDRS...")
N_TRIALS_REG = 50

def cv_mae(model_fn, X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    maes = []
    for tr_idx, vl_idx in gkf.split(X, y, groups=groups):
        m = model_fn()
        m.fit(X[tr_idx], y[tr_idx])
        maes.append(mean_absolute_error(y[vl_idx], m.predict(X[vl_idx])))
    return np.mean(maes)

# XGBoost Regressor
def obj_xgb_reg(trial):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "random_state": 42, "verbosity": 0,
    }
    return cv_mae(lambda: xgb.XGBRegressor(**p), X_tr, ym_tr, groups_train)

study_xgb_r = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study_xgb_r.optimize(obj_xgb_reg, n_trials=N_TRIALS_REG, show_progress_bar=False)
best_xgb_r = study_xgb_r.best_params
best_xgb_r.update({"random_state": 42, "verbosity": 0})
print(f"  XGB best CV MAE: {study_xgb_r.best_value:.4f}")

# LightGBM Regressor
def obj_lgb_reg(trial):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 100),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "random_state": 42, "n_jobs": -1, "verbose": -1,
    }
    return cv_mae(lambda: lgb.LGBMRegressor(**p), X_tr, ym_tr, groups_train)

study_lgb_r = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study_lgb_r.optimize(obj_lgb_reg, n_trials=N_TRIALS_REG, show_progress_bar=False)
best_lgb_r = study_lgb_r.best_params
best_lgb_r.update({"random_state": 42, "n_jobs": -1, "verbose": -1})
print(f"  LGB best CV MAE: {study_lgb_r.best_value:.4f}")

# Train tuned models
print("\nTraining tuned severity models...")
xgb_r_tuned = xgb.XGBRegressor(**best_xgb_r)
xgb_r_tuned, r_xgb_m = eval_regressor(xgb_r_tuned, X_tr, ym_tr, X_vl, ym_vl, X_ts, ym_ts,
                                        "XGBoost_Tuned", "motor_UPDRS")
cv_xgb_m = cv_groupkfold_regression(X_tr, ym_tr, groups_train, lambda: xgb.XGBRegressor(**best_xgb_r))
r_xgb_m.update(cv_xgb_m)
all_severity_results.append(r_xgb_m)
print(f"  XGBoost_Tuned motor_UPDRS: test_mae={r_xgb_m['test_mae']:.3f}, r2={r_xgb_m['test_r2']:.4f}")

lgb_r_tuned = lgb.LGBMRegressor(**best_lgb_r)
lgb_r_tuned, r_lgb_m = eval_regressor(lgb_r_tuned, X_tr, ym_tr, X_vl, ym_vl, X_ts, ym_ts,
                                        "LightGBM_Tuned", "motor_UPDRS")
cv_lgb_m = cv_groupkfold_regression(X_tr, ym_tr, groups_train, lambda: lgb.LGBMRegressor(**best_lgb_r))
r_lgb_m.update(cv_lgb_m)
all_severity_results.append(r_lgb_m)
print(f"  LightGBM_Tuned motor_UPDRS: test_mae={r_lgb_m['test_mae']:.3f}, r2={r_lgb_m['test_r2']:.4f}")

# Also for total_UPDRS
xgb_r_total = xgb.XGBRegressor(**best_xgb_r)
xgb_r_total, r_xgb_t = eval_regressor(xgb_r_total, X_tr, yt_tr, X_vl, yt_vl, X_ts, yt_ts,
                                        "XGBoost_Tuned", "total_UPDRS")
all_severity_results.append(r_xgb_t)
print(f"  XGBoost_Tuned total_UPDRS: test_mae={r_xgb_t['test_mae']:.3f}, r2={r_xgb_t['test_r2']:.4f}")

lgb_r_total = lgb.LGBMRegressor(**best_lgb_r)
lgb_r_total, r_lgb_t = eval_regressor(lgb_r_total, X_tr, yt_tr, X_vl, yt_vl, X_ts, yt_ts,
                                        "LightGBM_Tuned", "total_UPDRS")
all_severity_results.append(r_lgb_t)
print(f"  LightGBM_Tuned total_UPDRS: test_mae={r_lgb_t['test_mae']:.3f}, r2={r_lgb_t['test_r2']:.4f}")

# ─────────────────────────────────────────────
# Select best severity model
# ─────────────────────────────────────────────
sev_df = pd.DataFrame(all_severity_results)
sev_df.to_csv(RESULTS / "severity_results.csv", index=False)
print(f"\n  Saved: results/severity_results.csv")

# Best motor_UPDRS model
motor_df = sev_df[sev_df["target"] == "motor_UPDRS"]
best_motor_row = motor_df.loc[motor_df["test_r2"].idxmax()]
print(f"\n  BEST motor_UPDRS model: {best_motor_row['model']}")
print(f"    test_MAE:  {best_motor_row['test_mae']:.4f}")
print(f"    test_RMSE: {best_motor_row['test_rmse']:.4f}")
print(f"    test_R²:   {best_motor_row['test_r2']:.4f}")

# Best total_UPDRS model
total_df = sev_df[sev_df["target"] == "total_UPDRS"]
best_total_row = total_df.loc[total_df["test_r2"].idxmax()]
print(f"\n  BEST total_UPDRS model: {best_total_row['model']}")
print(f"    test_MAE:  {best_total_row['test_mae']:.4f}")
print(f"    test_RMSE: {best_total_row['test_rmse']:.4f}")
print(f"    test_R²:   {best_total_row['test_r2']:.4f}")

# Save best severity model (motor_UPDRS focused)
if best_motor_row["model"] == "XGBoost_Tuned":
    best_severity_model = xgb.XGBRegressor(**best_xgb_r)
    best_severity_model.fit(X_tr, ym_tr)
elif best_motor_row["model"] == "LightGBM_Tuned":
    best_severity_model = lgb.LGBMRegressor(**best_lgb_r)
    best_severity_model.fit(X_tr, ym_tr)
else:
    best_severity_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    best_severity_model.fit(X_tr, ym_tr)

joblib.dump(best_severity_model, MODELS / "severity_best_model.pkl")
print(f"\n  Saved: models/severity_best_model.pkl")

# Save model metadata
with open(MODELS / "severity_model_config.json", "w") as f:
    json.dump({
        "motor_UPDRS_model": best_motor_row["model"],
        "total_UPDRS_model": best_total_row["model"],
        "motor_test_mae": float(best_motor_row["test_mae"]),
        "motor_test_rmse": float(best_motor_row["test_rmse"]),
        "motor_test_r2": float(best_motor_row["test_r2"]),
        "total_test_mae": float(best_total_row["test_mae"]),
        "total_test_rmse": float(best_total_row["test_rmse"]),
        "total_test_r2": float(best_total_row["test_r2"]),
        "features_used": feat_3b,
        "n_features": len(feat_3b),
        "splitting": "subject-wise GroupKFold, 42 subjects",
    }, f, indent=2)

print("\n" + "="*60)
print("SEVERITY RESULTS SUMMARY")
print("="*60)
print(sev_df[["model", "target", "test_mae", "test_rmse", "test_r2", "cv_mae_mean", "cv_r2_mean"]
             ].sort_values(["target", "test_r2"], ascending=[True, False]).to_string(index=False))

print("\n" + "="*60)
print("Phase 15 COMPLETE")
print("="*60)
