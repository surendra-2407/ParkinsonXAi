#!/usr/bin/env python3
"""
Phase 5: Data Preprocessing
ParkinsonXAI Project

ALL preprocessing fitted ONLY on training data.
Produces fitted scaler/imputer objects saved to models/.

Outputs:
  models/detection_scaler.pkl
  models/severity_scaler.pkl
  processed/tabular/ds2_train.csv / ds2_val.csv / ds2_test.csv
  processed/tabular/ds3a_train.csv / ds3a_val.csv / ds3a_test.csv
  processed/tabular/ds3b_train.csv / ds3b_val.csv / ds3b_test.csv
  reports/preprocessing_report.md
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

ROOT = Path(__file__).resolve().parent.parent
TABULAR = ROOT / "processed" / "tabular"
SPLITS = ROOT / "processed" / "splits"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(parents=True, exist_ok=True)

print("="*60)
print("Phase 5: Data Preprocessing")
print("="*60)

report_lines = ["# ParkinsonXAI — Preprocessing Report\n\n"]

def document_removed_features(name, reason, cols):
    msg = f"  [{name}] Removed {len(cols)} features ({reason}): {cols[:5]}{'...' if len(cols)>5 else ''}"
    print(msg)
    return msg + "\n"

# ─────────────────────────────────────────────
# DATASET 2: pd_speech_features.csv
# ─────────────────────────────────────────────
print("\n[Dataset 2] pd_speech_features.csv")
report_lines.append("## Dataset 2: pd_speech_features.csv\n\n")

df2 = pd.read_csv(TABULAR / "pd_speech_features.csv")
with open(SPLITS / "dataset2_splits.json") as f:
    splits2 = json.load(f)

train_idx2 = splits2["train_indices"]
val_idx2 = splits2["val_indices"]
test_idx2 = splits2["test_indices"]

# Drop non-feature columns
feature_cols2 = [c for c in df2.columns if c not in ["id", "gender", "class"]]
X2 = df2[feature_cols2].copy()
y2 = df2["class"].values

X2_train = X2.iloc[train_idx2].copy()
X2_val = X2.iloc[val_idx2].copy()
X2_test = X2.iloc[test_idx2].copy()
y2_train = y2[train_idx2]
y2_val = y2[val_idx2]
y2_test = y2[test_idx2]

print(f"  Initial features: {X2_train.shape[1]}")
report_lines.append(f"- Initial features: {X2_train.shape[1]}\n")

# Step 1: Missing value imputation (fit on train only)
imputer2 = SimpleImputer(strategy="median")
X2_train = pd.DataFrame(imputer2.fit_transform(X2_train), columns=feature_cols2)
X2_val = pd.DataFrame(imputer2.transform(X2_val), columns=feature_cols2)
X2_test = pd.DataFrame(imputer2.transform(X2_test), columns=feature_cols2)
report_lines.append(f"- Missing values: {df2[feature_cols2].isnull().sum().sum()} total → imputed with median (fit on train)\n")

# Step 2: Constant / near-zero variance removal (fit on train only)
var_sel2 = VarianceThreshold(threshold=1e-6)
var_sel2.fit(X2_train)
nzv_mask2 = var_sel2.get_support()
removed_nzv2 = [c for c, keep in zip(feature_cols2, nzv_mask2) if not keep]
if removed_nzv2:
    report_lines.append(document_removed_features("DS2", "near-zero variance", removed_nzv2))
feature_cols2 = [c for c, keep in zip(feature_cols2, nzv_mask2) if keep]
X2_train = X2_train[feature_cols2]
X2_val = X2_val[feature_cols2]
X2_test = X2_test[feature_cols2]
print(f"  After NZV removal: {len(feature_cols2)} features")

# Step 3: High-correlation removal (fit on train only, threshold 0.97)
corr_matrix = X2_train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop_corr2 = [c for c in upper.columns if any(upper[c] > 0.97)]
if to_drop_corr2:
    report_lines.append(document_removed_features("DS2", "high correlation >0.97", to_drop_corr2))
feature_cols2 = [c for c in feature_cols2 if c not in to_drop_corr2]
X2_train = X2_train[feature_cols2]
X2_val = X2_val[feature_cols2]
X2_test = X2_test[feature_cols2]
print(f"  After high-corr removal: {len(feature_cols2)} features")

# Step 4: Outlier documentation (3*IQR, NOT removal)
Q1 = X2_train.quantile(0.25)
Q3 = X2_train.quantile(0.75)
IQR = Q3 - Q1
outlier_mask = ((X2_train < Q1 - 3*IQR) | (X2_train > Q3 + 3*IQR))
n_outlier_cells = outlier_mask.sum().sum()
n_outlier_rows = outlier_mask.any(axis=1).sum()
print(f"  Outliers (3*IQR): {n_outlier_cells} cells in {n_outlier_rows} rows (documented, NOT removed)")
report_lines.append(f"- Outliers (3*IQR): {n_outlier_cells} cells in {n_outlier_rows} rows — documented only, not removed\n")

# Step 5: Scaling (fit on train only)
scaler2 = RobustScaler()
X2_train_scaled = pd.DataFrame(scaler2.fit_transform(X2_train), columns=feature_cols2)
X2_val_scaled = pd.DataFrame(scaler2.transform(X2_val), columns=feature_cols2)
X2_test_scaled = pd.DataFrame(scaler2.transform(X2_test), columns=feature_cols2)
report_lines.append("- Scaling: RobustScaler fitted on training data only\n")

# Step 6: Class imbalance check
from collections import Counter
train_dist = Counter(y2_train)
print(f"  Class distribution (train): {dict(train_dist)}")
imbalance_ratio = max(train_dist.values()) / min(train_dist.values())
print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
report_lines.append(f"- Train class distribution: {dict(train_dist)} (ratio {imbalance_ratio:.2f}:1)\n")

# Save DS2 splits
train2_df = X2_train_scaled.copy()
train2_df["target"] = y2_train
val2_df = X2_val_scaled.copy()
val2_df["target"] = y2_val
test2_df = X2_test_scaled.copy()
test2_df["target"] = y2_test

train2_df.to_csv(TABULAR / "ds2_train.csv", index=False)
val2_df.to_csv(TABULAR / "ds2_val.csv", index=False)
test2_df.to_csv(TABULAR / "ds2_test.csv", index=False)

# Save feature list
ds2_feature_cols = feature_cols2
joblib.dump(scaler2, MODELS / "ds2_scaler.pkl")
joblib.dump(imputer2, MODELS / "ds2_imputer.pkl")
with open(MODELS / "ds2_feature_cols.json", "w") as f:
    json.dump({"features": ds2_feature_cols, "n_features": len(ds2_feature_cols)}, f)

print(f"  Final features: {len(feature_cols2)}")
report_lines.append(f"- **Final features after preprocessing: {len(feature_cols2)}**\n\n")

# ─────────────────────────────────────────────
# DATASET 3a: parkinsons.data (UCI)
# ─────────────────────────────────────────────
print("\n[Dataset 3a] parkinsons.data (UCI)")
report_lines.append("## Dataset 3a: parkinsons.data (UCI)\n\n")

df3a = pd.read_csv(TABULAR / "parkinsons_data.csv")
with open(SPLITS / "dataset3a_splits.json") as f:
    splits3a = json.load(f)

feature_cols3a = [c for c in df3a.columns if c not in ["name", "status", "subject_id"]]
X3a = df3a[feature_cols3a].copy()
y3a = df3a["status"].values

train_idx3a = splits3a["train_indices"]
val_idx3a = splits3a["val_indices"]
test_idx3a = splits3a["test_indices"]

X3a_train = X3a.iloc[train_idx3a].copy()
X3a_val = X3a.iloc[val_idx3a].copy()
X3a_test = X3a.iloc[test_idx3a].copy()
y3a_train = y3a[train_idx3a]
y3a_val = y3a[val_idx3a]
y3a_test = y3a[test_idx3a]

print(f"  Initial features: {X3a_train.shape[1]}")
report_lines.append(f"- Initial features: {X3a_train.shape[1]}\n")

# No missing values expected, but impute defensively
imputer3a = SimpleImputer(strategy="median")
X3a_train = pd.DataFrame(imputer3a.fit_transform(X3a_train), columns=feature_cols3a)
X3a_val = pd.DataFrame(imputer3a.transform(X3a_val), columns=feature_cols3a)
X3a_test = pd.DataFrame(imputer3a.transform(X3a_test), columns=feature_cols3a)

# NZV
var_sel3a = VarianceThreshold(threshold=1e-6)
var_sel3a.fit(X3a_train)
nzv_mask3a = var_sel3a.get_support()
removed_nzv3a = [c for c, k in zip(feature_cols3a, nzv_mask3a) if not k]
if removed_nzv3a:
    report_lines.append(document_removed_features("DS3a", "near-zero variance", removed_nzv3a))
feature_cols3a = [c for c, k in zip(feature_cols3a, nzv_mask3a) if k]
X3a_train = X3a_train[feature_cols3a]
X3a_val = X3a_val[feature_cols3a]
X3a_test = X3a_test[feature_cols3a]

# Scaling
scaler3a = RobustScaler()
X3a_train_sc = pd.DataFrame(scaler3a.fit_transform(X3a_train), columns=feature_cols3a)
X3a_val_sc = pd.DataFrame(scaler3a.transform(X3a_val), columns=feature_cols3a)
X3a_test_sc = pd.DataFrame(scaler3a.transform(X3a_test), columns=feature_cols3a)

train_dist3a = Counter(y3a_train)
print(f"  Class distribution (train): {dict(train_dist3a)}")
report_lines.append(f"- Train class distribution: {dict(train_dist3a)}\n")
report_lines.append(f"- Scaling: RobustScaler fitted on train only\n")
report_lines.append(f"- **Final features: {len(feature_cols3a)}**\n\n")

# Save
train3a_df = X3a_train_sc.copy(); train3a_df["target"] = y3a_train
val3a_df = X3a_val_sc.copy(); val3a_df["target"] = y3a_val
test3a_df = X3a_test_sc.copy(); test3a_df["target"] = y3a_test
train3a_df.to_csv(TABULAR / "ds3a_train.csv", index=False)
val3a_df.to_csv(TABULAR / "ds3a_val.csv", index=False)
test3a_df.to_csv(TABULAR / "ds3a_test.csv", index=False)

joblib.dump(scaler3a, MODELS / "ds3a_scaler.pkl")
joblib.dump(imputer3a, MODELS / "ds3a_imputer.pkl")
with open(MODELS / "ds3a_feature_cols.json", "w") as f:
    json.dump({"features": feature_cols3a, "n_features": len(feature_cols3a)}, f)
# This becomes the detection scaler (UCI benchmark model)
joblib.dump(scaler3a, MODELS / "detection_scaler.pkl")
print(f"  Final features: {len(feature_cols3a)}")

# ─────────────────────────────────────────────
# DATASET 3b: parkinsons_updrs.data (REGRESSION)
# ─────────────────────────────────────────────
print("\n[Dataset 3b] parkinsons_updrs.data (UPDRS regression)")
report_lines.append("## Dataset 3b: parkinsons_updrs.data (UPDRS Regression)\n\n")

df3b = pd.read_csv(TABULAR / "parkinsons_updrs.csv")
with open(SPLITS / "dataset3b_updrs_splits.json") as f:
    splits3b = json.load(f)

# Features: exclude identifiers and targets
feature_cols3b = [c for c in df3b.columns if c not in ["subject#", "motor_UPDRS", "total_UPDRS"]]
X3b = df3b[feature_cols3b].copy()
y3b_motor = df3b["motor_UPDRS"].values
y3b_total = df3b["total_UPDRS"].values

train_idx3b = splits3b["train_row_indices"]
val_idx3b = splits3b["val_row_indices"]
test_idx3b = splits3b["test_row_indices"]

X3b_train = X3b.iloc[train_idx3b].copy()
X3b_val = X3b.iloc[val_idx3b].copy()
X3b_test = X3b.iloc[test_idx3b].copy()

ym_train = y3b_motor[train_idx3b]; ym_val = y3b_motor[val_idx3b]; ym_test = y3b_motor[test_idx3b]
yt_train = y3b_total[train_idx3b]; yt_val = y3b_total[val_idx3b]; yt_test = y3b_total[test_idx3b]

print(f"  Initial features: {X3b_train.shape[1]}")
report_lines.append(f"- Initial features: {X3b_train.shape[1]}\n")

# Impute
imputer3b = SimpleImputer(strategy="median")
X3b_train = pd.DataFrame(imputer3b.fit_transform(X3b_train), columns=feature_cols3b)
X3b_val = pd.DataFrame(imputer3b.transform(X3b_val), columns=feature_cols3b)
X3b_test = pd.DataFrame(imputer3b.transform(X3b_test), columns=feature_cols3b)

# NZV
var_sel3b = VarianceThreshold(threshold=1e-6)
var_sel3b.fit(X3b_train)
nzv_mask3b = var_sel3b.get_support()
removed_nzv3b = [c for c, k in zip(feature_cols3b, nzv_mask3b) if not k]
if removed_nzv3b:
    report_lines.append(document_removed_features("DS3b", "near-zero variance", removed_nzv3b))
feature_cols3b_filtered = [c for c, k in zip(feature_cols3b, nzv_mask3b) if k]
X3b_train = X3b_train[feature_cols3b_filtered]
X3b_val = X3b_val[feature_cols3b_filtered]
X3b_test = X3b_test[feature_cols3b_filtered]

# Scale
scaler3b = RobustScaler()
X3b_train_sc = pd.DataFrame(scaler3b.fit_transform(X3b_train), columns=feature_cols3b_filtered)
X3b_val_sc = pd.DataFrame(scaler3b.transform(X3b_val), columns=feature_cols3b_filtered)
X3b_test_sc = pd.DataFrame(scaler3b.transform(X3b_test), columns=feature_cols3b_filtered)

report_lines.append(f"- motor_UPDRS train range: [{ym_train.min():.2f}, {ym_train.max():.2f}]\n")
report_lines.append(f"- total_UPDRS train range: [{yt_train.min():.2f}, {yt_train.max():.2f}]\n")
report_lines.append(f"- Scaling: RobustScaler fitted on train only\n")
report_lines.append(f"- **Final features: {len(feature_cols3b_filtered)}**\n\n")

# Save
train3b_df = X3b_train_sc.copy()
train3b_df["motor_UPDRS"] = ym_train
train3b_df["total_UPDRS"] = yt_train
val3b_df = X3b_val_sc.copy()
val3b_df["motor_UPDRS"] = ym_val
val3b_df["total_UPDRS"] = yt_val
test3b_df = X3b_test_sc.copy()
test3b_df["motor_UPDRS"] = ym_test
test3b_df["total_UPDRS"] = yt_test

train3b_df.to_csv(TABULAR / "ds3b_train.csv", index=False)
val3b_df.to_csv(TABULAR / "ds3b_val.csv", index=False)
test3b_df.to_csv(TABULAR / "ds3b_test.csv", index=False)

joblib.dump(scaler3b, MODELS / "severity_scaler.pkl")
joblib.dump(imputer3b, MODELS / "ds3b_imputer.pkl")
with open(MODELS / "ds3b_feature_cols.json", "w") as f:
    json.dump({"features": feature_cols3b_filtered, "n_features": len(feature_cols3b_filtered)}, f)
with open(MODELS / "severity_feature_cols.json", "w") as f:
    json.dump({"features": feature_cols3b_filtered, "n_features": len(feature_cols3b_filtered)}, f)

print(f"  Final features: {len(feature_cols3b_filtered)}")
print(f"  Train rows: {len(X3b_train_sc)}, Val rows: {len(X3b_val_sc)}, Test rows: {len(X3b_test_sc)}")

# Write report
with open(REPORTS / "preprocessing_report.md", "w", encoding="utf-8") as f:
    f.writelines(report_lines)

print("\n" + "="*60)
print("Phase 5 COMPLETE")
print("="*60)
print("Saved:")
print("  models/ds2_scaler.pkl, ds2_imputer.pkl, ds2_feature_cols.json")
print("  models/detection_scaler.pkl (ds3a)")
print("  models/severity_scaler.pkl (ds3b)")
print("  processed/tabular/ds2_train/val/test.csv")
print("  processed/tabular/ds3a_train/val/test.csv")
print("  processed/tabular/ds3b_train/val/test.csv")
print("  reports/preprocessing_report.md")
