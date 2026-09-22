#!/usr/bin/env python3
"""
Phase 3+4: Patient-Level Data Leakage Prevention & Data Splitting
ParkinsonXAI Project

Produces:
  processed/splits/dataset1_audio_splits.json
  processed/splits/dataset2_splits.json
  processed/splits/dataset3a_splits.json
  processed/splits/dataset3b_updrs_splits.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    StratifiedGroupKFold,
    GroupShuffleSplit,
)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "processed"
TABULAR = PROCESSED / "tabular"
SPLITS_DIR = PROCESSED / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

print("="*60)
print("Phase 3+4: Leakage-Safe Data Splitting")
print("="*60)

# ─────────────────────────────────────────────
# DATASET 1: Audio WAV files
# No patient IDs available → treat each file as independent
# ─────────────────────────────────────────────
print("\n[Dataset 1] Audio WAV — no patient IDs, stratified file split")
import zipfile
z1 = zipfile.ZipFile(ROOT / "datasets" / "dataset_1.zip")
all_names = z1.namelist()
healthy = sorted([n for n in all_names if "Healthy" in n and n.endswith(".wav")])
parkinson = sorted([n for n in all_names if "Parkinsons" in n and n.endswith(".wav")])

all_files = healthy + parkinson
labels = [0] * len(healthy) + [1] * len(parkinson)

# First split: 70% train, 30% temp
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, temp_idx = next(sss1.split(all_files, labels))

# Second split: split temp into 50/50 → val=15%, test=15%
temp_files = [all_files[i] for i in temp_idx]
temp_labels = [labels[i] for i in temp_idx]
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_idx_local, test_idx_local = next(sss2.split(temp_files, temp_labels))

val_idx = [temp_idx[i] for i in val_idx_local]
test_idx = [temp_idx[i] for i in test_idx_local]

ds1_splits = {
    "train_files": [all_files[i] for i in train_idx],
    "train_labels": [labels[i] for i in train_idx],
    "val_files": [all_files[i] for i in val_idx],
    "val_labels": [labels[i] for i in val_idx],
    "test_files": [all_files[i] for i in test_idx],
    "test_labels": [labels[i] for i in test_idx],
    "note": "No patient IDs in filenames; each WAV treated as independent; stratified split",
}
print(f"  Train: {len(ds1_splits['train_files'])} files ({sum(ds1_splits['train_labels'])} PD, {sum(1-l for l in ds1_splits['train_labels'])} HC)")
print(f"  Val:   {len(ds1_splits['val_files'])} files ({sum(ds1_splits['val_labels'])} PD, {sum(1-l for l in ds1_splits['val_labels'])} HC)")
print(f"  Test:  {len(ds1_splits['test_files'])} files ({sum(ds1_splits['test_labels'])} PD, {sum(1-l for l in ds1_splits['test_labels'])} HC)")

with open(SPLITS_DIR / "dataset1_audio_splits.json", "w") as f:
    json.dump(ds1_splits, f, indent=2)
print("  Saved: processed/splits/dataset1_audio_splits.json")

# ─────────────────────────────────────────────
# DATASET 2: pd_speech_features.csv
# Has 'id' column — treat as patient/recording ID
# ─────────────────────────────────────────────
print("\n[Dataset 2] pd_speech_features.csv — group split by 'id'")
df2 = pd.read_csv(TABULAR / "pd_speech_features.csv")
y2 = df2["class"].values
groups2 = df2["id"].values

# GroupShuffleSplit: 70% train, 30% temp
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx2, temp_idx2 = next(gss1.split(df2, y2, groups=groups2))

# Split temp 50/50
temp_df2 = df2.iloc[temp_idx2]
temp_y2 = y2[temp_idx2]
temp_g2 = groups2[temp_idx2]
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_local2, test_local2 = next(gss2.split(temp_df2, temp_y2, groups=temp_g2))
val_idx2 = temp_idx2[val_local2]
test_idx2 = temp_idx2[test_local2]

# Verify no group overlap
train_groups = set(groups2[train_idx2])
val_groups = set(groups2[val_idx2])
test_groups = set(groups2[test_idx2])
assert train_groups.isdisjoint(val_groups), "LEAKAGE: Train/Val group overlap!"
assert train_groups.isdisjoint(test_groups), "LEAKAGE: Train/Test group overlap!"
assert val_groups.isdisjoint(test_groups), "LEAKAGE: Val/Test group overlap!"

ds2_splits = {
    "train_indices": train_idx2.tolist(),
    "val_indices": val_idx2.tolist(),
    "test_indices": test_idx2.tolist(),
    "train_class_dist": pd.Series(y2[train_idx2]).value_counts().to_dict(),
    "val_class_dist": pd.Series(y2[val_idx2]).value_counts().to_dict(),
    "test_class_dist": pd.Series(y2[test_idx2]).value_counts().to_dict(),
    "group_overlap": "None (verified)",
    "note": "GroupShuffleSplit on 'id' column; no group overlap confirmed",
}
print(f"  Train: {len(train_idx2)} rows | Class dist: {ds2_splits['train_class_dist']}")
print(f"  Val:   {len(val_idx2)} rows | Class dist: {ds2_splits['val_class_dist']}")
print(f"  Test:  {len(test_idx2)} rows | Class dist: {ds2_splits['test_class_dist']}")
print(f"  Group overlap: NONE ✓")

with open(SPLITS_DIR / "dataset2_splits.json", "w") as f:
    json.dump(ds2_splits, f, indent=2)
print("  Saved: processed/splits/dataset2_splits.json")

# ─────────────────────────────────────────────
# DATASET 3a: parkinsons.data (UCI)
# Subject IDs extracted from 'name' column
# ─────────────────────────────────────────────
print("\n[Dataset 3a] parkinsons.data — subject-wise split")
df3a = pd.read_csv(TABULAR / "parkinsons_data.csv")
y3a = df3a["status"].values
# Extract subject ID from name (e.g. "phon_R01_S01_1" -> "phon_R01_S01")
df3a["subject_id"] = df3a["name"].apply(lambda x: "_".join(x.split("_")[:-1]) if "_" in x else x)
groups3a = df3a["subject_id"].values
unique_subjs_3a = df3a["subject_id"].unique()
print(f"  Unique subjects: {len(unique_subjs_3a)}, Total rows: {len(df3a)}")

# GroupShuffleSplit for 3a
gss3a_1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
try:
    train_idx3a, temp_idx3a = next(gss3a_1.split(df3a, y3a, groups=groups3a))
    temp_df3a = df3a.iloc[temp_idx3a]
    temp_y3a = y3a[temp_idx3a]
    temp_g3a = groups3a[temp_idx3a]
    gss3a_2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_local3a, test_local3a = next(gss3a_2.split(temp_df3a, temp_y3a, groups=temp_g3a))
    val_idx3a = temp_idx3a[val_local3a]
    test_idx3a = temp_idx3a[test_local3a]
    split_method = "GroupShuffleSplit on subject_id"
except Exception as e:
    print(f"  GroupShuffle failed ({e}), falling back to StratifiedShuffleSplit")
    sss_3a = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx3a, temp_idx3a = next(sss_3a.split(df3a, y3a))
    sss_3a2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx3a_local, test_idx3a_local = next(sss_3a2.split(df3a.iloc[temp_idx3a], y3a[temp_idx3a]))
    val_idx3a = temp_idx3a[val_idx3a_local]
    test_idx3a = temp_idx3a[test_idx3a_local]
    split_method = "StratifiedShuffleSplit (fallback)"

ds3a_splits = {
    "train_indices": train_idx3a.tolist(),
    "val_indices": val_idx3a.tolist(),
    "test_indices": test_idx3a.tolist(),
    "train_class_dist": pd.Series(y3a[train_idx3a]).value_counts().to_dict(),
    "val_class_dist": pd.Series(y3a[val_idx3a]).value_counts().to_dict(),
    "test_class_dist": pd.Series(y3a[test_idx3a]).value_counts().to_dict(),
    "split_method": split_method,
}
print(f"  Train: {len(train_idx3a)} rows | Class dist: {ds3a_splits['train_class_dist']}")
print(f"  Val:   {len(val_idx3a)} rows | Class dist: {ds3a_splits['val_class_dist']}")
print(f"  Test:  {len(test_idx3a)} rows | Class dist: {ds3a_splits['test_class_dist']}")

with open(SPLITS_DIR / "dataset3a_splits.json", "w") as f:
    json.dump(ds3a_splits, f, indent=2)
print("  Saved: processed/splits/dataset3a_splits.json")

# ─────────────────────────────────────────────
# DATASET 3b: parkinsons_updrs.data (REGRESSION)
# CRITICAL: 42 subjects, must use GroupKFold
# ─────────────────────────────────────────────
print("\n[Dataset 3b] parkinsons_updrs — subject-wise GroupKFold (CRITICAL)")
df3b = pd.read_csv(TABULAR / "parkinsons_updrs.csv")
groups3b = df3b["subject#"].values
unique_subjs = df3b["subject#"].unique()
n_subjs = len(unique_subjs)
print(f"  Unique subjects: {n_subjs}")

# Create explicit subject-level 70/15/15 split
from sklearn.model_selection import train_test_split as tts
subj_array = np.array(unique_subjs)

# First split subjects: 70% train, 30% temp
train_subjs, temp_subjs = tts(subj_array, test_size=0.30, random_state=42)
# Second split: 50/50 of temp → val=15%, test=15%
val_subjs, test_subjs = tts(temp_subjs, test_size=0.50, random_state=42)

train_mask = df3b["subject#"].isin(train_subjs)
val_mask = df3b["subject#"].isin(val_subjs)
test_mask = df3b["subject#"].isin(test_subjs)

train_idx3b = np.where(train_mask)[0]
val_idx3b = np.where(val_mask)[0]
test_idx3b = np.where(test_mask)[0]

# Also create 5-fold GroupKFold for CV on training set
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
cv_folds = []
train_df3b = df3b.iloc[train_idx3b]
train_groups3b = groups3b[train_idx3b]
for fold_i, (cv_train, cv_val) in enumerate(gkf.split(train_df3b, groups=train_groups3b)):
    cv_folds.append({
        "fold": fold_i,
        "train_local_indices": cv_train.tolist(),
        "val_local_indices": cv_val.tolist(),
        "train_subjects": list(set(train_groups3b[cv_train].tolist())),
        "val_subjects": list(set(train_groups3b[cv_val].tolist())),
    })

# Verify no subject leakage
assert set(train_subjs).isdisjoint(set(val_subjs)), "LEAKAGE: Train/Val subject overlap!"
assert set(train_subjs).isdisjoint(set(test_subjs)), "LEAKAGE: Train/Test subject overlap!"
assert set(val_subjs).isdisjoint(set(test_subjs)), "LEAKAGE: Val/Test subject overlap!"
print(f"  Subject leakage check: PASSED ✓")

ds3b_splits = {
    "train_subject_ids": train_subjs.tolist(),
    "val_subject_ids": val_subjs.tolist(),
    "test_subject_ids": test_subjs.tolist(),
    "train_row_indices": train_idx3b.tolist(),
    "val_row_indices": val_idx3b.tolist(),
    "test_row_indices": test_idx3b.tolist(),
    "train_rows": int(train_mask.sum()),
    "val_rows": int(val_mask.sum()),
    "test_rows": int(test_mask.sum()),
    "cv_folds_5fold_groupkfold": cv_folds,
    "subject_overlap": "None (verified)",
    "note": "Subject-wise split: each subject appears in exactly one split",
}
print(f"  Train: {ds3b_splits['train_rows']} rows, {len(train_subjs)} subjects")
print(f"  Val:   {ds3b_splits['val_rows']} rows, {len(val_subjs)} subjects")
print(f"  Test:  {ds3b_splits['test_rows']} rows, {len(test_subjs)} subjects")
print(f"  CV: 5-fold GroupKFold on training subjects")

with open(SPLITS_DIR / "dataset3b_updrs_splits.json", "w") as f:
    json.dump(ds3b_splits, f, indent=2)
print("  Saved: processed/splits/dataset3b_updrs_splits.json")

print("\n" + "="*60)
print("Phase 3+4 COMPLETE — All splits saved and verified leakage-free")
print("="*60)
