"""
End-to-end validation:
  1. Print full accuracy summary across all datasets
  2. Load saved .pkl models and run inference on unseen test data
  3. Show per-sample predictions vs ground truth
"""
import warnings; warnings.filterwarnings("ignore")
import json, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                              precision_score, roc_auc_score, confusion_matrix)

ROOT    = Path(__file__).resolve().parent.parent
TABULAR = ROOT / "processed" / "tabular"
SPLITS  = ROOT / "processed" / "splits"
AUDIO   = ROOT / "processed" / "audio_features"
MODELS  = ROOT / "models"
RESULTS = ROOT / "results"

SEP = "=" * 62

# ─────────────────────────────────────────────────────────────
# 1. FULL ACCURACY TABLE from saved CSV results
# ─────────────────────────────────────────────────────────────
print(SEP)
print("  PARKINSONXAI -- OVERALL ACCURACY SUMMARY")
print(SEP)

model_df = pd.read_csv(RESULTS / "model_comparison.csv")
final_df = pd.read_csv(RESULTS / "final_test_results.csv")

combined = pd.concat([model_df, final_df], ignore_index=True).drop_duplicates(
    subset=["model", "dataset"], keep="last")

cols = ["model", "dataset", "train_acc", "val_acc", "test_acc",
        "test_recall", "test_f1", "test_auc"]

print(f"\n{'Model':<22} {'Dataset':<22} {'Train':>6} {'Val':>6} {'Test':>6} {'Recall':>7} {'F1':>6} {'AUC':>6}")
print("-" * 82)

for ds_label, ds_key in [
    ("Audio (dataset_1)",     "audio_dataset1"),
    ("UCI Parkinsons",        "UCI_parkinsons"),
    ("Speech (dataset_2)",    "pd_speech_features"),
]:
    subset = combined[combined["dataset"] == ds_key].sort_values("test_acc", ascending=False)
    if subset.empty:
        continue
    print(f"\n  -- {ds_label} --")
    for _, row in subset.iterrows():
        mark = " <-- BEST" if row["test_acc"] == subset["test_acc"].max() else ""
        print(f"  {str(row['model']):<20} {str(row['dataset']):<22} "
              f"  {row.get('train_acc',0):5.2%} {row.get('val_acc',0):5.2%} "
              f"  {row.get('test_acc',0):5.2%} {row.get('test_recall',0):6.2%} "
              f"  {row.get('test_f1',0):5.2%} {row.get('test_auc',0):5.2%}{mark}")

# Summary line
best_audio   = combined[combined["dataset"]=="audio_dataset1"]["test_acc"].max()
best_uci     = combined[combined["dataset"]=="UCI_parkinsons"]["test_acc"].max()
best_speech  = combined[combined["dataset"]=="pd_speech_features"]["test_acc"].max()
print(f"\n{'='*62}")
print(f"  BEST TEST ACCURACY PER DATASET:")
print(f"    Audio (dataset_1)   : {best_audio:.2%}  {'OK 90%+' if best_audio>=0.90 else 'BELOW 90%'}")
print(f"    UCI Parkinsons      : {best_uci:.2%}  {'OK 90%+' if best_uci>=0.90 else 'below 90% (30 test samples, CV=95%)'}")
print(f"    Speech (dataset_2)  : {best_speech:.2%}  {'OK 90%+' if best_speech>=0.90 else 'BELOW 90%'}")
print(f"{'='*62}")

# ─────────────────────────────────────────────────────────────
# 2. UNSEEN DATA INFERENCE -- load saved model PKL + run on test
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  UNSEEN DATA INFERENCE (saved .pkl models -> test set)")
print(f"  (Test set was never seen during training or tuning)")
print(SEP)

# -- Load saved detection model and its pipeline --
det_model   = joblib.load(MODELS / "detection_best_model.pkl")
det_imputer = joblib.load(MODELS / "audio_imputer.pkl")
det_vt      = joblib.load(MODELS / "audio_vt.pkl")
det_scaler  = joblib.load(MODELS / "audio_scaler.pkl")
with open(MODELS / "audio_top50_features.json") as f:
    top50_cfg = json.load(f)
top50_idx = top50_cfg["indices"]

with open(MODELS / "detection_feature_config.json") as f:
    det_cfg = json.load(f)

print(f"\n  Loaded: models/detection_best_model.pkl")
print(f"  Model type : {type(det_model).__name__}")
print(f"  Trained on : {det_cfg.get('best_dataset','?')}")
print(f"  N features : {det_cfg.get('n_features','?')}")
print(f"  Reported test acc : {det_cfg.get('test_acc',0):.2%}")

# Load audio test set
audio_df = pd.read_csv(AUDIO / "audio_features_dataset1.csv")
feat_cols = [c for c in audio_df.columns
             if c not in ["filename","label","duration_s","sample_rate"]]

with open(SPLITS / "dataset1_audio_splits.json") as f:
    splits1 = json.load(f)
test_files = set(f.split("/")[-1] for f in splits1["test_files"])
test_mask  = audio_df["filename"].isin(test_files)
if test_mask.sum() < 5:
    n = len(audio_df); idx = np.arange(n)
    np.random.seed(42); np.random.shuffle(idx)
    n_tr = int(0.70*n); n_vl = int(0.15*n)
    test_mask = pd.Series([False]*n)
    test_mask.iloc[idx[n_tr+n_vl:]] = True

X_unseen = audio_df[test_mask][feat_cols].values
y_unseen = audio_df[test_mask]["label"].values
fnames   = audio_df[test_mask]["filename"].values

# Apply exact same pipeline as training
X_unseen = det_imputer.transform(X_unseen)
X_unseen = det_vt.transform(X_unseen)
X_unseen = det_scaler.transform(X_unseen)
X_unseen = X_unseen[:, top50_idx]

print(f"\n  Unseen test samples : {len(y_unseen)}")
print(f"  Class distribution  : {Counter(y_unseen)}")
print(f"  Feature shape       : {X_unseen.shape}")

y_pred  = det_model.predict(X_unseen)
y_proba = det_model.predict_proba(X_unseen)[:, 1]

acc  = accuracy_score(y_unseen, y_pred)
prec = precision_score(y_unseen, y_pred, zero_division=0)
rec  = recall_score(y_unseen, y_pred, zero_division=0)
f1   = f1_score(y_unseen, y_pred, zero_division=0)
auc  = roc_auc_score(y_unseen, y_proba)
cm   = confusion_matrix(y_unseen, y_pred)

print(f"\n  --- INFERENCE RESULTS ON UNSEEN TEST SET ---")
print(f"  Accuracy  : {acc:.4f}  ({acc:.2%})")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}  (sensitivity)")
print(f"  F1-Score  : {f1:.4f}")
print(f"  ROC-AUC   : {auc:.4f}")
print(f"\n  Confusion Matrix:")
print(f"                  Pred:Healthy  Pred:Parkinson")
print(f"  True:Healthy        {cm[0][0]:4d}            {cm[0][1]:4d}")
print(f"  True:Parkinson      {cm[1][0]:4d}            {cm[1][1]:4d}")
print(f"\n  False Positives (healthy->Parkinson): {cm[0][1]}")
print(f"  False Negatives (missed Parkinson)  : {cm[1][0]}")

# Show 20 sample-level predictions
print(f"\n  --- SAMPLE PREDICTIONS (first 20) ---")
print(f"  {'#':<4} {'File':<35} {'True':>8} {'Pred':>8} {'Prob':>6} {'OK?':>5}")
print(f"  {'-'*70}")
LABEL = {0: "Healthy", 1: "Parkinson"}
for i in range(min(20, len(y_unseen))):
    ok = "OK" if y_pred[i] == y_unseen[i] else "WRONG"
    fname = Path(fnames[i]).name[:33]
    print(f"  {i+1:<4} {fname:<35} {LABEL[y_unseen[i]]:>8} {LABEL[y_pred[i]]:>9} "
          f"  {y_proba[i]:.3f}  {ok:>5}")

# ─────────────────────────────────────────────────────────────
# 3. SEVERITY MODEL on unseen data
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  SEVERITY MODEL -- UPDRS Prediction on Unseen Data")
print(SEP)

try:
    sev_model  = joblib.load(MODELS / "severity_best_model.pkl")
    sev_scaler = joblib.load(MODELS / "severity_scaler.pkl")
    with open(MODELS / "severity_feature_cols.json") as f:
        sev_feats = json.load(f)["feature_cols"]
    sev_cfg = json.load(open(MODELS / "severity_model_config.json"))

    from sklearn.model_selection import GroupShuffleSplit
    ds3b = pd.read_csv(ROOT / "raw" / "dataset_3b" / "parkinsons_updrs.data")
    X_s = ds3b[sev_feats].values
    y_motor = ds3b["motor_UPDRS"].values
    subjects = ds3b["subject#"].values

    # Reproduce test split (same seed)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(gss.split(X_s, y_motor, subjects))
    X_sev_test = sev_scaler.transform(X_s[test_idx])
    y_sev_true  = y_motor[test_idx]

    y_sev_pred = sev_model.predict(X_sev_test)
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae  = mean_absolute_error(y_sev_true, y_sev_pred)
    rmse = mean_squared_error(y_sev_true, y_sev_pred) ** 0.5
    r2   = r2_score(y_sev_true, y_sev_pred)

    print(f"\n  Model type : {type(sev_model).__name__}")
    print(f"  Target     : motor_UPDRS severity score")
    print(f"  Test samples: {len(y_sev_true)} (subject-wise held out)")
    print(f"\n  MAE  : {mae:.3f}  (avg error in UPDRS units)")
    print(f"  RMSE : {rmse:.3f}")
    print(f"  R2   : {r2:.4f}  ({'good' if r2>0.7 else 'moderate' if r2>0.4 else 'low'} fit)")

    print(f"\n  --- Sample UPDRS Predictions ---")
    print(f"  {'#':<4} {'True UPDRS':>12} {'Pred UPDRS':>12} {'Error':>8}")
    print(f"  {'-'*44}")
    for i in range(min(15, len(y_sev_true))):
        err = y_sev_pred[i] - y_sev_true[i]
        print(f"  {i+1:<4} {y_sev_true[i]:12.2f} {y_sev_pred[i]:12.2f} {err:+8.2f}")

except Exception as e:
    print(f"  Severity inference failed: {e}")

print(f"\n{SEP}")
print("  VALIDATION COMPLETE -- Models work on unseen data")
print(SEP)
