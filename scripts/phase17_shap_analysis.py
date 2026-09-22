#!/usr/bin/env python3
"""
Phase 17: SHAP / Explainable AI Analysis
ParkinsonXAI Project

Generates:
  - Global SHAP feature importance (detection + severity)
  - SHAP summary plot
  - SHAP waterfall plots (5 sample predictions)
  - Feature importance bar chart
  - SHAP-based feature ranking saved to results/

Outputs:
  plots/shap_summary.png
  plots/shap_waterfall.png
  plots/feature_importance.png
  results/shap_feature_ranking.csv
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABULAR = ROOT / "processed" / "tabular"
SPLITS = ROOT / "processed" / "splits"
AUDIO_FEATURES = ROOT / "processed" / "audio_features"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"

print("="*60)
print("Phase 17: SHAP / Explainable AI")
print("="*60)

# ─────────────────────────────────────────────
# Load best detection model and its data
# ─────────────────────────────────────────────
detection_model = joblib.load(MODELS / "detection_best_model.pkl")
with open(MODELS / "detection_feature_config.json") as f:
    det_config = json.load(f)

best_dataset = det_config["best_dataset"]
features_used = det_config["features_used"]
print(f"  Detection model: {det_config['best_model']} on {best_dataset}")
print(f"  Features: {len(features_used)}")

# ─────────────────────────────────────────────
# Load the right feature data for SHAP
# ─────────────────────────────────────────────
if best_dataset == "audio_dataset1":
    audio_df = pd.read_csv(AUDIO_FEATURES / "audio_features_dataset1.csv")
    with open(SPLITS / "dataset1_audio_splits.json") as f:
        splits1 = json.load(f)
    feat_audio_all = [c for c in audio_df.columns
                      if c not in ["filename", "label", "duration_s", "sample_rate"]]
    y_full = audio_df["label"].values

    # Rebuild preprocessors
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler
    from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

    imp_a = joblib.load(MODELS / "audio_imputer.pkl") if (MODELS / "audio_imputer.pkl").exists() else SimpleImputer(strategy="median")
    vt_a = joblib.load(MODELS / "audio_vt.pkl") if (MODELS / "audio_vt.pkl").exists() else VarianceThreshold(1e-6)
    scl_a = joblib.load(MODELS / "audio_scaler.pkl") if (MODELS / "audio_scaler.pkl").exists() else RobustScaler()

    X_raw = audio_df[feat_audio_all].values
    X_raw = imp_a.fit_transform(X_raw) if not hasattr(imp_a, "statistics_") else imp_a.transform(X_raw)
    X_raw = vt_a.fit_transform(X_raw) if not hasattr(vt_a, "variances_") else vt_a.transform(X_raw)
    X_sc = scl_a.fit_transform(X_raw) if not hasattr(scl_a, "center_") else scl_a.transform(X_raw)
    feat_filt = [c for c, k in zip(feat_audio_all, vt_a.get_support()) if k]

    # Top-50 MI features
    if len(features_used) <= len(feat_filt):
        top50_path = MODELS / "audio_top50_features.json"
        if top50_path.exists():
            with open(top50_path) as f:
                top50_info = json.load(f)
            top50_idx = np.array(top50_info["indices"])
        else:
            mi_s = mutual_info_classif(X_sc[:500], y_full[:500], random_state=42)
            top50_idx = np.argsort(mi_s)[::-1][:50]
        X_shap = X_sc[:, top50_idx]
        shap_feature_names = [feat_filt[i] for i in top50_idx]
    else:
        X_shap = X_sc
        shap_feature_names = feat_filt

    # Split
    train_files_set = set([f.split("/")[-1] for f in splits1["train_files"]])
    test_files_set = set([f.split("/")[-1] for f in splits1["test_files"]])
    train_m = audio_df["filename"].isin(train_files_set)
    test_m = audio_df["filename"].isin(test_files_set)
    if train_m.sum() < 10:
        n = len(audio_df)
        idx = np.arange(n); np.random.seed(42); np.random.shuffle(idx)
        n_train = int(0.7*n); n_val = int(0.15*n)
        train_m = pd.Series([False]*n)
        test_m = pd.Series([False]*n)
        train_m.iloc[idx[:n_train]] = True
        test_m.iloc[idx[n_train+n_val:]] = True
    X_train_shap = X_shap[train_m.values]
    X_test_shap = X_shap[test_m.values]
    y_test_shap = y_full[test_m.values]

elif best_dataset == "UCI_parkinsons":
    df3a_train = pd.read_csv(TABULAR / "ds3a_train.csv")
    df3a_test = pd.read_csv(TABULAR / "ds3a_test.csv")
    feat_3a = [c for c in df3a_train.columns if c != "target"]
    X_train_shap = df3a_train[feat_3a].values
    X_test_shap = df3a_test[feat_3a].values
    y_test_shap = df3a_test["target"].values.astype(int)
    shap_feature_names = feat_3a

else:
    print("Unknown dataset, skipping SHAP")
    exit(0)

print(f"  SHAP features: {len(shap_feature_names)}")
print(f"  Background samples: {len(X_train_shap)}")
print(f"  Test samples: {len(X_test_shap)}")

# ─────────────────────────────────────────────
# Create SHAP explainer
# ─────────────────────────────────────────────
model_type = type(detection_model).__name__
print(f"\n  Creating SHAP explainer for {model_type}...")

# Try TreeExplainer first (fast for tree-based models)
try:
    if hasattr(detection_model, "estimators_") or "XGB" in model_type or "LGBM" in model_type \
       or "CatBoost" in model_type or "Forest" in model_type or "Tree" in model_type \
       or "Voting" in model_type or "Stacking" in model_type:

        if "Voting" in model_type or "Stacking" in model_type:
            # Use the first tree-based estimator
            for name, est in detection_model.estimators:
                if hasattr(est, "feature_importances_"):
                    explainer = shap.TreeExplainer(est)
                    break
            else:
                raise ValueError("No tree estimator found")
        else:
            explainer = shap.TreeExplainer(detection_model)
        print("  Using TreeExplainer (fast)")
        use_background_for_shap = False
    else:
        raise ValueError("Not a tree model")
except Exception as e:
    print(f"  TreeExplainer failed ({e}), using KernelExplainer...")
    # Use small background for speed
    n_bg = min(100, len(X_train_shap))
    bg_idx = np.random.choice(len(X_train_shap), n_bg, replace=False)
    bg_data = shap.kmeans(X_train_shap[bg_idx], 20)
    explainer = shap.KernelExplainer(
        lambda x: detection_model.predict_proba(x)[:, 1],
        bg_data
    )
    print(f"  Using KernelExplainer with {n_bg} background samples")
    use_background_for_shap = True

# Compute SHAP values
n_test_shap = min(200, len(X_test_shap))
X_test_shap_sample = X_test_shap[:n_test_shap]
y_test_sample = y_test_shap[:n_test_shap]

print(f"  Computing SHAP values for {n_test_shap} test samples...")
try:
    shap_values = explainer.shap_values(X_test_shap_sample)
    if isinstance(shap_values, list):
        # Binary classification: take class 1 values
        shap_values_cls1 = shap_values[1]
    else:
        shap_values_cls1 = shap_values
    print(f"  SHAP values shape: {np.array(shap_values_cls1).shape}")
except Exception as e:
    print(f"  SHAP computation failed: {e}")
    shap_values_cls1 = None

if shap_values_cls1 is not None:
    shap_arr = np.array(shap_values_cls1)
    # Handle extra dims
    if shap_arr.ndim == 3:
        shap_arr = shap_arr[:, :, 1]

    # ── Global SHAP summary plot ──
    print("\n  Generating SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_arr, X_test_shap_sample,
                      feature_names=shap_feature_names,
                      show=False, max_display=25)
    plt.title("SHAP Summary Plot — Detection Model", fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(PLOTS / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: plots/shap_summary.png")

    # ── Feature importance bar chart ──
    print("  Generating SHAP feature importance bar chart...")
    mean_abs_shap = np.abs(shap_arr).mean(axis=0)
    shap_importance_df = pd.DataFrame({
        "feature": shap_feature_names[:len(mean_abs_shap)],
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)
    shap_importance_df.to_csv(RESULTS / "shap_feature_ranking.csv", index=False)

    top_n = min(25, len(shap_importance_df))
    fig, ax = plt.subplots(figsize=(10, 7))
    top_feats = shap_importance_df.head(top_n)
    colors = plt.cm.YlOrRd(np.linspace(0.4, 0.9, top_n))[::-1]
    bars = ax.barh(range(top_n), top_feats["mean_abs_shap"].values, color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_feats["feature"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("SHAP Feature Importance — Top Features for PD Detection",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(PLOTS / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: plots/feature_importance.png")

    # ── SHAP Waterfall plots (5 samples) ──
    print("  Generating SHAP waterfall plots...")
    # Find interesting samples: mix of correct/incorrect and PD/Healthy
    y_pred_sample = detection_model.predict(X_test_shap_sample)
    sample_indices = []
    # True Positive, True Negative, False Positive, False Negative, random
    for tgt, pred in [(1, 1), (0, 0), (0, 1), (1, 0)]:
        matches = np.where((y_test_sample == tgt) & (y_pred_sample == pred))[0]
        if len(matches) > 0:
            sample_indices.append(matches[0])
    while len(sample_indices) < 5:
        candidate = np.random.randint(n_test_shap)
        if candidate not in sample_indices:
            sample_indices.append(candidate)
    sample_indices = sample_indices[:5]

    n_wf_feats = min(15, len(shap_feature_names))
    fig, axes = plt.subplots(1, len(sample_indices), figsize=(5 * len(sample_indices), 6))
    if len(sample_indices) == 1:
        axes = [axes]

    for plot_i, sample_i in enumerate(sample_indices):
        ax = axes[plot_i]
        sv = shap_arr[sample_i][:n_wf_feats]
        feat_names_short = [f[:15] for f in shap_feature_names[:n_wf_feats]]
        sorted_idx = np.argsort(np.abs(sv))[::-1]

        colors_wf = ["#E53935" if v > 0 else "#1E88E5" for v in sv[sorted_idx]]
        ax.barh(range(len(sorted_idx)), sv[sorted_idx], color=colors_wf)
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feat_names_short[i] for i in sorted_idx], fontsize=7)
        ax.invert_yaxis()
        true_label = "Parkinson" if y_test_sample[sample_i] == 1 else "Healthy"
        pred_label = "Parkinson" if y_pred_sample[sample_i] == 1 else "Healthy"
        correct = "✓" if y_test_sample[sample_i] == y_pred_sample[sample_i] else "✗"
        ax.set_title(f"Sample {sample_i}\nTrue: {true_label}\nPred: {pred_label} {correct}",
                     fontsize=8, fontweight="bold")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP Value", fontsize=7)
        ax.grid(True, alpha=0.2, axis="x")

    plt.suptitle("SHAP Waterfall Plots — Individual Predictions", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS / "shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: plots/shap_waterfall.png")

    # Print top features
    print(f"\n  Top 10 most important SHAP features:")
    for i, row in shap_importance_df.head(10).iterrows():
        print(f"    {row['feature']:<40} {row['mean_abs_shap']:.5f}")

# ─────────────────────────────────────────────
# SHAP for Severity Model
# ─────────────────────────────────────────────
print("\n" + "-"*50)
print("  SHAP for Severity Model (motor_UPDRS)")
print("-"*50)

severity_model = joblib.load(MODELS / "severity_best_model.pkl")
df3b_train = pd.read_csv(TABULAR / "ds3b_train.csv")
df3b_test = pd.read_csv(TABULAR / "ds3b_test.csv")
feat_sev = [c for c in df3b_train.columns if c not in ["motor_UPDRS", "total_UPDRS"]]
X_sev_tr = df3b_train[feat_sev].values
X_sev_ts = df3b_test[feat_sev].values

try:
    sev_explainer = shap.TreeExplainer(severity_model)
    shap_sev = sev_explainer.shap_values(X_sev_ts[:100])
    if isinstance(shap_sev, list):
        shap_sev = shap_sev[0]

    shap_sev_arr = np.array(shap_sev)
    mean_abs_sev = np.abs(shap_sev_arr).mean(axis=0)
    sev_importance_df = pd.DataFrame({
        "feature": feat_sev[:len(mean_abs_sev)],
        "mean_abs_shap": mean_abs_sev
    }).sort_values("mean_abs_shap", ascending=False)

    print("  Top 10 SHAP features for motor_UPDRS prediction:")
    for _, row in sev_importance_df.head(10).iterrows():
        print(f"    {row['feature']:<30} {row['mean_abs_shap']:.5f}")

    sev_importance_df.to_csv(RESULTS / "shap_severity_feature_ranking.csv", index=False)
    print("  Saved: results/shap_severity_feature_ranking.csv")
except Exception as e:
    print(f"  Severity SHAP failed: {e}")

print("\n" + "="*60)
print("Phase 17 COMPLETE")
print("="*60)
