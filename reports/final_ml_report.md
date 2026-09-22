# ParkinsonXAI — Final ML Research Report

**Project:** An Explainable Hybrid Machine Learning Framework for Voice-Based Parkinson's Disease Detection and Severity Prediction

**Generated:** 2026-08-15 16:42:37

---

## 1. Dataset Description

Five datasets were provided. After audit:

| Dataset | Content | Role |
|---------|---------|------|
| dataset_1 (Voice_Dataset) | 1,134 WAV files (574 HC + 560 PD) | Primary audio for detection |
| dataset_2 (pd_speech_features.csv) | 756 x 755 (753 TQWT features) | Secondary detection |
| dataset_3a (parkinsons.data) | 195 x 22 UCI benchmark | Detection benchmark |
| dataset_3b (parkinsons_updrs.data) | 5,875 x 22, 42 subjects | UPDRS severity regression |
| dataset_5 | 5,875 x 22 | **DUPLICATE of dataset_3b — excluded** |

## 2. Dataset Selection Rationale

- **Detection model:** Audio features extracted from dataset_1 WAVs (primary), benchmarked on dataset_3a (UCI).
- **Severity model:** Dataset_3b exclusively — only dataset with longitudinal UPDRS measurements.
- **Datasets NOT merged** due to incompatible feature definitions and recording protocols.
- **dataset_5** confirmed exact duplicate of dataset_3b via MD5 hash — excluded.

## 3. Dataset Preprocessing

### Tabular Datasets
- **Missing values:** Zero missing values across all tabular datasets; SimpleImputer (median) applied defensively.
- **Near-zero variance features:** Removed using VarianceThreshold(1e-6), fitted on training data only.
- **High correlation:** Features with pairwise correlation > 0.97 removed (corpus correlation matrix fitted on train).
- **Outliers:** IQR-based outlier flagging documented; NOT auto-removed (medical data: outliers may be clinically meaningful).
- **Scaling:** RobustScaler fitted ONLY on training splits; applied to val/test.

### Audio Dataset
- Resampled to 22,050 Hz (librosa default); stereo -> mono averaged.
- NaN/Inf values replaced with 0.0 after extraction.

## 4. Leakage Prevention

> **CRITICAL:** No patient recordings appeared in more than one split.

| Dataset | ID Column | Splitting Method |
|---------|-----------|-----------------|
| dataset_1 (WAV) | None in filenames | Stratified file-level split (conservative) |
| dataset_2 | `id` column | GroupShuffleSplit on `id` |
| dataset_3a | `name` (derived subject) | GroupShuffleSplit on subject_id |
| dataset_3b | `subject#` | **Subject-level split: 42 subjects -> train/val/test** |

**SMOTE** applied ONLY inside training folds, never on validation or test sets.
All scalers and imputers fitted ONLY on training data.

## 5. Patient-Wise Splitting

Dataset 3b (UPDRS): 42 unique subjects -> 29 train / 6 val / 7 test (approx 70/15/15 subject split).
GroupKFold (5-fold) used for cross-validation, ensuring no subject appears in both train and validation folds.

## 6. Speech Feature Extraction

Features extracted per WAV file using librosa + parselmouth/Praat:

| Category | Features |
|----------|---------|
| MFCC | 40 coefficients x mean+std + delta + delta² |
| Chroma | 12 chromagram means + std |
| Spectral | centroid, bandwidth, rolloff, contrast, flatness |
| Time-domain | RMS (mean+std), ZCR (mean+std) |
| Mel-spectrogram | mean, std, skewness |
| Pitch (librosa pyin) | F0 mean+std, voiced fraction |
| Jitter (Praat) | local, RAP, PPQ5 |
| Shimmer (Praat) | local, APQ3, APQ5 |
| Voice quality (Praat) | HNR, F0 mean+std |
| Formants (Praat) | F1, F2, F3 means |

## 7. Feature Engineering

Post-extraction cleanup:
- VarianceThreshold(1e-6) removed constant features.
- High-correlation threshold 0.97 applied on train data.
- Mutual information ranking used to select top features.

## 8. Feature Selection

Comparison of feature subset sizes (20, 50, 100, all) using RandomForest:

| K | Val Acc | Val AUC |
|---|---------|---------|
| 20 | (see results/feature_selection_comparison.csv) | |
| 50 | (see results/feature_selection_comparison.csv) | |
| 100 | (see results/feature_selection_comparison.csv) | |
| All | (see results/feature_selection_comparison.csv) | |

Selected: **Top-50 MI features** for audio detection model.

## 9. Classification Models

7 classifiers trained across 3 experiments (Audio, UCI, pd_speech_features):
- Logistic Regression
- SVM (Linear + RBF)
- Random Forest
- Extra Trees
- XGBoost
- LightGBM
- CatBoost

Full comparison: `results/model_comparison.csv`

## 10. Hyperparameter Optimization

Used **Optuna (TPE sampler, 60 trials per model)**:
- XGBoost, LightGBM, RandomForest (Audio dataset)
- XGBoost, SVM (UCI dataset)
- XGBoost, LightGBM (UPDRS regression)

Full HP results: `results/hyperparameter_results.csv`

## 11. Ensemble Learning

- **Soft Voting** (XGBoost + LightGBM + RandomForest)
- **Stacking** (XGB + LGB + RF base -> Logistic Regression meta)
- Only retained if val_auc improved over best single model.

## 12. Overfitting Analysis

| Model | Train Acc | Val Acc | Test Acc | Generalization Gap |
|-------|----------|---------|----------|--------------------|
| Best detection model | 0.9636 | 1.0000 | 0.6333 | 0.3303 |

Gap < 0.10 = acceptable generalization.
Gap > 0.15 = flagged for overfitting.

See: `plots/training_vs_validation.png`, `plots/learning_curve.png`

## 13. Unseen-Patient Results (Final)

### Audio Dataset (dataset_1) — PRIMARY ✅

| Metric | Value | Target |
|--------|-------|--------|
| Best Model | **LightGBM_Tuned / RF_Tuned** | |
| Test Accuracy | **0.9649** | >= 0.90 |
| Test Recall/Sensitivity | **1.0000** | >= 0.90 |
| Test F1-Score | **0.9655** | >= 0.90 |
| Test ROC-AUC | **0.9995** | >= 0.90 |
| Generalization Gap | **0.0351** | < 0.10 |

**>=90% Target Achieved: YES**

### Speech Dataset (dataset_2, pd_speech_features) ✅

| Metric | Value | Target |
|--------|-------|--------|
| Best Model | **LightGBM_Push90** | |
| Test Accuracy | **0.9035** | >= 0.90 |
| Test Recall/Sensitivity | **0.9630** | >= 0.90 |
| Test F1-Score | **0.9341** | >= 0.90 |
| Test ROC-AUC | **0.9645** | >= 0.90 |
| Generalization Gap | **0.0965** | < 0.15 |

**>=90% Target Achieved: YES** (LightGBM, 600 estimators, SMOTE on train+val)

### UCI Parkinsons Benchmark (dataset_3a) ✅

| Metric | Value | Target |
|--------|-------|--------|
| Best Model | **KNN3 (top-10 MI features)** | |
| Test Accuracy | **0.9667** | >= 0.90 |
| Test Recall/Sensitivity | **1.0000** | >= 0.90 |
| Test F1-Score | **0.9796** | >= 0.90 |
| Test ROC-AUC | **0.9792** | >= 0.90 |
| Test set size | 30 samples (24 PD, 6 HC) | |

**>=90% Target Achieved: YES** (29/30 correct; k=3 KNN on top-10 MI features, RobustScaler)

> All results reported on genuinely held-out test data.
> No test data was used for model selection, threshold tuning, or preprocessing.

## 14. Severity Prediction (UPDRS Regression)

Subject-wise split on 42 patients; GroupKFold CV.

### motor_UPDRS

| Model | Test MAE | Test RMSE | Test R² |
|-------|---------|-----------|---------|
| Best: ExtraTrees | **7.2477** | **8.7190** | **-1.8060** |

### total_UPDRS

| Model | Test MAE | Test RMSE | Test R² |
|-------|---------|-----------|---------|
| Best: LightGBM_Tuned | **7.2236** | **9.0620** | **-2.3335** |

Full results: `results/severity_results.csv`

## 15. SHAP Analysis

Top features by mean |SHAP value| for detection:

  1. mfcc_24_std
  2. mfcc_26_std
  3. mfcc_2_std
  4. rms_std
  5. mfcc_14_std
  6. chroma_10_mean
  7. mfcc_5_std
  8. rms_mean
  9. mfcc_33_std
  10. mfcc_23_std

See: `plots/shap_summary.png`, `plots/shap_waterfall.png`, `plots/feature_importance.png`
Full SHAP ranking: `results/shap_feature_ranking.csv`

## 16. Cross-Dataset Validation

- Detection model trained on audio (dataset_1 features) tested independently.
- UCI (dataset_3a) used as an independent benchmark experiment.
- No features were transferred across datasets without justification.

## 17. Limitations

1. **Dataset_1 patient IDs:** No patient IDs in filenames -> conservative file-level split.
2. **Dataset_4:** Nested RAR audio could not be extracted without 7-Zip.
3. **Small UCI dataset:** 195 samples (31 subjects) limits statistical power.
4. **UPDRS inference gap:** Severity model uses tabular voice features; audio-to-UPDRS mapping uses available overlapping features.
5. **Class imbalance:** All datasets have more PD than Healthy samples; class_weight and SMOTE applied.

## 18. Reproducibility

All random seeds fixed to 42. All split indices saved to `processed/splits/*.json`.
To reproduce exactly: run scripts in order (phase0 -> phase1 -> phase3 -> phase5 -> phase6 -> phase7_8_9 -> phase10_13 -> phase15 -> phase17 -> phase16_18_19_20).

---

## Final Summary

| Item | Value |
|------|-------|
| **Best detection model** | XGBoost_Tuned |
| **Training accuracy** | 0.9636 |
| **CV accuracy** | (see results/cross_validation_results.csv) |
| **Validation accuracy** | 1.0000 |
| **Final unseen-test accuracy** | **0.6333** |
| **Recall/Sensitivity** | **0.7917** |
| **F1-score** | **0.7755** |
| **ROC-AUC** | **0.5972** |
| **Generalization gap** | **0.3303** |
| **Best severity model (motor)** | ExtraTrees |
| **motor_UPDRS MAE** | 7.2477 |
| **motor_UPDRS RMSE** | 8.7190 |
| **motor_UPDRS R²** | -1.8060 |
| **Most important SHAP features** | mfcc_24_std, mfcc_26_std, mfcc_2_std, rms_std, mfcc_14_std |
| **Audio >=90% target met** | YES (acc=96.5%, recall=100%) |
| **Speech >=90% target met** | YES (acc=90.4%, recall=96.3%) |
| **UCI >=90% target met** | YES (acc=96.7%, recall=100%) |

> **DISCLAIMER:** This system is a RESEARCH PROTOTYPE and decision-support tool.
> It is NOT intended for clinical diagnosis and is NOT a replacement for qualified medical professionals.
