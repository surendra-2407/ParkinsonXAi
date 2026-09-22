# ParkinsonXAI — Dataset Roles

## Role Assignment

| Role | Dataset | Rationale |
|------|---------|-----------|
| **Primary Detection Training** | dataset_1 (Voice_Dataset WAV) | Raw audio → feature extraction → largest balanced dataset |
| **Secondary Detection (pre-extracted)** | dataset_2 (pd_speech_features.csv) | 753 rich features, useful for feature importance comparison |
| **Detection Benchmark** | dataset_3a (parkinsons.data) | UCI gold-standard 22-feature dataset; cross-dataset validation |
| **Primary Severity/UPDRS Regression** | dataset_3b (parkinsons_updrs.data) | 42 subjects, longitudinal UPDRS measurements |
| **EXCLUDED (Duplicate)** | dataset_5 | Exact copy of dataset_3b |
| **Excluded if inaccessible** | dataset_4 | RAR within ZIP; requires 7-Zip |

## Compatibility Assessment

### Can dataset_2 and dataset_3a be merged?

**NO.** Reasons:
- dataset_2 has 753 TQWT-derived features; dataset_3a has 22 classical features
- Different feature definitions and extraction methodologies
- No shared subject/patient identifiers
- Class proportions differ (but both binary PD/Healthy)
- They should be treated as SEPARATE EXPERIMENTS for cross-dataset validation

### Can dataset_3b UPDRS features be merged with detection datasets?

**PARTIALLY.** dataset_3b shares 7 features with dataset_3a (Jitter, Shimmer, NHR, HNR, RPDE, DFA, PPE),
but dataset_3b is a LONGITUDINAL dataset (multiple timepoints per subject) while dataset_3a is cross-sectional.
They are NOT directly mergeable without careful alignment.

### Decision: THREE SEPARATE EXPERIMENTS

1. **Experiment A:** Audio-based detection (dataset_1 WAV → extract features → detect PD)
2. **Experiment B:** Tabular detection benchmark (dataset_3a UCI → classical voice features)
3. **Experiment C:** UPDRS severity regression (dataset_3b → predict motor_UPDRS + total_UPDRS)

Cross-validation of Experiment B features against Experiment A features will be reported.
