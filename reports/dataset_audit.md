# ParkinsonXAI — Dataset Audit Report
**Generated:** 2026-08-15 01:44:04

## Summary

| Dataset | Type | Rows | Cols | Class/Target | Missing | Dup Rows | Notes |
|---------|------|------|------|--------------|---------|----------|-------|
| dataset_1 (Voice_Dataset WAV) | N/A | 1134 | N/A (audio) | label: {Healthy: 574, Parkinson: 560} | 0 | 0 | N/A |
| dataset_2 (pd_speech_features.csv) | CSV | 756 | 755 | class:{1: 564, 0: 192} | 0 | 1 | 753 TQWT + vocal features, id col, gender col, binary class target |
| dataset_3a (parkinsons.data) | CSV | 195 | 24 | status:{1: 147, 0: 48} | 0 | 0 | UCI Parkinson dataset; 22 voice features; 31 subjects with multiple recordings |
| dataset_3b (parkinsons_updrs.data) | CSV | 5875 | 22 | motor_UPDRS(min=5.04,max=39.51,mean=21.30) | total_UPDRS(min | 0 | 0 | UPDRS regression; 42 subjects, ~140 recordings/subject; motor + total UPDRS |
| dataset_4 (Parkinson_Multiple_Sound_Recording) | RAR nested in ZIP | UNKNOWN - RAR inaccessible | N/A | UNKNOWN | N/A | N/A | 7-Zip not installed; RAR could not be extracted; excluded from training |
| dataset_5 (parkinsons_updrs.data) | CSV | 5875 | 22 | motor_UPDRS(min=5.04,max=39.51,mean=21.30) | total_UPDRS(min | 0 | 0 | CONFIRMED DUPLICATE of dataset_3b (MD5 match: True). Excluded from training. |

## Dataset 1: Voice_Dataset (WAV Audio)
- **Healthy recordings:** 574
- **Parkinson recordings:** 560
- **Patient ID structure:** No embedded patient IDs in filenames. Each file treated as independent sample.
- **Role:** Primary audio dataset for detection model training.

## Dataset 2: pd_speech_features.csv
- **Shape:** (756, 755)
- **Features:** 753 TQWT-based vocal features + id + gender + binary class
- **Class distribution:** {1: 564, 0: 192}
- **Role:** Secondary detection dataset (pre-extracted features); rich feature space

## Dataset 3a: parkinsons.data (UCI)
- **Shape:** (195, 25)
- **Features:** 22 classical voice features (jitter, shimmer, HNR, RPDE, DFA, PPE, etc.)
- **Class distribution:** {1: 147, 0: 48}
- **Role:** Detection benchmark; also used for cross-dataset validation

## Dataset 3b: parkinsons_updrs.data
- **Shape:** (5875, 22)
- **Unique subjects:** 42
- **Features:** 16 voice features + demographics + test_time
- **Targets:** motor_UPDRS (5–40), total_UPDRS (7–55)
- **Role:** PRIMARY UPDRS severity regression dataset
- **⚠️ Critical:** Must use subject-wise GroupKFold to prevent data leakage

## Dataset 4: Parkinson_Multiple_Sound_Recording.rar
- **Status:** INACCESSIBLE (7-Zip not found)
- **Role:** Potentially additional audio; excluded if inaccessible

## Dataset 5: parkinsons_updrs.data
- **CONFIRMED EXACT DUPLICATE of Dataset 3b** (MD5 match: True)
- **Decision:** Excluded from training. Not treated as independent evidence.

## Duplicate Detection Summary

| Pair | Duplicate? | Action |
|------|-----------|--------|
| dataset_3b vs dataset_5 | **YES** (MD5: bf5a01c3eaf3a0d9...) | Use only dataset_3b |
| All other pairs | NO | Treated independently |
