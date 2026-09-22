# ParkinsonXAI — Preprocessing Report

## Dataset 2: pd_speech_features.csv

- Initial features: 752
- Missing values: 0 total → imputed with median (fit on train)
  [DS2] Removed 87 features (near-zero variance): ['stdDevPeriodPulses', 'locAbsJitter', 'mean_delta_delta_log_energy', 'mean_1st_delta_delta', 'mean_2nd_delta_delta']...
  [DS2] Removed 163 features (high correlation >0.97): ['numPeriodsPulses', 'ppq5Jitter', 'ddpJitter', 'locDbShimmer', 'apq3Shimmer']...
- Outliers (3*IQR): 15433 cells in 526 rows — documented only, not removed
- Scaling: RobustScaler fitted on training data only
- Train class distribution: {np.int64(1): 399, np.int64(0): 129} (ratio 3.09:1)
- **Final features after preprocessing: 502**

## Dataset 3a: parkinsons.data (UCI)

- Initial features: 22
  [DS3a] Removed 1 features (near-zero variance): ['MDVP:Jitter(Abs)']
- Train class distribution: {np.int64(1): 98, np.int64(0): 36}
- Scaling: RobustScaler fitted on train only
- **Final features: 21**

## Dataset 3b: parkinsons_updrs.data (UPDRS Regression)

- Initial features: 19
  [DS3b] Removed 1 features (near-zero variance): ['Jitter(Abs)']
- motor_UPDRS train range: [6.01, 39.51]
- total_UPDRS train range: [7.09, 54.99]
- Scaling: RobustScaler fitted on train only
- **Final features: 18**

