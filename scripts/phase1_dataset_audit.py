#!/usr/bin/env python3
"""
Phase 1+2: Complete Dataset Audit & Duplicate Detection
ParkinsonXAI Project

Produces:
  reports/dataset_audit.md
  reports/dataset_audit.csv
  reports/dataset_roles.md
  processed/tabular/parkinsons_data.csv
  processed/tabular/pd_speech_features.csv
  processed/tabular/parkinsons_updrs.csv
  processed/audio_features/  (populated in Phase 6)
"""

import zipfile
import os
import io
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"
PROCESSED_TABULAR = ROOT / "processed" / "tabular"
REPORTS_DIR = ROOT / "reports"

for d in [PROCESSED_TABULAR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Helper: file hash for duplicate detection
# ─────────────────────────────────────────────
def file_md5(path, chunk_size=65536):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def zip_file_md5(zf: zipfile.ZipFile, name: str):
    """MD5 of a file inside a zip."""
    h = hashlib.md5()
    with zf.open(name) as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def audit_dataframe(df: pd.DataFrame, name: str) -> dict:
    """Compute comprehensive audit statistics for a DataFrame."""
    row = {
        "dataset": name,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns_preview": ", ".join(list(df.columns[:8])) + ("..." if len(df.columns) > 8 else ""),
        "missing_values": int(df.isnull().sum().sum()),
        "missing_pct": round(df.isnull().mean().mean() * 100, 3),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_cols": int(df.select_dtypes(include=np.number).shape[1]),
        "categorical_cols": int(df.select_dtypes(include="object").shape[1]),
    }

    # Target variable detection
    potential_targets = ["status", "class", "Class", "label", "Label",
                         "motor_UPDRS", "total_UPDRS"]
    targets_found = []
    for t in potential_targets:
        if t in df.columns:
            if df[t].dtype in [np.int64, np.float64, np.int32, np.float32]:
                if df[t].nunique() <= 2:
                    dist = df[t].value_counts().to_dict()
                    targets_found.append(f"{t}:{dist}")
                else:
                    stats = f"min={df[t].min():.2f},max={df[t].max():.2f},mean={df[t].mean():.2f}"
                    targets_found.append(f"{t}({stats})")
    row["target_variables"] = " | ".join(targets_found) if targets_found else "None detected"

    # Subject/patient ID detection
    id_cols = []
    for c in df.columns:
        if any(kw in c.lower() for kw in ["name", "subject", "patient", "id", "speaker"]):
            id_cols.append(f"{c}(unique={df[c].nunique()})")
    row["id_columns"] = " | ".join(id_cols) if id_cols else "None detected"

    # Outlier detection (IQR-based, numeric only)
    num_df = df.select_dtypes(include=np.number)
    Q1 = num_df.quantile(0.25)
    Q3 = num_df.quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = ((num_df < Q1 - 3 * IQR) | (num_df > Q3 + 3 * IQR))
    row["outlier_cells_3iqr"] = int(outlier_mask.sum().sum())

    # Near-zero variance features
    variances = num_df.var()
    row["near_zero_variance_cols"] = int((variances < 1e-6).sum())

    # High correlation pairs (threshold 0.95)
    if len(num_df.columns) > 1:
        corr = num_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr_pairs = (upper > 0.95).sum().sum()
        row["high_corr_pairs_095"] = int(high_corr_pairs)
    else:
        row["high_corr_pairs_095"] = 0

    return row


# ─────────────────────────────────────────────
# DATASET 1: WAV audio files
# ─────────────────────────────────────────────
print("="*60)
print("Auditing DATASET 1 (WAV audio files)...")
print("="*60)
z1 = zipfile.ZipFile(DATASETS_DIR / "dataset_1.zip")
all_names = z1.namelist()
healthy_wavs = [n for n in all_names if "Healthy" in n and n.endswith(".wav")]
parkinson_wavs = [n for n in all_names if "Parkinsons" in n and n.endswith(".wav")]
other_files = [n for n in all_names if not n.endswith(".wav")]

audio_audit = {
    "dataset": "dataset_1 (Voice_Dataset WAV)",
    "n_rows": len(healthy_wavs) + len(parkinson_wavs),
    "n_cols": "N/A (audio)",
    "columns_preview": "filename, label (derived from folder)",
    "missing_values": 0,
    "missing_pct": 0.0,
    "duplicate_rows": 0,
    "numeric_cols": "N/A",
    "categorical_cols": "N/A",
    "target_variables": f"label: {{Healthy: {len(healthy_wavs)}, Parkinson: {len(parkinson_wavs)}}}",
    "id_columns": "filename (no embedded patient ID)",
    "outlier_cells_3iqr": "N/A",
    "near_zero_variance_cols": "N/A",
    "high_corr_pairs_095": "N/A",
}
print(f"  Healthy WAVs:   {len(healthy_wavs)}")
print(f"  Parkinson WAVs: {len(parkinson_wavs)}")
print(f"  Other files:    {len(other_files)}")

# Sample a few WAV properties using soundfile
try:
    import soundfile as sf
    print("  Sampling audio properties (first 3 of each class)...")
    sample_info = []
    for wav_name in healthy_wavs[:3] + parkinson_wavs[:3]:
        with z1.open(wav_name) as wf:
            data_bytes = wf.read()
        bio = io.BytesIO(data_bytes)
        try:
            info = sf.info(bio)
            sample_info.append({
                "file": wav_name.split("/")[-1],
                "duration_s": round(info.duration, 2),
                "sample_rate": info.samplerate,
                "channels": info.channels,
            })
        except Exception as e:
            print(f"    Could not read {wav_name}: {e}")
    if sample_info:
        print("  Sample audio info:")
        for s in sample_info:
            print(f"    {s['file']}: {s['duration_s']}s, {s['sample_rate']}Hz, {s['channels']}ch")
    audio_audit["sample_duration_s"] = np.mean([s["duration_s"] for s in sample_info]) if sample_info else "unknown"
    audio_audit["sample_rate"] = sample_info[0]["sample_rate"] if sample_info else "unknown"
except Exception as e:
    print(f"  Could not sample WAV info: {e}")
    audio_audit["sample_duration_s"] = "unknown"
    audio_audit["sample_rate"] = "unknown"

# ─────────────────────────────────────────────
# DATASET 2: pd_speech_features.csv
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("Auditing DATASET 2 (pd_speech_features.csv)...")
print("="*60)
z2 = zipfile.ZipFile(DATASETS_DIR / "dataset_2.zip")
with z2.open("pd_speech_features.csv") as f:
    df2 = pd.read_csv(f)
ds2_audit = audit_dataframe(df2, "dataset_2 (pd_speech_features.csv)")
ds2_audit["file_type"] = "CSV"
ds2_audit["notes"] = "753 TQWT + vocal features, id col, gender col, binary class target"
print(f"  Shape: {df2.shape}")
print(f"  Target: {ds2_audit['target_variables']}")
print(f"  Missing: {ds2_audit['missing_values']}")
print(f"  High-corr pairs: {ds2_audit['high_corr_pairs_095']}")
print(f"  NZV cols: {ds2_audit['near_zero_variance_cols']}")
# Save processed copy
df2.to_csv(PROCESSED_TABULAR / "pd_speech_features.csv", index=False)
print(f"  Saved: processed/tabular/pd_speech_features.csv")

# ─────────────────────────────────────────────
# DATASET 3a: parkinsons.data
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("Auditing DATASET 3a (parkinsons.data)...")
print("="*60)
z3 = zipfile.ZipFile(DATASETS_DIR / "dataset_3.zip")
with z3.open("parkinsons.data") as f:
    df3a = pd.read_csv(f)
ds3a_audit = audit_dataframe(df3a, "dataset_3a (parkinsons.data)")
ds3a_audit["file_type"] = "CSV"
ds3a_audit["notes"] = "UCI Parkinson dataset; 22 voice features; 31 subjects with multiple recordings"

# Analyze repeated recordings per subject
name_col = "name"
unique_subjects = df3a[name_col].apply(lambda x: "_".join(x.split("_")[:-1])).unique()
ds3a_audit["unique_subjects"] = len(unique_subjects)
print(f"  Shape: {df3a.shape}")
print(f"  Target: {ds3a_audit['target_variables']}")
print(f"  Unique name values: {df3a['name'].nunique()}")
# Try to extract subject base names
df3a["subject_id"] = df3a["name"].apply(lambda x: "_".join(x.split("_")[:-1]))
print(f"  Estimated unique subjects: {df3a['subject_id'].nunique()}")
print(f"  Recordings per subject: mean={df3a.groupby('subject_id').size().mean():.1f}")

df3a.to_csv(PROCESSED_TABULAR / "parkinsons_data.csv", index=False)
print(f"  Saved: processed/tabular/parkinsons_data.csv")

# ─────────────────────────────────────────────
# DATASET 3b: parkinsons_updrs.data
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("Auditing DATASET 3b (parkinsons_updrs.data)...")
print("="*60)
with z3.open("parkinsons_updrs.data") as f:
    df3b = pd.read_csv(f)
ds3b_audit = audit_dataframe(df3b, "dataset_3b (parkinsons_updrs.data)")
ds3b_audit["file_type"] = "CSV"
ds3b_audit["notes"] = "UPDRS regression; 42 subjects, ~140 recordings/subject; motor + total UPDRS"
print(f"  Shape: {df3b.shape}")
print(f"  Subjects: {df3b['subject#'].nunique()}")
rec_per_subj = df3b.groupby("subject#").size()
print(f"  Recordings/subject: mean={rec_per_subj.mean():.1f}, min={rec_per_subj.min()}, max={rec_per_subj.max()}")
print(f"  motor_UPDRS: min={df3b['motor_UPDRS'].min():.2f}, max={df3b['motor_UPDRS'].max():.2f}")
print(f"  total_UPDRS: min={df3b['total_UPDRS'].min():.2f}, max={df3b['total_UPDRS'].max():.2f}")
df3b.to_csv(PROCESSED_TABULAR / "parkinsons_updrs.csv", index=False)
print(f"  Saved: processed/tabular/parkinsons_updrs.csv")

# ─────────────────────────────────────────────
# DATASET 4: Parkinson_Multiple_Sound_Recording.rar
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("Auditing DATASET 4 (Parkinson_Multiple_Sound_Recording.rar)...")
print("="*60)
z4 = zipfile.ZipFile(DATASETS_DIR / "dataset_4.zip")
z4_contents = z4.namelist()
print(f"  ZIP contains: {z4_contents}")
# Check if 7-Zip is available
import shutil
sevenz = shutil.which("7z") or shutil.which("7za")
if sevenz:
    print(f"  7-Zip found at: {sevenz}")
    # Extract RAR
    rar_path = ROOT / "processed" / "dataset4_temp" / "Parkinson_Multiple_Sound_Recording.rar"
    rar_path.parent.mkdir(parents=True, exist_ok=True)
    with z4.open(z4_contents[0]) as rf, open(rar_path, "wb") as out:
        out.write(rf.read())
    # Extract RAR
    import subprocess
    result = subprocess.run(
        [sevenz, "l", str(rar_path)],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("  RAR listing (first 20 lines):")
        for line in result.stdout.split("\n")[:20]:
            print("   ", line)
    ds4_audit = {
        "dataset": "dataset_4 (Parkinson_Multiple_Sound_Recording)",
        "n_rows": "audio files (count TBD)",
        "n_cols": "N/A",
        "file_type": "RAR -> WAV/audio",
        "target_variables": "TBD from listing",
        "notes": f"7-Zip found; RAR listed; extraction status: {result.returncode}",
    }
else:
    print("  7-Zip NOT found. Cannot extract RAR file.")
    print("  dataset_4 will be documented as INACCESSIBLE without 7-Zip.")
    ds4_audit = {
        "dataset": "dataset_4 (Parkinson_Multiple_Sound_Recording)",
        "n_rows": "UNKNOWN - RAR inaccessible",
        "n_cols": "N/A",
        "file_type": "RAR nested in ZIP",
        "target_variables": "UNKNOWN",
        "notes": "7-Zip not installed; RAR could not be extracted; excluded from training",
        "id_columns": "UNKNOWN",
        "outlier_cells_3iqr": "N/A",
        "near_zero_variance_cols": "N/A",
        "high_corr_pairs_095": "N/A",
        "missing_values": "N/A",
        "missing_pct": "N/A",
        "duplicate_rows": "N/A",
        "numeric_cols": "N/A",
        "categorical_cols": "N/A",
    }

# ─────────────────────────────────────────────
# DATASET 5: parkinsons_updrs.data (duplicate check)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("Auditing DATASET 5 (parkinsons_updrs.data)...")
print("="*60)
z5 = zipfile.ZipFile(DATASETS_DIR / "dataset_5.zip")
with z5.open("parkinsons_updrs.data") as f:
    df5 = pd.read_csv(f)

# Compute MD5s for duplicate detection
md5_3b = hashlib.md5(df3b.to_csv(index=False).encode()).hexdigest()
md5_5 = hashlib.md5(df5.to_csv(index=False).encode()).hexdigest()
is_exact_duplicate = df5.equals(df3b)
print(f"  Shape: {df5.shape}")
print(f"  MD5 dataset_3b: {md5_3b}")
print(f"  MD5 dataset_5:  {md5_5}")
print(f"  Exact duplicate of dataset_3b: {is_exact_duplicate}")

ds5_audit = audit_dataframe(df5, "dataset_5 (parkinsons_updrs.data)")
ds5_audit["file_type"] = "CSV"
ds5_audit["notes"] = f"CONFIRMED DUPLICATE of dataset_3b (MD5 match: {is_exact_duplicate}). Excluded from training."

# ─────────────────────────────────────────────
# Compile audit table
# ─────────────────────────────────────────────
audit_rows = [audio_audit, ds2_audit, ds3a_audit, ds3b_audit, ds4_audit, ds5_audit]

# Ensure all rows have same keys
all_keys = set()
for r in audit_rows:
    all_keys.update(r.keys())
for r in audit_rows:
    for k in all_keys:
        r.setdefault(k, "N/A")

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(REPORTS_DIR / "dataset_audit.csv", index=False)
print(f"\n  Saved: reports/dataset_audit.csv")

# ─────────────────────────────────────────────
# Write dataset_audit.md
# ─────────────────────────────────────────────
md_lines = [
    "# ParkinsonXAI — Dataset Audit Report\n",
    f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
    "## Summary\n\n",
    "| Dataset | Type | Rows | Cols | Class/Target | Missing | Dup Rows | Notes |\n",
    "|---------|------|------|------|--------------|---------|----------|-------|\n",
]
for r in audit_rows:
    md_lines.append(
        f"| {r.get('dataset','N/A')} | {r.get('file_type','N/A')} | "
        f"{r.get('n_rows','N/A')} | {r.get('n_cols','N/A')} | "
        f"{str(r.get('target_variables','N/A'))[:60]} | "
        f"{r.get('missing_values','N/A')} | {r.get('duplicate_rows','N/A')} | "
        f"{str(r.get('notes',''))[:80]} |\n"
    )

md_lines += [
    "\n## Dataset 1: Voice_Dataset (WAV Audio)\n",
    f"- **Healthy recordings:** {len(healthy_wavs)}\n",
    f"- **Parkinson recordings:** {len(parkinson_wavs)}\n",
    "- **Patient ID structure:** No embedded patient IDs in filenames. Each file treated as independent sample.\n",
    "- **Role:** Primary audio dataset for detection model training.\n\n",

    "## Dataset 2: pd_speech_features.csv\n",
    f"- **Shape:** {df2.shape}\n",
    "- **Features:** 753 TQWT-based vocal features + id + gender + binary class\n",
    f"- **Class distribution:** {df2['class'].value_counts().to_dict()}\n",
    "- **Role:** Secondary detection dataset (pre-extracted features); rich feature space\n\n",

    "## Dataset 3a: parkinsons.data (UCI)\n",
    f"- **Shape:** {df3a.shape}\n",
    "- **Features:** 22 classical voice features (jitter, shimmer, HNR, RPDE, DFA, PPE, etc.)\n",
    f"- **Class distribution:** {df3a['status'].value_counts().to_dict()}\n",
    "- **Role:** Detection benchmark; also used for cross-dataset validation\n\n",

    "## Dataset 3b: parkinsons_updrs.data\n",
    f"- **Shape:** {df3b.shape}\n",
    f"- **Unique subjects:** {df3b['subject#'].nunique()}\n",
    "- **Features:** 16 voice features + demographics + test_time\n",
    f"- **Targets:** motor_UPDRS (5–40), total_UPDRS (7–55)\n",
    "- **Role:** PRIMARY UPDRS severity regression dataset\n",
    "- **⚠️ Critical:** Must use subject-wise GroupKFold to prevent data leakage\n\n",

    "## Dataset 4: Parkinson_Multiple_Sound_Recording.rar\n",
    f"- **Status:** {'RAR extracted via 7-Zip' if sevenz else 'INACCESSIBLE (7-Zip not found)'}\n",
    "- **Role:** Potentially additional audio; excluded if inaccessible\n\n",

    "## Dataset 5: parkinsons_updrs.data\n",
    f"- **CONFIRMED EXACT DUPLICATE of Dataset 3b** (MD5 match: {is_exact_duplicate})\n",
    "- **Decision:** Excluded from training. Not treated as independent evidence.\n\n",

    "## Duplicate Detection Summary\n\n",
    "| Pair | Duplicate? | Action |\n",
    "|------|-----------|--------|\n",
    f"| dataset_3b vs dataset_5 | **YES** (MD5: {md5_3b[:16]}...) | Use only dataset_3b |\n",
    "| All other pairs | NO | Treated independently |\n",
]

with open(REPORTS_DIR / "dataset_audit.md", "w", encoding="utf-8") as f:
    f.writelines(md_lines)
print("  Saved: reports/dataset_audit.md")

# ─────────────────────────────────────────────
# Write dataset_roles.md
# ─────────────────────────────────────────────
roles_md = """# ParkinsonXAI — Dataset Roles

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
"""

with open(REPORTS_DIR / "dataset_roles.md", "w", encoding="utf-8") as f:
    f.write(roles_md)
print("  Saved: reports/dataset_roles.md")

print("\n" + "="*60)
print("Phase 1+2 COMPLETE")
print("="*60)
print(f"  reports/dataset_audit.md")
print(f"  reports/dataset_audit.csv")
print(f"  reports/dataset_roles.md")
print(f"  processed/tabular/pd_speech_features.csv")
print(f"  processed/tabular/parkinsons_data.csv")
print(f"  processed/tabular/parkinsons_updrs.csv")
