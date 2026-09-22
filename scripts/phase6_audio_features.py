#!/usr/bin/env python3
"""
Phase 6: Audio Feature Extraction from dataset_1 WAV files
ParkinsonXAI Project

Extracts features using:
  - librosa (MFCC, delta MFCC, spectral, chroma, RMS, ZCR)
  - parselmouth/Praat (pitch, jitter, shimmer, HNR, formants)
  - scipy (additional statistics)

Output:
  processed/audio_features/audio_features_dataset1.csv
  models/audio_feature_config.json
"""

import zipfile
import io
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
AUDIO_OUT = ROOT / "processed" / "audio_features"
AUDIO_OUT.mkdir(parents=True, exist_ok=True)
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

import librosa
import soundfile as sf
try:
    # pyrefly: ignore [missing-import]
    import parselmouth
    # pyrefly: ignore [missing-import]
    from parselmouth.praat import call
    PARSELMOUTH_OK = True
except ImportError:
    PARSELMOUTH_OK = False
    print("WARNING: parselmouth not available, skipping Praat features")

SR_TARGET = 22050  # librosa default

# ─────────────────────────────────────────────
# Feature extraction functions
# ─────────────────────────────────────────────

def extract_librosa_features(y: np.ndarray, sr: int) -> dict:
    """Extract all librosa-based features."""
    feats = {}

    # MFCCs (40 coefficients) + delta + delta-delta
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    for i in range(40):
        feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))
        feats[f"delta_mfcc_{i+1}_mean"] = float(np.mean(delta_mfcc[i]))
        feats[f"delta2_mfcc_{i+1}_mean"] = float(np.mean(delta2_mfcc[i]))

    # Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        feats[f"chroma_{i+1}_mean"] = float(np.mean(chroma[i]))
        feats[f"chroma_{i+1}_std"] = float(np.std(chroma[i]))

    # Spectral features
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    feats["spectral_centroid_mean"] = float(np.mean(sc))
    feats["spectral_centroid_std"] = float(np.std(sc))

    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    feats["spectral_bandwidth_mean"] = float(np.mean(sb))
    feats["spectral_bandwidth_std"] = float(np.std(sb))

    sro = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    feats["spectral_rolloff_mean"] = float(np.mean(sro))
    feats["spectral_rolloff_std"] = float(np.std(sro))

    sc_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(sc_contrast.shape[0]):
        feats[f"spectral_contrast_{i+1}_mean"] = float(np.mean(sc_contrast[i]))

    sf_flat = librosa.feature.spectral_flatness(y=y)[0]
    feats["spectral_flatness_mean"] = float(np.mean(sf_flat))

    # RMS and Zero-Crossing Rate
    rms = librosa.feature.rms(y=y)[0]
    feats["rms_mean"] = float(np.mean(rms))
    feats["rms_std"] = float(np.std(rms))

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats["zcr_mean"] = float(np.mean(zcr))
    feats["zcr_std"] = float(np.std(zcr))

    # Mel-spectrogram statistics
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    feats["mel_mean"] = float(np.mean(mel_db))
    feats["mel_std"] = float(np.std(mel_db))
    feats["mel_skew"] = float(float(np.mean((mel_db - np.mean(mel_db))**3)) / (np.std(mel_db)**3 + 1e-8))

    # Pitch / F0 via librosa
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
        )
        f0_voiced = f0[voiced_flag] if f0 is not None else np.array([])
        feats["f0_mean"] = float(np.nanmean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
        feats["f0_std"] = float(np.nanstd(f0_voiced)) if len(f0_voiced) > 0 else 0.0
        feats["voiced_fraction"] = float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0
    except Exception:
        feats["f0_mean"] = 0.0
        feats["f0_std"] = 0.0
        feats["voiced_fraction"] = 0.0

    return feats


def extract_praat_features(y: np.ndarray, sr: int) -> dict:
    """Extract voice quality features using Praat/parselmouth."""
    feats = {}
    if not PARSELMOUTH_OK:
        return feats
    try:
        snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)

        # Pitch
        pitch = call(snd, "To Pitch", 0.0, 75, 600)
        pitch_values = pitch.selected_array["frequency"]
        pitch_values = pitch_values[pitch_values > 0]
        feats["praat_f0_mean"] = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0
        feats["praat_f0_std"] = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0

        # Jitter
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)
        # Local jitter
        try:
            jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            feats["jitter_local"] = float(jitter_local) if jitter_local is not None else 0.0
        except Exception:
            feats["jitter_local"] = 0.0
        # RAP jitter
        try:
            jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            feats["jitter_rap"] = float(jitter_rap) if jitter_rap is not None else 0.0
        except Exception:
            feats["jitter_rap"] = 0.0
        # PPQ5 jitter
        try:
            jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            feats["jitter_ppq5"] = float(jitter_ppq5) if jitter_ppq5 is not None else 0.0
        except Exception:
            feats["jitter_ppq5"] = 0.0

        # Shimmer
        try:
            shimmer_local = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            feats["shimmer_local"] = float(shimmer_local) if shimmer_local is not None else 0.0
        except Exception:
            feats["shimmer_local"] = 0.0
        try:
            shimmer_apq3 = call([snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            feats["shimmer_apq3"] = float(shimmer_apq3) if shimmer_apq3 is not None else 0.0
        except Exception:
            feats["shimmer_apq3"] = 0.0
        try:
            shimmer_apq5 = call([snd, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            feats["shimmer_apq5"] = float(shimmer_apq5) if shimmer_apq5 is not None else 0.0
        except Exception:
            feats["shimmer_apq5"] = 0.0

        # HNR
        try:
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            feats["hnr"] = float(hnr) if hnr is not None and not np.isnan(hnr) else 0.0
        except Exception:
            feats["hnr"] = 0.0

        # Formants
        try:
            formant = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
            feats["f1_mean"] = float(call(formant, "Get mean", 1, 0, 0, "Hertz"))
            feats["f2_mean"] = float(call(formant, "Get mean", 2, 0, 0, "Hertz"))
            feats["f3_mean"] = float(call(formant, "Get mean", 3, 0, 0, "Hertz"))
        except Exception:
            feats["f1_mean"] = 0.0
            feats["f2_mean"] = 0.0
            feats["f3_mean"] = 0.0

    except Exception as e:
        # If Praat fails for a recording, return zeros
        for k in ["praat_f0_mean", "praat_f0_std", "jitter_local", "jitter_rap",
                  "jitter_ppq5", "shimmer_local", "shimmer_apq3", "shimmer_apq5",
                  "hnr", "f1_mean", "f2_mean", "f3_mean"]:
            feats.setdefault(k, 0.0)

    return feats


def process_wav_from_zip(zf: zipfile.ZipFile, wav_name: str, label: int, sr_target: int = SR_TARGET) -> dict:
    """Load WAV from zip and extract all features."""
    try:
        with zf.open(wav_name) as wf:
            data_bytes = wf.read()

        bio = io.BytesIO(data_bytes)
        y_raw, sr_orig = sf.read(bio)

        # Convert to mono
        if y_raw.ndim > 1:
            y_raw = y_raw.mean(axis=1)
        y_raw = y_raw.astype(np.float32)

        # Resample if needed
        if sr_orig != sr_target:
            y = librosa.resample(y_raw, orig_sr=sr_orig, target_sr=sr_target)
            sr = sr_target
        else:
            y = y_raw
            sr = sr_orig

        # Skip extremely short recordings
        if len(y) < sr * 0.1:  # less than 0.1s
            return None

        # Extract features
        feats = {"filename": wav_name.split("/")[-1], "label": label}
        feats.update(extract_librosa_features(y, sr))
        feats.update(extract_praat_features(y, sr))
        feats["duration_s"] = float(len(y) / sr)
        feats["sample_rate"] = sr
        return feats

    except Exception as e:
        print(f"  WARNING: Failed to process {wav_name}: {e}")
        return None


# ─────────────────────────────────────────────
# Main extraction loop
# ─────────────────────────────────────────────
print("="*60)
print("Phase 6: Audio Feature Extraction")
print("="*60)
print(f"  Parselmouth available: {PARSELMOUTH_OK}")

z1 = zipfile.ZipFile(ROOT / "datasets" / "dataset_1.zip")
all_names = z1.namelist()
healthy_wavs = sorted([n for n in all_names if "Healthy" in n and n.endswith(".wav")])
parkinson_wavs = sorted([n for n in all_names if "Parkinsons" in n and n.endswith(".wav")])

print(f"  Healthy WAVs:   {len(healthy_wavs)}")
print(f"  Parkinson WAVs: {len(parkinson_wavs)}")
print(f"  Processing...")

all_rows = []
failed = 0

# Process healthy
for wav_name in tqdm(healthy_wavs, desc="Healthy"):
    row = process_wav_from_zip(z1, wav_name, label=0)
    if row:
        all_rows.append(row)
    else:
        failed += 1

# Process parkinson
for wav_name in tqdm(parkinson_wavs, desc="Parkinson"):
    row = process_wav_from_zip(z1, wav_name, label=1)
    if row:
        all_rows.append(row)
    else:
        failed += 1

print(f"\n  Successfully processed: {len(all_rows)} files")
print(f"  Failed: {failed} files")

# Create DataFrame
audio_df = pd.DataFrame(all_rows)
print(f"  Feature matrix: {audio_df.shape}")

# Verify label distribution
label_dist = audio_df["label"].value_counts().to_dict()
print(f"  Label distribution: {label_dist}")

# Check for NaN/Inf
nan_count = audio_df.isnull().sum().sum()
inf_count = np.isinf(audio_df.select_dtypes(include=np.number)).sum().sum()
print(f"  NaN values: {nan_count}, Inf values: {inf_count}")

# Clean infinities
audio_df = audio_df.replace([np.inf, -np.inf], np.nan)
audio_df = audio_df.fillna(0.0)

# Save
out_path = AUDIO_OUT / "audio_features_dataset1.csv"
audio_df.to_csv(out_path, index=False)
print(f"\n  Saved: {out_path}")

# Save feature config
feature_names = [c for c in audio_df.columns if c not in ["filename", "label", "duration_s", "sample_rate"]]
config = {
    "sr_target": SR_TARGET,
    "n_mfcc": 40,
    "n_mels": 64,
    "n_chroma": 12,
    "parselmouth_available": PARSELMOUTH_OK,
    "feature_names": feature_names,
    "n_features": len(feature_names),
    "label_encoding": {"0": "Healthy", "1": "Parkinson"},
    "feature_groups": {
        "mfcc": [f"mfcc_{i+1}_mean" for i in range(40)],
        "delta_mfcc": [f"delta_mfcc_{i+1}_mean" for i in range(40)],
        "delta2_mfcc": [f"delta2_mfcc_{i+1}_mean" for i in range(40)],
        "mfcc_std": [f"mfcc_{i+1}_std" for i in range(40)],
        "chroma": [f"chroma_{i+1}_mean" for i in range(12)],
        "spectral": ["spectral_centroid_mean", "spectral_bandwidth_mean",
                     "spectral_rolloff_mean", "spectral_flatness_mean"],
        "rms_zcr": ["rms_mean", "rms_std", "zcr_mean", "zcr_std"],
        "pitch": ["f0_mean", "f0_std", "voiced_fraction"],
        "praat_voice": ["praat_f0_mean", "praat_f0_std", "jitter_local",
                        "jitter_rap", "jitter_ppq5", "shimmer_local",
                        "shimmer_apq3", "shimmer_apq5", "hnr"],
        "formants": ["f1_mean", "f2_mean", "f3_mean"],
    }
}

with open(MODELS / "audio_feature_config.json", "w") as f:
    json.dump(config, f, indent=2)
print(f"  Saved: models/audio_feature_config.json")
print(f"  Total features: {len(feature_names)}")

print("\n" + "="*60)
print("Phase 6 COMPLETE")
print("="*60)
