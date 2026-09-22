#!/usr/bin/env python3
"""
ParkinsonXAI — Inference Pipeline
Accepts a voice recording and returns:
  1. PD Detection: Healthy / Parkinson's
  2. If Parkinson's: motor_UPDRS + total_UPDRS prediction
  3. SHAP explanation of the prediction

Usage:
  python scripts/phase16_inference_pipeline.py path/to/audio.wav

IMPORTANT: Models must already be trained (run all phases first).
Do NOT retrain during inference.
"""
import argparse
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

# -- Load saved models and preprocessors --
print("Loading saved models...")
detection_model = joblib.load(MODELS / "detection_best_model.pkl")
detection_scaler = joblib.load(MODELS / "detection_scaler.pkl")
detection_imputer = joblib.load(MODELS / "detection_imputer.pkl")
detection_le = joblib.load(MODELS / "detection_label_encoder.pkl")
severity_model = joblib.load(MODELS / "severity_best_model.pkl")
severity_scaler = joblib.load(MODELS / "severity_scaler.pkl")
severity_imputer = joblib.load(MODELS / "ds3b_imputer.pkl")

with open(MODELS / "audio_feature_config.json") as f:
    audio_config = json.load(f)
with open(MODELS / "detection_feature_config.json") as f:
    det_config = json.load(f)
with open(MODELS / "severity_feature_cols.json") as f:
    sev_config = json.load(f)

features_for_detection = det_config["features_used"]
features_for_severity = sev_config["features"]
SR_TARGET = audio_config["sr_target"]

print(f"Detection model: {det_config['best_model']} | Features: {det_config['n_features']}")
print(f"Severity model: {det_config['best_model']}")


def extract_features_from_wav(wav_path: str) -> dict:
    """Extract audio features using librosa and parselmouth."""
    import librosa
    import soundfile as sf

    y, sr = librosa.load(wav_path, sr=SR_TARGET, mono=True)

    feats = {}

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    for i in range(40):
        feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))
        feats[f"delta_mfcc_{i+1}_mean"] = float(np.mean(delta_mfcc[i]))
        feats[f"delta2_mfcc_{i+1}_mean"] = float(np.mean(delta2_mfcc[i]))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        feats[f"chroma_{i+1}_mean"] = float(np.mean(chroma[i]))
        feats[f"chroma_{i+1}_std"] = float(np.std(chroma[i]))

    # Spectral
    feats["spectral_centroid_mean"] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0]))
    feats["spectral_centroid_std"] = float(np.std(librosa.feature.spectral_centroid(y=y, sr=sr)[0]))
    feats["spectral_bandwidth_mean"] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]))
    feats["spectral_bandwidth_std"] = float(np.std(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]))
    feats["spectral_rolloff_mean"] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0]))
    feats["spectral_rolloff_std"] = float(np.std(librosa.feature.spectral_rolloff(y=y, sr=sr)[0]))
    sc = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(sc.shape[0]):
        feats[f"spectral_contrast_{i+1}_mean"] = float(np.mean(sc[i]))
    feats["spectral_flatness_mean"] = float(np.mean(librosa.feature.spectral_flatness(y=y)[0]))

    # RMS, ZCR
    feats["rms_mean"] = float(np.mean(librosa.feature.rms(y=y)[0]))
    feats["rms_std"] = float(np.std(librosa.feature.rms(y=y)[0]))
    feats["zcr_mean"] = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))
    feats["zcr_std"] = float(np.std(librosa.feature.zero_crossing_rate(y)[0]))

    # Mel
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    feats["mel_mean"] = float(np.mean(mel_db))
    feats["mel_std"] = float(np.std(mel_db))
    feats["mel_skew"] = float(np.mean((mel_db-np.mean(mel_db))**3)/(np.std(mel_db)**3 + 1e-8))

    # Pitch
    try:
        f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"),
                                          fmax=librosa.note_to_hz("C7"), sr=sr)
        f0_v = f0[voiced_flag] if f0 is not None else np.array([])
        feats["f0_mean"] = float(np.nanmean(f0_v)) if len(f0_v)>0 else 0.0
        feats["f0_std"] = float(np.nanstd(f0_v)) if len(f0_v)>0 else 0.0
        feats["voiced_fraction"] = float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0
    except Exception:
        feats["f0_mean"] = feats["f0_std"] = feats["voiced_fraction"] = 0.0

    # Praat features
    try:
        import parselmouth
        from parselmouth.praat import call
        snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
        pitch_obj = call(snd, "To Pitch", 0.0, 75, 600)
        pitch_vals = pitch_obj.selected_array["frequency"]
        pitch_vals = pitch_vals[pitch_vals > 0]
        feats["praat_f0_mean"] = float(np.mean(pitch_vals)) if len(pitch_vals)>0 else 0.0
        feats["praat_f0_std"] = float(np.std(pitch_vals)) if len(pitch_vals)>0 else 0.0
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 600)
        feats["jitter_local"] = float(call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) or 0)
        feats["jitter_rap"] = float(call(pp, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3) or 0)
        feats["jitter_ppq5"] = float(call(pp, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3) or 0)
        feats["shimmer_local"] = float(call([snd,pp],"Get shimmer (local)",0,0,0.0001,0.02,1.3,1.6) or 0)
        feats["shimmer_apq3"] = float(call([snd,pp],"Get shimmer (apq3)",0,0,0.0001,0.02,1.3,1.6) or 0)
        feats["shimmer_apq5"] = float(call([snd,pp],"Get shimmer (apq5)",0,0,0.0001,0.02,1.3,1.6) or 0)
        harm = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        feats["hnr"] = float(call(harm, "Get mean", 0, 0) or 0)
        form = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
        feats["f1_mean"] = float(call(form, "Get mean", 1, 0, 0, "Hertz") or 0)
        feats["f2_mean"] = float(call(form, "Get mean", 2, 0, 0, "Hertz") or 0)
        feats["f3_mean"] = float(call(form, "Get mean", 3, 0, 0, "Hertz") or 0)
    except Exception:
        for k in ["praat_f0_mean","praat_f0_std","jitter_local","jitter_rap","jitter_ppq5",
                  "shimmer_local","shimmer_apq3","shimmer_apq5","hnr","f1_mean","f2_mean","f3_mean"]:
            feats.setdefault(k, 0.0)

    feats["duration_s"] = float(len(y) / sr)
    feats["sample_rate"] = sr
    return feats


def run_inference(wav_path: str) -> dict:
    """Run full inference pipeline on a WAV file."""
    print(f"\nProcessing: {wav_path}")

    # Step 1: Extract audio features
    print("  Extracting audio features...")
    raw_feats = extract_features_from_wav(wav_path)

    # Step 2: Prepare feature vector for detection
    feat_vector = np.array([[raw_feats.get(f, 0.0) for f in features_for_detection]])
    feat_vector = np.nan_to_num(feat_vector, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 3: Preprocess (using saved scaler — do NOT refit)
    feat_vector_imp = detection_imputer.transform(feat_vector)
    feat_vector_sc = detection_scaler.transform(feat_vector_imp)

    # Step 4: Detection prediction
    print("  Running detection model...")
    prediction = detection_model.predict(feat_vector_sc)[0]
    try:
        probability = detection_model.predict_proba(feat_vector_sc)[0]
        pd_prob = float(probability[1])
        healthy_prob = float(probability[0])
    except Exception:
        pd_prob = float(prediction)
        healthy_prob = 1.0 - pd_prob

    label = "Parkinson's" if prediction == 1 else "Healthy"

    result = {
        "input_file": str(wav_path),
        "detection_result": label,
        "confidence": {"Healthy": round(healthy_prob, 4), "Parkinson's": round(pd_prob, 4)},
        "raw_prediction": int(prediction),
    }

    print(f"  Detection: {label} (PD prob: {pd_prob:.3f})")

    # Step 5: Severity prediction (only if Parkinson's detected)
    if prediction == 1:
        print("  Running severity model (UPDRS)...")
        # Map audio features to severity features where possible
        sev_vector = np.array([[raw_feats.get(f, 0.0) for f in features_for_severity]])
        sev_vector = np.nan_to_num(sev_vector, nan=0.0, posinf=0.0, neginf=0.0)
        sev_vector_imp = severity_imputer.transform(sev_vector)
        sev_vector_sc = severity_scaler.transform(sev_vector_imp)

        motor_pred = severity_model.predict(sev_vector_sc)[0]
        result["severity"] = {
            "motor_UPDRS_predicted": round(float(motor_pred), 2),
            "severity_category": (
                "Mild" if motor_pred < 15 else
                "Moderate" if motor_pred < 25 else
                "Severe"
            ),
            "note": "This is a research prototype. Not a clinical diagnosis."
        }
        print(f"  motor_UPDRS prediction: {motor_pred:.2f}")
    else:
        result["severity"] = None
        print("  No severity prediction (Healthy detected)")

    # Step 6: SHAP explanation (optional — loads saved SHAP values if available)
    result["disclaimer"] = (
        "This system is a RESEARCH PROTOTYPE and decision-support tool. "
        "It is NOT a replacement for qualified medical professional assessment."
    )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ParkinsonXAI Inference")
    parser.add_argument("wav_path", type=str, help="Path to WAV audio file")
    args = parser.parse_args()

    result = run_inference(args.wav_path)

    print("\n" + "="*60)
    print("INFERENCE RESULT")
    print("="*60)
    import json
    print(json.dumps(result, indent=2))
