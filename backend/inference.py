"""
inference.py — All ML inference logic for the ParkinsonXAI backend.
Mirrors the exact feature extraction pipeline from phase6_audio_features.py.
"""

import io
import base64
import logging
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import shap
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

SR_TARGET = 22050  # Must match phase6 training


# ── Audio Feature Extraction ──────────────────────────────────────────────────

def _extract_librosa_features(y: np.ndarray, sr: int) -> dict:
    """Exact mirror of phase6_audio_features.py extract_librosa_features()."""
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

    # Chroma
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

    # RMS + ZCR
    rms = librosa.feature.rms(y=y)[0]
    feats["rms_mean"] = float(np.mean(rms))
    feats["rms_std"] = float(np.std(rms))

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats["zcr_mean"] = float(np.mean(zcr))
    feats["zcr_std"] = float(np.std(zcr))

    # Mel-spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    feats["mel_mean"] = float(np.mean(mel_db))
    feats["mel_std"] = float(np.std(mel_db))
    feats["mel_skew"] = float(
        np.mean((mel_db - np.mean(mel_db)) ** 3) / (np.std(mel_db) ** 3 + 1e-8)
    )

    # F0 via librosa pyin
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
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


def _extract_praat_features(y: np.ndarray, sr: int) -> dict:
    """Praat/parselmouth features — optional, falls back to zeros if unavailable."""
    feats = {}
    praat_keys = [
        "praat_f0_mean", "praat_f0_std", "jitter_local", "jitter_rap",
        "jitter_ppq5", "shimmer_local", "shimmer_apq3", "shimmer_apq5",
        "hnr", "f1_mean", "f2_mean", "f3_mean",
    ]
    for k in praat_keys:
        feats[k] = 0.0

    try:
        import parselmouth  # type: ignore[import]
        from parselmouth.praat import call  # type: ignore[import]

        snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)

        # Pitch
        pitch = call(snd, "To Pitch", 0.0, 75, 600)
        pitch_values = pitch.selected_array["frequency"]
        pitch_values = pitch_values[pitch_values > 0]
        feats["praat_f0_mean"] = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0
        feats["praat_f0_std"] = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0

        # Point process for jitter/shimmer
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 600)

        for jit_name, jit_method in [
            ("jitter_local", "Get jitter (local)"),
            ("jitter_rap", "Get jitter (rap)"),
            ("jitter_ppq5", "Get jitter (ppq5)"),
        ]:
            try:
                v = call(pp, jit_method, 0, 0, 0.0001, 0.02, 1.3)
                feats[jit_name] = float(v) if v is not None else 0.0
            except Exception:
                feats[jit_name] = 0.0

        for shim_name, shim_method in [
            ("shimmer_local", "Get shimmer (local)"),
            ("shimmer_apq3", "Get shimmer (apq3)"),
            ("shimmer_apq5", "Get shimmer (apq5)"),
        ]:
            try:
                v = call([snd, pp], shim_method, 0, 0, 0.0001, 0.02, 1.3, 1.6)
                feats[shim_name] = float(v) if v is not None else 0.0
            except Exception:
                feats[shim_name] = 0.0

        # HNR
        try:
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            feats["hnr"] = float(hnr) if (hnr is not None and not np.isnan(hnr)) else 0.0
        except Exception:
            feats["hnr"] = 0.0

        # Formants
        try:
            formant = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
            feats["f1_mean"] = float(call(formant, "Get mean", 1, 0, 0, "Hertz"))
            feats["f2_mean"] = float(call(formant, "Get mean", 2, 0, 0, "Hertz"))
            feats["f3_mean"] = float(call(formant, "Get mean", 3, 0, 0, "Hertz"))
        except Exception:
            pass

    except ImportError:
        pass  # parselmouth not installed — zeros already set
    except Exception as e:
        logger.warning(f"Praat feature extraction failed: {e}")

    return feats


def extract_audio_features(wav_bytes: bytes) -> Dict[str, float]:
    """
    Load audio from bytes and extract features using the EXACT same pipeline
    as phase6_audio_features.py training script:
      sf.read() -> mono -> float32 -> resample to 22050 Hz -> librosa features
    NO additional preprocessing (no filter, no normalization, no trimming)
    to ensure features match the training distribution exactly.
    """
    bio = io.BytesIO(wav_bytes)

    # Try soundfile first (matches training: sf.read)
    try:
        y_raw, sr_orig = sf.read(bio)
        if y_raw.ndim > 1:
            y_raw = y_raw.mean(axis=1)          # mono — matches training
        y_raw = y_raw.astype(np.float32)        # float32 — matches training
    except Exception:
        # Fallback for WebM/OGG from browser recording
        bio.seek(0)
        try:
            y_raw, sr_orig = librosa.load(bio, sr=None, mono=True)
        except Exception as e:
            raise ValueError(
                f"Could not decode audio. Use WAV, WebM, OGG or MP3. Error: {e}"
            )

    # Resample to 22050 Hz — matches training
    if sr_orig != SR_TARGET:
        y_raw = librosa.resample(y_raw, orig_sr=sr_orig, target_sr=SR_TARGET)

    # Skip extremely short recordings (matches training: < 0.1 s skipped)
    if len(y_raw) < SR_TARGET * 0.1:
        raise ValueError("Audio too short (< 0.1 s). Please record at least 3 seconds.")

    # Pad to at least 0.5 s so librosa doesn't error on very short clips
    if len(y_raw) < SR_TARGET // 2:
        y_raw = np.pad(y_raw, (0, SR_TARGET // 2 - len(y_raw)))

    feats: Dict[str, float] = {}
    feats.update(_extract_librosa_features(y_raw, SR_TARGET))
    feats.update(_extract_praat_features(y_raw, SR_TARGET))
    return feats


# ── Detection ─────────────────────────────────────────────────────────────────

# Minimum confidence threshold below which we surface a note (no override)
_LOW_CONFIDENCE_THRESHOLD = 0.60


def predict_detection(
    raw_features: Dict[str, float],
    registry,
) -> Dict:
    """
    Run the detection pipeline:
      raw_features → imputer → VT → scaler → top50 → LightGBM
    Always returns the model's actual Healthy/Parkinson prediction.
    Voice quality is stored as metadata only — it never overrides the label.
    """
    # Compute voice quality for informational metadata only
    vq = check_voice_quality(raw_features)
    voice_quality  = vq["quality_score"]
    quality_warning = None  # no longer surfaced as a blocking warning

    # Build DataFrame in full feature order from audio_feature_config
    df = pd.DataFrame([raw_features])

    # 1) Imputer
    try:
        imp = registry.detection_imputer
        imp_feature_names = list(imp.feature_names_in_) if hasattr(imp, "feature_names_in_") else list(df.columns)
        df_imp = df.reindex(columns=imp_feature_names, fill_value=np.nan)
        X_imp = imp.transform(df_imp)
        df_imp_out = pd.DataFrame(X_imp, columns=imp_feature_names)
    except Exception as e:
        logger.warning(f"Imputer step failed: {e} — using raw features")
        df_imp_out = df.copy()
        imp_feature_names = list(df.columns)

    # 2) VarianceThreshold
    try:
        vt = registry.detection_vt
        X_vt = vt.transform(df_imp_out)
        vt_mask = vt.get_support()
        vt_cols = [imp_feature_names[i] for i, m in enumerate(vt_mask) if m]
        df_vt = pd.DataFrame(X_vt, columns=vt_cols)
    except Exception as e:
        logger.warning(f"VT step failed: {e}")
        df_vt = df_imp_out

    # 3) Scaler
    try:
        scaler = registry.detection_scaler
        X_scaled = scaler.transform(df_vt)
        df_scaled = pd.DataFrame(X_scaled, columns=df_vt.columns)
    except Exception as e:
        logger.warning(f"Scaler step failed: {e}")
        df_scaled = df_vt

    # 4) Top-50 feature selection
    top50 = registry.detection_top50_features
    df_final = df_scaled.reindex(columns=top50, fill_value=0.0)

    # 5) Predict — always trust the model output
    model = registry.detection_model
    proba = model.predict_proba(df_final)[0]
    classes = model.classes_

    label_enc = registry.detection_feature_config.get("label_encoding", {"0": "Healthy", "1": "Parkinson"})
    prob_dict = {}
    for cls, p in zip(classes, proba):
        label_name = label_enc.get(str(int(cls)), str(cls))
        prob_dict[label_name] = float(p)

    pred_idx = int(np.argmax(proba))
    pred_class = classes[pred_idx]
    pred_label = label_enc.get(str(int(pred_class)), str(pred_class))
    confidence = float(proba[pred_idx])

    # Low confidence is purely informational — no label change
    is_low_confidence = confidence < _LOW_CONFIDENCE_THRESHOLD

    logger.info(
        f"Detection: {pred_label} @ {confidence:.3f} "
        f"(voice_quality={voice_quality:.3f}, low_conf={is_low_confidence})"
    )

    return {
        "label": pred_label,            # always Healthy or Parkinson
        "confidence": confidence,
        "probabilities": prob_dict,
        "model_used": "LightGBM_Tuned",
        "voice_quality": voice_quality,
        "quality_warning": None,        # no blocking warnings
        "is_uncertain": False,          # always False — no uncertainty gate
        "is_low_confidence": is_low_confidence,
        "df_final": df_final,           # pass downstream for SHAP
    }


# ── Severity ──────────────────────────────────────────────────────────────────

def predict_severity(
    raw_features: Dict[str, float],
    registry,
) -> Dict:
    """
    Compute severity from acoustic biomarkers present in the audio recording.

    Strategy:
      - Try the UPDRS regression model first (requires UPDRS feature columns).
      - If UPDRS model columns don't match audio features, fall back to an
        *audio-proxy severity score* (0–100) computed from clinical voice markers
        that ARE available: jitter, shimmer, HNR, voiced_fraction, rms_mean.

    The proxy formula is grounded in published Parkinson's dysarthria research:
      - High jitter  → worse (tremor)
      - High shimmer → worse (hypophonia)
      - Low HNR      → worse (noise/breathiness)
      - Low voiced_fraction → worse (aperiodicity)
      - Low rms_mean → worse (hypophonia)
    """
    severity_basis = "audio_proxy"
    motor_updrs = 0.0
    total_updrs = 0.0
    severity_score = 0.0

    # ── Try UPDRS model first ─────────────────────────────────────────────────
    severity_cols = registry.severity_feature_cols
    df = pd.DataFrame([raw_features])
    df_sev = df.reindex(columns=severity_cols, fill_value=np.nan)

    # Only use UPDRS model if majority of columns are actually present in audio
    available_frac = df_sev.notna().sum().sum() / max(len(severity_cols), 1)
    if available_frac > 0.5:
        try:
            X_scaled = registry.severity_scaler.transform(df_sev.fillna(0.0))
            model = registry.severity_model
            motor_updrs = float(model.predict(X_scaled)[0])
            motor_updrs = max(0.0, min(108.0, motor_updrs))
            total_updrs = max(0.0, min(176.0, motor_updrs * 1.35))
            severity_score = round((motor_updrs / 108.0) * 100.0, 1)
            severity_basis = "updrs_model"
        except Exception as e:
            logger.warning(f"UPDRS model failed ({e}), falling back to audio proxy")

    # ── Audio-proxy severity score ─────────────────────────────────────────────
    if severity_basis == "audio_proxy":
        # Each component is normalized to [0, 1] where 1 = most Parkinsonian
        # Reference ranges from UCI parkinsons dataset and published literature

        # Jitter local: healthy ~0.004, PD can reach 0.03+
        jitter = raw_features.get("jitter_local", 0.0)
        jitter_score = min(jitter / 0.025, 1.0)  # saturates at 2.5%

        # Shimmer local: healthy ~0.03, PD can reach 0.15+
        shimmer = raw_features.get("shimmer_local", 0.0)
        shimmer_score = min(shimmer / 0.12, 1.0)  # saturates at 12%

        # HNR: healthy >20 dB, PD often <12 dB. Lower = worse.
        hnr = raw_features.get("hnr", 20.0)
        hnr_score = max(0.0, min(1.0, (20.0 - hnr) / 20.0))  # 0 at HNR=20, 1 at HNR≤0

        # Voiced fraction: healthy ~0.8+, PD often <0.5. Lower = worse.
        vf = raw_features.get("voiced_fraction", 0.8)
        vf_score = max(0.0, min(1.0, (0.8 - vf) / 0.6))  # 0 at vf=0.8, 1 at vf=0.2

        # RMS energy: lower = more hypophonia. Normalize relative — low RMS = worse.
        rms = raw_features.get("rms_mean", 0.05)
        rms_score = max(0.0, min(1.0, (0.05 - rms) / 0.04)) if rms < 0.05 else 0.0

        # Weighted combination (jitter+shimmer most clinical)
        raw_score = (
            0.30 * jitter_score +
            0.30 * shimmer_score +
            0.20 * hnr_score +
            0.12 * vf_score +
            0.08 * rms_score
        )
        severity_score = round(raw_score * 100.0, 1)

        # Map to approximate UPDRS for backward compat display
        motor_updrs = round(severity_score * 108.0 / 100.0, 2)
        total_updrs = round(motor_updrs * 1.35, 2)

    # ── Severity level ─────────────────────────────────────────────────────────
    if severity_score < 30:
        severity_level = "Mild"
    elif severity_score < 60:
        severity_level = "Moderate"
    else:
        severity_level = "Severe"

    return {
        "motor_updrs": round(motor_updrs, 2),
        "total_updrs": round(total_updrs, 2),
        "severity_level": severity_level,
        "model_used": "UPDRS_LightGBM" if severity_basis == "updrs_model" else "AudioProxy",
        "severity_score": severity_score,
        "severity_basis": severity_basis,
    }


# ── SHAP ──────────────────────────────────────────────────────────────────────

def compute_shap(
    df_final: pd.DataFrame,
    registry,
    n_top: int = 10,
) -> Dict:
    """
    Compute SHAP values for the detection model.
    Returns top-N feature contributions + base64 waterfall PNG.
    """
    try:
        model = registry.detection_model
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_final, check_additivity=False)

        # For binary classifiers, shap_values may be a list [class0, class1]
        if isinstance(shap_values, list):
            sv = shap_values[1]  # Parkinson class
        else:
            sv = shap_values

        sv_row = sv[0]  # single sample
        feature_names = list(df_final.columns)

        # Sort by absolute SHAP value
        abs_vals = np.abs(sv_row)
        top_idx = np.argsort(abs_vals)[::-1][:n_top]

        top_features = [
            {"feature": feature_names[i], "value": round(float(sv_row[i]), 5)}
            for i in top_idx
        ]

        # Waterfall chart
        plot_b64 = _render_shap_bar(top_features)

        return {"top_features": top_features, "plot_base64": plot_b64}

    except Exception as e:
        logger.error(f"SHAP computation failed: {type(e).__name__}: {e}", exc_info=True)
        # Return model metadata top features as fallback — SHAP failure is non-fatal
        meta_features = registry.model_metadata.get("shap_top_features", [])
        top_features = [{"feature": f, "value": 0.0} for f in meta_features[:n_top]]
        return {"top_features": top_features, "plot_base64": None}


def _render_shap_bar(top_features: List[Dict]) -> Optional[str]:
    """Render a horizontal bar chart of SHAP contributions, return as base64 PNG."""
    try:
        names = [f["feature"] for f in top_features][::-1]
        values = [f["value"] for f in top_features][::-1]
        colors = ["#ef4444" if v > 0 else "#14b8a6" for v in values]

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#0f0f1a")
        ax.set_facecolor("#0f0f1a")

        bars = ax.barh(names, values, color=colors, height=0.65, edgecolor="none")

        ax.axvline(0, color="#ffffff22", linewidth=1)
        ax.set_xlabel("SHAP Value (impact on Parkinson prediction)", color="#94a3b8", fontsize=9)
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#ffffff11")

        plt.title("Top Feature Contributions (SHAP)", color="#f8fafc", fontsize=11, pad=10)
        plt.tight_layout(pad=1.5)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"SHAP plot rendering failed: {e}")
        return None


# ── Voice Quality Guard ──────────────────────────────────────────────────────

def check_voice_quality(raw_features: Dict[str, float]) -> Dict:
    """
    Assess whether the uploaded audio contains a human voice.
    Returns a dict with:
      - is_voice     : bool  — True if likely a voiced recording
      - quality_score: float — 0.0 (silence/noise) to 1.0 (clear voice)
      - warning      : str | None — human-readable warning message
    
    Uses voiced_fraction, f0_mean, rms_mean, and hnr as indicators.
    These are exactly the features that are zero/near-zero for non-voice audio.
    """
    voiced_fraction = raw_features.get("voiced_fraction", 0.0)
    f0_mean         = raw_features.get("f0_mean", 0.0)
    rms_mean        = raw_features.get("rms_mean", 0.0)
    hnr             = raw_features.get("hnr", 0.0)

    # Score each component (0–1 each)
    # voiced_fraction: fraction of frames detected as voiced
    vf_score  = min(voiced_fraction / 0.3, 1.0)          # needs at least 30% voiced frames
    # f0_mean: fundamental frequency, 0 means no pitch detected
    f0_score  = 1.0 if f0_mean > 50 else (f0_mean / 50.0)  # expect 50–300 Hz for voice
    # rms_mean: energy — very low means silence
    rms_score = min(rms_mean / 0.005, 1.0)               # needs some energy
    # hnr: harmonics-to-noise ratio — very low means no harmonic structure
    hnr_score = min(max(hnr, 0.0) / 5.0, 1.0)            # needs at least 5 dB HNR

    # Weighted quality score
    quality_score = (
        0.40 * vf_score +
        0.30 * f0_score +
        0.20 * rms_score +
        0.10 * hnr_score
    )

    is_voice = quality_score >= 0.20  # minimum threshold

    warning = None
    if not is_voice:
        if rms_mean < 0.001:
            warning = (
                "The audio appears to be silent or nearly inaudible. "
                "Please record in a quiet environment and speak clearly."
            )
        elif voiced_fraction < 0.05 and f0_mean < 10:
            warning = (
                "No human voice was detected in the audio. "
                "This tool only works with voice/speech recordings. "
                "Please upload a sustained vowel sound (e.g. 'ahhh') or natural speech."
            )
        else:
            warning = (
                "The audio quality is too low for reliable Parkinson's analysis. "
                "Please use a clear speech recording (minimum 3 seconds of sustained voice)."
            )
    elif quality_score < 0.45:
        warning = (
            "Low voice quality detected — results may be less reliable. "
            "For best results use a sustained vowel recording in a quiet environment."
        )

    return {
        "is_voice": is_voice,
        "quality_score": round(quality_score, 3),
        "warning": warning,
        "details": {
            "voiced_fraction": round(voiced_fraction, 3),
            "f0_mean": round(f0_mean, 1),
            "rms_mean": round(rms_mean, 5),
            "hnr": round(hnr, 1),
        },
    }


# ── Snapshot helper ───────────────────────────────────────────────────────────

def build_audio_snapshot(raw_features: Dict[str, float], top_keys: List[str]) -> Dict[str, float]:
    """Return a small dict of key audio features for storage in MongoDB."""
    return {k: round(float(raw_features.get(k, 0.0)), 5) for k in top_keys if k in raw_features}
