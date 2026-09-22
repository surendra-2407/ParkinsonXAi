"""
models_loader.py — Loads all .pkl / .json model artifacts at startup.
All objects stored in a single ModelRegistry instance (module-level singleton).
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

logger = logging.getLogger(__name__)


@dataclass
class ModelRegistry:
    """Holds all loaded model artifacts."""
    # Detection pipeline (audio LightGBM — primary, 96.49%)
    detection_model: Any = None
    detection_imputer: Any = None
    detection_vt: Any = None          # VarianceThreshold
    detection_scaler: Any = None      # RobustScaler
    detection_top50_features: List[str] = field(default_factory=list)
    detection_feature_config: Dict = field(default_factory=dict)
    detection_label_encoder: Any = None

    # Severity pipeline (LightGBM UPDRS)
    severity_model: Any = None
    severity_scaler: Any = None
    severity_feature_cols: List[str] = field(default_factory=list)
    severity_model_config: Dict = field(default_factory=dict)

    # Global metadata
    model_metadata: Dict = field(default_factory=dict)

    @property
    def loaded(self) -> bool:
        return self.detection_model is not None and self.severity_model is not None


# Module-level singleton
registry = ModelRegistry()


def load_all_models(base_path: str = "../models") -> None:
    """Load all model artifacts from disk into the registry."""
    global registry
    models_dir = Path(base_path).resolve()
    logger.info(f"Loading models from: {models_dir}")

    def _load_pkl(name: str) -> Any:
        p = models_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")
        obj = joblib.load(p)
        logger.info(f"  ✓ Loaded {name} ({p.stat().st_size / 1024:.1f} KB)")
        return obj

    def _load_json(name: str) -> Any:
        p = models_dir / name
        if not p.exists():
            raise FileNotFoundError(f"JSON config not found: {p}")
        with open(p, "r") as f:
            return json.load(f)

    # ── Detection pipeline ──────────────────────────────────────────────
    registry.detection_model = _load_pkl("detection_best_model.pkl")
    registry.detection_imputer = _load_pkl("audio_imputer.pkl")
    registry.detection_vt = _load_pkl("audio_vt.pkl")
    registry.detection_scaler = _load_pkl("audio_scaler.pkl")
    registry.detection_label_encoder = _load_pkl("detection_label_encoder.pkl")

    top50_data = _load_json("audio_top50_features.json")
    registry.detection_top50_features = top50_data["features"]

    registry.detection_feature_config = _load_json("detection_feature_config.json")

    # ── Severity pipeline ───────────────────────────────────────────────
    registry.severity_model = _load_pkl("severity_best_model.pkl")
    registry.severity_scaler = _load_pkl("severity_scaler.pkl")

    severity_cols_data = _load_json("severity_feature_cols.json")
    # Handle both {"features": [...]} and plain list formats
    if isinstance(severity_cols_data, dict):
        registry.severity_feature_cols = severity_cols_data.get(
            "features", list(severity_cols_data.keys())
        )
    else:
        registry.severity_feature_cols = severity_cols_data

    registry.severity_model_config = _load_json("severity_model_config.json")

    # ── Metadata ────────────────────────────────────────────────────────
    registry.model_metadata = _load_json("model_metadata.json")

    logger.info(
        f"All models loaded successfully. "
        f"Detection: {type(registry.detection_model).__name__} | "
        f"Severity: {type(registry.severity_model).__name__}"
    )


def get_registry() -> ModelRegistry:
    """FastAPI dependency: returns the loaded model registry."""
    return registry
