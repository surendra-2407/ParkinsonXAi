"""
schemas.py — Pydantic request/response models for all API endpoints.
"""

from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ── SHAP ──────────────────────────────────────────────────────────────────────

class ShapFeature(BaseModel):
    feature: str
    value: float


class ShapResult(BaseModel):
    top_features: List[ShapFeature]
    plot_base64: Optional[str] = None  # Base64-encoded PNG waterfall chart


# ── Detection ─────────────────────────────────────────────────────────────────

class DetectionResult(BaseModel):
    label: str                            # "Parkinson" | "Healthy"
    confidence: float                     # 0.0 – 1.0
    probabilities: Dict[str, float]       # {"Healthy": 0.04, "Parkinson": 0.96}
    model_used: str


class PredictionResponse(BaseModel):
    prediction_id: str
    filename: str
    detection: DetectionResult
    processing_time_ms: float
    timestamp: datetime


# ── Severity ──────────────────────────────────────────────────────────────────

class SeverityResult(BaseModel):
    motor_updrs: float
    total_updrs: float
    severity_level: str          # "Mild" | "Moderate" | "Severe"
    model_used: str
    severity_score: float = 0.0        # 0–100 proxy score (always populated)
    severity_basis: str = "audio_proxy"  # "audio_proxy" | "updrs_model"


class SeverityResponse(BaseModel):
    prediction_id: str
    filename: str
    severity: SeverityResult
    processing_time_ms: float
    timestamp: datetime


# ── Full Analysis ─────────────────────────────────────────────────────────────

class FullAnalysisResponse(BaseModel):
    prediction_id: str
    filename: str
    detection: DetectionResult
    severity: SeverityResult
    shap: ShapResult
    audio_features_snapshot: Dict[str, float]
    processing_time_ms: float
    timestamp: datetime


# ── Dashboard ─────────────────────────────────────────────────────────────────

class ModelAccuracyCard(BaseModel):
    model_name: str
    dataset: str
    accuracy: Optional[float] = None
    auc: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    note: Optional[str] = None


class LiveStats(BaseModel):
    total_predictions: int
    parkinson_count: int
    healthy_count: int
    avg_confidence: float


class DashboardResponse(BaseModel):
    model_cards: List[ModelAccuracyCard]
    live_stats: LiveStats
    top_shap_features: List[ShapFeature]
    shap_severity_features: List[ShapFeature]


# ── History ───────────────────────────────────────────────────────────────────

class HistoryItem(BaseModel):
    prediction_id: str
    filename: str
    timestamp: datetime
    detection_label: Optional[str] = None
    confidence: Optional[float] = None
    motor_updrs: Optional[float] = None
    total_updrs: Optional[float] = None
    severity_level: Optional[str] = None
    shap_top_feature: Optional[str] = None
    processing_time_ms: Optional[float] = None


class HistoryResponse(BaseModel):
    items: List[HistoryItem]
    total: int
    page: int
    page_size: int
    total_pages: int


# ── Stats ─────────────────────────────────────────────────────────────────────

class StatsResponse(BaseModel):
    total_predictions: int
    parkinson_count: int
    healthy_count: int
    avg_confidence: float
    avg_processing_time_ms: float
    parkinson_pct: float
    healthy_pct: float


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    db: str
    models_loaded: bool
    timestamp: datetime
