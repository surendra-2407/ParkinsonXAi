"""
main.py — FastAPI application for ParkinsonXAI
All 8 API endpoints + MongoDB integration via Motor.
"""

import logging
import math
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

import database as db_module
import inference as inf
import schemas as sc
from models_loader import get_registry, load_all_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("parkinsonxai")

MONGO_URL = os.getenv("MONGODB_URL", "")
DB_NAME = os.getenv("DATABASE_NAME", "parkinson_xai")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
MODEL_BASE_PATH = os.getenv("MODEL_BASE_PATH", "../models")

# SHAP snapshot — key features to store
SNAPSHOT_KEYS = [
    "f0_mean", "f0_std", "rms_mean", "rms_std", "praat_f0_mean",
    "voiced_fraction", "mfcc_1_mean", "mfcc_2_std", "jitter_local",
    "spectral_rolloff_mean",
]


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models + connect MongoDB. Shutdown: close DB."""
    logger.info("=== ParkinsonXAI starting up ===")

    # Load ML models
    try:
        load_all_models(MODEL_BASE_PATH)
        logger.info("Models loaded ✓")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")

    # Connect to MongoDB
    if MONGO_URL and MONGO_URL != "PASTE_YOUR_MONGODB_ATLAS_URL_HERE":
        try:
            await db_module.connect_db(MONGO_URL, DB_NAME)
            logger.info("MongoDB connected ✓")
        except Exception as e:
            logger.error(f"MongoDB connection failed (running without DB): {e}")
    else:
        logger.warning("No MONGODB_URL set — running without database persistence.")

    yield  # App runs here

    logger.info("=== ParkinsonXAI shutting down ===")
    await db_module.close_db()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ParkinsonXAI API",
    description="Voice-Based Parkinson's Disease Detection & Severity Prediction with XAI",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Global exception handler — returns JSON with real error (not bare 500) ─────
@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    import traceback
    tb = traceback.format_exc()
    logger.error(f"Unhandled exception on {request.url}: {type(exc).__name__}: {exc}\n{tb}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {str(exc)}"},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _save_prediction(doc: dict) -> None:
    """Save prediction document to MongoDB (no-op if DB not connected)."""
    col = db_module.get_predictions_col()
    if col is not None:
        try:
            await col.insert_one(doc)
        except Exception as e:
            logger.warning(f"Failed to save prediction to DB: {e}")


async def _upsert_session(session_id: str) -> None:
    col = db_module.get_sessions_col()
    if col is None:
        return
    try:
        now = datetime.now(timezone.utc)
        await col.update_one(
            {"session_id": session_id},
            {"$set": {"last_active": now}, "$setOnInsert": {"created_at": now, "session_id": session_id}, "$inc": {"prediction_count": 1}},
            upsert=True,
        )
    except Exception as e:
        logger.warning(f"Session upsert failed: {e}")


def _validate_wav(file: UploadFile) -> None:
    ALLOWED_EXT = ('.wav', '.webm', '.ogg', '.mp3', '.flac')
    if not file.filename.lower().endswith(ALLOWED_EXT):
        raise HTTPException(status_code=400, detail=f"Unsupported format. Allowed: {', '.join(ALLOWED_EXT)}")
    if file.size and file.size > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 50 MB).")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=sc.HealthResponse, tags=["System"])
async def health_check():
    """Health check — verifies server, DB, and model status."""
    db_status = "connected" if db_module.db is not None else "disconnected"
    return sc.HealthResponse(
        status="ok",
        db=db_status,
        models_loaded=get_registry().loaded,
        timestamp=datetime.now(timezone.utc),
    )


@app.post("/api/predict", response_model=sc.PredictionResponse, tags=["Prediction"])
async def predict_detection(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(default=None),
):
    """Upload a WAV file → Parkinson / Healthy detection result."""
    _validate_wav(file)
    registry = get_registry()
    if not registry.loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    wav_bytes = await file.read()
    t0 = time.perf_counter()

    raw_features = inf.extract_audio_features(wav_bytes)
    result = inf.predict_detection(raw_features, registry)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    pred_id = str(uuid.uuid4())
    sid = session_id or "anonymous"
    now = datetime.now(timezone.utc)

    doc = {
        "_id": pred_id,
        "session_id": sid,
        "filename": file.filename,
        "file_size_bytes": len(wav_bytes),
        "prediction_type": "detection",
        "timestamp": now,
        "detection": {
            "label": result["label"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "model_used": result["model_used"],
        },
        "audio_features_snapshot": inf.build_audio_snapshot(raw_features, SNAPSHOT_KEYS),
        "processing_time_ms": elapsed_ms,
    }
    await _save_prediction(doc)
    await _upsert_session(sid)

    return sc.PredictionResponse(
        prediction_id=pred_id,
        filename=file.filename,
        detection=sc.DetectionResult(**result),
        processing_time_ms=elapsed_ms,
        timestamp=now,
    )


@app.post("/api/severity", response_model=sc.SeverityResponse, tags=["Prediction"])
async def predict_severity(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(default=None),
):
    """Upload a WAV file → UPDRS motor/total severity score."""
    _validate_wav(file)
    registry = get_registry()
    if not registry.loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    wav_bytes = await file.read()
    t0 = time.perf_counter()

    raw_features = inf.extract_audio_features(wav_bytes)
    sev_result = inf.predict_severity(raw_features, registry)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    pred_id = str(uuid.uuid4())
    sid = session_id or "anonymous"
    now = datetime.now(timezone.utc)

    doc = {
        "_id": pred_id,
        "session_id": sid,
        "filename": file.filename,
        "file_size_bytes": len(wav_bytes),
        "prediction_type": "severity",
        "timestamp": now,
        "severity": sev_result,
        "audio_features_snapshot": inf.build_audio_snapshot(raw_features, SNAPSHOT_KEYS),
        "processing_time_ms": elapsed_ms,
    }
    await _save_prediction(doc)
    await _upsert_session(sid)

    return sc.SeverityResponse(
        prediction_id=pred_id,
        filename=file.filename,
        severity=sc.SeverityResult(**sev_result),
        processing_time_ms=elapsed_ms,
        timestamp=now,
    )


@app.post("/api/predict-full", tags=["Prediction"])  # no response_model — avoid Pydantic serialisation errors
async def predict_full(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(default=None),
):
    """Upload a WAV/WebM/OGG file → full analysis: detection + severity + SHAP."""
    _validate_wav(file)
    registry = get_registry()
    if not registry.loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    wav_bytes = await file.read()
    logger.info(f"predict-full: '{file.filename}' ({len(wav_bytes)} bytes)")
    t0 = time.perf_counter()

    try:
        # Step 1: Feature extraction
        logger.info("Step 1: Extracting audio features...")
        raw_features = inf.extract_audio_features(wav_bytes)
        logger.info(f"  → {len(raw_features)} features extracted")

        # Step 2: Detection
        logger.info("Step 2: Running detection pipeline...")
        det_raw = inf.predict_detection(raw_features, registry)
        df_final = det_raw.pop("df_final")  # pull out internal DataFrame
        logger.info(f"  → {det_raw['label']} (conf={det_raw['confidence']:.3f})")

        # Step 3: Severity
        logger.info("Step 3: Running severity pipeline...")
        sev_result = inf.predict_severity(raw_features, registry)
        logger.info(f"  → motor={sev_result['motor_updrs']:.1f}")

        # Step 4: SHAP (non-fatal - falls back on error)
        logger.info("Step 4: Computing SHAP...")
        shap_result = inf.compute_shap(df_final, registry, n_top=10)

    except Exception as e:
        logger.error(f"predict-full failed: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    pred_id = str(uuid.uuid4())
    sid = session_id or "anonymous"
    now = datetime.now(timezone.utc)
    audio_snapshot = inf.build_audio_snapshot(raw_features, SNAPSHOT_KEYS)
    logger.info(f"predict-full complete in {elapsed_ms:.0f}ms")

    detection_model = sc.DetectionResult(
        label=det_raw["label"],
        confidence=det_raw["confidence"],
        probabilities=det_raw["probabilities"],
        model_used=det_raw["model_used"],
    )
    severity_model = sc.SeverityResult(
        motor_updrs=sev_result["motor_updrs"],
        total_updrs=sev_result["total_updrs"],
        severity_level=sev_result["severity_level"],
        model_used=sev_result["model_used"],
    )
    shap_model = sc.ShapResult(
        top_features=[sc.ShapFeature(**f) for f in shap_result["top_features"]],
        plot_base64=shap_result.get("plot_base64"),
    )

    doc = {
        "_id": pred_id,
        "session_id": sid,
        "filename": file.filename,
        "file_size_bytes": len(wav_bytes),
        "prediction_type": "both",
        "timestamp": now,
        "detection": {
            "label": det_raw["label"],
            "confidence": det_raw["confidence"],
            "probabilities": det_raw["probabilities"],
            "model_used": det_raw["model_used"],
        },
        "severity": sev_result,
        "shap": {
            "top_features": shap_result["top_features"],
            "plot_base64": shap_result.get("plot_base64"),
        },
        "audio_features_snapshot": audio_snapshot,
        "processing_time_ms": elapsed_ms,
    }
    await _save_prediction(doc)
    await _upsert_session(sid)

    # Build response dict with all numpy types cast to native Python
    def _f(v) -> float:
        """Cast any numeric to plain Python float, mapping NaN/Infinity to 0.0."""
        try:
            result = float(v)
            if result != result or result == float('inf') or result == float('-inf'):
                return 0.0
            return result
        except Exception:
            return 0.0

    response_payload = {
        "prediction_id": pred_id,
        "filename": file.filename,
        "processing_time_ms": _f(elapsed_ms),
        "timestamp": now.isoformat(),
        "detection": {
            "label": str(det_raw["label"]),
            "confidence": _f(det_raw["confidence"]),
            "probabilities": {str(k): _f(v) for k, v in det_raw["probabilities"].items()},
            "model_used": str(det_raw["model_used"]),
        },
        "severity": {
            "motor_updrs": _f(sev_result["motor_updrs"]),
            "total_updrs": _f(sev_result["total_updrs"]),
            "severity_level": str(sev_result["severity_level"]),
            "model_used": str(sev_result["model_used"]),
            "severity_score": _f(sev_result.get("severity_score", 0.0)),
            "severity_basis": str(sev_result.get("severity_basis", "audio_proxy")),
        },
        "shap": {
            "top_features": [
                {"feature": str(f["feature"]), "value": _f(f["value"])}
                for f in shap_result["top_features"]
            ],
            "plot_base64": shap_result.get("plot_base64"),
        },
        "audio_features_snapshot": {str(k): _f(v) for k, v in audio_snapshot.items()},
    }

    return JSONResponse(content=response_payload)


@app.get("/api/dashboard", response_model=sc.DashboardResponse, tags=["Analytics"])
async def get_dashboard():
    """Model accuracy cards + live MongoDB stats + top SHAP features."""
    registry = get_registry()
    meta = registry.model_metadata if registry.loaded else {}

    model_cards = [
        sc.ModelAccuracyCard(
            model_name="LightGBM (Audio)",
            dataset="Voice Dataset WAV",
            accuracy=0.9649,
            auc=0.9995,
            recall=1.0,
            f1=0.9655,
            note="Primary detection model",
        ),
        sc.ModelAccuracyCard(
            model_name="LightGBM (Speech Features)",
            dataset="pd_speech_features.csv",
            accuracy=0.9035,
            auc=0.9645,
            recall=0.963,
            f1=0.9341,
            note="90%+ target achieved",
        ),
        sc.ModelAccuracyCard(
            model_name="KNN-3 (UCI)",
            dataset="parkinsons.data UCI",
            accuracy=0.9667,
            auc=0.9792,
            recall=1.0,
            f1=0.9796,
            note="29/30 correct",
        ),
        sc.ModelAccuracyCard(
            model_name="LightGBM (Severity)",
            dataset="parkinsons_updrs.data",
            accuracy=None,
            note=f"Motor MAE: 7.25 | Total MAE: 7.22",
        ),
    ]

    # Live stats from MongoDB
    live_stats = sc.LiveStats(total_predictions=0, parkinson_count=0, healthy_count=0, avg_confidence=0.0)
    col = db_module.get_predictions_col()
    if col is not None:
        try:
            pipeline = [
                {"$group": {
                    "_id": None,
                    "total": {"$sum": 1},
                    "park_count": {"$sum": {"$cond": [{"$eq": ["$detection.label", "Parkinson"]}, 1, 0]}},
                    "health_count": {"$sum": {"$cond": [{"$eq": ["$detection.label", "Healthy"]}, 1, 0]}},
                    "avg_conf": {"$avg": "$detection.confidence"},
                }},
            ]
            async for doc in col.aggregate(pipeline):
                live_stats = sc.LiveStats(
                    total_predictions=doc.get("total", 0),
                    parkinson_count=doc.get("park_count", 0),
                    healthy_count=doc.get("health_count", 0),
                    avg_confidence=round(doc.get("avg_conf") or 0.0, 4),
                )
        except Exception as e:
            logger.warning(f"Dashboard DB query failed: {e}")

    # Top SHAP features from model metadata
    shap_features = [
        sc.ShapFeature(feature=f, value=0.0)
        for f in meta.get("shap_top_features", [])
    ]

    # Severity SHAP
    shap_sev_features = [
        sc.ShapFeature(feature=f, value=0.0)
        for f in ["subject_id", "test_time", "age", "sex", "motor_UPDRS"][:5]
    ]

    return sc.DashboardResponse(
        model_cards=model_cards,
        live_stats=live_stats,
        top_shap_features=shap_features,
        shap_severity_features=shap_sev_features,
    )


@app.get("/api/history", response_model=sc.HistoryResponse, tags=["Analytics"])
async def get_history(page: int = 1, page_size: int = 10):
    """Paginated prediction history from MongoDB."""
    col = db_module.get_predictions_col()
    if col is None:
        return sc.HistoryResponse(items=[], total=0, page=page, page_size=page_size, total_pages=0)

    page = max(1, page)
    page_size = min(50, max(1, page_size))
    skip = (page - 1) * page_size

    try:
        total = await col.count_documents({})
        cursor = col.find({}, {
            "_id": 1, "filename": 1, "timestamp": 1,
            "detection.label": 1, "detection.confidence": 1,
            "severity.motor_updrs": 1, "severity.total_updrs": 1,
            "severity.severity_level": 1,
            "shap.top_features": 1, "processing_time_ms": 1,
        }).sort("timestamp", -1).skip(skip).limit(page_size)

        items = []
        async for doc in cursor:
            det = doc.get("detection") or {}
            sev = doc.get("severity") or {}
            shap = doc.get("shap") or {}
            top_feat = (shap.get("top_features") or [{}])
            items.append(sc.HistoryItem(
                prediction_id=str(doc["_id"]),
                filename=doc.get("filename", ""),
                timestamp=doc.get("timestamp", datetime.now(timezone.utc)),
                detection_label=det.get("label"),
                confidence=det.get("confidence"),
                motor_updrs=sev.get("motor_updrs"),
                total_updrs=sev.get("total_updrs"),
                severity_level=sev.get("severity_level"),
                shap_top_feature=(top_feat[0].get("feature") if top_feat else None),
                processing_time_ms=doc.get("processing_time_ms"),
            ))

        return sc.HistoryResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=math.ceil(total / page_size),
        )
    except Exception as e:
        logger.error(f"History query failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch history.")


@app.get("/api/stats", response_model=sc.StatsResponse, tags=["Analytics"])
async def get_stats():
    """Live aggregate statistics from MongoDB."""
    col = db_module.get_predictions_col()
    if col is None:
        return sc.StatsResponse(
            total_predictions=0, parkinson_count=0, healthy_count=0,
            avg_confidence=0.0, avg_processing_time_ms=0.0,
            parkinson_pct=0.0, healthy_pct=0.0,
        )
    try:
        pipeline = [{"$group": {
            "_id": None,
            "total": {"$sum": 1},
            "park": {"$sum": {"$cond": [{"$eq": ["$detection.label", "Parkinson"]}, 1, 0]}},
            "healthy": {"$sum": {"$cond": [{"$eq": ["$detection.label", "Healthy"]}, 1, 0]}},
            "avg_conf": {"$avg": "$detection.confidence"},
            "avg_ms": {"$avg": "$processing_time_ms"},
        }}]
        doc = {}
        async for d in col.aggregate(pipeline):
            doc = d

        total = doc.get("total", 0)
        park = doc.get("park", 0)
        healthy = doc.get("healthy", 0)
        return sc.StatsResponse(
            total_predictions=total,
            parkinson_count=park,
            healthy_count=healthy,
            avg_confidence=round(doc.get("avg_conf") or 0.0, 4),
            avg_processing_time_ms=round(doc.get("avg_ms") or 0.0, 1),
            parkinson_pct=round(park / total * 100, 1) if total > 0 else 0.0,
            healthy_pct=round(healthy / total * 100, 1) if total > 0 else 0.0,
        )
    except Exception as e:
        logger.error(f"Stats query failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch stats.")


@app.get("/api/shap-plot/{prediction_id}", tags=["Analytics"])
async def get_shap_plot(prediction_id: str):
    """Return the stored SHAP plot PNG for a given prediction_id."""
    col = db_module.get_predictions_col()
    if col is None:
        raise HTTPException(status_code=503, detail="Database not connected.")

    doc = await col.find_one({"_id": prediction_id}, {"shap.plot_base64": 1})
    if not doc:
        raise HTTPException(status_code=404, detail="Prediction not found.")

    b64 = (doc.get("shap") or {}).get("plot_base64")
    if not b64:
        raise HTTPException(status_code=404, detail="No SHAP plot stored for this prediction.")

    import base64
    img_bytes = base64.b64decode(b64)
    from fastapi.responses import Response
    return Response(content=img_bytes, media_type="image/png")
