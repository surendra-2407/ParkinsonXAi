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
            "voice_quality": det_raw.get("voice_quality"),
            "is_uncertain": det_raw.get("is_uncertain", False),
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
            "voice_quality": _f(det_raw.get("voice_quality", 1.0)),
            "quality_warning": det_raw.get("quality_warning"),
            "is_uncertain": bool(det_raw.get("is_uncertain", False)),
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


# ── Enhanced History with search/filter ──────────────────────────────────────

@app.get("/api/history/detail/{prediction_id}", tags=["Analytics"])
async def get_prediction_detail(prediction_id: str):
    """Return full prediction document by ID."""
    col = db_module.get_predictions_col()
    if col is None:
        raise HTTPException(status_code=503, detail="Database not connected.")
    doc = await col.find_one({"_id": prediction_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    doc["_id"] = str(doc["_id"])
    # Serialize datetime
    if "timestamp" in doc and hasattr(doc["timestamp"], "isoformat"):
        doc["timestamp"] = doc["timestamp"].isoformat()
    return JSONResponse(content=doc)


@app.delete("/api/history/{prediction_id}", tags=["Analytics"])
async def delete_prediction(prediction_id: str):
    """Delete a prediction record from MongoDB."""
    col = db_module.get_predictions_col()
    if col is None:
        raise HTTPException(status_code=503, detail="Database not connected.")
    result = await col.delete_one({"_id": prediction_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return JSONResponse(content={"message": "Deleted", "prediction_id": prediction_id})


@app.get("/api/history-filtered", tags=["Analytics"])
async def get_history_filtered(
    page: int = 1,
    page_size: int = 10,
    search: Optional[str] = None,
    label: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
):
    """Paginated history with search (filename), label filter, and date range."""
    col = db_module.get_predictions_col()
    if col is None:
        return {"items": [], "total": 0, "page": page, "page_size": page_size, "total_pages": 0}

    page = max(1, page)
    page_size = min(50, max(1, page_size))
    skip = (page - 1) * page_size

    # Build filter query
    query: dict = {}
    if search:
        import re
        query["filename"] = {"$regex": re.escape(search), "$options": "i"}
    if label and label in ("Parkinson", "Healthy", "Uncertain"):
        query["detection.label"] = label
    if date_from or date_to:
        ts_filter: dict = {}
        try:
            if date_from:
                ts_filter["$gte"] = datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc)
            if date_to:
                ts_filter["$lte"] = datetime.fromisoformat(date_to).replace(tzinfo=timezone.utc)
            query["timestamp"] = ts_filter
        except ValueError:
            pass

    try:
        total = await col.count_documents(query)
        cursor = col.find(query, {
            "_id": 1, "filename": 1, "timestamp": 1,
            "detection.label": 1, "detection.confidence": 1,
            "detection.voice_quality": 1, "detection.is_uncertain": 1,
            "severity.motor_updrs": 1, "severity.total_updrs": 1,
            "severity.severity_level": 1, "severity.severity_score": 1,
            "shap.top_features": 1, "processing_time_ms": 1,
        }).sort("timestamp", -1).skip(skip).limit(page_size)

        items = []
        async for doc in cursor:
            det = doc.get("detection") or {}
            sev = doc.get("severity") or {}
            shap_d = doc.get("shap") or {}
            top_feat = shap_d.get("top_features") or []
            ts = doc.get("timestamp")
            items.append({
                "prediction_id": str(doc["_id"]),
                "filename": doc.get("filename", ""),
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "detection_label": det.get("label"),
                "confidence": det.get("confidence"),
                "voice_quality": det.get("voice_quality"),
                "is_uncertain": det.get("is_uncertain", False),
                "motor_updrs": sev.get("motor_updrs"),
                "total_updrs": sev.get("total_updrs"),
                "severity_level": sev.get("severity_level"),
                "severity_score": sev.get("severity_score"),
                "shap_top_feature": top_feat[0].get("feature") if top_feat else None,
                "processing_time_ms": doc.get("processing_time_ms"),
            })

        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": math.ceil(total / page_size) if total > 0 else 0,
        }
    except Exception as e:
        logger.error(f"Filtered history query failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch history.")


@app.get("/api/activity", tags=["Analytics"])
async def get_activity(days: int = 30):
    """Return daily prediction counts for the past N days (for activity chart)."""
    col = db_module.get_predictions_col()
    if col is None:
        return {"activity": []}
    try:
        from datetime import timedelta
        since = datetime.now(timezone.utc) - timedelta(days=days)
        pipeline = [
            {"$match": {"timestamp": {"$gte": since}}},
            {"$group": {
                "_id": {
                    "year": {"$year": "$timestamp"},
                    "month": {"$month": "$timestamp"},
                    "day": {"$dayOfMonth": "$timestamp"},
                },
                "count": {"$sum": 1},
                "parkinson": {"$sum": {"$cond": [{"$eq": ["$detection.label", "Parkinson"]}, 1, 0]}},
                "healthy": {"$sum": {"$cond": [{"$eq": ["$detection.label", "Healthy"]}, 1, 0]}},
            }},
            {"$sort": {"_id.year": 1, "_id.month": 1, "_id.day": 1}},
        ]
        result = []
        async for doc in col.aggregate(pipeline):
            d = doc["_id"]
            result.append({
                "date": f"{d['year']}-{d['month']:02d}-{d['day']:02d}",
                "count": doc["count"],
                "parkinson": doc["parkinson"],
                "healthy": doc["healthy"],
            })
        return {"activity": result}
    except Exception as e:
        logger.error(f"Activity query failed: {e}")
        return {"activity": []}


# ── Research Paper Data Endpoints ─────────────────────────────────────────────

@app.get("/api/models/performance", tags=["Research"])
async def get_model_performance():
    """
    Model performance data from the research paper (Table III).
    Labelled as reported held-out results — not live prediction results.
    """
    return JSONResponse(content={
        "source": "Research paper — Table III (held-out test set results)",
        "note": "D1 has no recoverable speaker IDs; record-level results cannot establish patient-independent generalization.",
        "models": [
            {"dataset": "D1", "dataset_name": "Voice WAV", "model": "LightGBM", "protocol": "Record-level",
             "accuracy": 0.9649, "f1": 0.9655, "recall": 1.000, "roc_auc": 0.9995, "is_primary": True},
            {"dataset": "D1", "dataset_name": "Voice WAV", "model": "Random Forest", "protocol": "Record-level",
             "accuracy": 0.9649, "f1": 0.9655, "recall": 1.000, "roc_auc": 0.9975, "is_primary": False},
            {"dataset": "D2", "dataset_name": "TQWT Speech", "model": "LightGBM", "protocol": "Speaker-disjoint",
             "accuracy": 0.8596, "f1": 0.9059, "recall": 0.951, "roc_auc": 0.9274, "is_primary": False},
            {"dataset": "D2", "dataset_name": "TQWT Speech", "model": "CatBoost", "protocol": "Speaker-disjoint",
             "accuracy": 0.7982, "f1": 0.8686, "recall": 0.938, "roc_auc": 0.8773, "is_primary": False},
            {"dataset": "D3a", "dataset_name": "UCI Parkinson's", "model": "SVM", "protocol": "Speaker-disjoint",
             "accuracy": 0.8000, "f1": 0.8889, "recall": 1.000, "roc_auc": 0.6319, "is_primary": False},
            {"dataset": "D3a", "dataset_name": "UCI Parkinson's", "model": "LightGBM", "protocol": "Speaker-disjoint",
             "accuracy": 0.7667, "f1": 0.8679, "recall": 0.958, "roc_auc": 0.4792, "is_primary": False},
            {"dataset": "D3a", "dataset_name": "UCI Parkinson's", "model": "XGBoost", "protocol": "Speaker-disjoint",
             "accuracy": 0.6333, "f1": 0.7755, "recall": 0.792, "roc_auc": 0.5972, "is_primary": False},
        ]
    })


@app.get("/api/explanations/global", tags=["Research"])
async def get_global_shap():
    """
    Global SHAP feature importance from the research paper.
    These are mean |SHAP| values averaged over the held-out D1 test set.
    Individual prediction SHAP values are computed live by /api/predict-full.
    """
    return JSONResponse(content={
        "source": "Research paper — global SHAP analysis on D1 test set",
        "note": "Mean absolute SHAP values represent average feature importance, not causation.",
        "features": [
            {"rank": 1, "feature": "mfcc_24_std", "display": "MFCC 24 Std", "mean_abs_shap": 1.958,
             "group": "MFCC", "description": "Standard deviation of 24th Mel-frequency cepstral coefficient — captures voice texture variation."},
            {"rank": 2, "feature": "mfcc_26_std", "display": "MFCC 26 Std", "mean_abs_shap": 1.206,
             "group": "MFCC", "description": "Std of 26th MFCC — related to spectral fine structure variability."},
            {"rank": 3, "feature": "mfcc_2_std", "display": "MFCC 2 Std", "mean_abs_shap": 0.963,
             "group": "MFCC", "description": "Std of 2nd MFCC — captures overall spectral shape variability."},
            {"rank": 4, "feature": "rms_std", "display": "RMS Energy Std", "mean_abs_shap": 0.812,
             "group": "Energy", "description": "Root-mean-square energy variability — reflects loudness instability (hypophonia)."},
            {"rank": 5, "feature": "mfcc_14_std", "display": "MFCC 14 Std", "mean_abs_shap": 0.703,
             "group": "MFCC", "description": "Std of 14th MFCC — mid-range spectral variability."},
            {"rank": 6, "feature": "jitter_local", "display": "Jitter (Local)", "mean_abs_shap": 0.196,
             "group": "Jitter", "description": "Cycle-to-cycle pitch period variation — a classic vocal tremor indicator in Parkinson's."},
        ]
    })


@app.get("/api/severity/evaluations", tags=["Research"])
async def get_severity_evaluations():
    """
    UPDRS regression evaluation data from the research paper (Table IV).
    Negative R² values indicate models do not generalize reliably to unseen subjects.
    """
    return JSONResponse(content={
        "source": "Research paper — Table IV (UPDRS regression, speaker-disjoint)",
        "note": (
            "Negative R² scores indicate the regression models do not yet achieve reliable "
            "generalization to unseen subjects on this dataset. "
            "The inference service uses a deterministic audio-proxy score when UPDRS inputs are unavailable — "
            "this proxy must NOT be interpreted as a clinical UPDRS score."
        ),
        "motor": [
            {"model": "Extra Trees", "mae": 7.25, "rmse": 8.72, "r2": -1.81},
            {"model": "XGBoost",     "mae": 7.96, "rmse": 8.72, "r2": -1.81},
            {"model": "LightGBM",    "mae": 8.01, "rmse": 9.09, "r2": -2.05},
            {"model": "Linear",      "mae": 9.20, "rmse": 10.41,"r2": -3.00},
        ],
        "total": [
            {"model": "LightGBM",     "mae": 7.22, "rmse": 9.06,  "r2": -2.33},
            {"model": "XGBoost",      "mae": 7.97, "rmse": 9.30,  "r2": -2.51},
            {"model": "Random Forest","mae": 8.98, "rmse": 11.70, "r2": -4.56},
            {"model": "Linear",       "mae": 11.81,"rmse": 12.94, "r2": -5.80},
        ]
    })


@app.get("/api/datasets", tags=["Research"])
async def get_datasets():
    """Dataset information from the research paper (Table I)."""
    return JSONResponse(content={
        "source": "Research paper — Table I",
        "datasets": [
            {
                "id": "D1", "name": "Voice WAV Dataset",
                "recordings": 1134, "speakers": None,
                "healthy_count": 574, "pd_count": 560,
                "target": "Binary (HC / PD)", "target_type": "detection",
                "purpose": "Primary detection model training & evaluation",
                "protocol": "Record-level (no recoverable speaker IDs)",
                "notes": "No speaker ID recovery possible; results cannot establish patient-independent generalization.",
                "model_used": "LightGBM (96.49% accuracy, AUC 0.9995)",
            },
            {
                "id": "D2", "name": "TQWT Speech Features",
                "recordings": 756, "speakers": 252,
                "healthy_count": 192, "pd_count": 564,
                "target": "Binary (HC / PD)", "target_type": "detection",
                "purpose": "Speaker-disjoint generalization benchmark",
                "protocol": "Speaker-disjoint cross-validation",
                "notes": "252 unique speakers; speaker-disjoint split ensures patient-independent evaluation.",
                "model_used": "LightGBM (85.96% accuracy, AUC 0.9274)",
            },
            {
                "id": "D3a", "name": "UCI Parkinson's Dataset",
                "recordings": 195, "speakers": 31,
                "healthy_count": 48, "pd_count": 147,
                "target": "Binary (HC / PD)", "target_type": "detection",
                "purpose": "Benchmark comparison against prior work",
                "protocol": "Speaker-disjoint (LOSO)",
                "notes": "Small dataset; SVM achieves 80% with 100% recall.",
                "model_used": "SVM (80.00%) / LightGBM (76.67%)",
            },
            {
                "id": "D3b", "name": "Parkinson's Telemonitoring",
                "recordings": 5875, "speakers": 42,
                "healthy_count": None, "pd_count": None,
                "target": "UPDRS Score (regression)", "target_type": "regression",
                "purpose": "Severity prediction (motor & total UPDRS)",
                "protocol": "Speaker-disjoint regression",
                "notes": "Negative R² values; models do not yet generalize reliably to unseen subjects.",
                "model_used": "LightGBM (Motor MAE: 7.25, Total MAE: 7.22)",
            },
        ]
    })


@app.get("/api/features", tags=["Research"])
async def get_features():
    """Feature engineering data from the research paper (Table II)."""
    return JSONResponse(content={
        "source": "Research paper — Table II",
        "total_raw": 220,
        "total_non_degenerate": 219,
        "total_selected": 50,
        "selection_method": "Mutual Information (top 50 from 219)",
        "feature_groups": [
            {"group": "MFCC", "count": 160, "description": "40 coefficients × mean + std + delta + delta-delta. Capture vocal tract shape and articulatory precision."},
            {"group": "Chroma", "count": 24, "description": "12 pitch classes × mean + std. Represent harmonic content and pitch stability."},
            {"group": "Spectral Centroid / Bandwidth / Roll-off", "count": 6, "description": "Mean + std for each. Capture spectral brightness, spread, and energy concentration."},
            {"group": "Spectral Contrast", "count": 7, "description": "7 sub-bands. Measures the difference between spectral peaks and valleys."},
            {"group": "Spectral Flatness", "count": 1, "description": "Single value. Distinguishes tonal voice from noisy, breathy speech."},
            {"group": "RMS / ZCR", "count": 4, "description": "Energy (mean + std) and zero-crossing rate (mean + std). Quantify loudness and speech rate."},
            {"group": "Mel Spectrogram", "count": 3, "description": "Mean, std, and skewness of 64-band log mel-spectrogram. Capture overall spectral texture."},
            {"group": "pYIN F0", "count": 3, "description": "Mean, std, and voiced fraction. Pitch estimation via probabilistic YIN algorithm."},
            {"group": "Praat F0", "count": 2, "description": "Mean and std of fundamental frequency via Praat/Parselmouth."},
            {"group": "Jitter", "count": 3, "description": "Local, RAP, PPQ5. Cycle-to-cycle pitch period variation — key Parkinson's tremor indicator."},
            {"group": "Shimmer", "count": 3, "description": "Local, APQ3, APQ5. Amplitude variation between consecutive cycles — hypophonia marker."},
            {"group": "HNR", "count": 1, "description": "Harmonics-to-Noise Ratio. Measures voice breathiness and hoarseness."},
            {"group": "Formants", "count": 3, "description": "F1, F2, F3 mean frequencies. Reflect vocal tract resonance and articulation quality."},
        ]
    })

