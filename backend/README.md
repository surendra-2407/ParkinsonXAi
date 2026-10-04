# ParkinsonXAI — Backend API Service

A high-performance asynchronous REST microservice built with **FastAPI**, **Uvicorn**, **Librosa**, **Praat-Parselmouth**, **LightGBM**, and **TreeSHAP**. The backend processes voice recordings, extracts ~220 acoustic features, executes gradient-boosted decision tree classification, generates game-theoretic SHAP explanations, and computes the AudioProxy acoustic severity index.

---

## Key Highlights

- **FastAPI Framework:** Native asynchronous request handling with automatic OpenAPI Swagger documentation at `/docs` and ReDoc at `/redoc`.
- **Digital Signal Processing (DSP):** ~220 raw acoustic descriptors extracted using `librosa` and `praat-parselmouth` (MFCCs, Deltas, Pitch $F_0$, Jitter, Shimmer, HNR, Voiced Fraction, RMS Energy, Spectral Centroid/Roll-off).
- **Production Preprocessing Pipeline:** Robust multi-stage pipeline utilizing Scikit-learn `SimpleImputer` (median), `VarianceThreshold`, and `RobustScaler` with top-50 Mutual Information feature selection.
- **Explainable AI (TreeSHAP):** Computes exact local Shapley values via `shap.TreeExplainer` on the tuned LightGBM classifier.
- **AudioProxy Severity Index:** Deterministic acoustic impairment score ($0–100$) reflecting voice symptom severity (Mild: $<30$, Moderate: $30–60$, Severe: $\ge 60$).
- **Zero-Setup Database Fallback:** Full support for MongoDB via `motor`/`pymongo`. If MongoDB is not running or unconfigured, the backend automatically falls back to an in-memory session store without breaking any API endpoints.
- **17 Dedicated REST Endpoints:** Complete API coverage for inference, telemetry, history management, and paper benchmark reproduction.

---

## Directory Structure

```
backend/
├── main.py                  # FastAPI application entry point, CORS, routes & lifecycle
├── inference.py             # Audio DSP extraction, ML pipeline transform, SHAP, AudioProxy
├── models_loader.py         # Model artifact loading, caching, and health verification
├── schemas.py               # Pydantic request/response data contracts
├── database.py              # MongoDB async client with graceful in-memory fallback
├── setup_mongodb.py         # MongoDB index initialization script
├── start_backend.ps1        # Windows PowerShell startup script
├── install_deps.ps1         # Windows PowerShell dependency installer script
├── requirements_backend.txt # Authoritative Python package dependencies
├── .env.example             # Environment variable template
└── .env                     # Local environment configuration
```

---

## Technology Stack & Dependencies

The authoritative list of dependencies is maintained in `requirements_backend.txt`:

| Package | Version | Purpose |
| :--- | :--- | :--- |
| `fastapi` | `0.115.0` | High-performance async web framework |
| `uvicorn[standard]` | `0.30.6` | Production ASGI web server |
| `librosa` | `0.10.2` | Audio signal processing & spectral feature extraction |
| `soundfile` | `0.12.1` | Audio I/O library |
| `praat-parselmouth` | `0.4.7` *(optional/recommended)* | Praat acoustic algorithms for Jitter, Shimmer, and HNR |
| `lightgbm` | `4.5.0` | Deployed gradient-boosted decision tree binary classifier |
| `scikit-learn` | `1.5.1` | Preprocessing transformers (`SimpleImputer`, `RobustScaler`, etc.) |
| `shap` | `0.46.0` | TreeSHAP explainability engine |
| `joblib` | `1.4.2` | Serialized model loading |
| `numpy` / `pandas` | `1.26.4` / `2.2.2` | High-performance numerical and matrix operations |
| `motor` / `pymongo` | `3.5.1` / `4.8.0` | Async MongoDB database driver |
| `python-multipart` | `0.0.12` | Multipart form-data parser for audio uploads |
| `matplotlib` / `Pillow` | `3.9.2` / `10.4.0` | SHAP waterfall plot rendering and base64 encoding |

---

## API Reference (All 17 Endpoints)

### 1. Clinical Inference
- **`POST /api/predict-full`**: Complete analysis pipeline. Uploads audio (`.wav`, `.mp3`, `.ogg`, `.flac`), extracts features, computes LightGBM prediction, AudioProxy severity, top SHAP positive/negative biomarkers, and persists the record.
- **`POST /api/predict`**: Fast binary classification returning `prediction`, `probability`, and `confidence`.
- **`POST /api/severity`**: Dedicated AudioProxy severity assessment returning composite score ($0–100$), stage classification, and individual biomarker penalty details.
- **`GET /api/shap-plot/{prediction_id}`**: Generates and serves a base64-encoded SHAP waterfall plot image for a specific historical prediction.

### 2. System Telemetry & Health
- **`GET /api/health`**: Service health probe returning model load status, backend version, and MongoDB connectivity status.
- **`GET /api/dashboard`**: Aggregated dashboard telemetry including active model parameters, accuracy benchmarks, and system activity.
- **`GET /api/stats`**: High-level platform statistics (total tests run, healthy vs. Parkinson's diagnosis counts).
- **`GET /api/activity`**: Chronological stream of recent prediction events.

### 3. History & Audit Log
- **`GET /api/history`**: Paginated history of past voice analyses (supports `limit` and `skip`).
- **`GET /api/history-filtered`**: Advanced filtering by diagnosis `status` (Healthy/PD), severity `stage` (Mild/Moderate/Severe), text `search`, `start_date`, and `end_date`.
- **`GET /api/history/detail/{prediction_id}`**: Retrieves the full 50-feature vector snapshot and individual SHAP attributions for a given prediction.
- **`DELETE /api/history/{prediction_id}`**: Deletes a specific prediction record from the database or in-memory cache.

### 4. Research Benchmarks & Datasets
- **`GET /api/models/performance`**: Benchmark comparison data across models (LightGBM, XGBoost, Random Forest, SVM, KNN) and datasets (Table III).
- **`GET /api/explanations/global`**: Global SHAP feature importance rankings and clinical descriptions derived from the Dataset 1 test set.
- **`GET /api/severity/evaluations`**: Audited MDS-UPDRS regression evaluations on telemonitoring data (Table IV).
- **`GET /api/datasets`**: Scientific profiles of Datasets 1, 2, 3a, and 3b including patient-leakage boundaries (Table I).
- **`GET /api/features`**: Complete feature engineering taxonomy (220 raw features across 13 acoustic groups into top-50 selected) (Table II).

---

## Getting Started

### 1. Create Virtual Environment

```bash
cd backend
python -m venv .venv
```

Activate the environment:
- **Windows PowerShell:** `.\.venv\Scripts\Activate.ps1`
- **Linux / macOS:** `source .venv/bin/activate`

### 2. Install Dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements_backend.txt
```

*(Optional Praat support)*:
```bash
pip install praat-parselmouth
```

### 3. Configure Environment

Copy `.env.example` to `.env`:
```ini
MONGODB_URL=
DATABASE_NAME=parkinson_xai
CORS_ORIGINS=http://localhost:5173
MODEL_BASE_PATH=../models
```

> **Note:** Leaving `MONGODB_URL` empty is fully supported. The backend will automatically run in zero-setup in-memory mode.

### 4. Run the Server

**Using the PowerShell Script (Windows):**
```powershell
.\start_backend.ps1
```

**Using Uvicorn Directly:**
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Interactive API documentation will be available at:
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`
