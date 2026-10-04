# ParkinsonXAI — Frontend Client

A responsive, clinical-grade web application built with **React 19**, **TypeScript**, **Vite**, **Recharts**, and a **Pure CSS Design System**. The frontend serves as the interactive decision-support interface for the **ParkinsonXAI** machine learning research framework.

---

## Key Highlights

- **Acoustic Audio Input & Recorder:** Multi-format audio file dropzone (`.wav`, `.mp3`, `.ogg`, `.flac`) and an in-browser live voice recorder (`LiveRecorder.tsx`) using the Web Audio API with a native PCM WAV encoder (`wavEncoder.ts`).
- **Explainable AI (TreeSHAP Visualizations):** Local SHAP attribution charts (`ShapChart.tsx`) displaying positive and negative acoustic biomarker contributions, paired with an interactive Global SHAP Explainability Hub (`SHAPExplainabilityPage.tsx`).
- **Dual Audience Explanations:**
  - **Patient-Friendly Summary (`PlainLanguageExplanation.tsx`):** Empathetic, jargon-free explanations translating acoustic deviations into accessible insights with actionable vocal wellness tips.
  - **Clinician Biomechanical Breakdown (`ClinicalExplanation.tsx`):** Detailed analysis of pitch perturbation (jitter), amplitude instability (shimmer), harmonics-to-noise ratio ($HNR$), and vocal tract resonances ($MFCCs$).
- **AudioProxy Severity Gauge (`SeverityGauge.tsx`):** A continuous 0–100 acoustic impairment meter categorized into *Mild (0–29)*, *Moderate (30–59)*, and *Severe (60–100)* with individual penalty breakdowns.
- **Scientific Research & Benchmark Hub (`ResearchOverviewPage.tsx`):** 5-tab portal reproducing experimental benchmark tables (Model Performance, UPDRS Severity, Datasets, Feature Engineering, and How It Works).
- **Comprehensive Audit Trail (`HistoryPage.tsx`):** Searchable, filterable history log with CSV/JSON export, prediction inspection drawer, and record management.
- **Zero-Dependency Styling & Icons:** Pure CSS design tokens with dark-mode clinical glassmorphism and embedded SVGs (no heavy UI frameworks or icon libraries required).

---

## Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Framework** | [React 19](https://react.dev/) | Component architecture & modern concurrent rendering |
| **Language** | [TypeScript 5 / 6](https://www.typescriptlang.org/) | Strict type checking & API data contracts |
| **Bundler & Dev Server** | [Vite 8](https://vite.dev/) | Instant HMR and optimized production bundling |
| **Routing** | [React Router 7](https://reactrouter.com/) | Client-side routing with clean URL navigation |
| **Charts & Data Viz** | [Recharts 3](https://recharts.org/) | Responsive SVG charts for SHAP and biomarker metrics |
| **HTTP Client** | [Axios](https://axios-http.com/) | REST API communication with the FastAPI backend |
| **Design System** | Pure CSS Tokens (`index.css`, `App.css`) | Clinical dark glassmorphism, responsive grids, micro-animations |
| **Linter** | [Oxlint](https://oxc.rs/) | High-performance JavaScript/TypeScript linter |

---

## Directory Structure

```
frontend/
├── index.html                   # HTML entry point with modern typography
├── package.json                 # Node dependencies and scripts
├── vite.config.ts               # Vite configuration with /api reverse proxy
├── tsconfig.json                # TypeScript strict compiler rules
│
└── src/
    ├── main.tsx                 # React DOM mount point
    ├── App.tsx                  # Top-level router and page route definitions
    ├── App.css                  # Clinical glassmorphism design tokens & card styles
    ├── index.css                # Base typography, layout resets, and CSS variables
    │
    ├── api/
    │   └── parkinsonApi.ts      # Axios client covering all 17 backend REST endpoints
    │
    ├── pages/
    │   ├── HomePage.tsx         # Voice dropzone, live recorder & analysis trigger
    │   ├── ResultsPage.tsx      # Comprehensive diagnosis, confidence, SHAP & explanations
    │   ├── DashboardPage.tsx    # Live system telemetry, active models & quick stats
    │   ├── HistoryPage.tsx      # Filterable audit log, CSV/JSON export & details drawer
    │   ├── SHAPExplainabilityPage.tsx # Global SHAP rankings & interactive waterfall simulator
    │   ├── ResearchOverviewPage.tsx   # 5-tab research portal hosting benchmark tables
    │   ├── ModelPerformancePage.tsx   # Comparative model benchmark table (Table III)
    │   ├── UPDRSSeverityPage.tsx      # MDS-UPDRS regression evaluations & R² limits (Table IV)
    │   ├── DatasetInfoPage.tsx        # Dataset provenance & patient leakage audit (Table I)
    │   ├── FeatureEngineeringPage.tsx # Acoustic feature extraction pipeline (Table II)
    │   └── HowItWorksPage.tsx         # Signal processing & methodology guide
    │
    ├── components/
    │   ├── Navigation.tsx             # Responsive header bar & mobile drawer
    │   ├── AudioInput.tsx             # Audio upload area with sample files
    │   ├── AudioUpload.tsx            # Drag-and-drop file uploader
    │   ├── LiveRecorder.tsx           # In-browser microphone recorder & waveform visualizer
    │   ├── ConfidenceRing.tsx         # Circular SVG confidence indicator
    │   ├── SeverityGauge.tsx          # AudioProxy 0–100 acoustic severity meter
    │   ├── ShapChart.tsx              # Interactive horizontal SHAP attribution chart
    │   ├── PlainLanguageExplanation.tsx # Patient summary with wellness recommendations
    │   ├── ClinicalExplanation.tsx    # Biomechanical acoustic breakdown for clinicians
    │   ├── ModelCard.tsx              # Reusable ML benchmark card
    │   ├── PredictionHistory.tsx      # Quick recent predictions widget
    │   └── FeatureGlossary.tsx        # Acoustic biomarker glossary modal
    │
    ├── hooks/
    │   ├── useAnalysis.ts             # Prediction execution state & async handlers
    │   └── useSession.ts              # Anonymous session management
    │
    └── utils/
        ├── plainLanguageExplainer.ts  # Patient explanation text generation logic
        ├── clinicalExplainer.ts       # Clinical acoustic breakdown synthesis
        └── wavEncoder.ts              # Native PCM WAV audio encoder for recordings
```

---

## Application Routes & Navigation

| Route | Page Component | Description |
| :--- | :--- | :--- |
| `/` | `HomePage.tsx` | **Voice Analysis:** Upload audio or record voice live, inspect waveform, submit for inference. |
| `/results` | `ResultsPage.tsx` | **Analysis Results:** Detection verdict, confidence gauge, AudioProxy severity, SHAP chart, and dual explanations. |
| `/dashboard` | `DashboardPage.tsx` | **Dashboard:** Live health telemetry, model registry cards, and aggregate diagnostic stats. |
| `/history` | `HistoryPage.tsx` | **History:** Searchable audit log, date/status filtering, export to CSV/JSON, and full feature vector inspection. |
| `/shap` | `SHAPExplainabilityPage.tsx` | **SHAP Explainability:** Global feature rankings from Dataset 1 test set, interactive simulator, and pathophysiology links. |
| `/research` | `ResearchOverviewPage.tsx` | **Research Hub:** 5-tab academic center reproducing paper tables and validation methodology. |

---

## Getting Started

### Prerequisites

- **Node.js:** `v18.x`, `v20.x`, or `v22.x`
- **npm:** `v9.x` or `v10.x`
- **Backend Service:** Running at `http://localhost:8000` (for live inference and API features)

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Start Development Server

```bash
npm run dev
```

The application will be available at **`http://localhost:5173`**.

> **API Proxying:** Requests to `/api/*` are automatically forwarded to `http://localhost:8000` via the Vite reverse proxy configured in `vite.config.ts`.

### 3. Build for Production

```bash
npm run build
```

This compiles TypeScript (`tsc -b`) and bundles assets into the `dist/` directory.

### 4. Run Linter

```bash
npm run lint
```

Runs **Oxlint** for ultra-fast static analysis across all TypeScript and React files.

---

## Architectural Notes

### 1. In-Browser WAV Recording
The browser microphone captures audio via `navigator.mediaDevices.getUserMedia`. Raw audio buffer chunks are collected using `AudioContext` and encoded into a standard 16-bit 22,050 Hz PCM `.wav` file entirely on the client side (`wavEncoder.ts`). This guarantees full compatibility with the backend `librosa` and `praat-parselmouth` processing pipeline without requiring external transcoding tools like ffmpeg.

### 2. Dual-Perspective Interpretability
Machine learning decisions in healthcare often struggle with communication gaps:
- **Patients** receive clear, non-alarmist summaries explaining what vocal changes mean in daily life, alongside vocal hygiene and lifestyle tips.
- **Clinicians** receive numerical biomechanical metrics with standard physiological reference ranges (e.g., Jitter threshold $1.04\%$, Shimmer threshold $3.81\%$, HNR threshold $20\text{ dB}$).

### 3. Graceful Fallbacks & Offline Resilience
If the backend or database is offline, the frontend provides clear error messaging, displays informative empty states, and allows browsing preloaded research benchmarks and model cards without UI crashes.
