import { useLocation, useNavigate } from 'react-router-dom'
import type { FullAnalysisResponse } from '../api/parkinsonApi'
import ConfidenceRing from '../components/ConfidenceRing'
import SeverityGauge from '../components/SeverityGauge'
import ShapChart from '../components/ShapChart'
import ClinicalExplanation from '../components/ClinicalExplanation'
import FeatureGlossary from '../components/FeatureGlossary'
import PlainLanguageExplanation from '../components/PlainLanguageExplanation'
import { buildClinicalSummary, FEATURE_TOOLTIP_MAP } from '../utils/clinicalExplainer'

export default function ResultsPage() {
  const location = useLocation()
  const navigate = useNavigate()

  const result: FullAnalysisResponse | null =
    location.state?.result ||
    (() => {
      try { return JSON.parse(localStorage.getItem('xai_last_result') || 'null') }
      catch { return null }
    })()

  if (!result) {
    return (
      <div className="page">
        <div className="container text-center" style={{ paddingTop: 80 }}>
          <div style={{ fontSize: '3rem', marginBottom: 16 }}>🎙️</div>
          <p style={{ fontSize: '1.1rem', color: 'var(--text-muted)' }}>No results to display.</p>
          <button className="btn-primary" style={{ marginTop: 20 }} onClick={() => navigate('/')}>
            Go to Analysis
          </button>
        </div>
      </div>
    )
  }

  const { detection, severity, shap, audio_features_snapshot, processing_time_ms, filename } = result

  // Always Healthy or Parkinson — no Uncertain
  const isParkinson = detection.label === 'Parkinson'
  const accentColor  = isParkinson ? '#ef4444' : '#10b981'
  const accentDim    = isParkinson ? 'rgba(239,68,68,0.10)' : 'rgba(16,185,129,0.10)'
  const accentBorder = isParkinson ? 'rgba(239,68,68,0.35)' : 'rgba(16,185,129,0.35)'
  const accentText   = isParkinson ? '#f87171' : '#34d399'

  const clinicalSummary = buildClinicalSummary(
    detection.label,
    detection.confidence,
    shap.top_features,
    severity.motor_updrs,
    severity.severity_level,
  )

  // Key voice metrics to show in the quick metrics row
  const voiceMetrics = [
    { key: 'voiced_fraction', label: 'Voiced Fraction', fmt: (v: number) => `${(v * 100).toFixed(1)}%` },
    { key: 'jitter_local',    label: 'Jitter (Local)',  fmt: (v: number) => v.toFixed(5) },
    { key: 'shimmer_local',   label: 'Shimmer (Local)', fmt: (v: number) => v.toFixed(5) },
    { key: 'hnr',             label: 'HNR (dB)',        fmt: (v: number) => v.toFixed(2) },
    { key: 'rms_mean',        label: 'RMS Energy',      fmt: (v: number) => v.toFixed(5) },
    { key: 'f0_mean',         label: 'F0 Mean (Hz)',    fmt: (v: number) => v.toFixed(1) },
  ]

  return (
    <div className="page">
      <div className="container">

        {/* ── Header ── */}
        <div className="flex justify-between items-center" style={{ marginBottom: 28, flexWrap: 'wrap', gap: 12 }}>
          <div>
            <p className="section-label">Analysis Results</p>
            <h1 style={{ fontSize: '1.6rem', fontWeight: 700, marginTop: 4 }}>🎵 {filename}</h1>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: 2 }}>
              Processed in {processing_time_ms?.toFixed(0)} ms · {new Date().toLocaleString()}
              {detection.model_used && ` · Model: ${detection.model_used}`}
            </p>
          </div>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            <button
              className="btn-secondary"
              onClick={() => {
                const el = document.getElementById('written-explanation-section')
                if (el) el.scrollIntoView({ behavior: 'smooth' })
              }}
              style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}
            >
              📖 Jump to Plain-Language Explanation
            </button>
            <button className="btn-secondary" onClick={() => navigate('/')}>← New Analysis</button>
          </div>
        </div>

        {/* ── Main result banner ── */}
        <div style={{
          padding: '20px 28px', marginBottom: 24, borderRadius: 'var(--radius-lg)',
          background: accentDim, border: `1px solid ${accentBorder}`,
          display: 'flex', alignItems: 'center', gap: 20,
        }}>
          <div style={{
            width: 56, height: 56, borderRadius: '50%',
            background: `${accentColor}22`, border: `2px solid ${accentColor}55`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '1.8rem', flexShrink: 0,
          }}>
            {isParkinson ? '⚠️' : '✅'}
          </div>
          <div style={{ flex: 1 }}>
            <p style={{ fontWeight: 800, fontSize: '1.2rem', color: accentText, marginBottom: 4 }}>
              {isParkinson ? "Parkinson's Disease Indicators Detected" : "No Parkinson's Indicators — Voice Appears Healthy"}
            </p>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              {isParkinson
                ? `The LightGBM model is ${(detection.confidence * 100).toFixed(1)}% confident that acoustic features match Parkinson's patterns.`
                : `Voice patterns appear healthy. The model is ${(detection.confidence * 100).toFixed(1)}% confident in a healthy classification.`}
            </p>
          </div>
          <div style={{ textAlign: 'right', flexShrink: 0 }}>
            <p style={{ fontSize: '2.2rem', fontWeight: 800, color: accentText, lineHeight: 1 }}>
              {(detection.confidence * 100).toFixed(1)}%
            </p>
            <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 2 }}>confidence</p>
          </div>
        </div>

        {/* ── Row 1: Confidence Ring + Severity ── */}
        <div className="grid-2" style={{ marginBottom: 20, alignItems: 'start' }}>

          {/* Detection card */}
          <div className="glass-card p-32">
            <p className="section-label" style={{ marginBottom: 20 }}>Detection Result</p>
            <div className="flex items-center gap-32" style={{ flexWrap: 'wrap' }}>
              <ConfidenceRing confidence={detection.confidence} label={detection.label} />

              <div style={{ flex: 1, minWidth: 160, display: 'flex', flexDirection: 'column', gap: 12 }}>
                {/* Class probability bars — always Healthy + Parkinson */}
                {Object.entries(detection.probabilities).map(([cls, prob]) => (
                  <div key={cls}>
                    <div className="flex justify-between" style={{ marginBottom: 6 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <span style={{
                          width: 8, height: 8, borderRadius: '50%',
                          background: cls === 'Parkinson' ? '#ef4444' : '#10b981',
                          display: 'inline-block', flexShrink: 0,
                        }} />
                        <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', fontWeight: 600 }}>{cls}</span>
                      </div>
                      <span style={{
                        fontSize: '0.9rem', fontWeight: 800,
                        color: cls === 'Parkinson' ? '#f87171' : '#34d399',
                      }}>
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="progress-bar-track" style={{ height: 10 }}>
                      <div
                        className="progress-bar-fill"
                        style={{
                          width: `${prob * 100}%`,
                          background: cls === 'Parkinson'
                            ? 'linear-gradient(90deg, #b91c1c, #ef4444)'
                            : 'linear-gradient(90deg, #059669, #10b981)',
                        }}
                      />
                    </div>
                  </div>
                ))}

                <div style={{
                  marginTop: 4, padding: '10px 14px',
                  background: 'var(--bg-card)', borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border)',
                }}>
                  <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: 2 }}>Model</p>
                  <p style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-primary)' }}>{detection.model_used}</p>
                  {detection.voice_quality != null && (
                    <>
                      <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 6, marginBottom: 2 }}>Voice Quality</p>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <div className="progress-bar-track" style={{ flex: 1, height: 6 }}>
                          <div className="progress-bar-fill" style={{
                            width: `${(detection.voice_quality * 100).toFixed(0)}%`,
                            background: detection.voice_quality > 0.6
                              ? 'linear-gradient(90deg, #059669, #10b981)'
                              : detection.voice_quality > 0.3
                                ? 'linear-gradient(90deg, #d97706, #f59e0b)'
                                : 'linear-gradient(90deg, #b91c1c, #ef4444)',
                          }} />
                        </div>
                        <span style={{
                          fontSize: '0.78rem', fontWeight: 700,
                          color: detection.voice_quality > 0.6 ? '#34d399' : detection.voice_quality > 0.3 ? '#fbbf24' : '#f87171',
                        }}>
                          {(detection.voice_quality * 100).toFixed(0)}%
                        </span>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Severity card */}
          <div className="glass-card p-32">
            <SeverityGauge
              motorUpdrs={severity.motor_updrs}
              totalUpdrs={severity.total_updrs}
              severityLevel={severity.severity_level}
              severityScore={severity.severity_score}
              severityBasis={severity.severity_basis}
            />
          </div>
        </div>

        {/* ── Row 2: Voice quality metrics quick view ── */}
        <div className="glass-card p-24" style={{ marginBottom: 20 }}>
          <div className="flex justify-between items-center" style={{ marginBottom: 16 }}>
            <p style={{ fontWeight: 700, color: 'var(--text-primary)' }}>Key Acoustic Features</p>
            <span className="badge badge-purple">Voice Biomarkers</span>
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
            {voiceMetrics.map(({ key, label, fmt }) => {
              const val = audio_features_snapshot?.[key] ?? result.detection?.voice_quality
              return (
                <div key={key} style={{
                  padding: '12px 16px', background: 'var(--bg-card)',
                  border: '1px solid var(--border)', borderRadius: 'var(--radius-md)',
                  minWidth: 130, flex: '1 1 130px',
                }}>
                  <p style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginBottom: 4, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                    {label}
                  </p>
                  <p style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--teal-light)', fontFamily: 'monospace' }}>
                    {val != null ? fmt(val as number) : '—'}
                  </p>
                </div>
              )
            })}
          </div>
        </div>

        {/* ── Row 3: SHAP chart ── */}
        <div className="glass-card p-32" style={{ marginBottom: 20 }}>
          <div className="flex justify-between items-center" style={{ marginBottom: 4 }}>
            <p style={{ fontWeight: 700, color: 'var(--text-primary)' }}>XAI Feature Explanation (SHAP)</p>
            <span className="badge badge-purple">TreeExplainer</span>
          </div>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 20 }}>
            Each bar shows how much a feature pushed the prediction toward Parkinson (red) or away from it (teal).
          </p>

          {shap.plot_base64 ? (
            <img
              src={`data:image/png;base64,${shap.plot_base64}`}
              alt="SHAP feature contributions"
              style={{ width: '100%', borderRadius: 8, marginBottom: 16 }}
            />
          ) : null}

          <ShapChart features={shap.top_features} title="Top Feature Contributions (SHAP TreeExplainer)" />

          <div style={{ marginTop: 24 }}>
            <FeatureGlossary defaultOpen={false} />
          </div>
        </div>

        {/* ── Row 4: Clinical Explanation ── */}
        <div className="glass-card p-32" style={{ marginBottom: 20 }}>
          <div className="flex justify-between items-center" style={{ marginBottom: 20 }}>
            <div>
              <p style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: '1rem' }}>Clinical Analysis</p>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 4 }}>
                Symptom-by-symptom breakdown of why the model reached this decision
              </p>
            </div>
            <span className="badge badge-purple">XAI Explanation</span>
          </div>
          <ClinicalExplanation summary={clinicalSummary} />
        </div>

        {/* ── Row 5: Full audio feature snapshot ── */}
        {Object.keys(audio_features_snapshot ?? {}).length > 0 && (
          <div className="glass-card p-24">
            <div className="flex justify-between items-center" style={{ marginBottom: 16 }}>
              <div>
                <p style={{ fontWeight: 700, color: 'var(--text-primary)' }}>Full Audio Feature Snapshot</p>
                <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: 2 }}>
                  All key acoustic values extracted from this recording
                </p>
              </div>
              <span className="badge badge-purple">{Object.keys(audio_features_snapshot).length} features</span>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {Object.entries(audio_features_snapshot).map(([k, v]) => {
                const info = FEATURE_TOOLTIP_MAP[k]
                return (
                  <div key={k} style={{
                    padding: '10px 14px', background: 'var(--bg-card)',
                    border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                    minWidth: 140, flex: '1 1 140px',
                  }}>
                    <p style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: 1, fontWeight: 600 }}>
                      {info?.clinical_name ?? k.replace(/_/g, ' ')}
                    </p>
                    {info?.symptom && (
                      <p style={{ fontSize: '0.6rem', color: 'var(--purple-light)', marginBottom: 2 }}>
                        {info.symptom}
                      </p>
                    )}
                    <p style={{ fontSize: '0.92rem', fontWeight: 700, color: 'var(--teal-light)', fontFamily: 'monospace' }}>
                      {typeof v === 'number' ? v.toFixed(4) : v}
                    </p>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* ── Row 6: Detailed Plain-Language Written Explanation (No Numeric Values) ── */}
        <PlainLanguageExplanation
          label={detection.label}
          confidence={detection.confidence}
          shapFeatures={shap.top_features}
          audioFeatures={audio_features_snapshot}
          severityLevel={severity.severity_level}
          filename={filename}
        />

        {/* ── Disclaimer ── */}
        <div style={{
          marginTop: 24, padding: '12px 16px', borderRadius: 'var(--radius-md)',
          background: 'var(--bg-card)', border: '1px solid var(--border)',
          fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: 1.6,
        }}>
          🔬 <strong style={{ color: 'var(--text-secondary)' }}>Research Output:</strong> This result is generated by a machine learning model trained on research datasets.
          It is an <strong>experimental research tool only</strong> and does not constitute a clinical diagnosis. Consult a qualified neurologist for medical advice.
        </div>

      </div>
    </div>
  )
}
