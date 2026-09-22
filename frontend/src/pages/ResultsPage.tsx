import { useLocation, useNavigate } from 'react-router-dom'
import type { FullAnalysisResponse } from '../api/parkinsonApi'
import ConfidenceRing from '../components/ConfidenceRing'
import SeverityGauge from '../components/SeverityGauge'
import ShapChart from '../components/ShapChart'
import ClinicalExplanation from '../components/ClinicalExplanation'
import FeatureGlossary from '../components/FeatureGlossary'
import { buildClinicalSummary, FEATURE_TOOLTIP_MAP } from '../utils/clinicalExplainer'

export default function ResultsPage() {
  const location = useLocation()
  const navigate = useNavigate()

  // Retrieve from navigation state or localStorage
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
          <p style={{ fontSize: '1.1rem', color: 'var(--text-muted)' }}>No results to display.</p>
          <button className="btn-primary" style={{ marginTop: 20 }} onClick={() => navigate('/')}>
            Go to Analysis
          </button>
        </div>
      </div>
    )
  }

  const { detection, severity, shap, audio_features_snapshot, processing_time_ms, filename } = result
  const isParkinson = detection.label === 'Parkinson'

  // Build clinical summary from SHAP + detection + severity
  const clinicalSummary = buildClinicalSummary(
    detection.label,
    detection.confidence,
    shap.top_features,
    severity.motor_updrs,
    severity.severity_level,
  )

  return (
    <div className="page">
      <div className="container">
        {/* Header */}
        <div className="flex justify-between items-center" style={{ marginBottom: 32 }}>
          <div>
            <p className="section-label">Analysis Results</p>
            <h1 style={{ fontSize: '1.6rem', fontWeight: 700, marginTop: 4 }}>
              {filename}
            </h1>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: 2 }}>
              Processed in {processing_time_ms?.toFixed(0)} ms · {new Date().toLocaleString()}
            </p>
          </div>
          <button className="btn-secondary" onClick={() => navigate('/')}>
            ← New Analysis
          </button>
        </div>

        {/* Result alert banner */}
        <div style={{
          padding: '16px 24px',
          marginBottom: 28,
          borderRadius: 'var(--radius-md)',
          background: isParkinson ? 'var(--negative-dim)' : 'var(--positive-dim)',
          border: `1px solid ${isParkinson ? 'rgba(239,68,68,0.35)' : 'rgba(16,185,129,0.35)'}`,
          display: 'flex', alignItems: 'center', gap: 16,
        }}>
          <span style={{ fontSize: '2rem' }}>{isParkinson ? '⚠️' : '✅'}</span>
          <div>
            <p style={{
              fontWeight: 700, fontSize: '1.05rem',
              color: isParkinson ? '#f87171' : '#34d399',
            }}>
              {isParkinson ? 'Parkinson\'s Disease Indicators Detected' : 'No Parkinson\'s Indicators Detected'}
            </p>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: 2 }}>
              {isParkinson
                ? `Model is ${(detection.confidence * 100).toFixed(1)}% confident in this prediction.`
                : `Voice patterns appear healthy with ${(detection.confidence * 100).toFixed(1)}% confidence.`}
            </p>
          </div>
        </div>

        {/* Row 1: Confidence Ring + Severity */}
        <div className="grid-2" style={{ marginBottom: 20, alignItems: 'start' }}>
          {/* Detection card */}
          <div className="glass-card p-32">
            <p className="section-label" style={{ marginBottom: 24 }}>Detection Result</p>

            <div className="flex items-center gap-32" style={{ flexWrap: 'wrap' }}>
              <ConfidenceRing confidence={detection.confidence} label={detection.label} />

              <div style={{ flex: 1, minWidth: 160, display: 'flex', flexDirection: 'column', gap: 12 }}>
                {Object.entries(detection.probabilities).map(([cls, prob]) => (
                  <div key={cls}>
                    <div className="flex justify-between" style={{ marginBottom: 4 }}>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{cls}</span>
                      <span style={{
                        fontSize: '0.8rem', fontWeight: 700,
                        color: cls === 'Parkinson' ? '#f87171' : '#34d399',
                      }}>
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="progress-bar-track">
                      <div
                        className="progress-bar-fill"
                        style={{
                          width: `${prob * 100}%`,
                          background: cls === 'Parkinson'
                            ? 'linear-gradient(90deg, #ef4444, #f87171)'
                            : 'linear-gradient(90deg, #10b981, #34d399)',
                        }}
                      />
                    </div>
                  </div>
                ))}

                <div style={{
                  marginTop: 8, padding: '8px 12px',
                  background: 'var(--bg-card)', borderRadius: 'var(--radius-sm)',
                }}>
                  <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                    Model: {detection.model_used}
                  </p>
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

        {/* Row 2: SHAP chart + Glossary */}
        <div className="glass-card p-32" style={{ marginBottom: 20 }}>
          <div className="flex justify-between items-center" style={{ marginBottom: 4 }}>
            <p style={{ fontWeight: 700, color: 'var(--text-primary)' }}>XAI Feature Explanation</p>
            <span className="badge badge-purple">SHAP Values</span>
          </div>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 20 }}>
            These features most influenced the model's prediction. <strong style={{ color: '#f87171' }}>Red</strong> = increases Parkinson risk &middot; <strong style={{ color: '#34d399' }}>Teal</strong> = decreases risk. Hover any bar for clinical meaning.
          </p>

          {shap.plot_base64 ? (
            <img
              src={`data:image/png;base64,${shap.plot_base64}`}
              alt="SHAP waterfall chart"
              style={{ width: '100%', borderRadius: 8, marginBottom: 16 }}
            />
          ) : null}

          <ShapChart features={shap.top_features} title="Top Feature Contributions (SHAP TreeExplainer)" />

          {/* Feature Glossary */}
          <div style={{ marginTop: 24 }}>
            <FeatureGlossary defaultOpen={false} />
          </div>
        </div>


        {/* Row 3: Clinical Explanation — Why Parkinson / Why Not */}
        <div className="glass-card p-32" style={{ marginBottom: 20 }}>
          <div className="flex justify-between items-center" style={{ marginBottom: 20 }}>
            <div>
              <p style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: '1rem' }}>
                Clinical Analysis
              </p>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 4 }}>
                Why the model made this decision — symptom-by-symptom breakdown
              </p>
            </div>
            <span className="badge badge-purple">XAI Explanation</span>
          </div>
          <ClinicalExplanation summary={clinicalSummary} />
        </div>

        {/* Row 4: Audio feature snapshot */}
        {Object.keys(audio_features_snapshot).length > 0 && (
          <div className="glass-card p-24">
            <p className="section-label" style={{ marginBottom: 16 }}>Key Audio Feature Values</p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
              {Object.entries(audio_features_snapshot).map(([k, v]) => {
                const info = FEATURE_TOOLTIP_MAP[k]
                return (
                  <div key={k} style={{
                    padding: '8px 14px',
                    background: 'var(--bg-card)',
                    border: '1px solid var(--border)',
                    borderRadius: 'var(--radius-sm)',
                    minWidth: 140,
                  }}>
                    <p style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginBottom: 2 }}>
                      {info?.clinical_name ?? k.replace(/_/g, ' ')}
                    </p>
                    {info?.symptom && (
                      <p style={{ fontSize: '0.62rem', color: 'var(--purple-light)', marginBottom: 2 }}>
                        {info.symptom}
                      </p>
                    )}
                    <p style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--teal-light)', fontFamily: 'monospace' }}>
                      {typeof v === 'number' ? v.toFixed(4) : v}
                    </p>
                  </div>
                )
              })}
            </div>
          </div>
        )}

      </div>
    </div>
  )
}
