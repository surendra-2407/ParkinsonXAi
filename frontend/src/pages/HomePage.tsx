import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import AudioInput from '../components/AudioInput'
import { useSession } from '../hooks/useSession'
import { useAnalysis } from '../hooks/useAnalysis'

export default function HomePage() {
  const sessionId = useSession()
  const { state, result, error, progress, analyze } = useAnalysis(sessionId)
  const navigate = useNavigate()
  const [file, setFile] = useState<File | null>(null)

  function handleAnalyze() {
    if (file) analyze(file)
  }

  // Redirect to results when done
  if (state === 'done' && result) {
    localStorage.setItem('xai_last_result', JSON.stringify(result))
    navigate('/results', { state: { result } })
    return null
  }

  return (
    <div className="page">
      <div className="container">
        {/* Hero */}
        <div className="text-center" style={{ maxWidth: 700, margin: '0 auto 60px', paddingTop: 20 }}>
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: 8,
            padding: '6px 16px', borderRadius: 999,
            background: 'var(--purple-dim)', border: '1px solid rgba(124,58,237,0.3)',
            marginBottom: 24,
          }}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--purple-light)', display: 'inline-block' }} />
            <span style={{ fontSize: '0.72rem', fontWeight: 600, letterSpacing: '0.08em', color: 'var(--purple-light)', textTransform: 'uppercase' }}>
              Explainable AI · Clinical Grade
            </span>
          </div>

          <h1 style={{
            fontSize: 'clamp(2.2rem, 5vw, 3.5rem)',
            fontWeight: 800,
            lineHeight: 1.1,
            letterSpacing: '-0.03em',
            marginBottom: 20,
          }}>
            <span className="gradient-text">Voice-Based</span>
            <br />
            Parkinson's Detection
          </h1>

          <p style={{
            fontSize: '1.05rem', color: 'var(--text-secondary)', lineHeight: 1.7, maxWidth: 520, margin: '0 auto 24px',
          }}>
            Upload a voice recording and our AI will analyze it for Parkinson's disease indicators,
            predict disease severity, and explain the key features driving the prediction.
          </p>

          {/* Feature pills */}
          <div className="flex items-center justify-center" style={{ gap: 10, flexWrap: 'wrap' }}>
            {[
              { label: '100% Recall', color: 'var(--teal)' },
              { label: 'SHAP Explained', color: 'var(--pink)' },
            ].map(({ label, color }) => (
              <span key={label} style={{
                padding: '4px 12px', borderRadius: 999, fontSize: '0.75rem', fontWeight: 600,
                background: `${color}15`, border: `1px solid ${color}40`, color,
              }}>
                {label}
              </span>
            ))}
          </div>
        </div>

        {/* Upload Card */}
        <div style={{ maxWidth: 620, margin: '0 auto' }}>
          <div className="glass-card" style={{ padding: 32, marginBottom: 20 }}>
            <AudioInput onFile={setFile} disabled={state === 'loading'} />

            {state === 'loading' && (
              <div style={{ marginTop: 24 }}>
                <div className="flex justify-between items-center" style={{ marginBottom: 8 }}>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    Extracting features · Running inference · Computing SHAP...
                  </span>
                  <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--purple-light)' }}>
                    {Math.round(progress)}%
                  </span>
                </div>
                <div className="progress-bar-track">
                  <div
                    className="progress-bar-fill"
                    style={{
                      width: `${progress}%`,
                      background: 'linear-gradient(90deg, var(--purple), var(--teal))',
                    }}
                  />
                </div>
              </div>
            )}

            {error && (
              <div style={{
                marginTop: 16, padding: '12px 16px',
                background: 'var(--negative-dim)', border: '1px solid rgba(239,68,68,0.3)',
                borderRadius: 'var(--radius-sm)', color: '#f87171', fontSize: '0.85rem',
              }}>
                ⚠️ {error}
              </div>
            )}

            <button
              className="btn-primary w-full"
              style={{ marginTop: 20, justifyContent: 'center', padding: '14px 28px', fontSize: '0.95rem' }}
              onClick={handleAnalyze}
              disabled={!file || state === 'loading'}
            >
              {state === 'loading' ? (
                <><div className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} /> Analyzing...</>
              ) : (
                <> Analyze Voice Recording</>
              )}
            </button>
          </div>

          {/* Info strip */}
          <div className="grid-3" style={{ gap: 12 }}>
            {[
              { icon: '🔒', title: 'Private', desc: 'Files processed locally' },
              { icon: '⚡', title: 'Fast', desc: 'Results in ~5 seconds' },
              { icon: '🧠', title: 'Explainable', desc: 'SHAP feature analysis' },
            ].map(({ icon, title, desc }) => (
              <div key={title} className="glass-card" style={{ padding: '14px 16px', textAlign: 'center' }}>
                <div style={{ fontSize: '1.3rem', marginBottom: 4 }}>{icon}</div>
                <p style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-primary)' }}>{title}</p>
                <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
