import { useState, useEffect } from 'react'
import { parkinsonApi } from '../api/parkinsonApi'
import type { SeverityEvalEntry } from '../api/parkinsonApi'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts'

type MetricKey = 'mae' | 'rmse' | 'r2'

export default function UPDRSSeverityPage() {
  const [motor, setMotor] = useState<SeverityEvalEntry[]>([])
  const [total, setTotal] = useState<SeverityEvalEntry[]>([])
  const [source, setSource] = useState('')
  const [note, setNote] = useState('')
  const [loading, setLoading] = useState(true)
  const [target, setTarget] = useState<'motor' | 'total'>('motor')
  const [metric, setMetric] = useState<MetricKey>('mae')

  useEffect(() => {
    parkinsonApi.severityEvaluations()
      .then(r => {
        setMotor(r.data.motor)
        setTotal(r.data.total)
        setSource(r.data.source)
        setNote(r.data.note)
      })
      .finally(() => setLoading(false))
  }, [])

  const activeData = target === 'motor' ? motor : total
  const metricLabel: Record<MetricKey, string> = {
    mae: 'MAE (↓ better)',
    rmse: 'RMSE (↓ better)',
    r2: 'R² Score (↑ better, negative = poor generalization)',
  }

  const chartData = activeData.map(m => ({
    model: m.model,
    value: m[metric],
    fill: metric === 'r2' ? (m.r2 < 0 ? '#ef4444' : '#10b981') : '#a855f7',
  }))

  if (loading) return (
    <div className="page flex items-center justify-center">
      <div className="flex flex-col items-center gap-16">
        <div className="spinner" style={{ width: 48, height: 48 }} />
        <p style={{ color: 'var(--text-muted)' }}>Loading severity evaluation data...</p>
      </div>
    </div>
  )

  return (
    <div className="page">
      <div className="container">
        {/* Header */}
        <div style={{ marginBottom: 32 }}>
          <p className="section-label">Research Paper — Table IV</p>
          <h1 style={{ fontSize: '1.8rem', fontWeight: 700, marginTop: 4 }}>UPDRS Severity Evaluation</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: 6, maxWidth: 700 }}>
            Regression model performance on Parkinson's Telemonitoring dataset (D3b).
            These results are from the research paper, not live predictions.
          </p>
        </div>

        {/* Critical warning about negative R² */}
        <div style={{
          padding: '16px 20px', marginBottom: 24, borderRadius: 'var(--radius-md)',
          background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.3)',
          display: 'flex', alignItems: 'flex-start', gap: 12,
        }}>
          <span style={{ fontSize: '1.3rem', flexShrink: 0 }}>🔬</span>
          <div>
            <p style={{ fontWeight: 700, color: '#f87171', fontSize: '0.9rem', marginBottom: 4 }}>
              Important: Negative R² Scores
            </p>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
              {note}
            </p>
          </div>
        </div>

        {/* Source */}
        <div style={{
          padding: '12px 16px', marginBottom: 24, borderRadius: 'var(--radius-md)',
          background: 'rgba(251,191,36,0.08)', border: '1px solid rgba(251,191,36,0.3)',
          fontSize: '0.8rem', color: '#fbbf24',
        }}>
          📄 {source}
        </div>

        {/* Metric definitions */}
        <div className="grid-3" style={{ marginBottom: 24 }}>
          {[
            { label: 'MAE', name: 'Mean Absolute Error', color: '#a855f7', desc: 'Average absolute difference between predicted and actual UPDRS. Lower is better.' },
            { label: 'RMSE', name: 'Root Mean Squared Error', color: '#14b8a6', desc: 'Penalizes large errors more. Lower is better.' },
            { label: 'R²', name: 'R² Coefficient of Determination', color: '#ef4444', desc: 'Proportion of variance explained. Negative = worse than a constant baseline.' },
          ].map(m => (
            <div key={m.label} className="glass-card p-24">
              <p style={{ fontWeight: 700, color: m.color, fontSize: '1.2rem', marginBottom: 4 }}>{m.label}</p>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-primary)', fontWeight: 600, marginBottom: 6 }}>{m.name}</p>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>{m.desc}</p>
            </div>
          ))}
        </div>

        {/* Controls */}
        <div className="glass-card p-24" style={{ marginBottom: 24 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'center' }}>
            <div>
              <label style={labelStyle}>UPDRS Target</label>
              <div style={{ display: 'flex', gap: 8 }}>
                {(['motor', 'total'] as const).map(t => (
                  <button
                    key={t}
                    onClick={() => setTarget(t)}
                    className={target === t ? 'btn-primary' : 'btn-secondary'}
                    style={{ padding: '8px 20px', fontSize: '0.82rem', textTransform: 'capitalize' }}
                  >
                    {t} UPDRS
                  </button>
                ))}
              </div>
            </div>
            <div>
              <label style={labelStyle}>Metric</label>
              <div style={{ display: 'flex', gap: 8 }}>
                {(['mae', 'rmse', 'r2'] as MetricKey[]).map(m => (
                  <button
                    key={m}
                    onClick={() => setMetric(m)}
                    className={metric === m ? 'btn-primary' : 'btn-secondary'}
                    style={{ padding: '8px 16px', fontSize: '0.82rem', textTransform: 'uppercase' }}
                  >
                    {m.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Chart */}
        <div className="glass-card p-24" style={{ marginBottom: 24 }}>
          <p style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: 4 }}>
            {target === 'motor' ? 'Motor' : 'Total'} UPDRS — {metricLabel[metric]}
          </p>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 20 }}>
            {metric === 'r2' ? 'All values negative — models do not generalize to unseen subjects.' : ''}
          </p>
          <div style={{ height: 280 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="model" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
                {metric === 'r2' && <ReferenceLine y={0} stroke="#ef4444" strokeDasharray="4 4" />}
                <Tooltip
                  formatter={(v: any) => [v.toFixed(2), metric.toUpperCase()]}
                  contentStyle={{ background: 'var(--bg-glass)', border: '1px solid var(--border)', borderRadius: 8, fontSize: '0.8rem' }}
                />
                <Bar dataKey="value" fill="#a855f7" radius={[4, 4, 0, 0]}
                  style={{ fill: metric === 'r2' ? '#ef4444' : '#a855f7' }}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Tables side by side */}
        <div className="grid-2">
          {[
            { label: 'Motor UPDRS', data: motor },
            { label: 'Total UPDRS', data: total },
          ].map(({ label, data }) => (
            <div key={label} className="glass-card p-24">
              <p style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: 16 }}>{label}</p>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th style={{ textAlign: 'right' }}>MAE ↓</th>
                    <th style={{ textAlign: 'right' }}>RMSE ↓</th>
                    <th style={{ textAlign: 'right' }}>R²</th>
                  </tr>
                </thead>
                <tbody>
                  {data.map((m, i) => (
                    <tr key={i}>
                      <td style={{ color: 'var(--text-primary)', fontWeight: i === 0 ? 700 : 400 }}>
                        {m.model} {i === 0 && <span style={{ color: 'var(--teal-light)', fontSize: '0.7rem' }}>★ Best</span>}
                      </td>
                      <td style={{ textAlign: 'right', color: '#a855f7', fontWeight: 600 }}>{m.mae.toFixed(2)}</td>
                      <td style={{ textAlign: 'right' }}>{m.rmse.toFixed(2)}</td>
                      <td style={{ textAlign: 'right', color: '#f87171', fontWeight: 600 }}>{m.r2.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </div>

        {/* Audio proxy note */}
        <div style={{
          padding: '16px 20px', marginTop: 24, borderRadius: 'var(--radius-md)',
          background: 'var(--teal-dim)', border: '1px solid rgba(20,184,166,0.3)',
          fontSize: '0.8rem', color: 'var(--text-secondary)',
        }}>
          <strong style={{ color: 'var(--teal-light)' }}>Audio Proxy Score:</strong> When UPDRS regression inputs are unavailable,
          the inference service computes a deterministic acoustic proxy score (0–100) from jitter, shimmer, HNR, and voiced fraction.
          This proxy is labeled separately and must <strong>never</strong> be interpreted as a clinical UPDRS score.
        </div>
      </div>
    </div>
  )
}

const labelStyle: React.CSSProperties = {
  fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block',
  marginBottom: 8, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase',
}
