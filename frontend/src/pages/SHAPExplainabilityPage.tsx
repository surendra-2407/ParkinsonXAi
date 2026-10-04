import { useState, useEffect } from 'react'
import { parkinsonApi } from '../api/parkinsonApi'
import type { GlobalShapFeature, HistoryItem } from '../api/parkinsonApi'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts'

const GROUP_COLORS: Record<string, string> = {
  MFCC: '#a855f7',
  Energy: '#14b8a6',
  Jitter: '#ef4444',
  Shimmer: '#f59e0b',
  Pitch: '#3b82f6',
  Spectral: '#ec4899',
}

export default function SHAPExplainabilityPage() {
  const [features, setFeatures] = useState<GlobalShapFeature[]>([])
  const [source, setSource] = useState('')
  const [note, setNote] = useState('')
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')

  // Per-prediction SHAP
  const [recentPreds, setRecentPreds] = useState<HistoryItem[]>([])
  const [selectedPred, setSelectedPred] = useState<any | null>(null)
  const [loadingPred, setLoadingPred] = useState(false)

  useEffect(() => {
    Promise.all([
      parkinsonApi.globalShap(),
      parkinsonApi.historyFiltered({ page: 1, page_size: 10 }),
    ]).then(([shapRes, histRes]) => {
      setFeatures(shapRes.data.features)
      setSource(shapRes.data.source)
      setNote(shapRes.data.note)
      setRecentPreds((histRes.data as any).items || [])
    }).finally(() => setLoading(false))
  }, [])

  const filtered = features.filter(f =>
    f.display.toLowerCase().includes(search.toLowerCase()) ||
    f.group.toLowerCase().includes(search.toLowerCase())
  )

  const maxShap = features[0]?.mean_abs_shap ?? 1

  async function loadPredShap(pred: HistoryItem) {
    setLoadingPred(true)
    try {
      const res = await parkinsonApi.predictionDetail(pred.prediction_id)
      setSelectedPred(res.data)
    } catch {
      setSelectedPred(null)
    } finally {
      setLoadingPred(false)
    }
  }

  if (loading) return (
    <div className="page flex items-center justify-center">
      <div className="flex flex-col items-center gap-16">
        <div className="spinner" style={{ width: 48, height: 48 }} />
        <p style={{ color: 'var(--text-muted)' }}>Loading SHAP data...</p>
      </div>
    </div>
  )

  return (
    <div className="page">
      <div className="container">
        {/* Header */}
        <div style={{ marginBottom: 32 }}>
          <p className="section-label">Explainable AI</p>
          <h1 style={{ fontSize: '1.8rem', fontWeight: 700, marginTop: 4 }}>SHAP Explainability</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: 6 }}>
            Feature importance from the LightGBM model using SHAP TreeExplainer. 
            SHAP values show which audio features most influence predictions.
          </p>
        </div>

        {/* Source banner */}
        <div style={{
          padding: '12px 16px', marginBottom: 16, borderRadius: 'var(--radius-md)',
          background: 'rgba(251,191,36,0.08)', border: '1px solid rgba(251,191,36,0.3)',
          fontSize: '0.8rem', color: '#fbbf24',
        }}>
          📄 {source}
        </div>
        <div style={{
          padding: '12px 16px', marginBottom: 24, borderRadius: 'var(--radius-md)',
          background: 'var(--purple-dim)', border: '1px solid rgba(124,58,237,0.3)',
          fontSize: '0.8rem', color: 'var(--text-secondary)',
        }}>
          ℹ️ {note}
        </div>

        {/* Global Chart */}
        <div className="glass-card p-24" style={{ marginBottom: 24 }}>
          <div className="flex justify-between items-center" style={{ marginBottom: 16 }}>
            <div>
              <p style={{ fontWeight: 700, color: 'var(--text-primary)' }}>Global Feature Importance (Mean |SHAP|)</p>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: 4 }}>
                Averaged over D1 held-out test set — not individual prediction values
              </p>
            </div>
            <span className="badge badge-purple">SHAP · LightGBM</span>
          </div>
          <div style={{ height: 280 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={[...features].reverse()}
                layout="vertical"
                margin={{ left: 100, right: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis
                  type="category"
                  dataKey="display"
                  tick={{ fill: '#94a3b8', fontSize: 11 }}
                  width={96}
                />
                <Tooltip
                  formatter={(v: any) => [v.toFixed(3), 'Mean |SHAP|']}
                  contentStyle={{ background: 'var(--bg-glass)', border: '1px solid var(--border)', borderRadius: 8, fontSize: '0.8rem' }}
                />
                <Bar dataKey="mean_abs_shap" radius={[0, 4, 4, 0]}>
                  {[...features].reverse().map((f, i) => (
                    <Cell key={i} fill={GROUP_COLORS[f.group] || '#a855f7'} fillOpacity={0.85} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Feature Table */}
        <div className="glass-card p-24" style={{ marginBottom: 24 }}>
          <div className="flex justify-between items-center" style={{ marginBottom: 16, flexWrap: 'wrap', gap: 12 }}>
            <p style={{ fontWeight: 700, color: 'var(--text-primary)' }}>Feature Importance Table</p>
            <input
              placeholder="Search features..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              style={{
                background: 'var(--bg-card)', border: '1px solid var(--border)',
                borderRadius: 'var(--radius-sm)', color: 'var(--text-primary)',
                padding: '8px 14px', fontSize: '0.85rem', outline: 'none', width: 220,
              }}
            />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {filtered.map((f) => (
              <div key={f.feature} style={{
                padding: '14px 16px', borderRadius: 'var(--radius-md)',
                background: 'var(--bg-card)', border: '1px solid var(--border)',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
                  <span style={{
                    width: 24, height: 24, borderRadius: '50%',
                    background: GROUP_COLORS[f.group] || '#a855f7',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '0.7rem', fontWeight: 700, color: 'white', flexShrink: 0,
                  }}>#{f.rank}</span>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: '0.9rem' }}>{f.display}</span>
                      <span style={{
                        fontSize: '0.68rem', padding: '1px 6px', borderRadius: 4,
                        background: `${GROUP_COLORS[f.group] || '#a855f7'}22`,
                        color: GROUP_COLORS[f.group] || '#a855f7',
                      }}>{f.group}</span>
                    </div>
                    <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: 2 }}>{f.description}</p>
                  </div>
                  <div style={{ textAlign: 'right', flexShrink: 0 }}>
                    <p style={{ fontWeight: 700, color: GROUP_COLORS[f.group] || '#a855f7', fontSize: '1rem' }}>
                      {f.mean_abs_shap.toFixed(3)}
                    </p>
                    <p style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Mean |SHAP|</p>
                  </div>
                </div>
                {/* Mini bar */}
                <div className="progress-bar-track" style={{ height: 4 }}>
                  <div className="progress-bar-fill" style={{
                    width: `${(f.mean_abs_shap / maxShap) * 100}%`,
                    background: `linear-gradient(90deg, ${GROUP_COLORS[f.group] || '#a855f7'}, ${GROUP_COLORS[f.group] || '#a855f7'}88)`,
                  }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Per-prediction SHAP */}
        <div className="glass-card p-24">
          <div style={{ marginBottom: 16 }}>
            <p style={{ fontWeight: 700, color: 'var(--text-primary)' }}>Individual Prediction SHAP</p>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: 4 }}>
              Select a recent prediction to view its individual SHAP explanation computed live by the inference pipeline.
            </p>
          </div>

          {recentPreds.length === 0 ? (
            <div style={{ textAlign: 'center', padding: 40, color: 'var(--text-muted)' }}>
              🎙️ No predictions yet. Upload a voice file on the Analyze page to generate individual SHAP explanations.
            </div>
          ) : (
            <div className="grid-2" style={{ gap: 12 }}>
              <div>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 10, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                  Recent Predictions
                </p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {recentPreds.map(p => (
                    <button
                      key={p.prediction_id}
                      onClick={() => loadPredShap(p)}
                      style={{
                        background: selectedPred?._id === p.prediction_id ? 'var(--purple-dim)' : 'var(--bg-card)',
                        border: `1px solid ${selectedPred?._id === p.prediction_id ? 'rgba(124,58,237,0.5)' : 'var(--border)'}`,
                        borderRadius: 'var(--radius-sm)',
                        padding: '10px 14px',
                        textAlign: 'left',
                        cursor: 'pointer',
                        transition: 'all 0.2s',
                        width: '100%',
                      }}
                    >
                      <p style={{ fontSize: '0.82rem', color: 'var(--text-primary)', fontWeight: 600, marginBottom: 2 }}>
                        🎵 {p.filename}
                      </p>
                      <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                        {p.detection_label} · {new Date(p.timestamp).toLocaleDateString()}
                      </p>
                    </button>
                  ))}
                </div>
              </div>

              <div>
                {loadingPred ? (
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 200 }}>
                    <div className="spinner" />
                  </div>
                ) : selectedPred ? (
                  <div>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 10, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                      SHAP for: {selectedPred.filename}
                    </p>
                    {selectedPred.shap?.plot_base64 ? (
                      <img
                        src={`data:image/png;base64,${selectedPred.shap.plot_base64}`}
                        alt="SHAP waterfall"
                        style={{ width: '100%', borderRadius: 8 }}
                      />
                    ) : (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                        {(selectedPred.shap?.top_features || []).map((f: any, i: number) => (
                          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', width: 120, flexShrink: 0 }}>
                              {f.feature}
                            </span>
                            <div className="progress-bar-track" style={{ flex: 1, height: 8 }}>
                              <div style={{
                                height: '100%', borderRadius: 999,
                                width: `${Math.min(Math.abs(f.value) * 20, 100)}%`,
                                background: f.value > 0 ? '#ef4444' : '#14b8a6',
                                marginLeft: f.value < 0 ? 'auto' : 0,
                              }} />
                            </div>
                            <span style={{ fontSize: '0.75rem', color: f.value > 0 ? '#f87171' : '#2dd4bf', width: 50, textAlign: 'right' }}>
                              {f.value > 0 ? '+' : ''}{f.value.toFixed(3)}
                            </span>
                          </div>
                        ))}
                        <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 8 }}>
                          🔴 Red = increases Parkinson probability · 🔵 Teal = decreases probability
                        </p>
                      </div>
                    )}
                  </div>
                ) : (
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 200, color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                    ← Select a prediction to view its SHAP explanation
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
