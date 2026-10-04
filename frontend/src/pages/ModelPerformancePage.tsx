import { useState, useEffect } from 'react'
import { parkinsonApi } from '../api/parkinsonApi'
import type { ModelPerfEntry } from '../api/parkinsonApi'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts'

const DATASET_COLORS: Record<string, string> = {
  D1: '#a855f7',
  D2: '#14b8a6',
  D3a: '#f59e0b',
}

const METRIC_OPTIONS = [
  { key: 'accuracy', label: 'Accuracy', fmt: (v: number) => `${(v * 100).toFixed(2)}%` },
  { key: 'f1',       label: 'F1 Score', fmt: (v: number) => v.toFixed(4) },
  { key: 'recall',   label: 'Recall',   fmt: (v: number) => v.toFixed(4) },
  { key: 'roc_auc',  label: 'ROC-AUC',  fmt: (v: number) => v.toFixed(4) },
]

export default function ModelPerformancePage() {
  const [models, setModels] = useState<ModelPerfEntry[]>([])
  const [source, setSource] = useState('')
  const [note, setNote] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Filters
  const [datasetFilter, setDatasetFilter] = useState<string>('All')
  const [protocolFilter, setProtocolFilter] = useState<string>('All')
  const [metricKey, setMetricKey] = useState('accuracy')

  useEffect(() => {
    parkinsonApi.modelPerformance()
      .then(r => { setModels(r.data.models); setSource(r.data.source); setNote(r.data.note) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const datasets = ['All', ...Array.from(new Set(models.map(m => m.dataset)))]
  const protocols = ['All', ...Array.from(new Set(models.map(m => m.protocol)))]

  const filtered = models.filter(m =>
    (datasetFilter === 'All' || m.dataset === datasetFilter) &&
    (protocolFilter === 'All' || m.protocol === protocolFilter)
  )

  const metricObj = METRIC_OPTIONS.find(o => o.key === metricKey)!
  const chartData = filtered.map(m => ({
    name: `${m.model}\n(${m.dataset})`,
    shortName: `${m.dataset}-${m.model.split(' ')[0]}`,
    value: (m as any)[metricKey],
    dataset: m.dataset,
    color: DATASET_COLORS[m.dataset] || '#94a3b8',
  }))

  if (loading) return (
    <div className="page flex items-center justify-center">
      <div className="flex flex-col items-center gap-16">
        <div className="spinner" style={{ width: 48, height: 48 }} />
        <p style={{ color: 'var(--text-muted)' }}>Loading model performance data...</p>
      </div>
    </div>
  )

  if (error) return (
    <div className="page"><div className="container text-center">
      <p style={{ color: 'var(--negative)' }}>Failed to load: {error}</p>
    </div></div>
  )

  return (
    <div className="page">
      <div className="container">
        {/* Header */}
        <div style={{ marginBottom: 32 }}>
          <p className="section-label">Research Paper — Table III</p>
          <h1 style={{ fontSize: '1.8rem', fontWeight: 700, marginTop: 4 }}>Model Performance</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: 6, maxWidth: 700 }}>
            Held-out test set results reported in the research paper. These are <strong style={{ color: '#fbbf24' }}>not live prediction outputs</strong> — 
            they reflect training-time evaluation.
          </p>
        </div>

        {/* Source banner */}
        <div style={{
          padding: '12px 16px', marginBottom: 24, borderRadius: 'var(--radius-md)',
          background: 'rgba(251,191,36,0.08)', border: '1px solid rgba(251,191,36,0.3)',
          fontSize: '0.8rem', color: '#fbbf24',
        }}>
          📄 Source: {source}
        </div>

        {/* Note */}
        <div style={{
          padding: '12px 16px', marginBottom: 24, borderRadius: 'var(--radius-md)',
          background: 'var(--purple-dim)', border: '1px solid rgba(124,58,237,0.3)',
          fontSize: '0.8rem', color: 'var(--text-secondary)',
        }}>
          ⚠️ {note}
        </div>

        {/* Filters */}
        <div className="glass-card p-24" style={{ marginBottom: 24 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'center' }}>
            <div>
              <label style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block', marginBottom: 4, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase' }}>Dataset</label>
              <select
                value={datasetFilter}
                onChange={e => setDatasetFilter(e.target.value)}
                style={selectStyle}
              >
                {datasets.map(d => <option key={d} value={d}>{d}</option>)}
              </select>
            </div>
            <div>
              <label style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block', marginBottom: 4, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase' }}>Protocol</label>
              <select
                value={protocolFilter}
                onChange={e => setProtocolFilter(e.target.value)}
                style={selectStyle}
              >
                {protocols.map(p => <option key={p} value={p}>{p}</option>)}
              </select>
            </div>
            <div>
              <label style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block', marginBottom: 4, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase' }}>Chart Metric</label>
              <select
                value={metricKey}
                onChange={e => setMetricKey(e.target.value)}
                style={selectStyle}
              >
                {METRIC_OPTIONS.map(o => <option key={o.key} value={o.key}>{o.label}</option>)}
              </select>
            </div>
            <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {Object.entries(DATASET_COLORS).map(([d, c]) => (
                <span key={d} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                  <span style={{ width: 10, height: 10, borderRadius: 2, background: c, display: 'inline-block' }} />
                  {d}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Bar Chart */}
        <div className="glass-card p-24" style={{ marginBottom: 24 }}>
          <p style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: 20 }}>
            {metricObj.label} Comparison
          </p>
          <div style={{ height: 320 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ left: 0, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis
                  dataKey="shortName"
                  tick={{ fill: '#94a3b8', fontSize: 11 }}
                  angle={-30}
                  textAnchor="end"
                />
                <YAxis
                  tick={{ fill: '#94a3b8', fontSize: 11 }}
                  domain={metricKey === 'accuracy' ? [0.5, 1] : [0, 1]}
                  tickFormatter={v => metricKey === 'accuracy' ? `${(v*100).toFixed(0)}%` : v.toFixed(2)}
                />
                <Tooltip
                  formatter={(v: any) => metricObj.fmt(v)}
                  contentStyle={{ background: 'var(--bg-glass)', border: '1px solid var(--border)', borderRadius: 8, fontSize: '0.8rem' }}
                />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {chartData.map((entry, i) => (
                    <Cell key={i} fill={entry.color} fillOpacity={0.85} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Data Table */}
        <div className="glass-card p-24">
          <p style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: 16 }}>
            Full Results Table ({filtered.length} entries)
          </p>
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Dataset</th>
                  <th>Model</th>
                  <th>Protocol</th>
                  <th style={{ textAlign: 'right' }}>Accuracy</th>
                  <th style={{ textAlign: 'right' }}>F1</th>
                  <th style={{ textAlign: 'right' }}>Recall</th>
                  <th style={{ textAlign: 'right' }}>ROC-AUC</th>
                  <th>Note</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((m, i) => (
                  <tr key={i}>
                    <td>
                      <span style={{
                        display: 'inline-flex', alignItems: 'center', gap: 6,
                        padding: '2px 8px', borderRadius: 4,
                        background: `${DATASET_COLORS[m.dataset]}22`,
                        border: `1px solid ${DATASET_COLORS[m.dataset]}44`,
                        color: DATASET_COLORS[m.dataset], fontSize: '0.75rem', fontWeight: 700,
                      }}>{m.dataset}</span>
                    </td>
                    <td>
                      <span style={{ color: 'var(--text-primary)', fontWeight: m.is_primary ? 700 : 400 }}>
                        {m.model} {m.is_primary && <span style={{ color: 'var(--purple-light)', fontSize: '0.7rem' }}>★ Primary</span>}
                      </span>
                    </td>
                    <td>
                      <span style={{
                        fontSize: '0.72rem', padding: '2px 8px', borderRadius: 4,
                        background: m.protocol === 'Record-level' ? 'rgba(245,158,11,0.15)' : 'rgba(20,184,166,0.15)',
                        color: m.protocol === 'Record-level' ? '#fbbf24' : '#2dd4bf',
                      }}>{m.protocol}</span>
                    </td>
                    <td style={{ textAlign: 'right', fontWeight: 600, color: 'var(--text-primary)' }}>
                      {(m.accuracy * 100).toFixed(2)}%
                    </td>
                    <td style={{ textAlign: 'right' }}>{m.f1.toFixed(4)}</td>
                    <td style={{ textAlign: 'right' }}>{m.recall.toFixed(3)}</td>
                    <td style={{ textAlign: 'right', color: 'var(--teal-light)', fontWeight: 600 }}>
                      {m.roc_auc.toFixed(4)}
                    </td>
                    <td style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                      {m.dataset === 'D1' ? 'No speaker IDs' : 'Speaker-disjoint'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}

const selectStyle: React.CSSProperties = {
  background: 'var(--bg-card)',
  border: '1px solid var(--border)',
  borderRadius: 'var(--radius-sm)',
  color: 'var(--text-primary)',
  padding: '8px 12px',
  fontSize: '0.85rem',
  cursor: 'pointer',
  outline: 'none',
}
