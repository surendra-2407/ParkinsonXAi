import { useState, useEffect } from 'react'
import { parkinsonApi } from '../api/parkinsonApi'
import type { DatasetInfo } from '../api/parkinsonApi'
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts'

const DATASET_COLORS: Record<string, string> = {
  D1: '#a855f7',
  D2: '#14b8a6',
  D3a: '#f59e0b',
  D3b: '#ef4444',
}

export default function DatasetInfoPage() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([])
  const [source, setSource] = useState('')
  const [loading, setLoading] = useState(true)
  const [selected, setSelected] = useState<string>('D1')

  useEffect(() => {
    parkinsonApi.datasets()
      .then(r => { setDatasets(r.data.datasets); setSource(r.data.source) })
      .finally(() => setLoading(false))
  }, [])

  const active = datasets.find(d => d.id === selected)

  // Recording distribution for pie
  const pieData = datasets
    .filter(d => d.target_type === 'detection')
    .map(d => ({
      name: d.id,
      value: d.recordings,
      color: DATASET_COLORS[d.id],
    }))

  if (loading) return (
    <div className="page flex items-center justify-center">
      <div className="flex flex-col items-center gap-16">
        <div className="spinner" style={{ width: 48, height: 48 }} />
        <p style={{ color: 'var(--text-muted)' }}>Loading dataset information...</p>
      </div>
    </div>
  )

  return (
    <div className="page">
      <div className="container">
        {/* Header */}
        <div style={{ marginBottom: 32 }}>
          <p className="section-label">Research Paper — Table I</p>
          <h1 style={{ fontSize: '1.8rem', fontWeight: 700, marginTop: 4 }}>Dataset Information</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: 6 }}>
            Four datasets used across detection and severity prediction experiments.
            No patient identities or confidential source data are exposed.
          </p>
        </div>

        <div style={{
          padding: '12px 16px', marginBottom: 24, borderRadius: 'var(--radius-md)',
          background: 'rgba(251,191,36,0.08)', border: '1px solid rgba(251,191,36,0.3)',
          fontSize: '0.8rem', color: '#fbbf24',
        }}>
          📄 {source}
        </div>

        {/* Summary Cards */}
        <div className="grid-4" style={{ marginBottom: 24 }}>
          {datasets.map(d => (
            <button
              key={d.id}
              onClick={() => setSelected(d.id)}
              style={{
                background: selected === d.id ? `${DATASET_COLORS[d.id]}18` : 'var(--bg-card)',
                border: `1px solid ${selected === d.id ? DATASET_COLORS[d.id] + '66' : 'var(--border)'}`,
                borderRadius: 'var(--radius-lg)',
                padding: 20,
                textAlign: 'left',
                cursor: 'pointer',
                transition: 'all 0.2s',
              }}
            >
              <div style={{
                display: 'inline-block', padding: '2px 10px', borderRadius: 6,
                background: `${DATASET_COLORS[d.id]}22`,
                color: DATASET_COLORS[d.id], fontSize: '0.75rem', fontWeight: 700,
                marginBottom: 10,
              }}>{d.id}</div>
              <p style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: '0.88rem', marginBottom: 6 }}>{d.name}</p>
              <p style={{ fontSize: '1.4rem', fontWeight: 800, color: DATASET_COLORS[d.id], lineHeight: 1 }}>
                {d.recordings.toLocaleString()}
              </p>
              <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 2 }}>recordings</p>
              <div style={{
                marginTop: 10, padding: '3px 8px', borderRadius: 4,
                background: d.target_type === 'detection' ? 'var(--positive-dim)' : 'rgba(59,130,246,0.15)',
                display: 'inline-block',
              }}>
                <span style={{
                  fontSize: '0.68rem', fontWeight: 600,
                  color: d.target_type === 'detection' ? '#34d399' : '#60a5fa',
                }}>
                  {d.target_type === 'detection' ? '🔍 Detection' : '📊 Regression'}
                </span>
              </div>
            </button>
          ))}
        </div>

        {/* Detail + Pie row */}
        <div className="grid-2" style={{ marginBottom: 24, alignItems: 'start' }}>
          {/* Detail card */}
          {active && (
            <div className="glass-card p-24">
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
                <div style={{
                  width: 40, height: 40, borderRadius: 10,
                  background: `${DATASET_COLORS[active.id]}22`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '1rem', fontWeight: 700, color: DATASET_COLORS[active.id],
                }}>{active.id}</div>
                <div>
                  <p style={{ fontWeight: 700, color: 'var(--text-primary)' }}>{active.name}</p>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{active.target}</p>
                </div>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {[
                  { label: 'Total Recordings', value: active.recordings.toLocaleString() },
                  { label: 'Speakers', value: active.speakers ? active.speakers.toLocaleString() : 'Unknown (no IDs)' },
                  { label: 'Healthy (HC)', value: active.healthy_count != null ? active.healthy_count.toLocaleString() : 'N/A' },
                  { label: 'Parkinson (PD)', value: active.pd_count != null ? active.pd_count.toLocaleString() : 'N/A' },
                  { label: 'Evaluation Protocol', value: active.protocol },
                  { label: 'Model Used', value: active.model_used },
                ].map(({ label, value }) => (
                  <div key={label} style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border)', paddingBottom: 10 }}>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{label}</span>
                    <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-primary)', textAlign: 'right', maxWidth: 220 }}>{value}</span>
                  </div>
                ))}
                <div>
                  <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: 4 }}>Purpose</p>
                  <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>{active.purpose}</p>
                </div>
                <div style={{
                  padding: '10px 14px', borderRadius: 'var(--radius-sm)',
                  background: 'rgba(251,191,36,0.08)', border: '1px solid rgba(251,191,36,0.2)',
                  fontSize: '0.78rem', color: 'var(--text-muted)',
                }}>
                  ⚠️ {active.notes}
                </div>
              </div>
            </div>
          )}

          {/* Pie chart */}
          <div className="glass-card p-24">
            <p style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: 4 }}>
              Detection Dataset Distribution
            </p>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 16 }}>
              Recording counts across D1, D2, D3a (binary classification datasets)
            </p>
            <div style={{ height: 240 }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={pieData} dataKey="value" cx="50%" cy="50%" outerRadius={90} innerRadius={50} strokeWidth={0}>
                    {pieData.map((entry, i) => (
                      <Cell key={i} fill={entry.color} fillOpacity={0.85} />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(v: any) => [`${v} recordings`]}
                    contentStyle={{ background: 'var(--bg-glass)', border: '1px solid var(--border)', borderRadius: 8, fontSize: '0.8rem' }}
                  />
                  <Legend formatter={(v) => <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>{v}</span>} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Full comparison table */}
        <div className="glass-card p-24" style={{ marginBottom: 24 }}>
          <p style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: 16 }}>Full Dataset Comparison</p>
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Dataset</th>
                  <th style={{ textAlign: 'right' }}>Recordings</th>
                  <th style={{ textAlign: 'right' }}>Speakers</th>
                  <th>Target</th>
                  <th>Purpose</th>
                  <th>Protocol</th>
                </tr>
              </thead>
              <tbody>
                {datasets.map(d => (
                  <tr key={d.id}>
                    <td>
                      <span style={{
                        display: 'inline-flex', alignItems: 'center', gap: 6,
                        padding: '2px 8px', borderRadius: 4,
                        background: `${DATASET_COLORS[d.id]}22`,
                        color: DATASET_COLORS[d.id], fontSize: '0.75rem', fontWeight: 700,
                      }}>{d.id} — {d.name}</span>
                    </td>
                    <td style={{ textAlign: 'right', fontWeight: 600, color: 'var(--text-primary)' }}>{d.recordings.toLocaleString()}</td>
                    <td style={{ textAlign: 'right' }}>{d.speakers ?? '—'}</td>
                    <td style={{ fontSize: '0.8rem' }}>{d.target}</td>
                    <td style={{ fontSize: '0.8rem', maxWidth: 200 }}>{d.purpose}</td>
                    <td>
                      <span style={{
                        fontSize: '0.7rem', padding: '2px 8px', borderRadius: 4,
                        background: d.protocol.includes('disjoint') ? 'rgba(20,184,166,0.15)' : 'rgba(245,158,11,0.15)',
                        color: d.protocol.includes('disjoint') ? '#2dd4bf' : '#fbbf24',
                      }}>{d.protocol.split('(')[0].trim()}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Evaluation protocol explainer */}
        <div className="grid-2">
          {[
            {
              title: 'Record-Level Evaluation',
              color: '#f59e0b',
              icon: '📋',
              points: [
                'Each audio recording is treated as an independent sample.',
                'Train/test split is done at the recording level.',
                'Same speaker may appear in both train and test.',
                'Cannot confirm patient-independent generalization.',
                'D1 uses this protocol (no speaker IDs available).',
              ]
            },
            {
              title: 'Speaker-Disjoint Evaluation',
              color: '#14b8a6',
              icon: '👥',
              points: [
                'Each speaker appears in only one split (train OR test).',
                'More realistic — simulates deployment on new patients.',
                'Typically yields lower accuracy than record-level.',
                'Used for D2, D3a, D3b where speaker IDs are known.',
                'The standard for clinical-grade model assessment.',
              ]
            }
          ].map(card => (
            <div key={card.title} className="glass-card p-24">
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
                <span style={{ fontSize: '1.3rem' }}>{card.icon}</span>
                <p style={{ fontWeight: 700, color: card.color, fontSize: '0.92rem' }}>{card.title}</p>
              </div>
              <ul style={{ display: 'flex', flexDirection: 'column', gap: 8, listStyle: 'none' }}>
                {card.points.map((p, i) => (
                  <li key={i} style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'flex', gap: 8 }}>
                    <span style={{ color: card.color, flexShrink: 0 }}>›</span>
                    {p}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
