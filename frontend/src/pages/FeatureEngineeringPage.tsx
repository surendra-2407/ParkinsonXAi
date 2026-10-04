import { useState, useEffect } from 'react'
import { parkinsonApi } from '../api/parkinsonApi'
import type { FeatureGroup } from '../api/parkinsonApi'
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts'

const GROUP_COLORS = [
  '#a855f7', '#14b8a6', '#f59e0b', '#ef4444', '#3b82f6',
  '#ec4899', '#10b981', '#8b5cf6', '#06b6d4', '#f97316',
  '#84cc16', '#e879f9', '#94a3b8',
]

export default function FeatureEngineeringPage() {
  const [groups, setGroups] = useState<FeatureGroup[]>([])
  const [meta, setMeta] = useState({ total_raw: 0, total_non_degenerate: 0, total_selected: 0, selection_method: '' })
  const [source, setSource] = useState('')
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')

  useEffect(() => {
    parkinsonApi.features()
      .then(r => {
        setGroups(r.data.feature_groups)
        setMeta({
          total_raw: r.data.total_raw,
          total_non_degenerate: r.data.total_non_degenerate,
          total_selected: r.data.total_selected,
          selection_method: r.data.selection_method,
        })
        setSource(r.data.source)
      })
      .finally(() => setLoading(false))
  }, [])

  const filtered = groups.filter(g =>
    g.group.toLowerCase().includes(search.toLowerCase()) ||
    g.description.toLowerCase().includes(search.toLowerCase())
  )

  const pieData = groups.map((g, i) => ({
    name: g.group,
    value: g.count,
    color: GROUP_COLORS[i % GROUP_COLORS.length],
  }))

  if (loading) return (
    <div className="page flex items-center justify-center">
      <div className="flex flex-col items-center gap-16">
        <div className="spinner" style={{ width: 48, height: 48 }} />
        <p style={{ color: 'var(--text-muted)' }}>Loading feature engineering data...</p>
      </div>
    </div>
  )

  return (
    <div className="page">
      <div className="container">
        {/* Header */}
        <div style={{ marginBottom: 32 }}>
          <p className="section-label">Research Paper — Table II</p>
          <h1 style={{ fontSize: '1.8rem', fontWeight: 700, marginTop: 4 }}>Feature Engineering</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: 6 }}>
            220 raw acoustic descriptors extracted per recording, reduced to 219 non-degenerate features on D1,
            and further selected to top 50 via Mutual Information.
          </p>
        </div>

        <div style={{
          padding: '12px 16px', marginBottom: 24, borderRadius: 'var(--radius-md)',
          background: 'rgba(251,191,36,0.08)', border: '1px solid rgba(251,191,36,0.3)',
          fontSize: '0.8rem', color: '#fbbf24',
        }}>📄 {source}</div>

        {/* Summary stats */}
        <div className="grid-4" style={{ marginBottom: 24 }}>
          {[
            { value: meta.total_raw, label: 'Raw Features', color: 'var(--purple-light)', icon: '🔢', desc: 'Initial extraction' },
            { value: meta.total_non_degenerate, label: 'Non-Degenerate', color: 'var(--teal-light)', icon: '✅', desc: 'After variance filter on D1' },
            { value: meta.total_selected, label: 'Final Selected', color: '#fbbf24', icon: '⭐', desc: meta.selection_method },
            { value: groups.length, label: 'Feature Groups', color: '#ec4899', icon: '📦', desc: 'Acoustic feature families' },
          ].map(card => (
            <div key={card.label} className="glass-card stat-card">
              <div style={{ fontSize: '1.4rem' }}>{card.icon}</div>
              <p className="stat-value" style={{ color: card.color, fontSize: '2rem' }}>{card.value}</p>
              <p className="stat-label">{card.label}</p>
              <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 2 }}>{card.desc}</p>
            </div>
          ))}
        </div>

        {/* Pipeline flow */}
        <div className="glass-card p-24" style={{ marginBottom: 24 }}>
          <p style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: 16 }}>Feature Selection Pipeline</p>
          <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
            {[
              { step: '1', label: 'Raw Audio WAV', val: '', color: '#475569' },
              { step: '2', label: 'Librosa + Praat', val: '220 features', color: '#7c3aed' },
              { step: '3', label: 'VarianceThreshold', val: '219 features', color: '#0891b2' },
              { step: '4', label: 'SimpleImputer', val: 'fill NaN', color: '#059669' },
              { step: '5', label: 'RobustScaler', val: 'normalize', color: '#d97706' },
              { step: '6', label: 'Mutual Info Select', val: 'top 50', color: '#dc2626' },
              { step: '7', label: 'LightGBM', val: 'predict', color: '#7c3aed' },
            ].map((s, i, arr) => (
              <div key={s.step} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <div style={{
                  padding: '10px 14px', borderRadius: 8,
                  background: `${s.color}22`, border: `1px solid ${s.color}44`,
                  textAlign: 'center', minWidth: 90,
                }}>
                  <p style={{ fontSize: '0.62rem', color: 'var(--text-muted)', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase' }}>Step {s.step}</p>
                  <p style={{ fontSize: '0.8rem', fontWeight: 700, color: s.color, marginTop: 2 }}>{s.label}</p>
                  {s.val && <p style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: 2 }}>{s.val}</p>}
                </div>
                {i < arr.length - 1 && (
                  <span style={{ color: 'var(--text-muted)', fontSize: '1rem' }}>→</span>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Charts row */}
        <div className="grid-2" style={{ marginBottom: 24 }}>
          {/* Pie */}
          <div className="glass-card p-24">
            <p style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: 16 }}>Distribution by Feature Group</p>
            <div style={{ height: 260 }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={pieData} dataKey="value" cx="50%" cy="50%" outerRadius={90} innerRadius={45} strokeWidth={0}>
                    {pieData.map((e, i) => <Cell key={i} fill={e.color} fillOpacity={0.85} />)}
                  </Pie>
                  <Tooltip
                    formatter={(v: any) => [`${v} features`]}
                    contentStyle={{ background: 'var(--bg-glass)', border: '1px solid var(--border)', borderRadius: 8, fontSize: '0.78rem' }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Bar */}
          <div className="glass-card p-24">
            <p style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: 16 }}>Feature Count per Group</p>
            <div style={{ height: 260 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={groups} layout="vertical" margin={{ left: 100, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                  <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                  <YAxis type="category" dataKey="group" tick={{ fill: '#94a3b8', fontSize: 10 }} width={96} />
                  <Tooltip
                    formatter={(v: any) => [`${v} features`]}
                    contentStyle={{ background: 'var(--bg-glass)', border: '1px solid var(--border)', borderRadius: 8, fontSize: '0.78rem' }}
                  />
                  <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                    {groups.map((_, i) => <Cell key={i} fill={GROUP_COLORS[i % GROUP_COLORS.length]} fillOpacity={0.85} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Feature group table */}
        <div className="glass-card p-24">
          <div className="flex justify-between items-center" style={{ marginBottom: 16, flexWrap: 'wrap', gap: 12 }}>
            <p style={{ fontWeight: 700, color: 'var(--text-primary)' }}>Feature Groups ({groups.length})</p>
            <input
              placeholder="Search groups..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              style={{
                background: 'var(--bg-card)', border: '1px solid var(--border)',
                borderRadius: 'var(--radius-sm)', color: 'var(--text-primary)',
                padding: '8px 14px', fontSize: '0.85rem', outline: 'none', width: 220,
              }}
            />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {filtered.map((g) => (
              <div key={g.group} style={{
                display: 'flex', alignItems: 'flex-start', gap: 14, padding: '14px 16px',
                borderRadius: 'var(--radius-md)', background: 'var(--bg-card)',
                border: '1px solid var(--border)',
              }}>
                <div style={{
                  width: 10, height: 10, borderRadius: '50%', flexShrink: 0, marginTop: 6,
                  background: GROUP_COLORS[groups.indexOf(g) % GROUP_COLORS.length],
                }} />
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
                    <span style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: '0.88rem' }}>{g.group}</span>
                    <span style={{
                      fontSize: '0.72rem', fontWeight: 700, padding: '1px 8px', borderRadius: 4,
                      background: `${GROUP_COLORS[groups.indexOf(g) % GROUP_COLORS.length]}22`,
                      color: GROUP_COLORS[groups.indexOf(g) % GROUP_COLORS.length],
                    }}>{g.count} features</span>
                  </div>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>{g.description}</p>
                </div>
              </div>
            ))}
            {/* Total row */}
            <div style={{
              display: 'flex', alignItems: 'center', gap: 14, padding: '14px 16px',
              borderRadius: 'var(--radius-md)',
              background: 'var(--purple-dim)', border: '1px solid rgba(124,58,237,0.4)',
            }}>
              <div style={{ width: 10, height: 10, borderRadius: '50%', flexShrink: 0, background: '#a855f7' }} />
              <div style={{ flex: 1 }}>
                <span style={{ fontWeight: 700, color: 'var(--purple-light)', fontSize: '0.88rem' }}>Total</span>
              </div>
              <span style={{
                fontSize: '0.9rem', fontWeight: 800, padding: '2px 12px', borderRadius: 6,
                background: '#a855f722', color: '#a855f7',
              }}>220 features</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
