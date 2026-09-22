import { useState, useEffect } from 'react'
import { parkinsonApi } from '../api/parkinsonApi'
import type { DashboardResponse, StatsResponse } from '../api/parkinsonApi'
import ModelCard from '../components/ModelCard'
import ShapChart from '../components/ShapChart'
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts'

export default function DashboardPage() {
  const [dashboard, setDashboard] = useState<DashboardResponse | null>(null)
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([parkinsonApi.dashboard(), parkinsonApi.stats()])
      .then(([d, s]) => { setDashboard(d.data); setStats(s.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="page flex items-center justify-center">
      <div className="flex flex-col items-center gap-16">
        <div className="spinner" style={{ width: 48, height: 48 }} />
        <p style={{ color: 'var(--text-muted)' }}>Loading dashboard...</p>
      </div>
    </div>
  )

  if (error || !dashboard) return (
    <div className="page">
      <div className="container text-center">
        <p style={{ color: 'var(--negative)' }}>Failed to load dashboard. Is the backend running?</p>
      </div>
    </div>
  )

  const { live_stats, model_cards, top_shap_features } = dashboard

  const pieData = [
    { name: 'Parkinson', value: live_stats.parkinson_count, color: '#ef4444' },
    { name: 'Healthy', value: live_stats.healthy_count, color: '#10b981' },
  ].filter(d => d.value > 0)

  return (
    <div className="page">
      <div className="container">
        <div style={{ marginBottom: 32 }}>
          <p className="section-label">Analytics Dashboard</p>
          <h1 style={{ fontSize: '1.8rem', fontWeight: 700, marginTop: 4 }}>Model Performance & Statistics</h1>
        </div>

        {/* Live stats row */}
        <div className="grid-4" style={{ marginBottom: 24 }}>
          {[
            {
              value: stats?.total_predictions ?? live_stats.total_predictions,
              label: 'Total Analyzed',
              color: 'var(--purple-light)',
              icon: '🎙️',
            },
            {
              value: stats?.parkinson_count ?? live_stats.parkinson_count,
              label: 'Parkinson Detected',
              color: '#f87171',
              icon: '⚠️',
            },
            {
              value: stats?.healthy_count ?? live_stats.healthy_count,
              label: 'Healthy Results',
              color: '#34d399',
              icon: '✅',
            },
            {
              value: `${((stats?.avg_confidence ?? live_stats.avg_confidence) * 100).toFixed(1)}%`,
              label: 'Avg Confidence',
              color: 'var(--teal-light)',
              icon: '📊',
            },
          ].map(({ value, label, color, icon }) => (
            <div key={label} className="glass-card stat-card">
              <div style={{ fontSize: '1.4rem' }}>{icon}</div>
              <p className="stat-value" style={{ color }}>{value}</p>
              <p className="stat-label">{label}</p>
            </div>
          ))}
        </div>

        {/* Row 2: Distribution + Model Cards */}
        <div className="grid-2" style={{ marginBottom: 24, alignItems: 'start' }}>
          {/* Distribution donut */}
          <div className="glass-card p-24">
            <p className="section-label" style={{ marginBottom: 16 }}>Prediction Distribution</p>
            {pieData.length === 0 ? (
              <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No predictions yet</p>
              </div>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
                <div style={{ width: 180, height: 180, flexShrink: 0 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={pieData}
                        dataKey="value"
                        cx="50%" cy="50%"
                        innerRadius={55} outerRadius={80}
                        strokeWidth={0}
                      >
                        {pieData.map((entry, i) => (
                          <Cell key={i} fill={entry.color} fillOpacity={0.85} />
                        ))}
                      </Pie>
                      <Tooltip
                        contentStyle={{
                          background: 'var(--bg-glass)', border: '1px solid var(--border)',
                          borderRadius: 8, fontSize: '0.8rem',
                        }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  {pieData.map(d => (
                    <div key={d.name} className="flex items-center gap-12">
                      <div style={{ width: 12, height: 12, borderRadius: 3, background: d.color, flexShrink: 0 }} />
                      <div>
                        <p style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-primary)' }}>{d.name}</p>
                        <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{d.value} predictions</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Quick stats */}
          <div className="glass-card p-24">
            <p className="section-label" style={{ marginBottom: 16 }}>System Performance</p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {[
                { label: 'Avg Processing Time', value: `${stats?.avg_processing_time_ms?.toFixed(0) ?? '—'} ms` },
                { label: 'Parkinson Detection Rate', value: `${stats?.parkinson_pct?.toFixed(1) ?? '—'}%` },
                { label: 'Primary Model', value: 'LightGBM (Audio)' },
                { label: 'SHAP Method', value: 'TreeExplainer' },
                { label: 'Feature Count', value: '50 top features' },
                { label: 'Training AUC', value: '0.9995' },
              ].map(({ label, value }) => (
                <div key={label} className="flex justify-between items-center"
                  style={{ borderBottom: '1px solid var(--border)', paddingBottom: 10 }}>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{label}</span>
                  <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>{value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Model cards */}
        <div style={{ marginBottom: 24 }}>
          <p className="section-label" style={{ marginBottom: 16 }}>Model Performance Cards</p>
          <div className="grid-4">
            {model_cards.map(card => <ModelCard key={card.model_name} card={card} />)}
          </div>
        </div>

        {/* SHAP feature importance */}
        {top_shap_features.length > 0 && (
          <div className="glass-card p-32">
            <div className="flex justify-between items-center" style={{ marginBottom: 20 }}>
              <div>
                <p style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: '1rem' }}>
                  Global Feature Importance
                </p>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 4 }}>
                  Top SHAP features from the detection model (aggregated over training set)
                </p>
              </div>
              <span className="badge badge-purple">SHAP · LightGBM</span>
            </div>
            <ShapChart features={top_shap_features} title="Top Detection Features" />
          </div>
        )}
      </div>
    </div>
  )
}
