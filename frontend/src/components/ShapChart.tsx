import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts'
import { useState } from 'react'
import type { ShapFeature } from '../api/parkinsonApi'
import { FEATURE_TOOLTIP_MAP } from '../utils/clinicalExplainer'

interface ShapChartProps {
  features: ShapFeature[]
  title?: string
}

const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null
  const d = payload[0]
  const info = FEATURE_TOOLTIP_MAP[d.payload.rawFeature]
  return (
    <div style={{
      background: 'var(--bg-glass)', border: '1px solid var(--border)',
      borderRadius: 10, padding: '12px 16px', backdropFilter: 'blur(20px)',
      maxWidth: 280,
    }}>
      <p style={{ color: 'var(--text-primary)', fontSize: '0.8rem', fontWeight: 700, marginBottom: 4 }}>
        {info?.clinical_name ?? d.payload.feature}
      </p>
      {info && (
        <>
          <p style={{ fontSize: '0.72rem', color: 'var(--purple-light)', marginBottom: 6 }}>
            🩺 {info.symptom}
          </p>
          <p style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: 8 }}>
            {info.description}
          </p>
        </>
      )}
      <p style={{ color: d.value > 0 ? '#f87171' : '#34d399', fontSize: '0.85rem', fontWeight: 700 }}>
        {d.value > 0 ? '+' : ''}{d.value.toFixed(5)}
      </p>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.72rem', marginTop: 2 }}>
        {d.value > 0 ? '↑ Increases Parkinson risk' : '↓ Decreases Parkinson risk'}
      </p>
    </div>
  )
}

export default function ShapChart({ features, title = 'SHAP Feature Contributions' }: ShapChartProps) {
  const [showShapInfo, setShowShapInfo] = useState(false)
  const sorted = [...features].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 10)
  const data = sorted.map(f => ({
    feature: f.feature.replace(/_/g, ' '),
    value: parseFloat(f.value.toFixed(5)),
    rawFeature: f.feature,
  }))

  return (
    <div style={{ width: '100%' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <p style={{
          fontSize: '0.7rem', fontWeight: 700, letterSpacing: '0.12em',
          textTransform: 'uppercase', color: 'var(--text-muted)',
        }}>
          {title}
        </p>
        <button
          id="shap-info-toggle"
          onClick={() => setShowShapInfo(s => !s)}
          style={{
            background: 'rgba(124,58,237,0.12)', border: '1px solid rgba(124,58,237,0.3)',
            borderRadius: 20, padding: '3px 12px', cursor: 'pointer',
            fontSize: '0.7rem', color: 'var(--purple-light)', fontWeight: 600,
            transition: 'all 0.2s',
          }}
        >
          {showShapInfo ? '✕ Close' : '❓ What is SHAP?'}
        </button>
      </div>

      {showShapInfo && (
        <div style={{
          padding: '14px 16px', marginBottom: 16,
          background: 'rgba(124,58,237,0.06)',
          border: '1px solid rgba(124,58,237,0.2)',
          borderRadius: 'var(--radius-sm)',
        }}>
          <p style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--purple-light)', marginBottom: 8 }}>
            🔍 What are SHAP Values?
          </p>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.65, marginBottom: 8 }}>
            <strong style={{ color: 'var(--text-primary)' }}>SHAP</strong> stands for <em>SHapley Additive exPlanations</em>.
            It answers: <strong>"Why did the model predict Parkinson's?"</strong> by assigning each voice feature a score that shows how much it pushed the prediction.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <div style={{ padding: '8px 10px', background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 6 }}>
              <p style={{ fontSize: '0.72rem', fontWeight: 700, color: '#f87171', marginBottom: 4 }}>🔴 Positive SHAP (Red bar)</p>
              <p style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                This feature's value in your voice <strong>pushed</strong> the model toward predicting Parkinson's disease. Higher bar = stronger push.
              </p>
            </div>
            <div style={{ padding: '8px 10px', background: 'rgba(20,184,166,0.08)', border: '1px solid rgba(20,184,166,0.2)', borderRadius: 6 }}>
              <p style={{ fontSize: '0.72rem', fontWeight: 700, color: '#34d399', marginBottom: 4 }}>🟢 Negative SHAP (Teal bar)</p>
              <p style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                This feature's value <strong>pushed against</strong> the Parkinson's prediction, acting as a protective indicator.
              </p>
            </div>
          </div>
          <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 10, fontStyle: 'italic' }}>
            💡 Hover over each bar to see what the feature means clinically.
          </p>
        </div>
      )}

      <div style={{ height: Math.max(260, data.length * 36) }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 0, right: 20, top: 0, bottom: 0 }}>
            <XAxis
              type="number"
              tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
              axisLine={{ stroke: 'var(--border)' }}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="feature"
              width={160}
              tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
            <ReferenceLine x={0} stroke="rgba(255,255,255,0.1)" strokeWidth={1} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={22}>
              {data.map((entry, i) => (
                <Cell
                  key={i}
                  fill={entry.value > 0 ? '#ef4444' : '#14b8a6'}
                  fillOpacity={0.85}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="flex gap-16" style={{ marginTop: 12 }}>
        <div className="flex items-center gap-8">
          <div style={{ width: 12, height: 12, borderRadius: 2, background: '#ef4444', opacity: 0.85 }} />
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Increases risk</span>
        </div>
        <div className="flex items-center gap-8">
          <div style={{ width: 12, height: 12, borderRadius: 2, background: '#14b8a6', opacity: 0.85 }} />
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Decreases risk</span>
        </div>
        <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginLeft: 'auto' }}>
          Hover bars for clinical meaning
        </span>
      </div>
    </div>
  )
}
