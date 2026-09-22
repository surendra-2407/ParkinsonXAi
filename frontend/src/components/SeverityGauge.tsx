import { useState } from 'react'

interface SeverityGaugeProps {
  motorUpdrs: number
  totalUpdrs: number
  severityLevel: string
  severityScore?: number         // 0–100 audio-proxy score
  severityBasis?: string         // "audio_proxy" | "updrs_model"
}

function SemiGauge({
  score,
}: {
  score: number
}) {
  const size = 200
  const strokeW = 14
  const r = (size - strokeW * 2) / 2
  const cx = size / 2
  const cy = size / 2
  const pct = Math.max(0, Math.min(1, score / 100))
  const arcLen = Math.PI * r
  const dash = pct * arcLen

  // Color gradient based on severity
  const gaugeColor =
    score < 30 ? '#10b981' :
    score < 60 ? '#f59e0b' : '#ef4444'

  const glowColor = gaugeColor + '55'

  // Needle position along the arc
  const angle = Math.PI + pct * Math.PI  // π to 2π (left to right)
  const needleX = cx + (r) * Math.cos(angle)
  const needleY = cy + (r) * Math.sin(angle)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <div style={{ position: 'relative', width: size, height: size / 2 + 30 }}>
        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          style={{ position: 'absolute', top: 0, left: 0 }}
        >
          {/* Track arc */}
          <path
            d={`M ${strokeW} ${cy} A ${r} ${r} 0 0 1 ${size - strokeW} ${cy}`}
            fill="none"
            stroke="rgba(255,255,255,0.06)"
            strokeWidth={strokeW}
            strokeLinecap="round"
          />
          {/* Colored fill arc */}
          <path
            d={`M ${strokeW} ${cy} A ${r} ${r} 0 0 1 ${size - strokeW} ${cy}`}
            fill="none"
            stroke={gaugeColor}
            strokeWidth={strokeW}
            strokeLinecap="round"
            strokeDasharray={`${dash} ${arcLen}`}
            style={{
              filter: `drop-shadow(0 0 8px ${glowColor})`,
              transition: 'stroke-dasharray 1.4s cubic-bezier(0.4,0,0.2,1)',
            }}
          />
          {/* Needle dot */}
          <circle
            cx={needleX}
            cy={needleY}
            r={6}
            fill={gaugeColor}
            style={{ filter: `drop-shadow(0 0 6px ${gaugeColor})`, transition: 'cx 1.4s ease, cy 1.4s ease' }}
          />
          {/* Center score */}
          <text
            x={cx} y={cy - 8}
            textAnchor="middle"
            fill="var(--text-primary)"
            fontSize={32}
            fontWeight={800}
            fontFamily="var(--font-display)"
          >
            {score.toFixed(0)}
          </text>
          <text
            x={cx} y={cy + 14}
            textAnchor="middle"
            fill="var(--text-muted)"
            fontSize={11}
          >
            / 100
          </text>
          {/* Scale labels */}
          <text x={strokeW + 2} y={cy + 22} fill="rgba(255,255,255,0.25)" fontSize={9} textAnchor="middle">0</text>
          <text x={cx} y={strokeW - 2} fill="rgba(255,255,255,0.25)" fontSize={9} textAnchor="middle">50</text>
          <text x={size - strokeW - 2} y={cy + 22} fill="rgba(255,255,255,0.25)" fontSize={9} textAnchor="middle">100</text>
        </svg>
      </div>
    </div>
  )
}

export default function SeverityGauge({
  motorUpdrs,
  totalUpdrs,
  severityLevel,
  severityScore,
  severityBasis = 'audio_proxy',
}: SeverityGaugeProps) {
  const [showInfo, setShowInfo] = useState(false)

  // Use severityScore if provided, else derive from motorUpdrs for backward compat
  const displayScore = severityScore !== undefined
    ? severityScore
    : Math.round((motorUpdrs / 108) * 100)

  const levelColor =
    severityLevel === 'Mild' ? '#10b981' :
    severityLevel === 'Moderate' ? '#f59e0b' : '#ef4444'

  const levelEmoji =
    severityLevel === 'Mild' ? '🟢' :
    severityLevel === 'Moderate' ? '🟡' : '🔴'

  const isProxy = severityBasis === 'audio_proxy'

  return (
    <div>
      <div className="flex justify-between items-center" style={{ marginBottom: 16 }}>
        <p className="section-label">Vocal Severity Score</p>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{
            padding: '4px 12px',
            borderRadius: 999,
            background: `${levelColor}20`,
            border: `1px solid ${levelColor}50`,
            color: levelColor,
            fontSize: '0.78rem',
            fontWeight: 700,
          }}>
            {levelEmoji} {severityLevel}
          </span>
        </div>
      </div>

      {/* Main gauge */}
      <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 8 }}>
        <SemiGauge score={displayScore} />
      </div>

      {/* Severity level description */}
      <div style={{
        display: 'flex', justifyContent: 'center', gap: 6, marginBottom: 14,
      }}>
        {(['Mild', 'Moderate', 'Severe'] as const).map(lvl => (
          <div
            key={lvl}
            style={{
              flex: 1,
              padding: '6px 10px',
              borderRadius: 6,
              textAlign: 'center',
              background: severityLevel === lvl
                ? (lvl === 'Mild' ? 'rgba(16,185,129,0.15)' : lvl === 'Moderate' ? 'rgba(245,158,11,0.15)' : 'rgba(239,68,68,0.15)')
                : 'rgba(255,255,255,0.03)',
              border: `1px solid ${severityLevel === lvl
                ? (lvl === 'Mild' ? 'rgba(16,185,129,0.4)' : lvl === 'Moderate' ? 'rgba(245,158,11,0.4)' : 'rgba(239,68,68,0.4)')
                : 'rgba(255,255,255,0.06)'}`,
            }}
          >
            <p style={{
              fontSize: '0.72rem',
              fontWeight: severityLevel === lvl ? 700 : 400,
              color: severityLevel === lvl
                ? (lvl === 'Mild' ? '#10b981' : lvl === 'Moderate' ? '#f59e0b' : '#ef4444')
                : 'var(--text-muted)',
            }}>
              {lvl}
            </p>
            <p style={{ fontSize: '0.62rem', color: 'var(--text-muted)' }}>
              {lvl === 'Mild' ? '0–29' : lvl === 'Moderate' ? '30–59' : '60–100'}
            </p>
          </div>
        ))}
      </div>

      {/* Source badge + info */}
      <div style={{
        padding: '10px 14px',
        background: 'var(--bg-card)',
        borderRadius: 'var(--radius-sm)',
        border: '1px solid var(--border)',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <p style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 4 }}>
              {isProxy
                ? '🎙️ Score based on acoustic biomarkers'
                : '📋 Score based on UPDRS regression model'}
            </p>
            <p style={{ fontSize: '0.68rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
              {isProxy
                ? 'Jitter (tremor) · Shimmer (hypophonia) · HNR (voice clarity) · Voiced fraction (breathiness)'
                : `Motor UPDRS: ${motorUpdrs.toFixed(1)} / 108 · Total UPDRS: ${totalUpdrs.toFixed(1)} / 176`}
            </p>
          </div>
          <button
            id="severity-info-toggle"
            onClick={() => setShowInfo(s => !s)}
            style={{
              background: 'transparent', border: 'none', cursor: 'pointer',
              color: 'var(--text-muted)', fontSize: '0.7rem', flexShrink: 0, paddingLeft: 8,
            }}
          >
            ❓
          </button>
        </div>

        {showInfo && (
          <div style={{
            marginTop: 10,
            padding: '10px 12px',
            background: 'rgba(124,58,237,0.06)',
            border: '1px solid rgba(124,58,237,0.15)',
            borderRadius: 6,
          }}>
            <p style={{ fontSize: '0.74rem', fontWeight: 700, color: 'var(--purple-light)', marginBottom: 6 }}>
              How is Severity Calculated?
            </p>
            {isProxy ? (
              <>
                <p style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: 8 }}>
                  Since your voice recording doesn't include clinical UPDRS test data, severity is estimated from <strong>acoustic biomarkers</strong> measured directly from your voice:
                </p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {[
                    ['Jitter (30%)', 'Vocal tremor — pitch cycle irregularity', '#f87171'],
                    ['Shimmer (30%)', 'Hypophonia — amplitude cycle irregularity', '#fb923c'],
                    ['HNR (20%)', 'Voice clarity — harmonics vs. noise', '#34d399'],
                    ['Voiced fraction (12%)', 'Breathiness / aperiodicity', '#60a5fa'],
                    ['RMS energy (8%)', 'Overall voice loudness', '#a78bfa'],
                  ].map(([label, desc, color]) => (
                    <div key={label} style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                      <div style={{ width: 8, height: 8, borderRadius: '50%', background: color, marginTop: 4, flexShrink: 0 }} />
                      <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                        <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>{label}</span> — {desc}
                      </p>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <p style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                UPDRS (Unified Parkinson's Disease Rating Scale) is the standard clinical tool for measuring Parkinson's severity.
                Motor UPDRS (0–108) assesses motor symptoms. Total UPDRS (0–176) includes non-motor domains.
                Lower scores = less severe symptoms.
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
