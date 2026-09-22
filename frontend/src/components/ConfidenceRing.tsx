interface ConfidenceRingProps {
  confidence: number       // 0.0 – 1.0
  label: string            // 'Parkinson' | 'Healthy'
  size?: number
}


export default function ConfidenceRing({ confidence, label, size = 180 }: ConfidenceRingProps) {
  const isParkinson = label === 'Parkinson'
  const color = isParkinson ? '#ef4444' : '#10b981'
  const glowColor = isParkinson ? 'rgba(239,68,68,0.4)' : 'rgba(16,185,129,0.4)'

  const pct = Math.max(0, Math.min(1, confidence))
  const r = (size - 20) / 2
  const circumference = 2 * Math.PI * r
  const dash = pct * circumference
  const center = size / 2

  return (
    <div className="confidence-ring-container">
      <div style={{ position: 'relative', width: size, height: size }}>
        <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
          {/* Track */}
          <circle
            cx={center} cy={center} r={r}
            fill="none"
            stroke="rgba(255,255,255,0.05)"
            strokeWidth={10}
          />
          {/* Progress arc */}
          <circle
            cx={center} cy={center} r={r}
            fill="none"
            stroke={color}
            strokeWidth={10}
            strokeLinecap="round"
            strokeDasharray={`${dash} ${circumference}`}
            style={{
              filter: `drop-shadow(0 0 8px ${glowColor})`,
              transition: 'stroke-dasharray 1.2s cubic-bezier(0.4,0,0.2,1)',
            }}
          />
        </svg>

        {/* Center text */}
        <div style={{
          position: 'absolute', inset: 0,
          display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center',
          gap: 2,
        }}>
          <span style={{
            fontSize: size > 150 ? '2rem' : '1.4rem',
            fontWeight: 800,
            fontFamily: 'var(--font-display)',
            color,
            lineHeight: 1,
          }}>
            {(pct * 100).toFixed(1)}%
          </span>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', letterSpacing: '0.06em' }}>
            confidence
          </span>
        </div>
      </div>

      {/* Label badge */}
      <div style={{
        display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4,
      }}>
        <span className={`badge ${isParkinson ? 'badge-parkinson' : 'badge-healthy'}`}
          style={{ fontSize: '0.85rem', padding: '6px 16px' }}>
          {isParkinson ? '⚠️' : '✅'} {label}
        </span>
        {isParkinson && (
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
            Parkinson's indicators detected
          </span>
        )}
      </div>
    </div>
  )
}
