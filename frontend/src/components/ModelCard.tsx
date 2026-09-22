import type { ModelAccuracyCard as ModelAccuracyCardType } from '../api/parkinsonApi'

interface ModelCardProps {
  card: ModelAccuracyCardType
}

export default function ModelCard({ card }: ModelCardProps) {
  const hasAccuracy = card.accuracy != null && card.accuracy > 0

  return (
    <div className="glass-card stat-card" style={{ gap: 12, height: '100%' }}>
      <div className="flex justify-between items-center">
        <span className="badge badge-purple" style={{ fontSize: '0.65rem' }}>Model</span>
        {hasAccuracy && (
          <div style={{
            width: 36, height: 36,
            borderRadius: '50%',
            background: 'conic-gradient(var(--purple) 0% ' + (card.accuracy! * 100) + '%, rgba(255,255,255,0.05) ' + (card.accuracy! * 100) + '% 100%)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <div style={{
              width: 26, height: 26, borderRadius: '50%',
              background: 'var(--bg-secondary)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <span style={{ fontSize: '0.55rem', fontWeight: 700, color: 'var(--purple-light)' }}>
                {(card.accuracy! * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        )}
      </div>

      <div>
        <p style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--text-primary)', lineHeight: 1.3 }}>
          {card.model_name}
        </p>
        <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 2 }}>
          {card.dataset}
        </p>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 4 }}>
        {hasAccuracy && (
          <>
            <MetricRow label="Accuracy" value={`${(card.accuracy! * 100).toFixed(2)}%`} color="var(--purple-light)" />
            {card.auc != null && <MetricRow label="AUC" value={card.auc.toFixed(4)} color="var(--teal)" />}
            {card.recall != null && <MetricRow label="Recall" value={`${(card.recall * 100).toFixed(1)}%`} color="var(--positive)" />}
            {card.f1 != null && <MetricRow label="F1" value={card.f1.toFixed(4)} color="var(--amber)" />}
          </>
        )}
        {card.note && (
          <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 4, lineHeight: 1.4 }}>
            {card.note}
          </p>
        )}
      </div>
    </div>
  )
}

function MetricRow({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="flex justify-between items-center">
      <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{label}</span>
      <span style={{ fontSize: '0.8rem', fontWeight: 700, color }}>{value}</span>
    </div>
  )
}
