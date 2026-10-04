import { useState, useEffect } from 'react'
import { parkinsonApi } from '../api/parkinsonApi'
import type { HistoryItem, HistoryResponse } from '../api/parkinsonApi'

export default function PredictionHistory({ refreshTick }: { refreshTick?: number }) {
  const [data, setData] = useState<HistoryResponse | null>(null)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    parkinsonApi.history(page, 10)
      .then(r => { setData(r.data); setError(null) })
      .catch(e => setError(e.message || 'Failed to load history'))
      .finally(() => setLoading(false))
  }, [page, refreshTick])

  if (loading) return (
    <div className="flex items-center justify-center" style={{ height: 200 }}>
      <div className="spinner" />
    </div>
  )

  if (error) return (
    <div style={{
      padding: 24, textAlign: 'center', color: 'var(--negative)',
      background: 'var(--negative-dim)', borderRadius: 'var(--radius-md)',
    }}>
      ⚠️ {error}
    </div>
  )

  if (!data || data.items.length === 0) return (
    <div style={{ padding: 48, textAlign: 'center', color: 'var(--text-muted)' }}>
      <div style={{ fontSize: '2.5rem', marginBottom: 12 }}>🎙️</div>
      <p style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-secondary)' }}>No predictions yet</p>
      <p style={{ fontSize: '0.85rem', marginTop: 4 }}>Upload a WAV file to get started.</p>
    </div>
  )

  return (
    <div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>File</th>
              <th>Result</th>
              <th>Confidence</th>
              <th>Motor UPDRS</th>
              <th>Severity</th>
              <th>Top Feature</th>
              <th>Time</th>
              <th>Processed</th>
            </tr>
          </thead>
          <tbody>
            {data.items.map(item => (
              <HistoryRow key={item.prediction_id} item={item} />
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {data.total_pages > 1 && (
        <div className="flex items-center justify-between" style={{ marginTop: 20 }}>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            {data.total} total · Page {data.page} of {data.total_pages}
          </p>
          <div className="flex gap-8">
            <button
              className="btn-secondary"
              style={{ padding: '6px 16px', fontSize: '0.8rem' }}
              disabled={page <= 1}
              onClick={() => setPage(p => p - 1)}
            >
              ← Prev
            </button>
            <button
              className="btn-secondary"
              style={{ padding: '6px 16px', fontSize: '0.8rem' }}
              disabled={page >= data.total_pages}
              onClick={() => setPage(p => p + 1)}
            >
              Next →
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

function HistoryRow({ item }: { item: HistoryItem }) {
  const isParkinson = item.detection_label === 'Parkinson'
  const date = new Date(item.timestamp)

  return (
    <tr>
      <td style={{ maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        <span title={item.filename} style={{ color: 'var(--text-primary)', fontSize: '0.82rem' }}>
          🎵 {item.filename}
        </span>
      </td>
      <td>
        {item.detection_label ? (
          <span className={`badge ${isParkinson ? 'badge-parkinson' : 'badge-healthy'}`}
            style={{ fontSize: '0.7rem' }}>
            {item.detection_label}
          </span>
        ) : '—'}
      </td>
      <td>
        {item.confidence != null ? (
          <span style={{ color: isParkinson ? '#f87171' : '#34d399', fontWeight: 600 }}>
            {(item.confidence * 100).toFixed(1)}%
          </span>
        ) : '—'}
      </td>
      <td style={{ color: 'var(--text-secondary)' }}>
        {item.motor_updrs != null ? item.motor_updrs.toFixed(1) : '—'}
      </td>
      <td>
        {item.severity_level ? (
          <span style={{
            color: item.severity_level === 'Mild' ? 'var(--positive)' :
                   item.severity_level === 'Moderate' ? 'var(--warning)' : 'var(--negative)',
            fontSize: '0.8rem', fontWeight: 600,
          }}>
            {item.severity_level}
          </span>
        ) : '—'}
      </td>
      <td style={{ maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        <span title={item.shap_top_feature || ''} style={{ fontSize: '0.75rem', color: 'var(--purple-light)' }}>
          {item.shap_top_feature || '—'}
        </span>
      </td>
      <td style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
        {item.processing_time_ms != null ? `${item.processing_time_ms.toFixed(0)} ms` : '—'}
      </td>
      <td style={{ fontSize: '0.75rem', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
        {date.toLocaleDateString()} {date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
      </td>
    </tr>
  )
}
