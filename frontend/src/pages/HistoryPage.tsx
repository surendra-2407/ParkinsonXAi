import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { parkinsonApi } from '../api/parkinsonApi'
import type { HistoryItem } from '../api/parkinsonApi'

interface FilteredHistoryResponse {
  items: HistoryItem[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

export default function HistoryPage() {
  const navigate = useNavigate()
  const [data, setData] = useState<FilteredHistoryResponse | null>(null)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Filters
  const [search, setSearch] = useState('')
  const [labelFilter, setLabelFilter] = useState('')
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')

  // Detail modal
  const [detailId, setDetailId] = useState<string | null>(null)
  const [detailData, setDetailData] = useState<any | null>(null)
  const [loadingDetail, setLoadingDetail] = useState(false)

  // Delete
  const [deleteId, setDeleteId] = useState<string | null>(null)
  const [deleteFilename, setDeleteFilename] = useState('')
  const [deleting, setDeleting] = useState(false)

  const fetchHistory = useCallback(() => {
    setLoading(true)
    parkinsonApi.historyFiltered({
      page, page_size: 10,
      search: search || undefined,
      label: labelFilter || undefined,
      date_from: dateFrom || undefined,
      date_to: dateTo || undefined,
    })
      .then(r => { setData(r.data as FilteredHistoryResponse); setError(null) })
      .catch(e => setError(e.message || 'Failed to load history'))
      .finally(() => setLoading(false))
  }, [page, search, labelFilter, dateFrom, dateTo])

  useEffect(() => { fetchHistory() }, [fetchHistory])

  // Reset to page 1 when filters change
  useEffect(() => { setPage(1) }, [search, labelFilter, dateFrom, dateTo])

  async function openDetail(id: string) {
    setDetailId(id)
    setLoadingDetail(true)
    try {
      const res = await parkinsonApi.predictionDetail(id)
      setDetailData(res.data)
    } catch {
      setDetailData(null)
    } finally {
      setLoadingDetail(false)
    }
  }

  async function confirmDelete() {
    if (!deleteId) return
    setDeleting(true)
    try {
      await parkinsonApi.deletePrediction(deleteId)
      setDeleteId(null)
      fetchHistory()
    } catch (e: any) {
      alert('Delete failed: ' + (e?.response?.data?.detail || e.message))
    } finally {
      setDeleting(false)
    }
  }

  function exportCSV() {
    if (!data?.items.length) return
    const headers = ['Prediction ID', 'Filename', 'Timestamp', 'Label', 'Confidence', 'Voice Quality', 'Severity Level', 'Motor UPDRS', 'Processing (ms)']
    const rows = data.items.map(item => [
      item.prediction_id,
      item.filename,
      item.timestamp,
      item.detection_label ?? '',
      item.confidence != null ? (item.confidence * 100).toFixed(1) + '%' : '',
      item.voice_quality != null ? (item.voice_quality * 100).toFixed(0) + '%' : '',
      item.severity_level ?? '',
      item.motor_updrs != null ? item.motor_updrs.toFixed(1) : '',
      item.processing_time_ms != null ? item.processing_time_ms.toFixed(0) : '',
    ])
    const csv = [headers, ...rows].map(r => r.map(v => `"${v}"`).join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = `parkinsonxai_history_${new Date().toISOString().slice(0, 10)}.csv`
    a.click(); URL.revokeObjectURL(url)
  }

  return (
    <div className="page">
      <div className="container">
        {/* Header */}
        <div style={{ marginBottom: 28, display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
          <div>
            <p className="section-label">MongoDB</p>
            <h1 style={{ fontSize: '1.8rem', fontWeight: 700, marginTop: 4 }}>Prediction History</h1>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: 6 }}>
              All analyses stored in MongoDB — newest first.
              {data && ` ${data.total} total records.`}
            </p>
          </div>
          <button
            className="btn-secondary"
            onClick={exportCSV}
            disabled={!data?.items.length}
            style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 20px', fontSize: '0.82rem' }}
          >
            ⬇️ Export CSV
          </button>
        </div>

        {/* Filters */}
        <div className="glass-card p-24" style={{ marginBottom: 20 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'flex-end' }}>
            <div style={{ flex: 1, minWidth: 180 }}>
              <label style={labelStyle}>Search Filename</label>
              <input
                placeholder="e.g. recording.wav"
                value={search}
                onChange={e => setSearch(e.target.value)}
                style={inputStyle}
              />
            </div>
            <div>
              <label style={labelStyle}>Label Filter</label>
              <select value={labelFilter} onChange={e => setLabelFilter(e.target.value)} style={inputStyle}>
                <option value="">All Labels</option>
                <option value="Parkinson">Parkinson</option>
                <option value="Healthy">Healthy</option>
                <option value="Uncertain">Uncertain</option>
              </select>
            </div>
            <div>
              <label style={labelStyle}>From Date</label>
              <input type="date" value={dateFrom} onChange={e => setDateFrom(e.target.value)} style={inputStyle} />
            </div>
            <div>
              <label style={labelStyle}>To Date</label>
              <input type="date" value={dateTo} onChange={e => setDateTo(e.target.value)} style={inputStyle} />
            </div>
            {(search || labelFilter || dateFrom || dateTo) && (
              <button
                className="btn-secondary"
                onClick={() => { setSearch(''); setLabelFilter(''); setDateFrom(''); setDateTo('') }}
                style={{ padding: '8px 16px', fontSize: '0.82rem' }}
              >
                ✕ Clear
              </button>
            )}
          </div>
        </div>

        {/* Table */}
        <div className="glass-card p-24">
          {loading ? (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 200 }}>
              <div className="spinner" />
            </div>
          ) : error ? (
            <div style={{ padding: 24, textAlign: 'center', color: 'var(--negative)' }}>⚠️ {error}</div>
          ) : !data || data.items.length === 0 ? (
            <div style={{ padding: 60, textAlign: 'center', color: 'var(--text-muted)' }}>
              <div style={{ fontSize: '2.5rem', marginBottom: 12 }}>🎙️</div>
              <p style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-secondary)' }}>No predictions found</p>
              <p style={{ fontSize: '0.85rem', marginTop: 4 }}>
                {search || labelFilter ? 'Try different filters.' : 'Upload a WAV file on the Analyze page to get started.'}
              </p>
            </div>
          ) : (
            <>
              <div style={{ overflowX: 'auto' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>File</th>
                      <th>Result</th>
                      <th style={{ textAlign: 'right' }}>Confidence</th>
                      <th style={{ textAlign: 'right' }}>Voice Quality</th>
                      <th>Severity</th>
                      <th style={{ textAlign: 'right' }}>Motor UPDRS</th>
                      <th>Top Feature</th>
                      <th style={{ textAlign: 'right' }}>Time (ms)</th>
                      <th>Date</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.items.map(item => (
                      <HistoryRow
                        key={item.prediction_id}
                        item={item}
                        onView={() => openDetail(item.prediction_id)}
                        onDelete={() => { setDeleteId(item.prediction_id); setDeleteFilename(item.filename) }}
                      />
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              {data.total_pages > 1 && (
                <div className="flex items-center justify-between" style={{ marginTop: 20 }}>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    {data.total} records · Page {data.page} of {data.total_pages}
                  </p>
                  <div className="flex gap-8">
                    <button className="btn-secondary" style={{ padding: '6px 16px', fontSize: '0.8rem' }}
                      disabled={page <= 1} onClick={() => setPage(p => p - 1)}>← Prev</button>
                    {/* Page numbers */}
                    {Array.from({ length: Math.min(data.total_pages, 5) }, (_, i) => {
                      const pg = Math.max(1, page - 2) + i
                      if (pg > data.total_pages) return null
                      return (
                        <button key={pg}
                          className={pg === page ? 'btn-primary' : 'btn-secondary'}
                          style={{ padding: '6px 12px', fontSize: '0.8rem', minWidth: 36 }}
                          onClick={() => setPage(pg)}>
                          {pg}
                        </button>
                      )
                    })}
                    <button className="btn-secondary" style={{ padding: '6px 16px', fontSize: '0.8rem' }}
                      disabled={page >= data.total_pages} onClick={() => setPage(p => p + 1)}>Next →</button>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Detail Modal */}
      {detailId && (
        <div style={backdropStyle} onClick={() => { setDetailId(null); setDetailData(null) }}>
          <div style={modalStyle} onClick={e => e.stopPropagation()}>
            <div className="flex justify-between items-center" style={{ marginBottom: 20 }}>
              <p style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: '1rem' }}>Prediction Detail</p>
              <button onClick={() => { setDetailId(null); setDetailData(null) }}
                style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '1.2rem' }}>✕</button>
            </div>
            {loadingDetail ? (
              <div style={{ display: 'flex', justifyContent: 'center', padding: 40 }}><div className="spinner" /></div>
            ) : detailData ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                <DetailRow label="Filename" value={detailData.filename} />
                <DetailRow label="Prediction ID" value={detailData._id} mono />
                <DetailRow label="Timestamp" value={detailData.timestamp ? new Date(detailData.timestamp).toLocaleString() : '—'} />
                <DetailRow label="Detection" value={detailData.detection?.label ?? '—'} />
                <DetailRow label="Confidence" value={detailData.detection?.confidence != null ? `${(detailData.detection.confidence * 100).toFixed(1)}%` : '—'} />
                <DetailRow label="Voice Quality" value={detailData.detection?.voice_quality != null ? `${(detailData.detection.voice_quality * 100).toFixed(0)}%` : '—'} />
                <DetailRow label="Severity Level" value={detailData.severity?.severity_level ?? '—'} />
                <DetailRow label="Motor UPDRS" value={detailData.severity?.motor_updrs != null ? detailData.severity.motor_updrs.toFixed(1) : '—'} />
                <DetailRow label="Total UPDRS" value={detailData.severity?.total_updrs != null ? detailData.severity.total_updrs.toFixed(1) : '—'} />
                <DetailRow label="Severity Basis" value={detailData.severity?.severity_basis ?? '—'} />
                <DetailRow label="Model" value={detailData.detection?.model_used ?? '—'} />
                <DetailRow label="Processing" value={detailData.processing_time_ms != null ? `${detailData.processing_time_ms.toFixed(0)} ms` : '—'} />
                {detailData.shap?.top_features?.length > 0 && (
                  <div>
                    <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: 8, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase' }}>Top SHAP Features</p>
                    {detailData.shap.top_features.slice(0, 5).map((f: any, i: number) => (
                      <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', borderBottom: '1px solid var(--border)', fontSize: '0.8rem' }}>
                        <span style={{ color: 'var(--text-secondary)' }}>{f.feature}</span>
                        <span style={{ color: f.value > 0 ? '#f87171' : '#2dd4bf', fontWeight: 600 }}>
                          {f.value > 0 ? '+' : ''}{f.value.toFixed(4)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                <button
                  className="btn-primary"
                  style={{
                    width: '100%',
                    marginTop: 20,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: 8,
                    padding: '10px 16px',
                  }}
                  onClick={() => {
                    localStorage.setItem('xai_last_result', JSON.stringify(detailData))
                    navigate('/results', { state: { result: detailData } })
                  }}
                >
                  📖 View Full Analysis & Written Explanation
                </button>
              </div>
            ) : (
              <p style={{ color: 'var(--negative)', fontSize: '0.85rem' }}>Failed to load prediction details.</p>
            )}
          </div>
        </div>
      )}

      {/* Delete Confirm Modal */}
      {deleteId && (
        <div style={backdropStyle} onClick={() => setDeleteId(null)}>
          <div style={{ ...modalStyle, maxWidth: 420 }} onClick={e => e.stopPropagation()}>
            <div style={{ textAlign: 'center', marginBottom: 20 }}>
              <div style={{ fontSize: '2.5rem', marginBottom: 12 }}>🗑️</div>
              <p style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: '1rem', marginBottom: 8 }}>Delete Prediction?</p>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                This will permanently remove <strong style={{ color: 'var(--text-primary)' }}>{deleteFilename}</strong> from MongoDB. This cannot be undone.
              </p>
            </div>
            <div style={{ display: 'flex', gap: 12 }}>
              <button className="btn-secondary" style={{ flex: 1, justifyContent: 'center' }} onClick={() => setDeleteId(null)}>
                Cancel
              </button>
              <button
                onClick={confirmDelete}
                disabled={deleting}
                style={{
                  flex: 1, padding: '10px 20px', borderRadius: 'var(--radius-md)',
                  background: '#ef4444', color: 'white', border: 'none',
                  fontWeight: 600, cursor: deleting ? 'not-allowed' : 'pointer',
                  opacity: deleting ? 0.6 : 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                }}
              >
                {deleting ? <><div className="spinner" style={{ width: 14, height: 14, borderWidth: 2, borderTopColor: 'white' }} /> Deleting...</> : 'Delete'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function HistoryRow({ item, onView, onDelete }: { item: HistoryItem; onView: () => void; onDelete: () => void }) {
  const isParkinson = item.detection_label === 'Parkinson'
  const date = new Date(item.timestamp)
  return (
    <tr>
      <td style={{ maxWidth: 150, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        <span title={item.filename} style={{ color: 'var(--text-primary)', fontSize: '0.82rem' }}>🎵 {item.filename}</span>
      </td>
      <td>
        {item.detection_label ? (
          <span className={`badge ${isParkinson ? 'badge-parkinson' : 'badge-healthy'}`} style={{ fontSize: '0.68rem' }}>
            {item.detection_label}
          </span>
        ) : '—'}
      </td>
      <td style={{ textAlign: 'right' }}>
        {item.confidence != null ? (
          <span style={{ color: isParkinson ? '#f87171' : '#34d399', fontWeight: 600 }}>
            {(item.confidence * 100).toFixed(1)}%
          </span>
        ) : '—'}
      </td>
      <td style={{ textAlign: 'right' }}>
        {item.voice_quality != null ? (
          <span style={{
            color: item.voice_quality > 0.6 ? '#34d399' : item.voice_quality > 0.3 ? '#fbbf24' : '#f87171',
            fontWeight: 600,
          }}>
            {(item.voice_quality * 100).toFixed(0)}%
          </span>
        ) : '—'}
      </td>
      <td>
        {item.severity_level ? (
          <span style={{
            fontSize: '0.75rem', fontWeight: 600,
            color: item.severity_level === 'Mild' ? 'var(--positive)' :
              item.severity_level === 'Moderate' ? 'var(--warning)' : 'var(--negative)',
          }}>{item.severity_level}</span>
        ) : '—'}
      </td>
      <td style={{ textAlign: 'right', color: 'var(--text-secondary)' }}>
        {item.motor_updrs != null ? item.motor_updrs.toFixed(1) : '—'}
      </td>
      <td style={{ maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        <span title={item.shap_top_feature || ''} style={{ fontSize: '0.72rem', color: 'var(--purple-light)' }}>
          {item.shap_top_feature || '—'}
        </span>
      </td>
      <td style={{ textAlign: 'right', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
        {item.processing_time_ms != null ? `${item.processing_time_ms.toFixed(0)} ms` : '—'}
      </td>
      <td style={{ fontSize: '0.72rem', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
        {date.toLocaleDateString()} {date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
      </td>
      <td>
        <div style={{ display: 'flex', gap: 6 }}>
          <button onClick={onView} style={actionBtnStyle('#7c3aed')}>View</button>
          <button onClick={onDelete} style={actionBtnStyle('#ef4444')}>Delete</button>
        </div>
      </td>
    </tr>
  )
}

function DetailRow({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border)', paddingBottom: 10 }}>
      <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>{label}</span>
      <span style={{ fontSize: '0.8rem', color: 'var(--text-primary)', fontWeight: 600, fontFamily: mono ? 'monospace' : undefined, textAlign: 'right', maxWidth: 250, wordBreak: 'break-all' }}>
        {value}
      </span>
    </div>
  )
}

const actionBtnStyle = (color: string): React.CSSProperties => ({
  padding: '4px 10px', borderRadius: 6, border: `1px solid ${color}44`,
  background: `${color}15`, color, fontSize: '0.72rem', fontWeight: 600,
  cursor: 'pointer', transition: 'all 0.2s',
})

const backdropStyle: React.CSSProperties = {
  position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)',
  backdropFilter: 'blur(4px)', display: 'flex', alignItems: 'center',
  justifyContent: 'center', zIndex: 1000, padding: 20,
}

const modalStyle: React.CSSProperties = {
  background: 'var(--bg-secondary)', border: '1px solid var(--border)',
  borderRadius: 'var(--radius-xl)', padding: 32, maxWidth: 560,
  width: '100%', maxHeight: '85vh', overflowY: 'auto',
}

const labelStyle: React.CSSProperties = {
  fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block',
  marginBottom: 6, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase',
}

const inputStyle: React.CSSProperties = {
  background: 'var(--bg-card)', border: '1px solid var(--border)',
  borderRadius: 'var(--radius-sm)', color: 'var(--text-primary)',
  padding: '8px 12px', fontSize: '0.85rem', outline: 'none',
  width: '100%',
}
