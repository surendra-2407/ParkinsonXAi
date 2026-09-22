import PredictionHistory from '../components/PredictionHistory'

export default function HistoryPage() {
  return (
    <div className="page">
      <div className="container">
        <div style={{ marginBottom: 32 }}>
          <h1 style={{ fontSize: '1.8rem', fontWeight: 700, marginTop: 4 }}>Prediction History</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: 6 }}>
            All past analyses, paginated by date, newest first.
          </p>
        </div>

        <div className="glass-card p-24">
          <PredictionHistory />
        </div>
      </div>
    </div>
  )
}
