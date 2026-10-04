import { useState, useEffect } from 'react'
import { parkinsonApi } from '../api/parkinsonApi'
import type { ModelPerfEntry, SeverityEvalEntry, DatasetInfo, FeatureGroup } from '../api/parkinsonApi'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, PieChart, Pie, Legend,
} from 'recharts'

type Tab = 'model' | 'updrs' | 'datasets' | 'features' | 'howitworks'

const TABS: { id: Tab; label: string; icon: string }[] = [
  { id: 'model',      label: 'Model Performance', icon: '🏆' },
  { id: 'updrs',      label: 'UPDRS Severity',     icon: '📈' },
  { id: 'datasets',   label: 'Datasets',           icon: '🗄️' },
  { id: 'features',   label: 'Feature Engineering',icon: '⚙️' },
  { id: 'howitworks', label: 'How It Works',       icon: '💡' },
]

const DS_COLORS: Record<string, string> = {
  D1: '#3b82f6', D2: '#0ea5e9', D3a: '#38bdf8', D3b: '#60a5fa',
}
const GRP_COLORS = ['#1d4ed8','#2563eb','#3b82f6','#60a5fa','#93c5fd','#0ea5e9','#38bdf8','#67e8f9','#a5f3fc','#0284c7','#0369a1','#1e40af','#1e3a8a']

export default function ResearchOverviewPage() {
  const [tab, setTab] = useState<Tab>('model')

  // Model perf
  const [models, setModels] = useState<ModelPerfEntry[]>([])
  const [modelsNote, setModelsNote] = useState('')

  // UPDRS
  const [motor, setMotor] = useState<SeverityEvalEntry[]>([])
  const [totalSev, setTotalSev] = useState<SeverityEvalEntry[]>([])
  const [updrsNote, setUpdrsNote] = useState('')
  const [updrsTarget, setUpdrsTarget] = useState<'motor'|'total'>('motor')

  // Datasets
  const [datasets, setDatasets] = useState<DatasetInfo[]>([])
  const [selDs, setSelDs] = useState('D1')

  // Features
  const [groups, setGroups] = useState<FeatureGroup[]>([])
  const [featMeta, setFeatMeta] = useState({ total_raw: 0, total_selected: 0, selection_method: '' })

  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      parkinsonApi.modelPerformance(),
      parkinsonApi.severityEvaluations(),
      parkinsonApi.datasets(),
      parkinsonApi.features(),
    ]).then(([m, s, d, f]) => {
      setModels(m.data.models); setModelsNote(m.data.note)
      setMotor(s.data.motor); setTotalSev(s.data.total); setUpdrsNote(s.data.note)
      setDatasets(d.data.datasets)
      setGroups(f.data.feature_groups)
      setFeatMeta({ total_raw: f.data.total_raw, total_selected: f.data.total_selected, selection_method: f.data.selection_method })
    }).finally(() => setLoading(false))
  }, [])

  const activeDs = datasets.find(d => d.id === selDs)
  const updrsData = updrsTarget === 'motor' ? motor : totalSev

  return (
    <div className="page">
      <div className="container">
        <div style={{ marginBottom: 24 }}>
          <p className="section-label">Research Details</p>
          <h1 style={{ fontSize: '1.6rem', fontWeight: 700, marginTop: 4 }}>Research Overview</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem', marginTop: 4 }}>
            All research paper data — model evaluations, datasets, feature engineering, and methodology.
          </p>
        </div>

        {/* Source banner */}
        <div style={{
          padding: '10px 16px', marginBottom: 20, borderRadius: 'var(--radius-md)',
          background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.25)',
          fontSize: '0.78rem', color: '#60a5fa',
        }}>
          📄 All values are from the research paper (held-out test results). Not live prediction outputs.
        </div>

        {/* Tab bar */}
        <div style={{
          display: 'flex', gap: 6, marginBottom: 24, flexWrap: 'wrap',
          padding: '6px', background: 'var(--bg-card)', borderRadius: 'var(--radius-md)',
          border: '1px solid var(--border)',
        }}>
          {TABS.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)} style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '8px 16px', borderRadius: 'var(--radius-sm)', border: 'none',
              background: tab === t.id ? 'linear-gradient(135deg,#1d4ed8,#3b82f6)' : 'transparent',
              color: tab === t.id ? 'white' : 'var(--text-secondary)',
              fontWeight: tab === t.id ? 700 : 500, fontSize: '0.82rem',
              cursor: 'pointer', transition: 'all 0.2s',
              boxShadow: tab === t.id ? '0 4px 14px rgba(29,78,216,0.4)' : 'none',
            }}>
              <span>{t.icon}</span><span>{t.label}</span>
            </button>
          ))}
        </div>

        {loading ? (
          <div style={{ display: 'flex', justifyContent: 'center', padding: 60 }}>
            <div className="spinner" style={{ width: 40, height: 40 }} />
          </div>
        ) : (
          <>
            {/* ── MODEL PERFORMANCE ── */}
            {tab === 'model' && (
              <div>
                <div style={{ padding: '12px 16px', marginBottom: 20, borderRadius: 'var(--radius-md)', background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.25)', fontSize: '0.78rem', color: '#fbbf24' }}>
                  ⚠️ {modelsNote}
                </div>
                {/* Bar chart */}
                <div className="glass-card p-24" style={{ marginBottom: 20 }}>
                  <p style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: 16 }}>Accuracy by Model & Dataset</p>
                  <div style={{ height: 280 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={models} margin={{ bottom: 40 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="model" tick={{ fill: '#94a3b8', fontSize: 11 }} angle={-25} textAnchor="end" />
                        <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} domain={[0.5, 1]} tickFormatter={v => `${(v*100).toFixed(0)}%`} />
                        <Tooltip formatter={(v: any) => `${(v*100).toFixed(2)}%`} contentStyle={{ background: 'var(--bg-glass)', border: '1px solid var(--border)', borderRadius: 8, fontSize: '0.78rem' }} />
                        <Bar dataKey="accuracy" radius={[4,4,0,0]}>
                          {models.map((m, i) => <Cell key={i} fill={DS_COLORS[m.dataset] || '#3b82f6'} fillOpacity={0.85} />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div style={{ display: 'flex', gap: 12, justifyContent: 'center', marginTop: 8, flexWrap: 'wrap' }}>
                    {Object.entries(DS_COLORS).map(([d, c]) => (
                      <span key={d} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                        <span style={{ width: 10, height: 10, borderRadius: 2, background: c, display: 'inline-block' }} />{d}
                      </span>
                    ))}
                  </div>
                </div>
                {/* Table */}
                <div className="glass-card p-24">
                  <p style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: 14 }}>Full Results Table</p>
                  <div style={{ overflowX: 'auto' }}>
                    <table className="data-table">
                      <thead><tr>
                        <th>Dataset</th><th>Model</th><th>Protocol</th>
                        <th style={{ textAlign: 'right' }}>Accuracy</th>
                        <th style={{ textAlign: 'right' }}>F1</th>
                        <th style={{ textAlign: 'right' }}>ROC-AUC</th>
                      </tr></thead>
                      <tbody>
                        {models.map((m, i) => (
                          <tr key={i}>
                            <td><span style={{ padding: '2px 8px', borderRadius: 4, background: `${DS_COLORS[m.dataset]}22`, color: DS_COLORS[m.dataset], fontSize: '0.75rem', fontWeight: 700 }}>{m.dataset}</span></td>
                            <td style={{ color: 'var(--text-primary)', fontWeight: m.is_primary ? 700 : 400 }}>{m.model}{m.is_primary && <span style={{ color: '#60a5fa', fontSize: '0.68rem', marginLeft: 6 }}>★ Primary</span>}</td>
                            <td><span style={{ fontSize: '0.7rem', padding: '2px 8px', borderRadius: 4, background: m.protocol === 'Record-level' ? 'rgba(245,158,11,0.15)' : 'rgba(14,165,233,0.15)', color: m.protocol === 'Record-level' ? '#fbbf24' : '#38bdf8' }}>{m.protocol}</span></td>
                            <td style={{ textAlign: 'right', fontWeight: 700, color: 'var(--text-primary)' }}>{(m.accuracy*100).toFixed(2)}%</td>
                            <td style={{ textAlign: 'right' }}>{m.f1.toFixed(4)}</td>
                            <td style={{ textAlign: 'right', color: '#38bdf8', fontWeight: 600 }}>{m.roc_auc.toFixed(4)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            {/* ── UPDRS SEVERITY ── */}
            {tab === 'updrs' && (
              <div>
                <div style={{ padding: '12px 16px', marginBottom: 20, borderRadius: 'var(--radius-md)', background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.25)', fontSize: '0.78rem', color: '#f87171' }}>
                  🔬 {updrsNote}
                </div>
                <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
                  {(['motor','total'] as const).map(t => (
                    <button key={t} onClick={() => setUpdrsTarget(t)}
                      className={updrsTarget === t ? 'btn-primary' : 'btn-secondary'}
                      style={{ padding: '8px 20px', fontSize: '0.82rem', textTransform: 'capitalize' }}>
                      {t} UPDRS
                    </button>
                  ))}
                </div>
                <div className="grid-2">
                  <div className="glass-card p-24">
                    <p style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: 14 }}>{updrsTarget === 'motor' ? 'Motor' : 'Total'} UPDRS — MAE</p>
                    <div style={{ height: 220 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={updrsData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                          <XAxis dataKey="model" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                          <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} />
                          <Tooltip contentStyle={{ background: 'var(--bg-glass)', border: '1px solid var(--border)', borderRadius: 8, fontSize: '0.78rem' }} />
                          <Bar dataKey="mae" fill="#3b82f6" radius={[4,4,0,0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  <div className="glass-card p-24">
                    <p style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: 14 }}>Results Table</p>
                    <table className="data-table">
                      <thead><tr><th>Model</th><th style={{ textAlign:'right' }}>MAE↓</th><th style={{ textAlign:'right' }}>RMSE↓</th><th style={{ textAlign:'right' }}>R²</th></tr></thead>
                      <tbody>
                        {updrsData.map((m, i) => (
                          <tr key={i}>
                            <td style={{ color: 'var(--text-primary)', fontWeight: i===0?700:400 }}>{m.model}{i===0&&<span style={{ color:'#38bdf8',fontSize:'0.68rem',marginLeft:6 }}>★</span>}</td>
                            <td style={{ textAlign:'right',color:'#60a5fa',fontWeight:600 }}>{m.mae.toFixed(2)}</td>
                            <td style={{ textAlign:'right' }}>{m.rmse.toFixed(2)}</td>
                            <td style={{ textAlign:'right',color:'#f87171',fontWeight:600 }}>{m.r2.toFixed(2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <div style={{ marginTop: 14, padding: '10px 14px', borderRadius: 'var(--radius-sm)', background: 'rgba(14,165,233,0.08)', border: '1px solid rgba(14,165,233,0.2)', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      <strong style={{ color: '#38bdf8' }}>Audio Proxy:</strong> When UPDRS inputs are unavailable, the system uses a deterministic acoustic proxy (0–100) from jitter, shimmer, HNR. Never interpret as a clinical UPDRS score.
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* ── DATASETS ── */}
            {tab === 'datasets' && (
              <div>
                <div className="grid-4" style={{ marginBottom: 20 }}>
                  {datasets.map(d => (
                    <button key={d.id} onClick={() => setSelDs(d.id)} style={{
                      background: selDs===d.id ? `${DS_COLORS[d.id]}18` : 'var(--bg-card)',
                      border: `1px solid ${selDs===d.id ? DS_COLORS[d.id]+'66' : 'var(--border)'}`,
                      borderRadius: 'var(--radius-lg)', padding: 16, textAlign: 'left', cursor: 'pointer', transition: 'all 0.2s',
                    }}>
                      <div style={{ display: 'inline-block', padding: '2px 8px', borderRadius: 4, background: `${DS_COLORS[d.id]}22`, color: DS_COLORS[d.id], fontSize: '0.72rem', fontWeight: 700, marginBottom: 8 }}>{d.id}</div>
                      <p style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: '0.82rem', marginBottom: 4 }}>{d.name}</p>
                      <p style={{ fontSize: '1.4rem', fontWeight: 800, color: DS_COLORS[d.id], lineHeight: 1 }}>{d.recordings.toLocaleString()}</p>
                      <p style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>recordings</p>
                    </button>
                  ))}
                </div>
                {activeDs && (
                  <div className="grid-2" style={{ alignItems: 'start' }}>
                    <div className="glass-card p-24">
                      <p style={{ fontWeight: 700, color: DS_COLORS[activeDs.id], marginBottom: 16 }}>{activeDs.id} — {activeDs.name}</p>
                      {[
                        ['Recordings', activeDs.recordings.toLocaleString()],
                        ['Speakers', activeDs.speakers?.toLocaleString() ?? 'Unknown'],
                        ['Healthy', activeDs.healthy_count?.toLocaleString() ?? 'N/A'],
                        ['Parkinson', activeDs.pd_count?.toLocaleString() ?? 'N/A'],
                        ['Target', activeDs.target],
                        ['Protocol', activeDs.protocol],
                        ['Model', activeDs.model_used],
                      ].map(([l, v]) => (
                        <div key={l} style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border)', paddingBottom: 8, marginBottom: 8 }}>
                          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>{l}</span>
                          <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)', textAlign: 'right', maxWidth: 200 }}>{v}</span>
                        </div>
                      ))}
                      <div style={{ marginTop: 8, padding: '8px 12px', borderRadius: 'var(--radius-sm)', background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.2)', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        ⚠️ {activeDs.notes}
                      </div>
                    </div>
                    <div className="glass-card p-24">
                      <p style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: 12 }}>Recording Distribution</p>
                      <div style={{ height: 200 }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <PieChart>
                            <Pie data={datasets.filter(d=>d.target_type==='detection').map(d=>({ name:d.id, value:d.recordings, color: DS_COLORS[d.id] }))} dataKey="value" cx="50%" cy="50%" outerRadius={80} innerRadius={40} strokeWidth={0}>
                              {datasets.filter(d=>d.target_type==='detection').map((d,i)=><Cell key={i} fill={DS_COLORS[d.id]} fillOpacity={0.85}/>)}
                            </Pie>
                            <Tooltip formatter={(v:any)=>[`${v} recordings`]} contentStyle={{ background:'var(--bg-glass)',border:'1px solid var(--border)',borderRadius:8,fontSize:'0.75rem' }}/>
                            <Legend formatter={v=><span style={{ color:'var(--text-secondary)',fontSize:'0.75rem' }}>{v}</span>}/>
                          </PieChart>
                        </ResponsiveContainer>
                      </div>
                      <div style={{ marginTop: 12 }}>
                        <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 8, letterSpacing: '0.06em', textTransform: 'uppercase' }}>Evaluation Protocol</p>
                        <div className="grid-2" style={{ gap: 8 }}>
                          {[{title:'Record-Level',color:'#f59e0b',note:'Same speaker may appear in train & test. Cannot confirm patient-independent generalization (D1).'},{title:'Speaker-Disjoint',color:'#0ea5e9',note:'Speaker only in train OR test. More realistic — used for D2, D3a, D3b.'}].map(c=>(
                            <div key={c.title} style={{ padding: '10px 12px', borderRadius: 'var(--radius-sm)', background: `${c.color}10`, border: `1px solid ${c.color}33` }}>
                              <p style={{ fontWeight: 700, color: c.color, fontSize: '0.78rem', marginBottom: 4 }}>{c.title}</p>
                              <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>{c.note}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* ── FEATURES ── */}
            {tab === 'features' && (
              <div>
                <div className="grid-4" style={{ marginBottom: 20 }}>
                  {[
                    { v: featMeta.total_raw, l: 'Raw Features', c: '#3b82f6' },
                    { v: 219, l: 'Non-Degenerate', c: '#0ea5e9' },
                    { v: featMeta.total_selected, l: 'Selected (MI)', c: '#38bdf8' },
                    { v: groups.length, l: 'Feature Groups', c: '#60a5fa' },
                  ].map(card => (
                    <div key={card.l} className="glass-card stat-card">
                      <p className="stat-value" style={{ color: card.c }}>{card.v}</p>
                      <p className="stat-label">{card.l}</p>
                    </div>
                  ))}
                </div>
                {/* Pipeline */}
                <div className="glass-card p-24" style={{ marginBottom: 20 }}>
                  <p style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: 14 }}>Extraction Pipeline</p>
                  <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 6 }}>
                    {[
                      { l: 'WAV Audio', c: '#475569' },
                      { l: '220 Features\n(Librosa+Praat)', c: '#1d4ed8' },
                      { l: 'VT Filter\n219 kept', c: '#0284c7' },
                      { l: 'Imputer\n+Scaler', c: '#0369a1' },
                      { l: 'MI Select\ntop 50', c: '#0ea5e9' },
                      { l: 'LightGBM\nPredict', c: '#38bdf8' },
                    ].map((s, i, arr) => (
                      <div key={s.l} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <div style={{ padding: '8px 12px', borderRadius: 8, background: `${s.c}20`, border: `1px solid ${s.c}44`, textAlign: 'center', whiteSpace: 'pre-line', fontSize: '0.72rem', fontWeight: 700, color: s.c, minWidth: 80 }}>{s.l}</div>
                        {i < arr.length - 1 && <span style={{ color: 'var(--text-muted)' }}>→</span>}
                      </div>
                    ))}
                  </div>
                </div>
                <div className="grid-2">
                  <div className="glass-card p-24">
                    <p style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: 14 }}>Feature Groups</p>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                      {groups.map((g, i) => (
                        <div key={g.group} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                          <div style={{ width: 8, height: 8, borderRadius: '50%', background: GRP_COLORS[i%GRP_COLORS.length], flexShrink: 0 }} />
                          <span style={{ flex: 1, fontSize: '0.82rem', color: 'var(--text-secondary)' }}>{g.group}</span>
                          <span style={{ fontSize: '0.8rem', fontWeight: 700, color: GRP_COLORS[i%GRP_COLORS.length] }}>{g.count}</span>
                        </div>
                      ))}
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10, borderTop: '1px solid var(--border)', paddingTop: 8, marginTop: 4 }}>
                        <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#60a5fa' }} />
                        <span style={{ flex: 1, fontSize: '0.82rem', fontWeight: 700, color: '#60a5fa' }}>Total</span>
                        <span style={{ fontSize: '0.9rem', fontWeight: 800, color: '#60a5fa' }}>220</span>
                      </div>
                    </div>
                  </div>
                  <div className="glass-card p-24">
                    <p style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: 14 }}>Distribution</p>
                    <div style={{ height: 260 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={groups} layout="vertical" margin={{ left: 100, right: 20 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                          <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 9 }} />
                          <YAxis type="category" dataKey="group" tick={{ fill: '#94a3b8', fontSize: 9 }} width={96} />
                          <Tooltip contentStyle={{ background:'var(--bg-glass)',border:'1px solid var(--border)',borderRadius:8,fontSize:'0.75rem' }}/>
                          <Bar dataKey="count" radius={[0,4,4,0]}>
                            {groups.map((_,i)=><Cell key={i} fill={GRP_COLORS[i%GRP_COLORS.length]} fillOpacity={0.85}/>)}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* ── HOW IT WORKS ── */}
            {tab === 'howitworks' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                {[
                  { icon: '🎙️', title: 'Step 1 — Record or Upload Voice', color: '#3b82f6', body: 'Upload a WAV audio file of a sustained vowel sound (e.g., holding "ahh" for 3–5 seconds). The system accepts WAV, WebM, and OGG formats recorded at any sample rate — it resamples to 22,050 Hz internally to match training conditions.' },
                  { icon: '⚙️', title: 'Step 2 — Feature Extraction', color: '#0ea5e9', body: '220 acoustic features are extracted using Librosa and Parselmouth (Praat). These include MFCCs (160), Chroma (24), Jitter, Shimmer, HNR, F0, RMS energy, Spectral features, and Formants. A VarianceThreshold filter removes near-constant features, leaving 219.' },
                  { icon: '🔢', title: 'Step 3 — Preprocessing Pipeline', color: '#38bdf8', body: 'The 219 features pass through: SimpleImputer (fills missing values with training medians), RobustScaler (normalizes to training distribution), and Mutual Information feature selection (retains top 50 features most predictive of Parkinson\'s).' },
                  { icon: '🤖', title: 'Step 4 — LightGBM Prediction', color: '#60a5fa', body: 'The 50 selected features are fed into a tuned LightGBM gradient boosting classifier. The model outputs a probability for each class: Healthy and Parkinson\'s. The class with the higher probability is the prediction. Reported accuracy on D1: 96.49%, AUC: 0.9995 (record-level).' },
                  { icon: '🔬', title: 'Step 5 — SHAP Explainability', color: '#93c5fd', body: 'SHAP TreeExplainer computes individual feature contributions for each prediction. Red bars indicate features pushing the model toward Parkinson\'s; teal bars push toward Healthy. Global importance is averaged over the D1 test set. Top feature: MFCC 24 Std (mean |SHAP| = 1.958).' },
                  { icon: '📊', title: 'Step 6 — Severity Estimate', color: '#1d4ed8', body: 'A separate LightGBM model estimates UPDRS severity (Motor and Total). If UPDRS-specific inputs are unavailable from the audio, an acoustic proxy score (0–100) is computed from jitter, shimmer, HNR, and voiced fraction. This proxy is never a clinical UPDRS score.' },
                  { icon: '⚠️', title: 'Important Limitations', color: '#f59e0b', body: 'This is a research prototype, not a clinical diagnostic tool. Results must not replace neurological assessment. The 96.49% accuracy applies only to D1-style sustained vowel recordings under record-level evaluation. Speaker-disjoint accuracy on D2 is 85.96%, and R² for UPDRS regression is negative — indicating the severity model does not generalize reliably to new subjects.' },
                ].map(s => (
                  <div key={s.title} className="glass-card p-24" style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
                    <div style={{ width: 40, height: 40, borderRadius: 10, background: `${s.color}20`, border: `1px solid ${s.color}44`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.2rem', flexShrink: 0 }}>{s.icon}</div>
                    <div>
                      <p style={{ fontWeight: 700, color: s.color, fontSize: '0.9rem', marginBottom: 6 }}>{s.title}</p>
                      <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.7 }}>{s.body}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
