import type { ClinicalSummary, ClinicalFinding } from '../utils/clinicalExplainer'

interface ClinicalExplanationProps {
  summary: ClinicalSummary
}

export default function ClinicalExplanation({ summary }: ClinicalExplanationProps) {
  const isParkinson = summary.verdict === 'Parkinson'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

      {/* Headline verdict */}
      <div style={{
        padding: '20px 24px',
        borderRadius: 'var(--radius-md)',
        background: isParkinson
          ? 'linear-gradient(135deg, rgba(239,68,68,0.08), rgba(239,68,68,0.03))'
          : 'linear-gradient(135deg, rgba(16,185,129,0.08), rgba(16,185,129,0.03))',
        border: `1px solid ${isParkinson ? 'rgba(239,68,68,0.25)' : 'rgba(16,185,129,0.25)'}`,
      }}>
        <p style={{
          fontSize: '1rem', fontWeight: 700,
          color: isParkinson ? '#f87171' : '#34d399',
          marginBottom: 12,
        }}>
          {isParkinson ? '⚠️' : '✅'} {summary.headline}
        </p>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.7 }}>
          {summary.explanation}
        </p>
      </div>

      {/* Why Parkinson / Why Not — two column indicator lists */}
      {(summary.positive_signs.length > 0 || summary.negative_signs.length > 0) && (
        <div className="grid-2" style={{ gap: 14 }}>
          {/* Risk indicators */}
          <div style={{
            padding: 16,
            background: 'rgba(239,68,68,0.06)',
            border: '1px solid rgba(239,68,68,0.2)',
            borderRadius: 'var(--radius-sm)',
          }}>
            <p style={{
              fontSize: '0.68rem', fontWeight: 700, letterSpacing: '0.1em',
              textTransform: 'uppercase', color: '#f87171', marginBottom: 10,
            }}>
              🔴 Indicators Pointing To Parkinson's
            </p>
            {summary.positive_signs.length > 0 ? (
              <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 6 }}>
                {summary.positive_signs.map(s => (
                  <li key={s} style={{
                    fontSize: '0.8rem', color: 'var(--text-secondary)',
                    display: 'flex', alignItems: 'center', gap: 8,
                  }}>
                    <span style={{
                      width: 6, height: 6, borderRadius: '50%',
                      background: '#ef4444', flexShrink: 0,
                    }} />
                    {s}
                  </li>
                ))}
              </ul>
            ) : (
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>None detected</p>
            )}
          </div>

          {/* Protective indicators */}
          <div style={{
            padding: 16,
            background: 'rgba(16,185,129,0.06)',
            border: '1px solid rgba(16,185,129,0.2)',
            borderRadius: 'var(--radius-sm)',
          }}>
            <p style={{
              fontSize: '0.68rem', fontWeight: 700, letterSpacing: '0.1em',
              textTransform: 'uppercase', color: '#34d399', marginBottom: 10,
            }}>
              🟢 Indicators Against Parkinson's
            </p>
            {summary.negative_signs.length > 0 ? (
              <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 6 }}>
                {summary.negative_signs.map(s => (
                  <li key={s} style={{
                    fontSize: '0.8rem', color: 'var(--text-secondary)',
                    display: 'flex', alignItems: 'center', gap: 8,
                  }}>
                    <span style={{
                      width: 6, height: 6, borderRadius: '50%',
                      background: '#10b981', flexShrink: 0,
                    }} />
                    {s}
                  </li>
                ))}
              </ul>
            ) : (
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>None detected</p>
            )}
          </div>
        </div>
      )}

      {/* Key findings with symptom cards */}
      {summary.key_findings.length > 0 && (
        <div>
          <p style={{
            fontSize: '0.68rem', fontWeight: 700, letterSpacing: '0.1em',
            textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: 12,
          }}>
            Key Clinical Findings
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {summary.key_findings.map(finding => (
              <FindingCard key={finding.feature} finding={finding} />
            ))}
          </div>
        </div>
      )}

      {/* Recommendation */}
      <div style={{
        padding: '14px 18px',
        background: 'rgba(124,58,237,0.06)',
        border: '1px solid rgba(124,58,237,0.2)',
        borderRadius: 'var(--radius-sm)',
        display: 'flex', gap: 12, alignItems: 'flex-start',
      }}>
        <span style={{ fontSize: '1.2rem', flexShrink: 0 }}>🩺</span>
        <div>
          <p style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--purple-light)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
            Clinical Recommendation
          </p>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            {summary.recommendation}
          </p>
          <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 8, fontStyle: 'italic' }}>
            ⚠️ This AI analysis is for research purposes only and does not constitute medical advice.
          </p>
        </div>
      </div>
    </div>
  )
}

function FindingCard({ finding }: { finding: ClinicalFinding }) {
  const isRisk = finding.direction === 'increases_risk'
  const severityColor = finding.severity === 'strong'
    ? (isRisk ? '#ef4444' : '#10b981')
    : finding.severity === 'moderate'
    ? (isRisk ? '#f97316' : '#34d399')
    : (isRisk ? '#fbbf24' : '#6ee7b7')

  const severityBarW = finding.severity === 'strong' ? '100%'
    : finding.severity === 'moderate' ? '60%' : '30%'

  return (
    <div style={{
      padding: '12px 16px',
      background: 'var(--bg-card)',
      border: '1px solid var(--border)',
      borderLeft: `3px solid ${severityColor}`,
      borderRadius: 'var(--radius-sm)',
    }}>
      <div className="flex justify-between items-center" style={{ marginBottom: 6 }}>
        <div className="flex items-center gap-8">
          <span style={{
            fontSize: '0.68rem', fontWeight: 700, padding: '2px 8px',
            borderRadius: 999, textTransform: 'uppercase', letterSpacing: '0.08em',
            background: isRisk ? 'rgba(239,68,68,0.12)' : 'rgba(16,185,129,0.12)',
            color: isRisk ? '#f87171' : '#34d399',
          }}>
            {isRisk ? '↑ Raises Risk' : '↓ Lowers Risk'}
          </span>
          <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)' }}>
            {finding.symptom}
          </span>
        </div>
        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'monospace' }}>
          {finding.shap_value > 0 ? '+' : ''}{finding.shap_value.toFixed(4)}
        </span>
      </div>

      <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: 1.5, marginBottom: 8 }}>
        <span style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>{finding.clinical_name}: </span>
        {finding.description}
      </p>

      {/* Severity bar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', width: 50 }}>
          {finding.severity}
        </span>
        <div style={{
          flex: 1, height: 3, background: 'rgba(255,255,255,0.06)',
          borderRadius: 2, overflow: 'hidden',
        }}>
          <div style={{
            height: '100%', width: severityBarW,
            background: severityColor,
            transition: 'width 0.8s ease',
          }} />
        </div>
      </div>
    </div>
  )
}
