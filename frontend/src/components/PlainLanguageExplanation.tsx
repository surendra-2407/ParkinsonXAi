import { useState, useEffect } from 'react'
import {
  generatePlainLanguageReport,
  type PlainWrittenReport,
  type AcousticObservation,
} from '../utils/plainLanguageExplainer'
import type { ShapFeature } from '../api/parkinsonApi'

interface PlainLanguageExplanationProps {
  label: 'Parkinson' | 'Healthy' | string
  confidence?: number
  shapFeatures?: ShapFeature[]
  audioFeatures?: Record<string, number>
  severityLevel?: string
  filename?: string
}

export default function PlainLanguageExplanation({
  label,
  confidence,
  shapFeatures,
  audioFeatures,
  severityLevel,
  filename,
}: PlainLanguageExplanationProps) {
  const [report, setReport] = useState<PlainWrittenReport>(() =>
    generatePlainLanguageReport({
      label,
      confidence,
      shapFeatures,
      audioFeatures,
      severityLevel,
    })
  )

  const [activeTab, setActiveTab] = useState<'overview' | 'voice' | 'neuro' | 'why' | 'next_steps'>('overview')
  const [isPlayingAudio, setIsPlayingAudio] = useState(false)
  const [copied, setCopied] = useState(false)

  // Re-generate report whenever inputs change
  useEffect(() => {
    setReport(
      generatePlainLanguageReport({
        label,
        confidence,
        shapFeatures,
        audioFeatures,
        severityLevel,
      })
    )
  }, [label, confidence, shapFeatures, audioFeatures, severityLevel])

  // Stop speech when component unmounts
  useEffect(() => {
    return () => {
      if (window.speechSynthesis) {
        window.speechSynthesis.cancel()
      }
    }
  }, [])

  const isParkinson = report.isParkinson
  const themeColor = isParkinson ? '#dc2626' : '#059669'
  const themeBg = isParkinson ? 'rgba(239, 68, 68, 0.06)' : 'rgba(16, 185, 129, 0.06)'
  const themeBorder = isParkinson ? 'rgba(239, 68, 68, 0.25)' : 'rgba(16, 185, 129, 0.25)'

  // Text-to-speech audio reader
  function handleToggleReadAloud() {
    if (!('speechSynthesis' in window)) {
      alert('Text-to-speech is not supported in this browser.')
      return
    }

    if (isPlayingAudio) {
      window.speechSynthesis.cancel()
      setIsPlayingAudio(false)
      return
    }

    window.speechSynthesis.cancel()

    // Build plain-spoken script without any numbers or symbols
    const speechScript = [
      `Detailed Voice Analysis Report for ${report.verdictHeading}.`,
      report.verdictSummary,
      ...report.executiveOverview,
      report.neurologicalContext.headline,
      report.neurologicalContext.explanation,
      ...report.neurologicalContext.keyBiologicalInsights,
      report.decisionReasoning.headline,
      report.decisionReasoning.narrative,
      report.practicalNextSteps.headline,
      report.practicalNextSteps.counselingMessage,
    ].join(' ')

    const utterance = new SpeechSynthesisUtterance(speechScript)
    utterance.rate = 0.95
    utterance.pitch = 1.0

    utterance.onend = () => setIsPlayingAudio(false)
    utterance.onerror = () => setIsPlayingAudio(false)

    window.speechSynthesis.speak(utterance)
    setIsPlayingAudio(true)
  }

  // Copy plain text report to clipboard
  function handleCopyReport() {
    const textLines = [
      `============================================================`,
      `PARKINSONXAI — PLAIN-LANGUAGE WRITTEN REPORT`,
      `Recording: ${filename || 'Voice Sample'}`,
      `Verdict: ${report.verdictHeading}`,
      `============================================================\n`,
      `EXECUTIVE SUMMARY:`,
      report.verdictSummary,
      '',
      ...report.executiveOverview,
      '',
      `WHAT WAS OBSERVED IN THE VOICE:`,
      ...report.vocalPillars.flatMap(p => [
        `• ${p.title} [${p.statusBadge}]`,
        `  Observation: ${p.whatWeObserved}`,
        `  Everyday Analogy: ${p.everydayAnalogy}`,
        `  Vocal Health Context: ${p.howItRelatesToVocalHealth}`,
        '',
      ]),
      `THE NEUROLOGICAL CONNECTION:`,
      report.neurologicalContext.explanation,
      ...report.neurologicalContext.keyBiologicalInsights.map(b => `• ${b}`),
      '',
      `WHY THE AI REACHED THIS DECISION:`,
      report.decisionReasoning.narrative,
      ...report.decisionReasoning.contributingObservations.map(c => `• ${c}`),
      report.decisionReasoning.conclusionText,
      '',
      `RECOMMENDED NEXT STEPS:`,
      report.practicalNextSteps.counselingMessage,
      ...report.practicalNextSteps.recommendedActions.map(a => `• ${a}`),
      '',
      `============================================================`,
      `Note: This is an AI-assisted screening assessment and not a standalone medical diagnosis.`,
    ].join('\n')

    navigator.clipboard.writeText(textLines).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2500)
    })
  }

  return (
    <div
      id="written-explanation-section"
      className="glass-card"
      style={{
        padding: '32px 36px',
        marginBottom: 28,
        border: `1.5px solid ${themeBorder}`,
        background: 'var(--bg-card)',
        borderRadius: 'var(--radius-xl)',
        boxShadow: '0 8px 30px rgba(0, 0, 0, 0.04)',
      }}
    >
      {/* ── Section Header ── */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          flexWrap: 'wrap',
          gap: 16,
          paddingBottom: 24,
          borderBottom: '1px solid var(--border)',
          marginBottom: 24,
        }}
      >
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 6,
                padding: '4px 12px',
                borderRadius: 999,
                fontSize: '0.72rem',
                fontWeight: 700,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                background: themeBg,
                color: themeColor,
                border: `1px solid ${themeBorder}`,
              }}
            >
              <span>{isParkinson ? '⚠️' : '✅'}</span>
              <span>Written Clinical Narrative</span>
            </span>
            <span
              style={{
                padding: '4px 10px',
                borderRadius: 999,
                fontSize: '0.7rem',
                fontWeight: 600,
                background: 'rgba(37, 99, 235, 0.08)',
                color: 'var(--primary)',
                border: '1px solid rgba(37, 99, 235, 0.2)',
              }}
            >
              Plain English (No Values)
            </span>
          </div>

          <h2 style={{ fontSize: '1.45rem', fontWeight: 800, color: 'var(--text-primary)', marginTop: 4 }}>
            Detailed Explanation: Why This Result Was Reached
          </h2>
          <p style={{ fontSize: '0.86rem', color: 'var(--text-secondary)', marginTop: 4, maxWidth: 680 }}>
            A plain-language, non-technical translation of the voice analysis. Written in conversational words so that
            patients, family members, and clinicians can clearly understand what was heard in the voice and why.
          </p>
        </div>

        {/* Action Controls */}
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          <button
            onClick={handleToggleReadAloud}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 8,
              padding: '8px 16px',
              borderRadius: 'var(--radius-md)',
              border: '1px solid var(--border)',
              background: isPlayingAudio ? '#fee2e2' : 'var(--bg-secondary)',
              color: isPlayingAudio ? '#dc2626' : 'var(--text-primary)',
              fontSize: '0.8rem',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.2s',
            }}
            title="Listen to written explanation read aloud"
          >
            <span>{isPlayingAudio ? '⏹️ Stop Reading' : '🔊 Listen (Read Aloud)'}</span>
          </button>

          <button
            onClick={handleCopyReport}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 8,
              padding: '8px 16px',
              borderRadius: 'var(--radius-md)',
              border: '1px solid var(--border)',
              background: copied ? 'rgba(16, 185, 129, 0.12)' : 'var(--bg-secondary)',
              color: copied ? '#059669' : 'var(--text-primary)',
              fontSize: '0.8rem',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.2s',
            }}
            title="Copy entire written narrative to clipboard"
          >
            <span>{copied ? '✓ Copied to Clipboard!' : '📋 Copy Written Summary'}</span>
          </button>
        </div>
      </div>

      {/* ── Key Verdict Card (Plain Language) ── */}
      <div
        style={{
          padding: '20px 24px',
          borderRadius: 'var(--radius-lg)',
          background: themeBg,
          border: `1.5px solid ${themeBorder}`,
          marginBottom: 24,
          display: 'flex',
          gap: 18,
          alignItems: 'flex-start',
        }}
      >
        <div
          style={{
            fontSize: '2rem',
            lineHeight: 1,
            padding: 12,
            borderRadius: '50%',
            background: 'white',
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.06)',
            flexShrink: 0,
          }}
        >
          {isParkinson ? '⚠️' : '✅'}
        </div>
        <div>
          <h3 style={{ fontSize: '1.15rem', fontWeight: 800, color: themeColor, marginBottom: 6 }}>
            {report.verdictHeading}
          </h3>
          <p style={{ fontSize: '0.92rem', color: 'var(--text-primary)', lineHeight: 1.6, fontWeight: 500 }}>
            {report.verdictSummary}
          </p>
        </div>
      </div>

      {/* ── Navigation Tabs ── */}
      <div
        style={{
          display: 'flex',
          gap: 8,
          overflowX: 'auto',
          paddingBottom: 12,
          marginBottom: 24,
          borderBottom: '1px solid var(--border)',
        }}
      >
        {[
          { id: 'overview', label: '📖 Plain-Language Summary', icon: '📝' },
          { id: 'voice', label: '🎙️ What We Heard in the Voice', icon: '🔍' },
          { id: 'neuro', label: '🧠 The Brain & Voice Link', icon: '🧬' },
          { id: 'why', label: '💡 Why the AI Decided This', icon: '⚖️' },
          { id: 'next_steps', label: '🩺 Recommended Next Steps', icon: '📋' },
        ].map(tab => {
          const isActive = activeTab === tab.id
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 8,
                padding: '9px 18px',
                borderRadius: 'var(--radius-md)',
                fontSize: '0.84rem',
                fontWeight: isActive ? 700 : 500,
                border: isActive ? '1px solid var(--primary)' : '1px solid transparent',
                background: isActive ? 'var(--primary-subtle)' : 'transparent',
                color: isActive ? 'var(--primary)' : 'var(--text-secondary)',
                cursor: 'pointer',
                whiteSpace: 'nowrap',
                transition: 'all 0.15s ease',
              }}
            >
              <span>{tab.label}</span>
            </button>
          )
        })}
      </div>

      {/* ── Tab Content ── */}
      <div style={{ minHeight: 300 }}>
        {/* Tab 1: Executive Overview */}
        {activeTab === 'overview' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            <div
              style={{
                padding: '24px 28px',
                background: 'var(--bg-secondary)',
                borderRadius: 'var(--radius-lg)',
                border: '1px solid var(--border)',
              }}
            >
              <h4
                style={{
                  fontSize: '1rem',
                  fontWeight: 700,
                  color: 'var(--text-primary)',
                  marginBottom: 16,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                }}
              >
                <span>📝</span> Executive Narrative Summary
              </h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                {report.executiveOverview.map((paragraph, idx) => (
                  <p
                    key={idx}
                    style={{
                      fontSize: '0.9rem',
                      lineHeight: 1.75,
                      color: 'var(--text-secondary)',
                      margin: 0,
                    }}
                  >
                    {paragraph}
                  </p>
                ))}
              </div>
            </div>

            {/* Quick summary box */}
            <div
              style={{
                padding: '20px 24px',
                borderRadius: 'var(--radius-lg)',
                background: 'rgba(37, 99, 235, 0.04)',
                border: '1px solid rgba(37, 99, 235, 0.18)',
                display: 'flex',
                gap: 16,
                alignItems: 'center',
              }}
            >
              <span style={{ fontSize: '1.8rem' }}>💡</span>
              <div style={{ flex: 1 }}>
                <p style={{ fontSize: '0.84rem', fontWeight: 700, color: 'var(--primary)', marginBottom: 4 }}>
                  Why This Written Explanation Matters
                </p>
                <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.6, margin: 0 }}>
                  Standard laboratory tests produce numbers and charts that only engineers or specialized clinicians
                  understand. This report translates those signals into concrete physical realities — such as vocal
                  cord flutter, breath control, and muscle agility — so you have full clarity over the findings.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Tab 2: What We Heard in the Voice */}
        {activeTab === 'voice' && (
          <div>
            <div style={{ marginBottom: 18 }}>
              <h4 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                Acoustic Observations (In Plain Everyday Terms)
              </h4>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: 2 }}>
                We examined four fundamental mechanics of vocal production. Here is how your voice behaved in each one:
              </p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
              {report.vocalPillars.map((pillar, idx) => (
                <ObservationCard key={idx} observation={pillar} />
              ))}
            </div>
          </div>
        )}

        {/* Tab 3: The Neurological Connection */}
        {activeTab === 'neuro' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            <div
              style={{
                padding: '24px 28px',
                background: 'var(--bg-secondary)',
                borderRadius: 'var(--radius-lg)',
                border: '1px solid var(--border)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
                <span style={{ fontSize: '1.6rem' }}>🧠</span>
                <div>
                  <h4 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                    {report.neurologicalContext.headline}
                  </h4>
                  <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                    How brain dopamine pathways directly influence vocal cord mechanics
                  </p>
                </div>
              </div>

              <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: 1.7, marginBottom: 20 }}>
                {report.neurologicalContext.explanation}
              </p>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {report.neurologicalContext.keyBiologicalInsights.map((insight, idx) => {
                  const [title, desc] = insight.split(': ')
                  return (
                    <div
                      key={idx}
                      style={{
                        padding: '14px 18px',
                        borderRadius: 'var(--radius-md)',
                        background: 'white',
                        border: '1px solid var(--border)',
                        display: 'flex',
                        gap: 12,
                        alignItems: 'flex-start',
                      }}
                    >
                      <span
                        style={{
                          width: 22,
                          height: 22,
                          borderRadius: '50%',
                          background: 'rgba(37, 99, 235, 0.1)',
                          color: 'var(--primary)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: '0.72rem',
                          fontWeight: 700,
                          flexShrink: 0,
                          marginTop: 2,
                        }}
                      >
                        {idx + 1}
                      </span>
                      <div>
                        <strong style={{ fontSize: '0.86rem', color: 'var(--text-primary)', display: 'block', marginBottom: 2 }}>
                          {title}
                        </strong>
                        <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                          {desc || title}
                        </span>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        )}

        {/* Tab 4: Why the AI Decided This */}
        {activeTab === 'why' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            <div
              style={{
                padding: '24px 28px',
                background: 'var(--bg-secondary)',
                borderRadius: 'var(--radius-lg)',
                border: '1px solid var(--border)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
                <span style={{ fontSize: '1.6rem' }}>💡</span>
                <div>
                  <h4 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                    {report.decisionReasoning.headline}
                  </h4>
                  <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                    Understanding the AI's pattern recognition process
                  </p>
                </div>
              </div>

              <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: 1.75, marginBottom: 20 }}>
                {report.decisionReasoning.narrative}
              </p>

              <div style={{ marginBottom: 20 }}>
                <p
                  style={{
                    fontSize: '0.75rem',
                    fontWeight: 700,
                    textTransform: 'uppercase',
                    letterSpacing: '0.08em',
                    color: 'var(--text-muted)',
                    marginBottom: 10,
                  }}
                >
                  Key Deciding Observations:
                </p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {report.decisionReasoning.contributingObservations.map((obs, idx) => (
                    <div
                      key={idx}
                      style={{
                        padding: '12px 16px',
                        background: 'white',
                        border: '1px solid var(--border)',
                        borderRadius: 'var(--radius-md)',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 12,
                      }}
                    >
                      <span style={{ color: isParkinson ? '#dc2626' : '#059669', fontSize: '1.1rem' }}>
                        {isParkinson ? '⚠️' : '✓'}
                      </span>
                      <span style={{ fontSize: '0.84rem', color: 'var(--text-primary)', lineHeight: 1.5 }}>
                        {obs}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <div
                style={{
                  padding: '16px 20px',
                  borderRadius: 'var(--radius-md)',
                  background: themeBg,
                  border: `1px solid ${themeBorder}`,
                }}
              >
                <p style={{ fontSize: '0.85rem', color: themeColor, fontWeight: 600, lineHeight: 1.6, margin: 0 }}>
                  {report.decisionReasoning.conclusionText}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Tab 5: Recommended Next Steps */}
        {activeTab === 'next_steps' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            <div
              style={{
                padding: '24px 28px',
                background: 'var(--bg-secondary)',
                borderRadius: 'var(--radius-lg)',
                border: '1px solid var(--border)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
                <span style={{ fontSize: '1.6rem' }}>🩺</span>
                <div>
                  <h4 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                    {report.practicalNextSteps.headline}
                  </h4>
                  <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                    Practical advice and proactive health suggestions
                  </p>
                </div>
              </div>

              <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: 1.7, marginBottom: 20 }}>
                {report.practicalNextSteps.counselingMessage}
              </p>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {report.practicalNextSteps.recommendedActions.map((action, idx) => {
                  const [title, desc] = action.split(': ')
                  return (
                    <div
                      key={idx}
                      style={{
                        padding: '14px 18px',
                        borderRadius: 'var(--radius-md)',
                        background: 'white',
                        border: '1px solid var(--border)',
                        display: 'flex',
                        gap: 12,
                        alignItems: 'flex-start',
                      }}
                    >
                      <span
                        style={{
                          width: 24,
                          height: 24,
                          borderRadius: '50%',
                          background: 'rgba(37, 99, 235, 0.1)',
                          color: 'var(--primary)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: '0.75rem',
                          fontWeight: 700,
                          flexShrink: 0,
                          marginTop: 1,
                        }}
                      >
                        {idx + 1}
                      </span>
                      <div>
                        <strong style={{ fontSize: '0.86rem', color: 'var(--text-primary)', display: 'block', marginBottom: 2 }}>
                          {title}
                        </strong>
                        <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                          {desc || title}
                        </span>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── Footer note / Compassionate reassurance ── */}
      <div
        style={{
          marginTop: 24,
          paddingTop: 18,
          borderTop: '1px solid var(--border)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 12,
          fontSize: '0.75rem',
          color: 'var(--text-muted)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span>🕊️</span>
          <span>
            This written narrative is designed to bring clarity, calm understanding, and transparency to AI healthcare assessments.
          </span>
        </div>
        <button
          onClick={() => window.print()}
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--primary)',
            fontSize: '0.75rem',
            fontWeight: 600,
            cursor: 'pointer',
            textDecoration: 'underline',
          }}
        >
          🖨️ Print or Save as PDF
        </button>
      </div>
    </div>
  )
}

function ObservationCard({ observation }: { observation: AcousticObservation }) {
  const isHealthy = observation.statusType === 'healthy'
  const badgeColor = isHealthy ? '#059669' : '#dc2626'
  const badgeBg = isHealthy ? 'rgba(16, 185, 129, 0.08)' : 'rgba(239, 68, 68, 0.08)'
  const badgeBorder = isHealthy ? 'rgba(16, 185, 129, 0.25)' : 'rgba(239, 68, 68, 0.25)'

  return (
    <div
      style={{
        padding: '20px',
        borderRadius: 'var(--radius-lg)',
        background: 'var(--bg-secondary)',
        border: '1px solid var(--border)',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        gap: 14,
      }}
    >
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: '1.4rem' }}>{observation.icon}</span>
            <h5 style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>
              {observation.title}
            </h5>
          </div>
        </div>

        <div style={{ marginBottom: 10 }}>
          <span
            style={{
              padding: '3px 8px',
              borderRadius: 999,
              fontSize: '0.68rem',
              fontWeight: 700,
              color: badgeColor,
              background: badgeBg,
              border: `1px solid ${badgeBorder}`,
              display: 'inline-block',
            }}
          >
            {observation.statusBadge}
          </span>
        </div>

        <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: 12 }}>
          {observation.whatWeObserved}
        </p>

        {/* Everyday analogy box */}
        <div
          style={{
            padding: '10px 12px',
            borderRadius: 'var(--radius-sm)',
            background: 'white',
            border: '1px solid var(--border)',
            fontSize: '0.78rem',
            color: 'var(--text-primary)',
            lineHeight: 1.55,
            marginBottom: 10,
          }}
        >
          <strong style={{ color: 'var(--primary)', display: 'block', fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 2 }}>
            💡 Everyday Analogy:
          </strong>
          {observation.everydayAnalogy}
        </div>
      </div>

      <div style={{ borderTop: '1px solid var(--border)', paddingTop: 8 }}>
        <p style={{ fontSize: '0.73rem', color: 'var(--text-muted)', lineHeight: 1.5, margin: 0 }}>
          <strong style={{ color: 'var(--text-secondary)' }}>Medical Context: </strong>
          {observation.howItRelatesToVocalHealth}
        </p>
      </div>
    </div>
  )
}
