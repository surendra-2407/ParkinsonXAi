import { useState } from 'react'

interface GlossaryItem {
  term: string
  icon: string
  tagline: string
  explanation: string
  clinical: string
  example?: string
}

const GLOSSARY: GlossaryItem[] = [
  {
    term: 'MFCC',
    icon: '🎵',
    tagline: 'Mel-Frequency Cepstral Coefficients',
    explanation:
      'MFCCs are a compact "fingerprint" of sound, inspired by how the human ear perceives audio. They capture the shape of the vocal tract — how your tongue, lips, and throat position themselves when speaking. The model uses 40 MFCC coefficients, each describing a different frequency band of your voice.',
    clinical:
      "In Parkinson's disease, muscle rigidity and bradykinesia (slowness of movement) affect the fine motor control of the articulators. This shows up as subtle changes across MFCC coefficients — the model has learned which patterns distinguish Parkinson's speech from healthy speech.",
    example: 'Think of MFCC like an equalizer display — each bar represents how much energy is in a frequency band.',
  },
  {
    term: 'Delta MFCC',
    icon: '📈',
    tagline: 'Rate of Change of MFCCs',
    explanation:
      'Delta MFCCs measure how fast the MFCC values change over time — essentially the "velocity" of the voice spectrum. Delta-delta (Δ²) measures the acceleration.',
    clinical:
      "Parkinson's dysarthria involves irregular, jerky movements of the vocal tract. Delta MFCCs capture this temporal instability — abnormal rates of change in vocal features reveal neuromotor control issues.",
  },
  {
    term: 'Jitter',
    icon: '〰️',
    tagline: 'Cycle-to-cycle pitch variation (vocal tremor)',
    explanation:
      'Jitter measures tiny irregularities in the length of each vocal fold vibration cycle. A healthy voice has very consistent vibration; a tremor-affected voice shows cycle-to-cycle variation.',
    clinical:
      'Elevated jitter is one of the most reliable acoustic markers of Parkinson\'s laryngeal tremor. Healthy speakers have jitter ≈ 0.4%; Parkinson\'s patients often exceed 1–3%.',
    example: 'Local Jitter, RAP, PPQ5 are different ways of measuring the same thing over different time windows.',
  },
  {
    term: 'Shimmer',
    icon: '📊',
    tagline: 'Cycle-to-cycle amplitude variation (hypophonia)',
    explanation:
      'Shimmer measures variations in the loudness of each vocal fold vibration cycle. A weak, breathy voice (hypophonia) shows high shimmer because the vocal folds aren\'t closing fully.',
    clinical:
      'Hypophonia — abnormally soft voice — is present in 70–90% of Parkinson\'s patients. High shimmer reflects incomplete glottal closure caused by vocal fold muscle weakness or rigidity.',
    example: 'If you imagine each glottal pulse as a drum hit, shimmer measures how inconsistent the hits are in volume.',
  },
  {
    term: 'HNR',
    icon: '🔊',
    tagline: 'Harmonics-to-Noise Ratio (voice clarity)',
    explanation:
      'HNR measures the ratio of the clean, periodic part of the voice (harmonics) to the noisy, aperiodic part. A clear, healthy voice has high HNR (>20 dB). A breathy or hoarse voice has low HNR.',
    clinical:
      'Low HNR indicates vocal fold irregularity and breathiness — signs of the neuromotor dysfunction seen in Parkinson\'s. HNR below 12 dB is clinically significant.',
    example: 'Think of HNR like signal-to-noise ratio (SNR) in audio engineering — higher is clearer.',
  },
  {
    term: 'SHAP Values',
    icon: '🔍',
    tagline: 'SHapley Additive exPlanations — WHY the model decided',
    explanation:
      'SHAP values explain how much each feature pushed the model\'s prediction toward Parkinson\'s (positive, red) or away from it (negative, teal). They come from game theory: each feature is treated like a player who gets credit/blame for the final prediction.',
    clinical:
      'A SHAP value of +0.05 for "jitter_local" means that this person\'s jitter measurement pushed the Parkinson\'s probability up by 5 percentage points compared to the average prediction. SHAP makes the "black box" model transparent.',
    example: 'Base prediction: 50%. jitter_local adds +12%, shimmer_local adds +8%, HNR subtracts -4% → Final: 66% Parkinson.',
  },
  {
    term: 'Voiced Fraction',
    icon: '🗣️',
    tagline: 'Proportion of speech that is voiced',
    explanation:
      'During speech, sounds can be voiced (vocal folds vibrate — e.g., vowels) or unvoiced (no vibration — e.g., "s", "f"). Voiced fraction is the percentage of the recording that contains voiced speech.',
    clinical:
      'Parkinson\'s patients often exhibit excessive breathiness and aperiodic voice breaks, leading to a lower voiced fraction than healthy speakers. This reflects the incomplete and irregular glottal closure caused by rigidity.',
  },
  {
    term: 'Spectral Features',
    icon: '📡',
    tagline: 'Distribution of energy across frequencies',
    explanation:
      'Spectral features (centroid, bandwidth, rolloff, flatness) describe the overall frequency content of the voice. Spectral centroid = "brightness", bandwidth = "spread" of frequencies, rolloff = frequency below which most energy sits.',
    clinical:
      'Changes in spectral features reflect alterations in vocal tract resonance caused by articulatory imprecision — the imprecise, "mumbled" quality of Parkinson\'s speech (hypokinetic dysarthria).',
  },
]

interface FeatureGlossaryProps {
  defaultOpen?: boolean
  highlight?: string  // feature name to highlight
}

export default function FeatureGlossary({ defaultOpen = false, highlight }: FeatureGlossaryProps) {
  const [open, setOpen] = useState(defaultOpen)
  const [activeIdx, setActiveIdx] = useState<number | null>(null)

  return (
    <div style={{
      border: '1px solid rgba(124,58,237,0.25)',
      borderRadius: 'var(--radius-md)',
      overflow: 'hidden',
      background: 'rgba(124,58,237,0.03)',
    }}>
      {/* Header / toggle */}
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: '100%', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          padding: '14px 20px', background: 'transparent', border: 'none', cursor: 'pointer',
          color: 'var(--text-primary)',
        }}
        id="glossary-toggle"
      >
        <span style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: '1rem' }}>📖</span>
          <span style={{ fontSize: '0.82rem', fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--purple-light)' }}>
            Feature Glossary — What Do These Terms Mean?
          </span>
        </span>
        <span style={{
          fontSize: '0.7rem', color: 'var(--text-muted)',
          transform: open ? 'rotate(180deg)' : 'rotate(0deg)',
          transition: 'transform 0.3s ease', display: 'inline-block',
        }}>▼</span>
      </button>

      {open && (
        <div style={{ padding: '0 20px 20px' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 10 }}>
            {GLOSSARY.map((item, idx) => {
              const isHighlighted = highlight && item.term.toLowerCase().includes(highlight.toLowerCase())
              const isActive = activeIdx === idx
              return (
                <div
                  key={item.term}
                  id={`glossary-${item.term.replace(/\s+/g, '-').toLowerCase()}`}
                  onClick={() => setActiveIdx(isActive ? null : idx)}
                  style={{
                    padding: '12px 14px',
                    background: isActive
                      ? 'rgba(124,58,237,0.12)'
                      : isHighlighted
                      ? 'rgba(124,58,237,0.08)'
                      : 'var(--bg-card)',
                    border: `1px solid ${isActive ? 'rgba(124,58,237,0.4)' : isHighlighted ? 'rgba(124,58,237,0.3)' : 'var(--border)'}`,
                    borderRadius: 'var(--radius-sm)',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10, marginBottom: isActive ? 10 : 0 }}>
                    <span style={{ fontSize: '1.1rem', flexShrink: 0 }}>{item.icon}</span>
                    <div style={{ flex: 1 }}>
                      <p style={{ fontWeight: 700, fontSize: '0.82rem', color: 'var(--text-primary)', marginBottom: 2 }}>
                        {item.term}
                      </p>
                      <p style={{ fontSize: '0.7rem', color: 'var(--purple-light)', lineHeight: 1.4 }}>
                        {item.tagline}
                      </p>
                    </div>
                    <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', flexShrink: 0 }}>
                      {isActive ? '▲' : '▼'}
                    </span>
                  </div>

                  {isActive && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                      <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                        {item.explanation}
                      </p>
                      <div style={{
                        padding: '8px 12px',
                        background: 'rgba(20,184,166,0.08)',
                        border: '1px solid rgba(20,184,166,0.2)',
                        borderRadius: 6,
                      }}>
                        <p style={{ fontSize: '0.68rem', fontWeight: 700, color: 'var(--teal-light)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                          🩺 Clinical Relevance
                        </p>
                        <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                          {item.clinical}
                        </p>
                      </div>
                      {item.example && (
                        <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontStyle: 'italic', lineHeight: 1.5 }}>
                          💡 {item.example}
                        </p>
                      )}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
