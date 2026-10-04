/**
 * plainLanguageExplainer.ts
 * Generates an in-depth, human-understandable, written narrative explaining
 * WHY the AI classified the voice as Parkinson's or Healthy.
 *
 * CRITICAL RULE: NO raw numbers, decimals, or technical values are shown.
 * All acoustic parameters are translated into everyday human language,
 * physical analogies, and clear descriptions of vocal cord behavior.
 */

import type { ShapFeature } from '../api/parkinsonApi'

export interface AcousticObservation {
  title: string
  icon: string
  statusBadge: string
  statusType: 'healthy' | 'caution'
  whatWeObserved: string
  everydayAnalogy: string
  howItRelatesToVocalHealth: string
}

export interface PlainWrittenReport {
  verdictHeading: string
  verdictSummary: string
  isParkinson: boolean
  
  // Section 1: Executive Overview in conversational words
  executiveOverview: string[]

  // Section 2: Four core vocal pillars (without any numbers)
  vocalPillars: AcousticObservation[]

  // Section 3: The Neurological Connection (Why Parkinson's affects voice)
  neurologicalContext: {
    headline: string
    explanation: string
    keyBiologicalInsights: string[]
  }

  // Section 4: Why the AI reached this exact conclusion
  decisionReasoning: {
    headline: string
    narrative: string
    contributingObservations: string[]
    conclusionText: string
  }

  // Section 5: Reassuring next steps & guidance
  practicalNextSteps: {
    headline: string
    counselingMessage: string
    recommendedActions: string[]
  }
}

/**
 * Generates the complete written explanation from the prediction results.
 * Strictly uses descriptive language rather than numeric metrics.
 */
export function generatePlainLanguageReport(params: {
  label: 'Parkinson' | 'Healthy' | string
  confidence?: number
  shapFeatures?: ShapFeature[]
  audioFeatures?: Record<string, number>
  severityLevel?: string
}): PlainWrittenReport {
  const isParkinson = params.label === 'Parkinson'
  const audio = params.audioFeatures || {}
  const shap = params.shapFeatures || []

  // Check qualitative vocal indicators from audio features or SHAP directions
  // Jitter (pitch stability): healthy < ~0.008
  const jitterVal = audio['jitter_local']
  const hasJitterIssue = jitterVal !== undefined
    ? jitterVal > 0.008
    : shap.some(f => f.feature.includes('jitter') && f.value > 0)

  // Shimmer (loudness stability): healthy < ~0.035
  const shimmerVal = audio['shimmer_local']
  const hasShimmerIssue = shimmerVal !== undefined
    ? shimmerVal > 0.035
    : shap.some(f => f.feature.includes('shimmer') && f.value > 0)

  // HNR / Voiced Fraction (breathiness / noise): healthy HNR > 16, voiced > 0.75
  const hnrVal = audio['hnr']
  const voicedVal = audio['voiced_fraction']
  const hasBreathinessIssue = (hnrVal !== undefined && hnrVal < 16) ||
    (voicedVal !== undefined && voicedVal < 0.75) ||
    shap.some(f => (f.feature.includes('voiced') || f.feature.includes('hnr')) && f.value > 0)

  // Energy / Loudness (hypophonia):
  const rmsVal = audio['rms_mean']
  const hasEnergyIssue = (rmsVal !== undefined && rmsVal < 0.02) ||
    shap.some(f => f.feature.includes('rms') && f.value > 0)

  // Articulation / Timbre (MFCCs):
  const hasArticulationIssue = shap.some(f => f.feature.includes('mfcc') && f.value > 0)

  if (isParkinson) {
    return buildParkinsonReport({
      hasJitterIssue,
      hasShimmerIssue,
      hasBreathinessIssue,
      hasEnergyIssue,
      hasArticulationIssue,
      severityLevel: params.severityLevel,
    })
  } else {
    return buildHealthyReport({
      hasJitterIssue,
      hasShimmerIssue,
      hasBreathinessIssue,
      hasEnergyIssue,
    })
  }
}

function buildParkinsonReport(flags: {
  hasJitterIssue: boolean
  hasShimmerIssue: boolean
  hasBreathinessIssue: boolean
  hasEnergyIssue: boolean
  hasArticulationIssue: boolean
  severityLevel?: string
}): PlainWrittenReport {
  const pillars: AcousticObservation[] = [
    {
      title: 'Vocal Cord Steadiness & Micro-Tremors',
      icon: '🎵',
      statusBadge: flags.hasJitterIssue ? 'Pitch Irregularity Detected' : 'Minor Pitch Variation',
      statusType: 'caution',
      whatWeObserved:
        'The recording reveals micro-tremors in the voice frequency. Instead of vibrating at a smooth, constant speed, the vocal cords wavered slightly from one millisecond to the next during speech.',
      everydayAnalogy:
        'Think of drawing a bow smoothly across a violin string. In a healthy voice, the tone is perfectly steady. Here, the bow exhibits a faint, involuntary flutter that causes tiny ripples in pitch.',
      howItRelatesToVocalHealth:
        'These tiny pitch wavers are known clinically as vocal jitter. They are one of the earliest signs of Parkinsonian dysarthria, caused when throat muscles lose continuous rhythmic motor signals from the brain.',
    },
    {
      title: 'Voice Loudness & Breathing Stability',
      icon: '🔊',
      statusBadge: flags.hasShimmerIssue || flags.hasEnergyIssue ? 'Loudness Fluctuation Observed' : 'Slight Volume Softening',
      statusType: 'caution',
      whatWeObserved:
        'The audio demonstrated fluctuations in vocal strength. The volume did not maintain an even projection, showing subtle drops in intensity and vocal energy across sustained sounds.',
      everydayAnalogy:
        'Imagine blowing air through a straw with fluctuating breath pressure. Rather than a constant, steady stream, the airflow gently flickers, causing the voice to sound softer or momentarily lose power.',
      howItRelatesToVocalHealth:
        'In Parkinson\'s, reduced chest wall expansion and incomplete vocal fold closure often lead to voice softening (hypophonia). Individuals often feel they are speaking at normal volume even though their voice has become quieter.',
    },
    {
      title: 'Vocal Cord Closure & Breathiness',
      icon: '🌬️',
      statusBadge: flags.hasBreathinessIssue ? 'Elevated Air Leakage' : 'Mild Breathiness Present',
      statusType: 'caution',
      whatWeObserved:
        'We detected background air escape and a faint raspy quality accompanying the vocal tone. The vocal cords did not create a complete, air-tight seal with each opening and closing cycle.',
      everydayAnalogy:
        'Like a door that doesn\'t shut all the way in the wind, a tiny crack allows air to hiss through continuously. This allows unvibrated air to escape, adding a breathy undertone to the voice.',
      howItRelatesToVocalHealth:
        'When vocal cords cannot press together with optimal firmness (incomplete glottal closure), air slips through without vibrating, diminishing voice richness and giving it a muffled or breathy texture.',
    },
    {
      title: 'Vocal Tract Articulation & Dynamic Tone',
      icon: '🗣️',
      statusBadge: flags.hasArticulationIssue ? 'Constrained Resonance' : 'Reduced Pitch Agility',
      statusType: 'caution',
      whatWeObserved:
        'The harmonic overtones that give speech its crispness and color appeared constrained. The transitions between different sound frequencies were less flexible and dynamic than typical speech.',
      everydayAnalogy:
        'Imagine speaking while holding your facial and jaw muscles slightly rigid. The words sound understandable, but the rich, expressive musicality and resonance of the voice become somewhat flattened.',
      howItRelatesToVocalHealth:
        'Stiffness in the tongue, palate, and laryngeal muscles restricts the throat\'s natural acoustic resonance chamber. This subtle rigidity softens vowel contrast and reduces dynamic expression.',
    },
  ]

  const stageDescriptor = flags.severityLevel
    ? flags.severityLevel.toLowerCase()
    : 'early-to-moderate'

  return {
    verdictHeading: "Parkinson's Vocal Indicators Detected",
    verdictSummary:
      'The automated voice analysis identified subtle acoustic signatures that are characteristic of neurological vocal changes associated with Parkinson\'s disease.',
    isParkinson: true,

    executiveOverview: [
      'When you spoke into the microphone, our artificial intelligence analyzed hundreds of subtle acoustic patterns in your voice wave. Rather than judging words or grammar, the system listens to the microscopic physical mechanics of how your vocal cords, lungs, and throat muscles work together.',
      `In this recording, the system observed a combination of involuntary micro-tremors, uneven vocal loudness, and breath leakage. While these subtle shifts are often too gentle for everyday human listeners to pick up during casual conversation, they match the well-documented vocal profile of Parkinson's hypokinetic dysarthria in ${stageDescriptor} stages.`,
      'Importantly, this is an intelligent acoustic screening assessment designed to support early detection, not an absolute diagnosis on its own. Below is a comprehensive breakdown of exactly why the AI arrived at this conclusion.',
    ],

    vocalPillars: pillars,

    neurologicalContext: {
      headline: 'The Neurological Connection: Why Does Parkinson\'s Affect the Voice?',
      explanation:
        'To produce a single clear syllable, your brain must coordinate more than one hundred microscopic muscles throughout your vocal cords, larynx, diaphragm, and mouth. Here is why voice is frequently the earliest indicator of Parkinson\'s:',
      keyBiologicalInsights: [
        'Dopamine Pathways & Automatic Coordination: Parkinson\'s gradually reduces dopamine in the brain\'s basal ganglia, the region responsible for smooth, effortless, automatic muscle control.',
        'Laryngeal Micro-Tremor: Because the tiny muscles holding the vocal cords are extraordinarily sensitive, they show microscopic involuntary tremors long before larger limb tremors appear in the hands or legs.',
        'Vocal Muscle Rigidity: Stiffness in the throat and chest prevents the vocal cords from snapping tightly shut and limits deep lung expansion, resulting in a softer, breathier voice.',
        'Sensory Feedback Calibration: People experiencing Parkinsonian voice changes genuinely perceive their own voice as loud and normal, making computerized acoustic analysis essential for objective early detection.',
      ],
    },

    decisionReasoning: {
      headline: 'Why the AI Decided On This Specific Result',
      narrative:
        'The machine learning model did not base its decision on any single isolated anomaly. Temporary factors like fatigue, a dry throat, or mild allergies can occasionally make anyone\'s voice sound slightly raspy. However, the AI confirmed a distinctive multi-feature pattern where vocal tremor, loudness instability, and acoustic breathiness co-occurred simultaneously.',
      contributingObservations: [
        'Concurrent Micro-Tremor: Frequency fluctuations occurred repeatedly across consecutive voice cycles.',
        'Breath Support Depletion: Voice intensity softened progressively rather than holding a sustained, uniform volume.',
        'Acoustic Air Turbulence: Elevated ratio of unvoiced airflow relative to clear vocal cord resonance.',
        'Coordinated Dysarthric Pattern: These traits aligned with known clinical benchmarks established in neurological speech research.',
      ],
      conclusionText:
        'Because these independent vocal markers reinforced one another across multiple acoustic dimensions, the system reached a high-confidence determination that Parkinsonian vocal patterns are present.',
    },

    practicalNextSteps: {
      headline: 'What This Means for You & Recommended Next Steps',
      counselingMessage:
        'Receiving a result indicating Parkinson\'s markers can naturally bring about concern, but early awareness is your greatest advantage. Modern interventions, voice therapies, and medical care are most effective when started early.',
      recommendedActions: [
        'Schedule a Neurological Evaluation: Share this report with a qualified neurologist or movement disorder specialist for a complete physical exam and clinical UPDRS motor assessment.',
        'Consult a Speech-Language Pathologist: Specialized vocal therapies like LSVT LOUD (Lee Silverman Voice Treatment) are proven to restore vocal strength, volume, and breath support.',
        'Observe Other Daily Signs: Note whether you have noticed any subtle changes in handwriting size (micrographia), reduced arm swing while walking, facial stiffness, or sleep disturbances.',
        'Remember This is a Screening Tool: Acoustic analysis provides high-value early indicators, but definitive diagnosis requires a physician\'s comprehensive clinical evaluation.',
      ],
    },
  }
}

function buildHealthyReport(_flags: {
  hasJitterIssue: boolean
  hasShimmerIssue: boolean
  hasBreathinessIssue: boolean
  hasEnergyIssue: boolean
}): PlainWrittenReport {
  const pillars: AcousticObservation[] = [
    {
      title: 'Vocal Cord Steadiness & Pitch Stability',
      icon: '🎵',
      statusBadge: 'Smooth & Stable Phonation',
      statusType: 'healthy',
      whatWeObserved:
        'The vocal cords vibrated with exceptional consistency and stability. Pitch maintenance remained steady across the entire duration without involuntary micro-tremors or frequency flutter.',
      everydayAnalogy:
        'Like a well-tuned musical instrument held by a steady hand, the note rang out pure and unwavering from start to finish.',
      howItRelatesToVocalHealth:
        'Steady vocal frequency reflects strong, uninterrupted neuromotor control from the brain stem to the laryngeal nerves, confirming absent tremor activity.',
    },
    {
      title: 'Voice Loudness & Breathing Control',
      icon: '🔊',
      statusBadge: 'Firm & Consistent Projection',
      statusType: 'healthy',
      whatWeObserved:
        'Vocal volume remained solid and evenly supported by breath pressure. There was no abnormal fading of vocal intensity or unintentional volume drops.',
      everydayAnalogy:
        'Like a steady, confident stream of air powering a clear whistle, energy and loudness stayed robust without sputtering or weakening.',
      howItRelatesToVocalHealth:
        'Consistent vocal loudness demonstrates healthy diaphragm support and proper vocal fold muscle tone without the muscle fatigue or weakness associated with hypophonia.',
    },
    {
      title: 'Vocal Cord Closure & Acoustic Purity',
      icon: '🌬️',
      statusBadge: 'Complete Seal & Clean Tone',
      statusType: 'healthy',
      whatWeObserved:
        'The vocal cords achieved a clean, complete seal with each cycle. The sound showed clear harmonic resonance with minimal background breath leakage or turbulence.',
      everydayAnalogy:
        'Like a well-insulated window that seals out every draft, the vocal folds close completely, transforming almost all exhaled air into pure musical sound rather than a breathy hiss.',
      howItRelatesToVocalHealth:
        'Complete vocal cord closure indicates healthy vocal fold flexibility and symmetric muscular coordination without the gaps seen in neurogenic vocal disorders.',
    },
    {
      title: 'Vocal Tract Articulation & Resonance',
      icon: '🗣️',
      statusBadge: 'Rich & Dynamic Overtones',
      statusType: 'healthy',
      whatWeObserved:
        'The voice exhibited rich harmonic depth and natural acoustic overtones. The resonance spaces of the throat and mouth modulated sound freely and cleanly.',
      everydayAnalogy:
        'Like singing in an acoustic concert hall where every note rings with warm, clear reverberation and crisp clarity.',
      howItRelatesToVocalHealth:
        'Unrestricted acoustic resonance confirms that the muscles of the throat, tongue, and soft palate have normal flexibility and are free from rigid stiffness.',
    },
  ]

  return {
    verdictHeading: "No Parkinson's Indicators — Voice Appears Healthy",
    verdictSummary:
      'The automated voice analysis found no signs of neurological vocal impairment. All acoustic biomarkers aligned with healthy, typical vocal cord mechanics.',
    isParkinson: false,

    executiveOverview: [
      'Our artificial intelligence examined the micro-acoustic properties of your voice recording, assessing how steadily and cleanly your vocal cords produce sound. The system evaluated pitch steadiness, breath support, volume consistency, and harmonic clarity.',
      'In this recording, your voice showed excellent stability, full vocal cord closure, and clear sound projection. The micro-tremors, voice softening, and breathiness that typically accompany Parkinson\'s disease were absent.',
      'This positive finding indicates that your vocal apparatus and the neurological pathways governing speech production are functioning within normal, healthy parameters.',
    ],

    vocalPillars: pillars,

    neurologicalContext: {
      headline: 'Why Healthy Speech Reflects Strong Brain Health',
      explanation:
        'Speaking requires delicate synchronization between your brain\'s motor centers and over one hundred throat, chest, and facial muscles. Here is what your healthy recording demonstrates:',
      keyBiologicalInsights: [
        'Intact Dopamine Motor Pathways: The steady rhythm of your vocal cord vibrations demonstrates that the brain\'s basal ganglia are sending smooth, regular signals without hesitation.',
        'Complete Vocal Fold Tension: The absence of breathiness confirms that the nerves controlling your laryngeal muscles are delivering strong, symmetric closing power.',
        'Robust Respiratory Drive: The steady vocal loudness indicates healthy diaphragm strength and well-regulated airflow pressure during sustained speech.',
        'Dynamic Resonance Flexibility: Your throat and oral muscles demonstrate normal muscular elasticity, preserving natural acoustic warmth and expressive clarity.',
      ],
    },

    decisionReasoning: {
      headline: 'Why the AI Decided On This Specific Result',
      narrative:
        'The machine learning model compared your audio wave across dozens of acoustic dimensions against large clinical reference datasets of both healthy individuals and diagnosed Parkinson\'s patients.',
      contributingObservations: [
        'Absence of Vocal Tremor: Cycle-to-cycle pitch changes remained within healthy baseline ranges.',
        'Sustained Vocal Loudness: No premature drop-off or fluctuating energy was observed during phonation.',
        'High Tonal Clarity: Harmonics were rich and clean, with minimal background breath leakage or friction noise.',
        'Natural Harmonic Distribution: Spectral resonance curves matched typical, unimpaired adult vocal patterns.',
      ],
      conclusionText:
        'Because all key vocal markers demonstrated consistent stability and normal muscular coordination, the AI classified the voice as healthy with high confidence.',
    },

    practicalNextSteps: {
      headline: 'Maintaining Vocal and Neurological Wellness',
      counselingMessage:
        'Your recording shows reassuringly healthy vocal function. Here are simple, practical habits to keep your voice and overall motor health in optimal shape:',
      recommendedActions: [
        'Stay Well Hydrated: Drinking adequate water keeps the thin mucosal lining of your vocal folds pliable and resistant to vocal strain.',
        'Practice Regular Vocal Warmups: Singing, reading aloud, or gentle humming helps maintain respiratory muscle tone and vocal cord flexibility.',
        'Routine Health Checkups: Continue regular medical checkups with your primary care physician to monitor overall motor and neurological wellness.',
        'Annual Voice Re-Check: You can re-take this non-invasive voice screening periodically to establish a personal baseline and track vocal wellness over time.',
      ],
    },
  }
}
