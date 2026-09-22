/**
 * clinicalExplainer.ts
 * Maps SHAP feature names to clinical Parkinson's symptom descriptions.
 * Generates plain-English explanations for the results page.
 */

import type { ShapFeature } from '../api/parkinsonApi'

// ── Feature → Clinical Symptom Mapping ─────────────────────────────────────

interface FeatureInfo {
  symptom: string
  description: string
  parkinsonian_direction: 'high' | 'low' // high value = more Parkinsonian
  clinical_name: string
}

export type { FeatureInfo }
export const FEATURE_TOOLTIP_MAP: Record<string, FeatureInfo> = {
  // Jitter — pitch micro-variations (tremor marker)
  jitter_local: {
    symptom: 'Vocal Tremor',
    clinical_name: 'Local Jitter',
    description: 'Cycle-to-cycle variation in pitch. Elevated jitter reflects laryngeal tremor, a hallmark of Parkinson\'s dysarthria.',
    parkinsonian_direction: 'high',
  },
  jitter_ppq5: {
    symptom: 'Vocal Tremor',
    clinical_name: 'PPQ5 Jitter',
    description: '5-point period perturbation quotient — measures pitch instability over longer time windows.',
    parkinsonian_direction: 'high',
  },

  // Shimmer — amplitude variations (voice weakness)
  shimmer_local: {
    symptom: 'Voice Weakness (Hypophonia)',
    clinical_name: 'Local Shimmer',
    description: 'Cycle-to-cycle amplitude variation. High shimmer indicates incomplete glottal closure, causing weak or breathy voice.',
    parkinsonian_direction: 'high',
  },
  shimmer_apq11: {
    symptom: 'Voice Weakness (Hypophonia)',
    clinical_name: 'APQ11 Shimmer',
    description: '11-point amplitude perturbation — sustained amplitude irregularity associated with vocal fold dysfunction.',
    parkinsonian_direction: 'high',
  },

  // Fundamental frequency (pitch)
  f0_mean: {
    symptom: 'Pitch Control',
    clinical_name: 'Mean F0',
    description: 'Average fundamental frequency (vocal pitch). Parkinson\'s patients often show reduced or abnormal pitch due to muscle rigidity.',
    parkinsonian_direction: 'low',
  },
  praat_f0_mean: {
    symptom: 'Pitch Control',
    clinical_name: 'Praat Mean F0',
    description: 'Fundamental frequency measured via autocorrelation. Reduced pitch range is a common Parkinson\'s speech symptom.',
    parkinsonian_direction: 'low',
  },
  f0_std: {
    symptom: 'Pitch Variability',
    clinical_name: 'F0 Std Dev',
    description: 'Standard deviation of pitch. Both very high (tremor) and very low (monotone) variability can indicate neurological voice disorder.',
    parkinsonian_direction: 'high',
  },
  praat_f0_std: {
    symptom: 'Pitch Variability',
    clinical_name: 'Praat F0 Std',
    description: 'Pitch variability measured via Praat. Monotonous speech (low variability) is a classic Parkinson\'s symptom.',
    parkinsonian_direction: 'high',
  },

  // Voiced fraction (breathiness)
  voiced_fraction: {
    symptom: 'Breathiness / Aperiodicity',
    clinical_name: 'Voiced Fraction',
    description: 'Proportion of speech that is voiced. Low voiced fraction reflects excessive breathiness, a key symptom of Parkinson\'s hypokinetic dysarthria.',
    parkinsonian_direction: 'low',
  },

  // Energy / RMS
  rms_mean: {
    symptom: 'Voice Loudness (Hypophonia)',
    clinical_name: 'RMS Energy',
    description: 'Root-mean-square energy (loudness). Reduced loudness (hypophonia) is one of the earliest and most common Parkinson\'s speech symptoms.',
    parkinsonian_direction: 'low',
  },
  rms_std: {
    symptom: 'Loudness Variability',
    clinical_name: 'RMS Variability',
    description: 'Variability in loudness over time. Irregular loudness may reflect respiratory control difficulties in Parkinson\'s.',
    parkinsonian_direction: 'high',
  },

  // ZCR
  zcr_mean: {
    symptom: 'Breathiness / Noise',
    clinical_name: 'Zero-Crossing Rate',
    description: 'Measures signal noise and breathiness. Elevated ZCR can indicate noisy, aperiodic vocal fold vibration.',
    parkinsonian_direction: 'high',
  },

  // MFCCs — articulation
  mfcc_1_mean: {
    symptom: 'Articulation / Vocal Tract',
    clinical_name: 'MFCC-1',
    description: 'Captures overall vocal tract energy. Changes reflect hypernasality and articulatory imprecision in Parkinson\'s dysarthria.',
    parkinsonian_direction: 'high',
  },
  mfcc_2_mean: {
    symptom: 'Articulation / Vocal Tract',
    clinical_name: 'MFCC-2',
    description: 'Second cepstral coefficient — related to front/back tongue position. Altered by Parkinson\'s muscle rigidity.',
    parkinsonian_direction: 'high',
  },

  // Spectral features
  spectral_rolloff_mean: {
    symptom: 'Spectral Quality',
    clinical_name: 'Spectral Rolloff',
    description: 'Frequency below which 85% of spectral energy is concentrated. Shifts in rolloff indicate changes in voice quality and breathiness.',
    parkinsonian_direction: 'high',
  },
  spectral_bandwidth_std: {
    symptom: 'Spectral Variability',
    clinical_name: 'Spectral Bandwidth',
    description: 'Variability of the spectral bandwidth. Reflects instability in voice quality over time.',
    parkinsonian_direction: 'high',
  },

  // Chroma (harmonics)
  mel_mean: {
    symptom: 'Vocal Quality',
    clinical_name: 'Mel Energy',
    description: 'Average mel-scale energy. Changes in mel energy distribution reflect overall voice quality degradation.',
    parkinsonian_direction: 'high',
  },
}

// Keep internal alias for backward compat
const FEATURE_MAP = FEATURE_TOOLTIP_MAP

function genericInfo(featureName: string): FeatureInfo {
  const isJitter = featureName.includes('jitter')
  const isShimmer = featureName.includes('shimmer')
  const isMfcc = featureName.includes('mfcc')
  const isChroma = featureName.includes('chroma')
  const isDelta = featureName.includes('delta')

  if (isJitter) return { ...FEATURE_MAP['jitter_local'], clinical_name: featureName }
  if (isShimmer) return { ...FEATURE_MAP['shimmer_local'], clinical_name: featureName }
  if (isMfcc) return { ...FEATURE_MAP['mfcc_1_mean'], clinical_name: featureName }
  if (isChroma) return {
    symptom: 'Harmonic Structure',
    clinical_name: featureName,
    description: 'Chromatic energy — reflects harmonicity of the voice signal.',
    parkinsonian_direction: 'high',
  }
  if (isDelta) return {
    symptom: 'Temporal Dynamics',
    clinical_name: featureName,
    description: 'Rate of change of vocal features over time. Altered dynamics reflect neuromotor control issues.',
    parkinsonian_direction: 'high',
  }

  return {
    symptom: 'Acoustic Feature',
    clinical_name: featureName,
    description: 'Acoustic property of the voice signal used by the model for classification.',
    parkinsonian_direction: 'high',
  }
}

// ── Main Explainer Functions ────────────────────────────────────────────────

export interface ClinicalFinding {
  symptom: string
  clinical_name: string
  feature: string
  shap_value: number
  direction: 'increases_risk' | 'decreases_risk'
  description: string
  severity: 'mild' | 'moderate' | 'strong'
}

export interface ClinicalSummary {
  verdict: 'Parkinson' | 'Healthy'
  confidence: number
  headline: string
  explanation: string
  key_findings: ClinicalFinding[]
  positive_signs: string[]   // signs pointing TO Parkinson
  negative_signs: string[]   // signs pointing AWAY from Parkinson
  recommendation: string
}

export function buildClinicalSummary(
  label: 'Parkinson' | 'Healthy' | string,
  confidence: number,
  shapFeatures: ShapFeature[],
  motorUpdrs: number,
  _severityLevel: string,
): ClinicalSummary {
  const isParkinson = label === 'Parkinson'
  const confPct = (confidence * 100).toFixed(1)

  // Map SHAP features to clinical findings
  const findings: ClinicalFinding[] = shapFeatures
    .filter(f => Math.abs(f.value) > 0.0001)
    .slice(0, 10)
    .map(f => {
      const info = FEATURE_MAP[f.feature] ?? genericInfo(f.feature)
      const raisesRisk = f.value > 0
      const absVal = Math.abs(f.value)
      return {
        symptom: info.symptom,
        clinical_name: info.clinical_name,
        feature: f.feature,
        shap_value: f.value,
        direction: raisesRisk ? 'increases_risk' : 'decreases_risk',
        description: info.description,
        severity: absVal > 0.05 ? 'strong' : absVal > 0.02 ? 'moderate' : 'mild',
      }
    })

  const positive_signs = findings
    .filter(f => f.direction === 'increases_risk')
    .map(f => `${f.symptom} (${f.clinical_name})`)
    .filter((v, i, a) => a.indexOf(v) === i)  // unique

  const negative_signs = findings
    .filter(f => f.direction === 'decreases_risk')
    .map(f => `${f.symptom} (${f.clinical_name})`)
    .filter((v, i, a) => a.indexOf(v) === i)

  const headline = isParkinson
    ? `Parkinson's indicators detected with ${confPct}% confidence`
    : `No significant Parkinson's indicators — ${confPct}% confidence`

  const explanation = isParkinson
    ? buildParkinsonExplanation(findings, motorUpdrs, _severityLevel, confidence)
    : buildHealthyExplanation(findings, confidence)

  const recommendation = isParkinson
    ? `The acoustic analysis reveals vocal biomarkers consistent with Parkinson's hypokinetic dysarthria. This result should be reviewed by a neurologist or movement disorder specialist. Clinical UPDRS assessment and additional motor/cognitive evaluations are recommended.`
    : `The voice recording shows acoustic properties within normal ranges for the measured parameters. Continue regular health monitoring. If you experience any motor symptoms such as tremor, rigidity, or bradykinesia, consult a neurologist.`

  return {
    verdict: label as 'Parkinson' | 'Healthy',
    confidence,
    headline,
    explanation,
    key_findings: findings.slice(0, 6),
    positive_signs,
    negative_signs,
    recommendation,
  }
}

function buildParkinsonExplanation(
  findings: ClinicalFinding[],
  motorUpdrs: number,
  _severityLevel: string,
  confidence: number
): string {
  const topSymptoms = [...new Set(
    findings.filter(f => f.direction === 'increases_risk').map(f => f.symptom)
  )].slice(0, 3)

  const symptomText = topSymptoms.length > 0
    ? topSymptoms.join(', ')
    : 'altered vocal biomarkers'

  const updrsContext = motorUpdrs < 15
    ? 'mild motor involvement (early stage)'
    : motorUpdrs < 30
    ? 'moderate motor involvement'
    : 'significant motor involvement'

  return `The LightGBM model analyzed 50 acoustic features extracted from your voice recording and identified patterns strongly associated with Parkinson's disease. The primary indicators are: ${symptomText}. These features reflect the neuromotor control difficulties characteristic of hypokinetic dysarthria — a speech disorder caused by the basal ganglia dysfunction in Parkinson's disease. The UPDRS severity estimate suggests ${updrsContext} (Motor UPDRS: ${motorUpdrs.toFixed(1)}). The model assigns ${(confidence * 100).toFixed(1)}% probability to Parkinson's classification.`
}

function buildHealthyExplanation(
  findings: ClinicalFinding[],
  confidence: number
): string {
  const protectiveFeatures = findings
    .filter(f => f.direction === 'decreases_risk')
    .map(f => f.symptom)
    .slice(0, 2)

  const protective = protectiveFeatures.length > 0
    ? ` Key features showing normal ranges include ${protectiveFeatures.join(' and ')}.`
    : ''

  return `The acoustic analysis of your voice recording does not show the vocal biomarkers typically associated with Parkinson's disease.${protective} Your pitch stability, voice energy, and spectral characteristics are within the range expected for healthy speakers. The LightGBM model assigns ${(confidence * 100).toFixed(1)}% probability to the Healthy classification. Note: this tool analyzes acoustic features and is not a medical diagnosis.`
}
