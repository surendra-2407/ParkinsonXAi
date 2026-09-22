import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export interface DetectionResult {
  label: 'Parkinson' | 'Healthy'
  confidence: number
  probabilities: Record<string, number>
  model_used: string
}

export interface SeverityResult {
  motor_updrs: number
  total_updrs: number
  severity_level: 'Mild' | 'Moderate' | 'Severe'
  model_used: string
  severity_score: number        // 0–100 proxy score
  severity_basis: 'audio_proxy' | 'updrs_model'
}

export interface ShapFeature { feature: string; value: number }

export interface FullAnalysisResponse {
  prediction_id: string
  filename: string
  detection: DetectionResult
  severity: SeverityResult
  shap: { top_features: ShapFeature[]; plot_base64?: string }
  audio_features_snapshot: Record<string, number>
  processing_time_ms: number
  timestamp: string
}

export interface HistoryItem {
  prediction_id: string
  filename: string
  timestamp: string
  detection_label?: string
  confidence?: number
  motor_updrs?: number
  total_updrs?: number
  severity_level?: string
  shap_top_feature?: string
  processing_time_ms?: number
}

export interface HistoryResponse {
  items: HistoryItem[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

export interface ModelAccuracyCard {
  model_name: string
  dataset: string
  accuracy?: number
  auc?: number
  recall?: number
  f1?: number
  note?: string
}

export interface LiveStats {
  total_predictions: number
  parkinson_count: number
  healthy_count: number
  avg_confidence: number
}

export interface DashboardResponse {
  model_cards: ModelAccuracyCard[]
  live_stats: LiveStats
  top_shap_features: ShapFeature[]
  shap_severity_features: ShapFeature[]
}

export interface StatsResponse {
  total_predictions: number
  parkinson_count: number
  healthy_count: number
  avg_confidence: number
  avg_processing_time_ms: number
  parkinson_pct: number
  healthy_pct: number
}

export const parkinsonApi = {
  health: () => api.get('/health'),

  predict: (file: File, sessionId?: string) => {
    const fd = new FormData()
    fd.append('file', file)
    if (sessionId) fd.append('session_id', sessionId)
    return api.post<FullAnalysisResponse>('/predict-full', fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 120_000, // 2 minutes for SHAP
    })
  },

  dashboard: () => api.get<DashboardResponse>('/dashboard'),
  stats: () => api.get<StatsResponse>('/stats'),
  history: (page = 1, pageSize = 10) =>
    api.get<HistoryResponse>('/history', { params: { page, page_size: pageSize } }),
}
