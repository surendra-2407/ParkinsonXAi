import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export interface DetectionResult {
  label: 'Parkinson' | 'Healthy'    // always one of these two
  confidence: number
  probabilities: Record<string, number>
  model_used: string
  voice_quality?: number             // informational only, 0–1
  quality_warning?: string | null    // always null now
  is_uncertain?: boolean             // always false now
  is_low_confidence?: boolean        // true when confidence < 0.60
}

export interface SeverityResult {
  motor_updrs: number
  total_updrs: number
  severity_level: 'Mild' | 'Moderate' | 'Severe'
  model_used: string
  severity_score: number
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
  voice_quality?: number
  is_uncertain?: boolean
  motor_updrs?: number
  total_updrs?: number
  severity_level?: string
  severity_score?: number
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

export interface ModelPerfEntry {
  dataset: string
  dataset_name: string
  model: string
  protocol: string
  accuracy: number
  f1: number
  recall: number
  roc_auc: number
  is_primary: boolean
}

export interface GlobalShapFeature {
  rank: number
  feature: string
  display: string
  mean_abs_shap: number
  group: string
  description: string
}

export interface SeverityEvalEntry {
  model: string
  mae: number
  rmse: number
  r2: number
}

export interface DatasetInfo {
  id: string
  name: string
  recordings: number
  speakers: number | null
  healthy_count: number | null
  pd_count: number | null
  target: string
  target_type: string
  purpose: string
  protocol: string
  notes: string
  model_used: string
}

export interface FeatureGroup {
  group: string
  count: number
  description: string
}

export interface ActivityPoint {
  date: string
  count: number
  parkinson: number
  healthy: number
}

export const parkinsonApi = {
  health: () => api.get('/health'),

  predict: (file: File, sessionId?: string) => {
    const fd = new FormData()
    fd.append('file', file)
    if (sessionId) fd.append('session_id', sessionId)
    return api.post<FullAnalysisResponse>('/predict-full', fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 120_000,
    })
  },

  dashboard: () => api.get<DashboardResponse>('/dashboard'),
  stats: () => api.get<StatsResponse>('/stats'),
  history: (page = 1, pageSize = 10) =>
    api.get<HistoryResponse>('/history', { params: { page, page_size: pageSize } }),

  historyFiltered: (params: {
    page?: number; page_size?: number; search?: string;
    label?: string; date_from?: string; date_to?: string
  }) => api.get('/history-filtered', { params }),

  predictionDetail: (id: string) => api.get(`/history/detail/${id}`),
  deletePrediction: (id: string) => api.delete(`/history/${id}`),

  activity: (days = 30) => api.get<{ activity: ActivityPoint[] }>('/activity', { params: { days } }),

  modelPerformance: () => api.get<{ source: string; note: string; models: ModelPerfEntry[] }>('/models/performance'),
  globalShap: () => api.get<{ source: string; note: string; features: GlobalShapFeature[] }>('/explanations/global'),
  severityEvaluations: () => api.get<{ source: string; note: string; motor: SeverityEvalEntry[]; total: SeverityEvalEntry[] }>('/severity/evaluations'),
  datasets: () => api.get<{ source: string; datasets: DatasetInfo[] }>('/datasets'),
  features: () => api.get<{ source: string; total_raw: number; total_non_degenerate: number; total_selected: number; selection_method: string; feature_groups: FeatureGroup[] }>('/features'),
}
