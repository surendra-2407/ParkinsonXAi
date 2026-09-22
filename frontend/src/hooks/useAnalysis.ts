import { useState, useCallback } from 'react'
import { parkinsonApi } from '../api/parkinsonApi'
import type { FullAnalysisResponse } from '../api/parkinsonApi'

type AnalysisState = 'idle' | 'loading' | 'done' | 'error'

export function useAnalysis(sessionId: string) {
  const [state, setState] = useState<AnalysisState>('idle')
  const [result, setResult] = useState<FullAnalysisResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)

  const analyze = useCallback(async (file: File) => {
    setState('loading')
    setError(null)
    setResult(null)
    setProgress(0)

    // Simulate progress during processing
    const timer = setInterval(() => {
      setProgress(p => Math.min(p + Math.random() * 15, 85))
    }, 600)

    try {
      const { data } = await parkinsonApi.predict(file, sessionId)
      clearInterval(timer)
      setProgress(100)
      setTimeout(() => {
        setResult(data)
        setState('done')
      }, 300)
    } catch (err: any) {
      clearInterval(timer)
      setProgress(0)
      const msg = err?.response?.data?.detail || err.message || 'Analysis failed.'
      setError(msg)
      setState('error')
    }
  }, [sessionId])

  const reset = useCallback(() => {
    setState('idle')
    setResult(null)
    setError(null)
    setProgress(0)
  }, [])

  return { state, result, error, progress, analyze, reset }
}
