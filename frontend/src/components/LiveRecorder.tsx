import { useRef, useState, useEffect } from 'react'
import { blobToWav } from '../utils/wavEncoder'

interface LiveRecorderProps {
  onRecording: (file: File) => void
  disabled?: boolean
}

type RecordState = 'idle' | 'recording' | 'converting' | 'done'

export default function LiveRecorder({ onRecording, disabled }: LiveRecorderProps) {
  const [recState, setRecState] = useState<RecordState>('idle')
  const [seconds, setSeconds] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [analyserData, setAnalyserData] = useState<Uint8Array>(new Uint8Array(32))

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const animFrameRef = useRef<number>(0)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const streamRef = useRef<MediaStream | null>(null)

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
      cancelAnimationFrame(animFrameRef.current)
      streamRef.current?.getTracks().forEach(t => t.stop())
      if (audioUrl) URL.revokeObjectURL(audioUrl)
    }
  }, [audioUrl])

  function drawWaveform() {
    if (!analyserRef.current) return
    const data = new Uint8Array(analyserRef.current.frequencyBinCount)
    analyserRef.current.getByteFrequencyData(data)
    setAnalyserData(new Uint8Array(data))
    animFrameRef.current = requestAnimationFrame(drawWaveform)
  }

  async function startRecording() {
    setError(null)
    setAudioUrl(null)
    chunksRef.current = []

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 22050, channelCount: 1, echoCancellation: true }
      })
      streamRef.current = stream

      // Live waveform via Web Audio API
      const audioCtx = new AudioContext()
      const source = audioCtx.createMediaStreamSource(stream)
      const analyser = audioCtx.createAnalyser()
      analyser.fftSize = 64
      source.connect(analyser)
      analyserRef.current = analyser
      drawWaveform()

      // Choose best available format
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')
        ? 'audio/ogg;codecs=opus'
        : 'audio/webm'

      const mr = new MediaRecorder(stream, { mimeType })
      mediaRecorderRef.current = mr

      mr.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data) }
      mr.onstop = async () => {
        cancelAnimationFrame(animFrameRef.current)
        stream.getTracks().forEach(t => t.stop())
        audioCtx.close()

        const rawBlob = new Blob(chunksRef.current, { type: mimeType })
        // Show preview of raw recording immediately
        const url = URL.createObjectURL(rawBlob)
        setAudioUrl(url)
        setRecState('converting')

        // Convert WebM/OGG → WAV (16-bit PCM 22050 Hz) in the browser
        // so the backend's soundfile parser can read it without ffmpeg
        try {
          const wavFile = await blobToWav(rawBlob)
          setRecState('done')
          onRecording(wavFile)
        } catch (convErr: any) {
          setError(`Audio conversion failed: ${convErr.message}`)
          setRecState('idle')
        }
      }

      mr.start(100) // collect chunks every 100ms
      setRecState('recording')
      setSeconds(0)
      timerRef.current = setInterval(() => setSeconds(s => s + 1), 1000)

    } catch (err: any) {
      setError(err.message?.includes('Permission')
        ? 'Microphone permission denied. Please allow access.'
        : `Microphone error: ${err.message}`)
    }
  }

  function stopRecording() {
    if (timerRef.current) clearInterval(timerRef.current)
    mediaRecorderRef.current?.stop()
  }

  function reset() {
    setRecState('idle')
    setSeconds(0)
    setAudioUrl(null)
    setAnalyserData(new Uint8Array(32))
  }

  const barCount = 28
  const bars = Array.from({ length: barCount }, (_, i) => {
    const dataIdx = Math.floor((i / barCount) * analyserData.length)
    const raw = analyserData[dataIdx] || 0
    const h = recState === 'recording' ? Math.max(6, (raw / 255) * 64) : 6
    return h
  })

  return (
    <div style={{ textAlign: 'center' }}>
      {/* Waveform visualizer */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        gap: 4, height: 72, marginBottom: 24,
      }}>
        {bars.map((h, i) => (
          <div key={i} style={{
            width: 5, height: `${h}px`,
            borderRadius: 3,
            background: recState === 'recording'
              ? `linear-gradient(to top, #7c3aed, #14b8a6)`
              : recState === 'done'
              ? '#10b981'
              : recState === 'converting'
              ? '#f59e0b'
              : 'rgba(255,255,255,0.1)',
            transition: recState === 'recording' ? 'height 0.1s ease' : 'height 0.4s ease',
          }} />
        ))}
      </div>

      {/* Timer */}
      {recState === 'recording' && (
        <div style={{ marginBottom: 16 }}>
          <span style={{
            fontFamily: 'var(--font-display)', fontSize: '2rem', fontWeight: 700,
            color: '#ef4444', letterSpacing: '0.05em',
          }}>
            {String(Math.floor(seconds / 60)).padStart(2, '0')}:{String(seconds % 60).padStart(2, '0')}
          </span>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, marginTop: 4 }}>
            <span style={{
              width: 8, height: 8, borderRadius: '50%', background: '#ef4444',
              display: 'inline-block',
              animation: 'pulse-glow 1s ease-in-out infinite',
            }} />
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Recording live audio</span>
          </div>
        </div>
      )}

      {/* Converting state */}
      {recState === 'converting' && (
        <div style={{ marginBottom: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10, marginBottom: 8 }}>
            <div className="spinner" style={{ width: 18, height: 18, borderWidth: 2, borderTopColor: '#f59e0b' }} />
            <span style={{ color: '#f59e0b', fontWeight: 600, fontSize: '0.9rem' }}>Converting to WAV...</span>
          </div>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Encoding audio for analysis</p>
          {audioUrl && (
            <audio controls src={audioUrl}
              style={{ width: '100%', maxWidth: 340, marginTop: 10, filter: 'invert(1) hue-rotate(180deg)', borderRadius: 8 }}
            />
          )}
        </div>
      )}

      {/* Done state */}
      {recState === 'done' && (
        <div style={{ marginBottom: 16 }}>
          <p style={{ color: '#34d399', fontWeight: 600, marginBottom: 8, fontSize: '0.95rem' }}>
            ✓ Recording complete — {seconds}s captured
          </p>
          {audioUrl && (
            <audio
              controls
              src={audioUrl}
              style={{
                width: '100%', maxWidth: 340,
                filter: 'invert(1) hue-rotate(180deg)',
                borderRadius: 8,
              }}
            />
          )}
        </div>
      )}

      {/* Idle state label */}
      {recState === 'idle' && (
        <div style={{ marginBottom: 20 }}>
          <div style={{
            width: 60, height: 60, borderRadius: '50%',
            background: 'var(--purple-dim)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            margin: '0 auto 12px',
            fontSize: '1.8rem',
            animation: 'float 3s ease-in-out infinite',
          }}>
            🎤
          </div>
          <p style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>
            Record your voice
          </p>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
            Speak a sustained vowel (e.g. "aaaaah") for 3–10 seconds
          </p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{
          padding: '10px 14px', marginBottom: 16,
          background: 'var(--negative-dim)', border: '1px solid rgba(239,68,68,0.3)',
          borderRadius: 'var(--radius-sm)', color: '#f87171', fontSize: '0.82rem',
        }}>
          ⚠️ {error}
        </div>
      )}

      {/* Controls */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: 10, flexWrap: 'wrap' }}>
        {recState === 'idle' && (
          <button
            className="btn-primary"
            onClick={startRecording}
            disabled={disabled}
            style={{
              background: 'linear-gradient(135deg, #ef4444, #f87171)',
              boxShadow: '0 4px 20px rgba(239,68,68,0.4)',
              fontSize: '0.9rem',
            }}
          >
            ⏺ Start Recording
          </button>
        )}

        {recState === 'recording' && (
          <button
            className="btn-primary"
            onClick={stopRecording}
            style={{
              background: 'linear-gradient(135deg, #1e293b, #334155)',
              boxShadow: '0 4px 20px rgba(0,0,0,0.3)',
              fontSize: '0.9rem',
            }}
          >
            ⏹ Stop Recording
          </button>
        )}

        {recState === 'converting' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, opacity: 0.7 }}>
            <div className="spinner" style={{ width: 16, height: 16, borderWidth: 2, borderTopColor: '#f59e0b' }} />
            <span style={{ fontSize: '0.85rem', color: '#f59e0b' }}>Encoding WAV...</span>
          </div>
        )}

        {recState === 'done' && (
          <>
            <button className="btn-secondary" onClick={reset} style={{ fontSize: '0.85rem' }}>
              🔄 Re-record
            </button>
            <span style={{
              padding: '8px 16px', borderRadius: 'var(--radius-md)',
              background: 'var(--positive-dim)', border: '1px solid rgba(16,185,129,0.3)',
              color: '#34d399', fontSize: '0.82rem', fontWeight: 600,
              display: 'flex', alignItems: 'center', gap: 6,
            }}>
              ✓ Ready to analyze
            </span>
          </>
        )}
      </div>

      {/* Tips */}
      {recState === 'idle' && (
        <div style={{
          marginTop: 20, display: 'flex', justifyContent: 'center', gap: 16, flexWrap: 'wrap',
        }}>
          {['Quiet room', '3–10 seconds', 'Say "aaah"'].map(tip => (
            <span key={tip} style={{
              fontSize: '0.7rem', color: 'var(--text-muted)',
              padding: '3px 10px', borderRadius: 999,
              border: '1px solid var(--border)',
            }}>
              {tip}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
