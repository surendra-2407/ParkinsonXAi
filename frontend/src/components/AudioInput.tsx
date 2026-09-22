import { useRef, useState } from 'react'
import type { DragEvent, ChangeEvent } from 'react'
import LiveRecorder from './LiveRecorder'

interface AudioInputProps {
  onFile: (file: File) => void
  disabled?: boolean
}

type Tab = 'upload' | 'record'

export default function AudioInput({ onFile, disabled }: AudioInputProps) {
  const [tab, setTab] = useState<Tab>('upload')

  return (
    <div>
      {/* Tab switcher */}
      <div style={{
        display: 'flex',
        background: 'rgba(255,255,255,0.04)',
        borderRadius: 'var(--radius-md)',
        padding: 4,
        marginBottom: 24,
        border: '1px solid var(--border)',
        gap: 4,
      }}>
        {(['upload', 'record'] as Tab[]).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              flex: 1,
              padding: '9px 0',
              borderRadius: 10,
              border: 'none',
              cursor: 'pointer',
              fontFamily: 'var(--font-sans)',
              fontSize: '0.85rem',
              fontWeight: 600,
              transition: 'all 0.2s ease',
              background: tab === t
                ? 'linear-gradient(135deg, var(--purple), var(--purple-light))'
                : 'transparent',
              color: tab === t ? '#fff' : 'var(--text-muted)',
              boxShadow: tab === t ? '0 2px 12px rgba(124,58,237,0.4)' : 'none',
            }}
          >
            {t === 'upload' ? '📁  Upload File' : '🎤  Live Record'}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {tab === 'upload'
        ? <FileUploadZone onFile={onFile} disabled={disabled} />
        : <LiveRecorder onRecording={onFile} disabled={disabled} />
      }
    </div>
  )
}

/* ── File Upload (inner component) ──────────────────────────────────────────── */

function FileUploadZone({ onFile, disabled }: AudioInputProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  function handleFile(file: File) {
    const name = file.name.toLowerCase()
    if (!name.endsWith('.wav') && !name.endsWith('.webm') && !name.endsWith('.ogg') && !name.endsWith('.mp3')) {
      alert('Supported formats: WAV, WebM, OGG, MP3')
      return
    }
    setSelectedFile(file)
    onFile(file)
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  const onChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
  }

  return (
    <div
      className={`upload-zone ${dragging ? 'dragging' : ''} ${disabled ? 'opacity-50' : ''}`}
      onClick={() => !disabled && inputRef.current?.click()}
      onDragOver={e => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      style={{ cursor: disabled ? 'not-allowed' : 'pointer' }}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".wav,.webm,.ogg,.mp3,audio/*"
        onChange={onChange}
        style={{ display: 'none' }}
        disabled={disabled}
      />

      {/* Waveform bars */}
      <div className="waveform" style={{ marginBottom: 20 }}>
        {Array.from({ length: 16 }).map((_, i) => (
          <div
            key={i}
            className="waveform-bar"
            style={{
              height: `${20 + Math.sin(i * 0.8) * 16}px`,
              animationDelay: `${i * 0.08}s`,
              opacity: selectedFile ? 1 : 0.35,
            }}
          />
        ))}
      </div>

      {selectedFile ? (
        <div>
          <p style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--teal-light)', marginBottom: 4 }}>
            ✓ {selectedFile.name}
          </p>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            {(selectedFile.size / 1024).toFixed(1)} KB · Click to change
          </p>
        </div>
      ) : (
        <div>
          <div style={{
            width: 52, height: 52, borderRadius: '50%',
            background: 'var(--purple-dim)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            margin: '0 auto 14px', fontSize: '1.5rem',
            animation: 'float 3s ease-in-out infinite',
          }}>
            📂
          </div>
          <p style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: 6 }}>
            Drop your audio file here
          </p>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: 12 }}>
            or click to browse
          </p>
          <span className="badge badge-purple">WAV · WebM · OGG · MP3</span>
        </div>
      )}
    </div>
  )
}
