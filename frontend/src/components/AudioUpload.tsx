import { useRef, useState } from 'react'
import type { DragEvent, ChangeEvent } from 'react'

interface AudioUploadProps {
  onFile: (file: File) => void
  disabled?: boolean
}

export default function AudioUpload({ onFile, disabled }: AudioUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  function handleFile(file: File) {
    if (!file.name.toLowerCase().endsWith('.wav')) {
      alert('Please upload a WAV audio file.')
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
        accept=".wav,audio/wav"
        onChange={onChange}
        style={{ display: 'none' }}
        disabled={disabled}
      />

      {/* Waveform animation */}
      <div className="waveform" style={{ marginBottom: 24 }}>
        {Array.from({ length: 16 }).map((_, i) => (
          <div
            key={i}
            className="waveform-bar"
            style={{
              height: `${24 + Math.sin(i * 0.8) * 20}px`,
              animationDelay: `${i * 0.08}s`,
              opacity: selectedFile ? 1 : 0.4,
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
            {(selectedFile.size / 1024).toFixed(1)} KB — Click to change file
          </p>
        </div>
      ) : (
        <div>
          <div
            style={{
              width: 56, height: 56,
              background: 'var(--purple-dim)',
              borderRadius: '50%',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              margin: '0 auto 16px',
              fontSize: '1.6rem',
              animation: 'float 3s ease-in-out infinite',
            }}
          >
            🎙️
          </div>
          <p style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: 6 }}>
            Drop your voice recording here
          </p>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: 12 }}>
            or click to browse — WAV files only
          </p>
          <span className="badge badge-purple">Supports WAV · Up to 50 MB</span>
        </div>
      )}
    </div>
  )
}
