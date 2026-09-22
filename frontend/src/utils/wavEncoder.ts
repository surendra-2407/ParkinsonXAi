/**
 * wavEncoder.ts
 * Converts a browser AudioBuffer to a WAV Blob (PCM 16-bit mono, 22050 Hz).
 * No external dependencies — pure Web Audio API.
 */

export async function blobToWav(blob: Blob): Promise<File> {
  // Decode the browser audio blob (WebM/OGG/MP3/WAV) to raw PCM
  const arrayBuffer = await blob.arrayBuffer()
  const audioCtx = new AudioContext({ sampleRate: 22050 })

  let audioBuffer: AudioBuffer
  try {
    audioBuffer = await audioCtx.decodeAudioData(arrayBuffer)
  } catch (e) {
    audioCtx.close()
    throw new Error(`Cannot decode audio: ${e}`)
  }

  // Mix down to mono (take channel 0 or average all channels)
  let samples: Float32Array
  if (audioBuffer.numberOfChannels === 1) {
    samples = audioBuffer.getChannelData(0)
  } else {
    const length = audioBuffer.length
    samples = new Float32Array(length)
    for (let ch = 0; ch < audioBuffer.numberOfChannels; ch++) {
      const chData = audioBuffer.getChannelData(ch)
      for (let i = 0; i < length; i++) {
        samples[i] += chData[i] / audioBuffer.numberOfChannels
      }
    }
  }

  audioCtx.close()

  const wavBlob = encodeWav(samples, audioBuffer.sampleRate)
  return new File([wavBlob], 'live_recording.wav', { type: 'audio/wav' })
}

function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const numChannels = 1
  const bitDepth = 16
  const bytesPerSample = bitDepth / 8
  const blockAlign = numChannels * bytesPerSample
  const byteRate = sampleRate * blockAlign
  const dataSize = samples.length * bytesPerSample
  const headerSize = 44

  const buffer = new ArrayBuffer(headerSize + dataSize)
  const view = new DataView(buffer)

  // RIFF chunk
  writeStr(view, 0, 'RIFF')
  view.setUint32(4, 36 + dataSize, true)
  writeStr(view, 8, 'WAVE')

  // fmt chunk
  writeStr(view, 12, 'fmt ')
  view.setUint32(16, 16, true)          // chunk size
  view.setUint16(20, 1, true)           // PCM format
  view.setUint16(22, numChannels, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, byteRate, true)
  view.setUint16(32, blockAlign, true)
  view.setUint16(34, bitDepth, true)

  // data chunk
  writeStr(view, 36, 'data')
  view.setUint32(40, dataSize, true)

  // Write PCM samples (float → int16)
  let offset = 44
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true)
    offset += 2
  }

  return new Blob([buffer], { type: 'audio/wav' })
}

function writeStr(view: DataView, offset: number, str: string) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i))
  }
}
