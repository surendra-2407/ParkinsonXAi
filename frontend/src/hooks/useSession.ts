import { useRef } from 'react'

const SESSION_KEY = 'parkinsonxai_session_id'

function generateUUID(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0
    const v = c === 'x' ? r : (r & 0x3 | 0x8)
    return v.toString(16)
  })
}

export function useSession(): string {
  const sessionRef = useRef<string>('')
  if (!sessionRef.current) {
    let sessionId = localStorage.getItem(SESSION_KEY)
    if (!sessionId) {
      sessionId = generateUUID()
      localStorage.setItem(SESSION_KEY, sessionId)
    }
    sessionRef.current = sessionId
  }
  return sessionRef.current
}
