import { useState } from 'react'
import { NavLink } from 'react-router-dom'

const NAV_ITEMS = [
  { to: '/',          label: 'Voice Analysis',      icon: '🎙️', end: true },
  { to: '/dashboard', label: 'Dashboard',           icon: '📊' },
  { to: '/history',   label: 'History',             icon: '🕒' },
  { to: '/shap',      label: 'SHAP Explainability', icon: '🔬' },
  { to: '/research',  label: 'Research Overview',   icon: '📋' },
]

export default function Navigation() {
  const [menuOpen, setMenuOpen] = useState(false)

  return (
    <>
      <nav className="nav">
        <div className="nav-inner">
          {/* Logo */}
          <NavLink to="/" style={{ display: 'flex', alignItems: 'center', gap: 10, textDecoration: 'none' }}>
            <div style={{
              width: 34, height: 34, borderRadius: 9,
              background: 'linear-gradient(135deg, #1d4ed8, #38bdf8)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '0.9rem', flexShrink: 0,
              boxShadow: '0 4px 14px rgba(29,78,216,0.45)',
            }}>🧠</div>
            <span>
              <span className="gradient-text" style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: '1.15rem' }}>Parkinson</span>
              <span style={{ color: 'var(--text-secondary)', fontWeight: 400, fontSize: '1.15rem' }}>XAI</span>
            </span>
          </NavLink>

          {/* Desktop links */}
          <ul className="nav-links" style={{ display: 'flex' }}>
            {NAV_ITEMS.map(item => (
              <li key={item.to}>
                <NavLink
                  to={item.to}
                  end={item.end}
                  className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
                >
                  <span style={{ marginRight: 6 }}>{item.icon}</span>
                  {item.label}
                </NavLink>
              </li>
            ))}
          </ul>

          {/* Mobile hamburger */}
          <button
            onClick={() => setMenuOpen(o => !o)}
            aria-label="Toggle menu"
            style={{
              display: 'none', background: 'none',
              border: '1px solid var(--border)', borderRadius: 8,
              padding: '6px 12px', color: 'var(--text-secondary)',
              cursor: 'pointer', fontSize: '1rem',
            }}
            className="nav-hamburger"
          >
            {menuOpen ? '✕' : '☰'}
          </button>
        </div>
      </nav>

      {/* Mobile drawer */}
      {menuOpen && (
        <div
          style={{ position: 'fixed', inset: 0, zIndex: 200, background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(6px)' }}
          onClick={() => setMenuOpen(false)}
        >
          <div
            style={{
              position: 'absolute', top: 0, left: 0, bottom: 0, width: 260,
              background: 'var(--bg-secondary)', borderRight: '1px solid var(--border)',
              padding: '24px 16px', display: 'flex', flexDirection: 'column', gap: 4,
            }}
            onClick={e => e.stopPropagation()}
          >
            <div style={{ marginBottom: 20, paddingBottom: 16, borderBottom: '1px solid var(--border)' }}>
              <p className="gradient-text" style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: '1.1rem' }}>ParkinsonXAI</p>
              <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 2 }}>AI Research Dashboard</p>
            </div>
            {NAV_ITEMS.map(item => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                onClick={() => setMenuOpen(false)}
                style={({ isActive }) => ({
                  display: 'flex', alignItems: 'center', gap: 12,
                  padding: '12px 14px', borderRadius: 10, textDecoration: 'none',
                  color: isActive ? '#60a5fa' : 'var(--text-secondary)',
                  background: isActive ? 'rgba(29,78,216,0.18)' : 'transparent',
                  fontSize: '0.88rem', fontWeight: isActive ? 600 : 400,
                  transition: 'all 0.15s',
                })}
              >
                <span style={{ fontSize: '1.1rem', width: 22 }}>{item.icon}</span>
                {item.label}
              </NavLink>
            ))}
          </div>
        </div>
      )}

      <style>{`
        @media (max-width: 720px) {
          .nav-links { display: none !important; }
          .nav-hamburger { display: flex !important; }
        }
      `}</style>
    </>
  )
}
