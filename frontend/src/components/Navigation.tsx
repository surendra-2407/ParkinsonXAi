import { NavLink } from 'react-router-dom'

export default function Navigation() {
  return (
    <nav className="nav">
      <div className="nav-inner">
        <NavLink to="/" className="nav-logo">
          <span className="gradient-text">Parkinson</span>
          <span style={{ color: 'var(--text-secondary)', fontWeight: 400 }}>XAI</span>
        </NavLink>

        <ul className="nav-links">
          <li>
            <NavLink
              to="/"
              end
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
            >
              Analyze
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/dashboard"
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
            >
              Dashboard
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/history"
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
            >
              History
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/how-it-works"
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
            >
              How It Works
            </NavLink>
          </li>
        </ul>

      </div>
    </nav>
  )
}
