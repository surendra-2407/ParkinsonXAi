import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navigation from './components/Navigation'
import HomePage from './pages/HomePage'
import ResultsPage from './pages/ResultsPage'
import DashboardPage from './pages/DashboardPage'
import HistoryPage from './pages/HistoryPage'
import SHAPExplainabilityPage from './pages/SHAPExplainabilityPage'
import ResearchOverviewPage from './pages/ResearchOverviewPage'
import './index.css'

export default function App() {
  return (
    <BrowserRouter>
      <Navigation />
      <Routes>
        <Route path="/"          element={<HomePage />} />
        <Route path="/results"   element={<ResultsPage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/history"   element={<HistoryPage />} />
        <Route path="/shap"      element={<SHAPExplainabilityPage />} />
        <Route path="/research"  element={<ResearchOverviewPage />} />
      </Routes>
    </BrowserRouter>
  )
}
