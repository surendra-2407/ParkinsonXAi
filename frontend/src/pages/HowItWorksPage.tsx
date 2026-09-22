import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import FeatureGlossary from '../components/FeatureGlossary'

interface AccordionItem {
  id: string
  title: string
  icon: string
  content: React.ReactNode
}

function Accordion({ items }: { items: AccordionItem[] }) {
  const [openId, setOpenId] = useState<string | null>(items[0]?.id ?? null)
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {items.map(item => (
        <div
          key={item.id}
          style={{
            border: `1px solid ${openId === item.id ? 'rgba(124,58,237,0.4)' : 'var(--border)'}`,
            borderRadius: 'var(--radius-md)',
            overflow: 'hidden',
            background: openId === item.id ? 'rgba(124,58,237,0.04)' : 'var(--bg-card)',
            transition: 'all 0.2s',
          }}
        >
          <button
            id={`accordion-${item.id}`}
            onClick={() => setOpenId(openId === item.id ? null : item.id)}
            style={{
              width: '100%', display: 'flex', alignItems: 'center', gap: 14,
              padding: '16px 20px', background: 'transparent', border: 'none',
              cursor: 'pointer', textAlign: 'left',
            }}
          >
            <span style={{ fontSize: '1.3rem', flexShrink: 0 }}>{item.icon}</span>
            <span style={{ flex: 1, fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              {item.title}
            </span>
            <span style={{
              fontSize: '0.7rem', color: 'var(--text-muted)',
              transform: openId === item.id ? 'rotate(180deg)' : 'rotate(0)',
              transition: 'transform 0.3s',
            }}>▼</span>
          </button>
          {openId === item.id && (
            <div style={{ padding: '0 20px 20px' }}>
              {item.content}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

function PipelineStep({ step, title, detail, color }: { step: string; title: string; detail: string; color: string }) {
  return (
    <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
      <div style={{
        width: 36, height: 36, borderRadius: '50%', flexShrink: 0,
        background: `${color}20`, border: `2px solid ${color}60`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: '0.8rem', fontWeight: 800, color,
      }}>
        {step}
      </div>
      <div style={{ flex: 1 }}>
        <p style={{ fontWeight: 700, fontSize: '0.85rem', color: 'var(--text-primary)', marginBottom: 4 }}>{title}</p>
        <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>{detail}</p>
      </div>
    </div>
  )
}

const MODEL_TRAINING_ITEMS: AccordionItem[] = [
  {
    id: 'datasets',
    icon: '🗃️',
    title: 'What datasets were used to train the model?',
    content: (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
          The model was trained on <strong style={{ color: 'var(--text-primary)' }}>three separate Parkinson's voice datasets</strong>,
          each providing a different type of acoustic data. Training across multiple datasets makes the model more robust.
        </p>
        {[
          {
            name: 'Dataset 1 — Raw WAV Files',
            desc: 'Voice recordings (WAV) of healthy speakers and Parkinson\'s patients. Each recording was processed to extract 200+ acoustic features using librosa and Praat. This is the primary dataset for the live microphone analysis.',
            color: '#7c3aed',
            badge: 'Audio',
          },
          {
            name: 'Dataset 2 — pd_speech_features.csv',
            desc: 'A tabular speech feature dataset from sustained phonation tasks, containing 754 features per subject including MFCC, chroma, and spectral features.',
            color: '#0ea5e9',
            badge: 'Tabular',
          },
          {
            name: 'Dataset 3 — UCI Parkinson\'s (parkinsons.data)',
            desc: 'The classic UCI Machine Learning Repository Parkinson\'s dataset with 22 voice features (jitter, shimmer, NHR, HNR, RPDE, DFA, PPE). Widely used as benchmark.',
            color: '#10b981',
            badge: 'UCI Benchmark',
          },
        ].map(d => (
          <div key={d.name} style={{
            padding: '12px 14px', border: '1px solid var(--border)',
            borderLeft: `3px solid ${d.color}`, borderRadius: 8, background: 'var(--bg-card)',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
              <p style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-primary)' }}>{d.name}</p>
              <span style={{
                fontSize: '0.65rem', padding: '2px 8px', borderRadius: 20,
                background: `${d.color}20`, color: d.color, border: `1px solid ${d.color}40`,
              }}>{d.badge}</span>
            </div>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>{d.desc}</p>
          </div>
        ))}
      </div>
    ),
  },
  {
    id: 'features',
    icon: '🎙️',
    title: 'How are voice features extracted from a recording?',
    content: (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
          When you record or upload audio, the backend runs <strong style={{ color: 'var(--text-primary)' }}>200+ feature extractors</strong> on the WAV signal.
          These are grouped into categories:
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: 10 }}>
          {[
            { group: 'MFCC (160 features)', tools: 'librosa', desc: '40 coefficients × mean/std/delta/delta² — captures vocal tract shape over time', color: '#7c3aed' },
            { group: 'Spectral Features', tools: 'librosa', desc: 'Centroid, bandwidth, rolloff, contrast, flatness — frequency distribution', color: '#0ea5e9' },
            { group: 'Chroma Features', tools: 'librosa', desc: '12 pitch class energies — harmonic structure of voice', color: '#10b981' },
            { group: 'Pitch & F0', tools: 'librosa pyin', desc: 'Fundamental frequency, voiced fraction — pitch tracking', color: '#f59e0b' },
            { group: 'Jitter & Shimmer', tools: 'Praat', desc: 'Period perturbation (jitter), amplitude perturbation (shimmer) — tremor markers', color: '#ef4444' },
            { group: 'HNR & Formants', tools: 'Praat', desc: 'Harmonics-to-noise ratio, F1/F2/F3 formant frequencies', color: '#8b5cf6' },
            { group: 'RMS & ZCR', tools: 'librosa', desc: 'Energy (loudness) and zero-crossing rate (noisiness)', color: '#14b8a6' },
          ].map(g => (
            <div key={g.group} style={{
              padding: '10px 12px', background: 'var(--bg-card)',
              border: `1px solid ${g.color}30`, borderRadius: 8,
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <p style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-primary)' }}>{g.group}</p>
                <span style={{ fontSize: '0.62rem', color: g.color }}>via {g.tools}</span>
              </div>
              <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>{g.desc}</p>
            </div>
          ))}
        </div>
        <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
          💡 Features are extracted using the <strong>same pipeline</strong> during training and inference — so the model always sees consistent inputs.
        </p>
      </div>
    ),
  },
  {
    id: 'training-pipeline',
    icon: '⚙️',
    title: 'How was the model trained step-by-step?',
    content: (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {[
            { step: '1', title: 'Feature Selection — Top 50 by Mutual Information', detail: 'From 200+ features, the 50 most informative features are selected using mutual information scoring (how much each feature tells us about the label). This reduces dimensionality and overfitting.', color: '#7c3aed' },
            { step: '2', title: 'Class Imbalance — SMOTE', detail: 'Parkinson\'s datasets often have more Parkinson\'s samples than healthy ones. SMOTE (Synthetic Minority Oversampling TEchnique) generates synthetic healthy samples to balance the classes — but ONLY inside each training fold, never on the test data.', color: '#0ea5e9' },
            { step: '3', title: '5-Fold Cross-Validation', detail: 'The training data is split into 5 folds. The model is trained on 4 folds and validated on the 5th, repeated 5 times. This gives a reliable estimate of generalization performance and prevents overfitting to a single train/test split.', color: '#10b981' },
            { step: '4', title: 'Baseline Models — 7 Classifiers Tested', detail: 'Seven classifiers are trained: Logistic Regression, SVM (RBF + Linear), Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost. Each is evaluated on accuracy, F1, recall, and AUC.', color: '#f59e0b' },
            { step: '5', title: 'Optuna Hyperparameter Tuning', detail: 'The best baseline model (typically LightGBM) is tuned using Optuna Bayesian optimization — automatically finding the best learning rate, max depth, num_leaves, regularization, and more.', color: '#ef4444' },
            { step: '6', title: 'Final Model — LightGBM (96.49% accuracy)', detail: 'The tuned LightGBM model achieves 96.49% accuracy and >0.98 AUC on the test set. It is saved to disk and loaded by the FastAPI backend for real-time inference. SHAP TreeExplainer is used to explain each prediction.', color: '#8b5cf6' },
          ].map(s => (
            <PipelineStep key={s.step} {...s} />
          ))}
        </div>
      </div>
    ),
  },
  {
    id: 'model-choice',
    icon: '🏆',
    title: 'Why LightGBM? What makes it good for this task?',
    content: (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
          LightGBM (Light Gradient Boosting Machine) consistently outperformed other models on this task for several reasons:
        </p>
        {[
          { title: 'Handles small datasets well', desc: 'Medical datasets are small. Tree boosting models excel with hundreds to low thousands of samples, unlike deep learning which needs millions.' },
          { title: 'Fast & efficient', desc: 'LightGBM uses histogram-based splitting and leaf-wise tree growth, making it far faster than XGBoost on small-to-medium datasets.' },
          { title: 'Handles imbalanced classes', desc: 'Built-in class_weight="balanced" and is_unbalance options handle the Parkinson\'s/Healthy class imbalance natively.' },
          { title: 'SHAP compatible', desc: 'Tree-based models have exact, fast SHAP explainers (TreeSHAP) — giving us precise feature attribution without approximation.' },
          { title: 'High performance', desc: '96.49% accuracy, 0.98+ AUC, 97% recall on the test set — outperforming SVM, Random Forest, and XGBoost in comparative evaluation.' },
        ].map(r => (
          <div key={r.title} style={{
            display: 'flex', gap: 10, padding: '10px 12px',
            background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8,
          }}>
            <span style={{ color: '#10b981', fontSize: '0.9rem', flexShrink: 0 }}>✓</span>
            <div>
              <p style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: 2 }}>{r.title}</p>
              <p style={{ fontSize: '0.73rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>{r.desc}</p>
            </div>
          </div>
        ))}
      </div>
    ),
  },
]

export default function HowItWorksPage() {
  const navigate = useNavigate()

  return (
    <div className="page">
      <div className="container">
        {/* Header */}
        <div style={{ marginBottom: 36 }}>
          <p className="section-label">Understanding the Model</p>
          <h1 style={{ fontSize: '1.8rem', fontWeight: 800, marginTop: 6, marginBottom: 10 }}>
            How ParkinsonXAI Works
          </h1>
          <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', maxWidth: 640, lineHeight: 1.7 }}>
            A plain-English guide to the machine learning pipeline, what acoustic features mean clinically,
            and how SHAP explains each prediction.
          </p>
          <div style={{ display: 'flex', gap: 10, marginTop: 16 }}>
            <button id="try-analysis-btn" className="btn-primary" onClick={() => navigate('/')}>
              🎙️ Try Live Analysis
            </button>
            <button id="view-dashboard-btn" className="btn-secondary" onClick={() => navigate('/dashboard')}>
              📊 View Dashboard
            </button>
          </div>
        </div>

        {/* Pipeline overview */}
        <div className="glass-card p-32" style={{ marginBottom: 24 }}>
          <p className="section-label" style={{ marginBottom: 20 }}>The Complete Pipeline</p>
          <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 0 }}>
            {[
              { label: 'Voice Recording', icon: '🎙️', color: '#7c3aed' },
              { label: 'Feature Extraction\n(200+ features)', icon: '📊', color: '#0ea5e9' },
              { label: 'Feature Selection\n(Top 50)', icon: '🔍', color: '#10b981' },
              { label: 'LightGBM\nClassifier', icon: '🤖', color: '#f59e0b' },
              { label: 'SHAP\nExplanation', icon: '💡', color: '#ef4444' },
              { label: 'Clinical\nReport', icon: '📋', color: '#8b5cf6' },
            ].map((node, i, arr) => (
              <div key={node.label} style={{ display: 'flex', alignItems: 'center' }}>
                <div style={{
                  display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8,
                  padding: '14px 16px', background: `${node.color}10`,
                  border: `1px solid ${node.color}40`, borderRadius: 12, minWidth: 100,
                }}>
                  <span style={{ fontSize: '1.5rem' }}>{node.icon}</span>
                  <p style={{
                    fontSize: '0.68rem', fontWeight: 600, color: node.color,
                    textAlign: 'center', lineHeight: 1.4, whiteSpace: 'pre-line',
                  }}>{node.label}</p>
                </div>
                {i < arr.length - 1 && (
                  <span style={{ fontSize: '1rem', color: 'var(--text-muted)', padding: '0 6px' }}>→</span>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Model Training Q&A */}
        <div className="glass-card p-32" style={{ marginBottom: 24 }}>
          <p className="section-label" style={{ marginBottom: 20 }}>Model Training — Deep Dive</p>
          <Accordion items={MODEL_TRAINING_ITEMS} />
        </div>

        {/* Feature Glossary */}
        <div className="glass-card p-32" style={{ marginBottom: 24 }}>
          <p className="section-label" style={{ marginBottom: 6 }}>Acoustic Feature Glossary</p>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 20 }}>
            Click any card to expand a plain-English explanation + clinical significance.
          </p>
          <FeatureGlossary defaultOpen={true} />
        </div>

        {/* Disclaimer */}
        <div style={{
          padding: '16px 20px', marginBottom: 40,
          background: 'rgba(245,158,11,0.06)',
          border: '1px solid rgba(245,158,11,0.25)',
          borderRadius: 'var(--radius-md)',
          display: 'flex', gap: 12,
        }}>
          <span style={{ fontSize: '1.2rem', flexShrink: 0 }}>⚠️</span>
          <div>
            <p style={{ fontSize: '0.8rem', fontWeight: 700, color: '#f59e0b', marginBottom: 4 }}>
              Research Tool — Not a Medical Device
            </p>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
              ParkinsonXAI is an academic research project demonstrating explainable AI for voice-based Parkinson's screening.
              It is not FDA-cleared, not CE-marked, and should not be used as a substitute for clinical diagnosis.
              Always consult a neurologist or movement disorder specialist for medical evaluation.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
