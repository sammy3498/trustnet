import { useEffect, useState } from 'react'
import './App.css'

const API_BASE = 'http://127.0.0.1:8000'

const factorNames = {
  evidence_strength: 'Evidence Strength',
  factual_consistency: 'Factual Consistency',
  source_provenance: 'Source Provenance',
  bias_framing: 'Bias & Framing',
  incentive_conflict: 'Incentive / Conflict',
}

function App() {
  const [claim, setClaim] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [backendOnline, setBackendOnline] = useState(null)

  useEffect(() => {
    async function checkBackend() {
      try {
        const response = await fetch(`${API_BASE}/health`)
        const data = await response.json()

        setBackendOnline(
          response.ok && data.status === 'ok'
        )
      } catch {
        setBackendOnline(false)
      }
    }

    checkBackend()
  }, [])

  async function handleAnalyze(event) {
    event.preventDefault()

    const trimmedClaim = claim.trim()

    if (!trimmedClaim) {
      setError('Enter a claim before starting analysis.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const response = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          input: trimmedClaim,
          input_type: 'claim',
          mode: 'normal',
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(
          data.detail || 'Credibility analysis failed.'
        )
      }

      setResult(data)
    } catch (requestError) {
      setError(
        requestError.message ||
          'Unable to connect to the TrustNet backend.'
      )
    } finally {
      setLoading(false)
    }
  }

  function getScoreClass(score) {
    if (score === null || score === undefined) {
      return 'score-insufficient'
    }

    if (score >= 80) {
      return 'score-high'
    }

    if (score >= 60) {
      return 'score-moderate'
    }

    if (score >= 40) {
      return 'score-low'
    }

    return 'score-poor'
  }

  function getStanceClass(stance) {
    return `stance stance-${stance}`
  }

  function getQualityClass(quality) {
    return `quality quality-${quality}`
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">T</div>

          <div>
            <h1>TrustNet</h1>
            <p>Evidence-first credibility analysis</p>
          </div>
        </div>

        <div
          className={`backend-status ${
            backendOnline === true
              ? 'online'
              : backendOnline === false
                ? 'offline'
                : ''
          }`}
        >
          <span className="status-dot" />

          {backendOnline === null && 'Checking API'}

          {backendOnline === true && 'API Online'}

          {backendOnline === false && 'API Offline'}
        </div>
      </header>

      <main className="main-content">
        <section className="hero">
          <div className="hero-copy">
            <span className="eyebrow">
              Intelligent credibility analysis
            </span>

            <h2>
              Check a claim against
              <span> real-world evidence.</span>
            </h2>

            <p>
              TrustNet retrieves relevant evidence, analyzes
              factual consistency and source quality, and
              produces an evidence-first credibility assessment.
            </p>
          </div>

          <form
            className="analysis-card"
            onSubmit={handleAnalyze}
          >
            <div className="input-heading">
              <div>
                <h3>Analyze a claim</h3>
                <p>
                  Enter one factual claim for the current MVP.
                </p>
              </div>

              <span className="mode-badge">
                Normal mode
              </span>
            </div>

            <textarea
              value={claim}
              onChange={(event) =>
                setClaim(event.target.value)
              }
              placeholder="Example: The Earth orbits the Sun."
              rows={6}
              disabled={loading}
            />

            <div className="form-footer">
              <span className="character-count">
                {claim.length} characters
              </span>

              <button
                type="submit"
                disabled={
                  loading ||
                  !claim.trim() ||
                  backendOnline === false
                }
              >
                {loading
                  ? 'Analyzing...'
                  : 'Analyze credibility'}
              </button>
            </div>
          </form>
        </section>

        {error && (
          <section className="message error-message">
            <strong>Analysis failed</strong>
            <span>{error}</span>
          </section>
        )}

        {loading && (
          <section className="loading-card">
            <div className="loader" />

            <div>
              <h3>Analyzing evidence</h3>
              <p>
                Retrieving sources and evaluating the claim.
                This can take several seconds.
              </p>
            </div>
          </section>
        )}

        {result && (
          <section className="results">
            <div className="results-header">
              <div>
                <span className="eyebrow">
                  Analysis result
                </span>

                <h2>{result.label}</h2>

                <p className="analyzed-claim">
                  “{result.claim}”
                </p>
              </div>

              <div
                className={`score-circle ${getScoreClass(
                  result.overall_score
                )}`}
              >
                <span className="score-value">
                  {result.overall_score ?? '—'}
                </span>

                <span className="score-total">
                  {result.overall_score === null
                    ? 'N/A'
                    : '/ 100'}
                </span>
              </div>
            </div>

            <div className="summary-grid">
              <article className="summary-card">
                <span>Confidence</span>
                <strong className="capitalize">
                  {result.confidence}
                </strong>
              </article>

              <article className="summary-card">
                <span>Uncertainty</span>
                <strong className="capitalize">
                  {result.uncertainty.level}
                </strong>
              </article>

              <article className="summary-card">
                <span>Evidence sources</span>
                <strong>{result.evidence.length}</strong>
              </article>

              <article className="summary-card">
                <span>Latency</span>
                <strong>
                  {(result.latency_ms / 1000).toFixed(1)}s
                </strong>
              </article>
            </div>

            {result.uncertainty.gate_triggered && (
              <div className="uncertainty-warning">
                <strong>
                  Insufficient evidence gate triggered
                </strong>

                <span>
                  {result.uncertainty.reason}
                </span>
              </div>
            )}

            {result.factors && (
              <section className="result-section">
                <div className="section-heading">
                  <div>
                    <span className="eyebrow">
                      Credibility framework
                    </span>

                    <h3>Factor breakdown</h3>
                  </div>
                </div>

                <div className="factor-list">
                  {Object.entries(result.factors).map(
                    ([key, factor]) => (
                      <article
                        className="factor-card"
                        key={key}
                      >
                        <div className="factor-top">
                          <strong>
                            {factorNames[key] || key}
                          </strong>

                          <span>{factor.score}/100</span>
                        </div>

                        <div className="factor-track">
                          <div
                            className="factor-fill"
                            style={{
                              width: `${factor.score}%`,
                            }}
                          />
                        </div>

                        <p>{factor.justification}</p>
                      </article>
                    )
                  )}
                </div>
              </section>
            )}

            <section className="result-section">
              <div className="section-heading">
                <div>
                  <span className="eyebrow">
                    Retrieved evidence
                  </span>

                  <h3>Evidence sources</h3>
                </div>

                <span className="section-count">
                  {result.evidence.length}
                </span>
              </div>

              {result.evidence.length === 0 ? (
                <div className="empty-state">
                  No public evidence items were returned.
                </div>
              ) : (
                <div className="evidence-list">
                  {result.evidence.map(
                    (item, index) => (
                      <article
                        className="evidence-card"
                        key={`${item.url}-${index}`}
                      >
                        <div className="evidence-top">
                          <div className="evidence-number">
                            {index + 1}
                          </div>

                          <div className="evidence-title">
                            <a
                              href={item.url}
                              target="_blank"
                              rel="noreferrer"
                            >
                              {item.title}
                            </a>

                            <span>{item.source}</span>
                          </div>
                        </div>

                        <div className="badges">
                          <span
                            className={getStanceClass(
                              item.stance
                            )}
                          >
                            {item.stance}
                          </span>

                          <span
                            className={getQualityClass(
                              item.quality
                            )}
                          >
                            {item.quality} quality
                          </span>
                        </div>

                        <p>{item.summary}</p>
                      </article>
                    )
                  )}
                </div>
              )}
            </section>

            {result.flags.length > 0 && (
              <section className="result-section">
                <div className="section-heading">
                  <div>
                    <span className="eyebrow">
                      Additional signals
                    </span>

                    <h3>Flags</h3>
                  </div>
                </div>

                <div className="flags">
                  {result.flags.map((flag, index) => (
                    <span
                      className="flag"
                      key={`${flag}-${index}`}
                    >
                      {flag}
                    </span>
                  ))}
                </div>
              </section>
            )}
          </section>
        )}
      </main>

      <footer>
        <strong>TrustNet</strong>
        <span>
          Evidence-first credibility analysis using LLMs
          and retrieval-augmented generation.
        </span>
      </footer>
    </div>
  )
}

export default App