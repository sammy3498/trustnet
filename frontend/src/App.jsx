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

  const [configChecked, setConfigChecked] = useState(false)
  const [configured, setConfigured] = useState(true)
  const [configSource, setConfigSource] = useState(null)
  const [savingConfig, setSavingConfig] = useState(false)
  const [configError, setConfigError] = useState('')

  const [llmApiKey, setLlmApiKey] = useState('')
  const [tavilyApiKey, setTavilyApiKey] = useState('')
  const [llmModel, setLlmModel] = useState('')

  async function checkBackend() {
    try {
      const response = await fetch(
        `${API_BASE}/health`
      )

      const data = await response.json()

      const online =
        response.ok
        && data.status === 'ok'

      setBackendOnline(
        online
      )

      return online
    } catch {
      setBackendOnline(false)
      return false
    }
  }

  useEffect(() => {
    async function initializeApp() {
      if (
        window.trustnetDesktop
        && window.trustnetDesktop.getConfigStatus
      ) {
        try {
          const status =
            await window.trustnetDesktop
              .getConfigStatus()

          setConfigured(
            status.configured
          )

          setConfigSource(
            status.source
          )

          setConfigChecked(true)

          if (
            status.configured
          ) {
            await checkBackend()
          } else {
            setBackendOnline(false)
          }

          return
        } catch {
          setConfigured(true)
          setConfigChecked(true)
          await checkBackend()
          return
        }
      }

      setConfigured(true)
      setConfigChecked(true)
      await checkBackend()
    }

    initializeApp()
  }, [])

  async function handleSaveConfig(event) {
    event.preventDefault()

    const openAiKey =
      llmApiKey.trim()

    const tavilyKey =
      tavilyApiKey.trim()

    const model =
      llmModel.trim()

    if (
      !openAiKey
      || !tavilyKey
      || !model
    ) {
      setConfigError(
        'OpenAI API key, Tavily API key, and model are required.'
      )
      return
    }

    if (
      !window.trustnetDesktop
      || !window.trustnetDesktop.saveConfig
    ) {
      setConfigError(
        'Desktop configuration is unavailable in this environment.'
      )
      return
    }

    setSavingConfig(true)
    setConfigError('')

    try {
      const response =
        await window.trustnetDesktop.saveConfig({
          llmApiKey: openAiKey,
          tavilyApiKey: tavilyKey,
          llmModel: model,
        })

      if (
        !response.ok
      ) {
        throw new Error(
          response.error
          || 'Unable to save TrustNet configuration.'
        )
      }

      setConfigured(true)
      setConfigSource('secure')

      setLlmApiKey('')
      setTavilyApiKey('')

      await checkBackend()
    } catch (saveError) {
      setConfigError(
        saveError.message
        || 'Unable to save TrustNet configuration.'
      )
    } finally {
      setSavingConfig(false)
    }
  }

  async function handleAnalyze(event) {
    event.preventDefault()

    const trimmedClaim = claim.trim()

    if (!trimmedClaim) {
      setError(
        'Enter a claim before starting analysis.'
      )
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const response = await fetch(
        `${API_BASE}/analyze`,
        {
          method: 'POST',
          headers: {
            'Content-Type':
              'application/json',
          },
          body: JSON.stringify({
            input: trimmedClaim,
            input_type: 'claim',
            mode: 'normal',
          }),
        }
      )

      const data = await response.json()

      if (!response.ok) {
        throw new Error(
          data.detail
          || 'Credibility analysis failed.'
        )
      }

      setResult(data)
    } catch (requestError) {
      setError(
        requestError.message
        || 'Unable to connect to the TrustNet backend.'
      )
    } finally {
      setLoading(false)
    }
  }

  function getScoreClass(score) {
    if (
      score === null
      || score === undefined
    ) {
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

  if (!configChecked) {
    return (
      <div className="app">
        <div className="startup-screen">
          <div className="loader" />

          <div>
            <h2>Starting TrustNet</h2>
            <p>
              Checking desktop configuration.
            </p>
          </div>
        </div>
      </div>
    )
  }

  if (!configured) {
    return (
      <div className="app">
        <header className="topbar">
          <div className="brand">
            <div className="brand-mark">
              T
            </div>

            <div>
              <h1>TrustNet</h1>
              <p>
                Evidence-first credibility analysis
              </p>
            </div>
          </div>

          <div className="backend-status offline">
            <span className="status-dot" />
            Setup required
          </div>
        </header>

        <main className="setup-page">
          <section className="setup-intro">
            <span className="eyebrow">
              First-time setup
            </span>

            <h2>
              Connect TrustNet to its
              <span> analysis services.</span>
            </h2>

            <p>
              TrustNet uses OpenAI for structured
              credibility reasoning and Tavily for
              evidence retrieval. Your credentials are
              encrypted locally on this Windows device
              and are not bundled into the application.
            </p>
          </section>

          <form
            className="setup-card"
            onSubmit={handleSaveConfig}
          >
            <div className="setup-heading">
              <div>
                <h3>API configuration</h3>
                <p>
                  Complete this once to activate
                  credibility analysis.
                </p>
              </div>

              <span className="secure-badge">
                Local encrypted storage
              </span>
            </div>

            <label className="config-field">
              <span>OpenAI API key</span>

              <input
                type="password"
                value={llmApiKey}
                onChange={(event) =>
                  setLlmApiKey(
                    event.target.value
                  )
                }
                placeholder="sk-..."
                autoComplete="off"
                disabled={savingConfig}
              />
            </label>

            <label className="config-field">
              <span>Tavily API key</span>

              <input
                type="password"
                value={tavilyApiKey}
                onChange={(event) =>
                  setTavilyApiKey(
                    event.target.value
                  )
                }
                placeholder="tvly-..."
                autoComplete="off"
                disabled={savingConfig}
              />
            </label>

            <label className="config-field">
              <span>OpenAI model</span>

              <input
                type="text"
                value={llmModel}
                onChange={(event) =>
                  setLlmModel(
                    event.target.value
                  )
                }
                placeholder="Enter the model available to your API account"
                autoComplete="off"
                disabled={savingConfig}
              />
            </label>

            {configError && (
              <div className="setup-error">
                {configError}
              </div>
            )}

            <div className="setup-note">
              TrustNet never stores these credentials
              inside the installer or Git repository.
            </div>

            <button
              className="setup-button"
              type="submit"
              disabled={
                savingConfig
                || !llmApiKey.trim()
                || !tavilyApiKey.trim()
                || !llmModel.trim()
              }
            >
              {savingConfig
                ? 'Saving and starting TrustNet...'
                : 'Save and start TrustNet'}
            </button>
          </form>
        </main>

        <footer>
          <strong>TrustNet</strong>

          <span>
            Evidence-first credibility analysis using
            LLMs and retrieval-augmented generation.
          </span>
        </footer>
      </div>
    )
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            T
          </div>

          <div>
            <h1>TrustNet</h1>
            <p>
              Evidence-first credibility analysis
            </p>
          </div>
        </div>

        <div className="topbar-actions">
          {configSource === 'secure' && (
            <span className="config-indicator">
              Secure config
            </span>
          )}

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

            {backendOnline === null
              && 'Checking API'}

            {backendOnline === true
              && 'API Online'}

            {backendOnline === false
              && 'API Offline'}
          </div>
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
              TrustNet retrieves relevant evidence,
              analyzes factual consistency and source
              quality, and produces an evidence-first
              credibility assessment.
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
                  Enter one factual claim for the
                  current MVP.
                </p>
              </div>

              <span className="mode-badge">
                Normal mode
              </span>
            </div>

            <textarea
              value={claim}
              onChange={(event) =>
                setClaim(
                  event.target.value
                )
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
                  loading
                  || !claim.trim()
                  || backendOnline !== true
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
            <strong>
              Analysis failed
            </strong>

            <span>
              {error}
            </span>
          </section>
        )}

        {loading && (
          <section className="loading-card">
            <div className="loader" />

            <div>
              <h3>
                Analyzing evidence
              </h3>

              <p>
                Retrieving sources and evaluating
                the claim. This can take several
                seconds.
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

                <h2>
                  {result.label}
                </h2>

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
                <span>
                  Confidence
                </span>

                <strong className="capitalize">
                  {result.confidence}
                </strong>
              </article>

              <article className="summary-card">
                <span>
                  Uncertainty
                </span>

                <strong className="capitalize">
                  {result.uncertainty.level}
                </strong>
              </article>

              <article className="summary-card">
                <span>
                  Evidence sources
                </span>

                <strong>
                  {result.evidence.length}
                </strong>
              </article>

              <article className="summary-card">
                <span>
                  Latency
                </span>

                <strong>
                  {(result.latency_ms / 1000)
                    .toFixed(1)}s
                </strong>
              </article>
            </div>

            {result.uncertainty
              .gate_triggered && (
              <div className="uncertainty-warning">
                <strong>
                  Insufficient evidence gate
                  triggered
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

                    <h3>
                      Factor breakdown
                    </h3>
                  </div>
                </div>

                <div className="factor-list">
                  {Object.entries(
                    result.factors
                  ).map(
                    ([key, factor]) => (
                      <article
                        className="factor-card"
                        key={key}
                      >
                        <div className="factor-top">
                          <strong>
                            {factorNames[key]
                              || key}
                          </strong>

                          <span>
                            {factor.score}/100
                          </span>
                        </div>

                        <div className="factor-track">
                          <div
                            className="factor-fill"
                            style={{
                              width:
                                `${factor.score}%`,
                            }}
                          />
                        </div>

                        <p>
                          {factor.justification}
                        </p>
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

                  <h3>
                    Evidence sources
                  </h3>
                </div>

                <span className="section-count">
                  {result.evidence.length}
                </span>
              </div>

              {result.evidence.length === 0 ? (
                <div className="empty-state">
                  No public evidence items were
                  returned.
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

                            <span>
                              {item.source}
                            </span>
                          </div>
                        </div>

                        <div className="badges">
                          <span
                            className={
                              getStanceClass(
                                item.stance
                              )
                            }
                          >
                            {item.stance}
                          </span>

                          <span
                            className={
                              getQualityClass(
                                item.quality
                              )
                            }
                          >
                            {item.quality}
                            {' '}
                            quality
                          </span>
                        </div>

                        <p>
                          {item.summary}
                        </p>
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

                    <h3>
                      Flags
                    </h3>
                  </div>
                </div>

                <div className="flags">
                  {result.flags.map(
                    (flag, index) => (
                      <span
                        className="flag"
                        key={`${flag}-${index}`}
                      >
                        {flag}
                      </span>
                    )
                  )}
                </div>
              </section>
            )}
          </section>
        )}
      </main>

      <footer>
        <strong>
          TrustNet
        </strong>

        <span>
          Evidence-first credibility analysis using
          LLMs and retrieval-augmented generation.
        </span>
      </footer>
    </div>
  )
}

export default App