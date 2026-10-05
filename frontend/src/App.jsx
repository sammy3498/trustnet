import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react'

import './App.css'

const API_BASE =
  'http://127.0.0.1:8000'

const HEALTH_FAILURE_LIMIT = 8

const factorNames = {
  evidence_strength:
    'Evidence Strength',
  factual_consistency:
    'Factual Consistency',
  source_provenance:
    'Source Provenance',
  bias_framing:
    'Bias & Framing',
  incentive_conflict:
    'Incentive / Conflict',
}

function App() {
  const [claim, setClaim] =
    useState('')

  const [result, setResult] =
    useState(null)

  const [error, setError] =
    useState('')

  const [loading, setLoading] =
    useState(false)

  const [
    backendOnline,
    setBackendOnline,
  ] = useState(null)

  const backendFailureCount =
    useRef(0)

  const [
    configChecked,
    setConfigChecked,
  ] = useState(false)

  const [
    configStatus,
    setConfigStatus,
  ] = useState(null)

  const [
    configured,
    setConfigured,
  ] = useState(true)

  const [
    savingConfig,
    setSavingConfig,
  ] = useState(false)

  const [
    configError,
    setConfigError,
  ] = useState('')

  const [
    llmApiKey,
    setLlmApiKey,
  ] = useState('')

  const [
    tavilyApiKey,
    setTavilyApiKey,
  ] = useState('')

  const [
    llmModel,
    setLlmModel,
  ] = useState('')

  const [
    settingsOpen,
    setSettingsOpen,
  ] = useState(false)

  const [
    settingsLlmApiKey,
    setSettingsLlmApiKey,
  ] = useState('')

  const [
    settingsTavilyApiKey,
    setSettingsTavilyApiKey,
  ] = useState('')

  const [
    settingsLlmModel,
    setSettingsLlmModel,
  ] = useState('')

  const [
    settingsSaving,
    setSettingsSaving,
  ] = useState(false)

  const [
    settingsError,
    setSettingsError,
  ] = useState('')

  const markBackendStarting =
    useCallback(
      () => {
        backendFailureCount.current = 0
        setBackendOnline(null)
      },
      [],
    )

  const checkBackend =
    useCallback(
      async () => {
        try {
          const response =
            await fetch(
              `${API_BASE}/health`,
            )

          const data =
            await response.json()

          const online =
            response.ok
            && data.status === 'ok'

          if (online) {
            backendFailureCount.current = 0
            setBackendOnline(true)

            return true
          }
        } catch {
          // Handled below as a failed health check.
        }

        backendFailureCount.current += 1

        if (
          backendFailureCount.current
          >= HEALTH_FAILURE_LIMIT
        ) {
          setBackendOnline(false)
        } else {
          setBackendOnline(null)
        }

        return false
      },
      [],
    )

  useEffect(
    () => {
      let cancelled = false

      async function initializeApp() {
        if (
          window.trustnetDesktop
          && window
            .trustnetDesktop
            .getConfigStatus
        ) {
          try {
            const status =
              await window
                .trustnetDesktop
                .getConfigStatus()

            if (cancelled) {
              return
            }

            setConfigStatus(
              status,
            )

            setConfigured(
              status.configured,
            )

            if (
              status.llmModel
            ) {
              setLlmModel(
                status.llmModel,
              )
            }

            setConfigChecked(true)

            if (
              status.configured
            ) {
              markBackendStarting()
              await checkBackend()
            } else {
              setBackendOnline(
                false,
              )
            }

            return
          } catch {
            if (cancelled) {
              return
            }

            setConfigured(true)
            setConfigChecked(true)

            markBackendStarting()
            await checkBackend()

            return
          }
        }

        setConfigured(true)
        setConfigChecked(true)

        markBackendStarting()
        await checkBackend()
      }

      initializeApp()

      return () => {
        cancelled = true
      }
    },
    [
      checkBackend,
      markBackendStarting,
    ],
  )

  useEffect(
    () => {
      if (
        !configChecked
        || !configured
      ) {
        return undefined
      }

      const intervalId =
        window.setInterval(
          () => {
            checkBackend()
          },
          2000,
        )

      return () => {
        window.clearInterval(
          intervalId,
        )
      }
    },
    [
      checkBackend,
      configChecked,
      configured,
    ],
  )

  async function handleSaveConfig(
    event,
  ) {
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
        'OpenAI API key, Tavily API key, and model are required.',
      )

      return
    }

    if (
      !window.trustnetDesktop
      || !window
        .trustnetDesktop
        .saveConfig
    ) {
      setConfigError(
        'Desktop configuration is unavailable in this environment.',
      )

      return
    }

    setSavingConfig(true)
    setConfigError('')

    try {
      const response =
        await window
          .trustnetDesktop
          .saveConfig({
            llmApiKey:
              openAiKey,

            tavilyApiKey:
              tavilyKey,

            llmModel:
              model,
          })

      if (
        !response.ok
      ) {
        throw new Error(
          response.error
          || 'Unable to save TrustNet configuration.',
        )
      }

      const nextStatus =
        response.status
        || {
          configured: true,
          source: 'secure',
          editable: true,
          llmModel: model,
          hasLlmApiKey: true,
          hasTavilyApiKey: true,
        }

      setConfigStatus(
        nextStatus,
      )

      setConfigured(true)

      setLlmApiKey('')
      setTavilyApiKey('')

      backendFailureCount.current = 0
      setBackendOnline(true)

      await checkBackend()
    } catch (saveError) {
      setConfigError(
        saveError.message
        || 'Unable to save TrustNet configuration.',
      )
    } finally {
      setSavingConfig(false)
    }
  }

  function openSettings() {
    setSettingsLlmApiKey('')
    setSettingsTavilyApiKey('')

    setSettingsLlmModel(
      configStatus?.llmModel
      || '',
    )

    setSettingsError('')
    setSettingsOpen(true)
  }

  function closeSettings() {
    if (
      settingsSaving
    ) {
      return
    }

    setSettingsOpen(false)
    setSettingsError('')

    setSettingsLlmApiKey('')
    setSettingsTavilyApiKey('')
  }

  async function handleSaveSettings(
    event,
  ) {
    event.preventDefault()

    if (
      !window.trustnetDesktop
      || !window
        .trustnetDesktop
        .saveConfig
    ) {
      setSettingsError(
        'Desktop configuration is unavailable.',
      )

      return
    }

    const model =
      settingsLlmModel.trim()

    if (
      !model
    ) {
      setSettingsError(
        'An OpenAI model is required.',
      )

      return
    }

    setSettingsSaving(true)
    setSettingsError('')

    markBackendStarting()

    try {
      const response =
        await window
          .trustnetDesktop
          .saveConfig({
            llmApiKey:
              settingsLlmApiKey
                .trim(),

            tavilyApiKey:
              settingsTavilyApiKey
                .trim(),

            llmModel:
              model,
          })

      if (
        !response.ok
      ) {
        throw new Error(
          response.error
          || 'Unable to update TrustNet configuration.',
        )
      }

      if (
        response.status
      ) {
        setConfigStatus(
          response.status,
        )
      } else {
        setConfigStatus(
          (previous) => ({
            ...previous,
            configured: true,
            source: 'secure',
            editable: true,
            llmModel: model,
            hasLlmApiKey: true,
            hasTavilyApiKey: true,
          }),
        )
      }

      setConfigured(true)

      backendFailureCount.current = 0
      setBackendOnline(true)

      setSettingsLlmApiKey('')
      setSettingsTavilyApiKey('')
      setSettingsOpen(false)

      await checkBackend()
    } catch (saveError) {
      setSettingsError(
        saveError.message
        || 'Unable to update TrustNet configuration.',
      )

      await checkBackend()
    } finally {
      setSettingsSaving(false)
    }
  }

  async function handleAnalyze(
    event,
  ) {
    event.preventDefault()

    const trimmedClaim =
      claim.trim()

    if (
      !trimmedClaim
    ) {
      setError(
        'Enter a claim before starting analysis.',
      )

      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const response =
        await fetch(
          `${API_BASE}/analyze`,
          {
            method: 'POST',

            headers: {
              'Content-Type':
                'application/json',
            },

            body:
              JSON.stringify({
                input:
                  trimmedClaim,

                input_type:
                  'claim',

                mode:
                  'normal',
              }),
          },
        )

      const data =
        await response.json()

      if (
        !response.ok
      ) {
        throw new Error(
          data.detail
          || 'Credibility analysis failed.',
        )
      }

      setResult(data)
    } catch (
      requestError
    ) {
      setError(
        requestError.message
        || 'Unable to connect to the TrustNet backend.',
      )
    } finally {
      setLoading(false)
    }
  }

  function getScoreClass(
    score,
  ) {
    if (
      score === null
      || score === undefined
    ) {
      return 'score-insufficient'
    }

    if (
      score >= 80
    ) {
      return 'score-high'
    }

    if (
      score >= 60
    ) {
      return 'score-moderate'
    }

    if (
      score >= 40
    ) {
      return 'score-low'
    }

    return 'score-poor'
  }

  function getStanceClass(
    stance,
  ) {
    return (
      `stance stance-${stance}`
    )
  }

  function getQualityClass(
    quality,
  ) {
    return (
      `quality quality-${quality}`
    )
  }

  if (
    !configChecked
  ) {
    return (
      <div className="app">
        <div className="startup-screen">
          <div className="loader" />

          <div>
            <h2>
              Starting TrustNet
            </h2>

            <p>
              Checking desktop
              configuration.
            </p>
          </div>
        </div>
      </div>
    )
  }

  if (
    !configured
  ) {
    return (
      <div className="app">
        <header className="topbar">
          <div className="brand">
            <div className="brand-mark">
              T
            </div>

            <div>
              <h1>
                TrustNet
              </h1>

              <p>
                Evidence-first
                credibility analysis
              </p>
            </div>
          </div>

          <div
            className=
              "backend-status offline"
          >
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
              Connect TrustNet
              to its
              <span>
                {' '}
                analysis services.
              </span>
            </h2>

            <p>
              TrustNet uses OpenAI
              for structured credibility
              reasoning and Tavily for
              evidence retrieval. Your
              credentials are encrypted
              locally on this Windows
              device and are not bundled
              into the application.
            </p>
          </section>

          <form
            className="setup-card"
            onSubmit={
              handleSaveConfig
            }
          >
            <div className="window-bar" aria-hidden="true">TRUSTNET / SETUP</div>
            <div className="setup-heading">
              <div>
                <h3>
                  API configuration
                </h3>

                <p>
                  Complete this once
                  to activate
                  credibility analysis.
                </p>
              </div>

              <span className="secure-badge">
                Local encrypted storage
              </span>
            </div>

            <label className="config-field">
              <span>
                OpenAI API key
              </span>

              <input
                type="password"
                value={llmApiKey}
                onChange={
                  (event) =>
                    setLlmApiKey(
                      event.target.value,
                    )
                }
                placeholder="sk-..."
                autoComplete="off"
                disabled={
                  savingConfig
                }
              />
            </label>

            <label className="config-field">
              <span>
                Tavily API key
              </span>

              <input
                type="password"
                value={tavilyApiKey}
                onChange={
                  (event) =>
                    setTavilyApiKey(
                      event.target.value,
                    )
                }
                placeholder="tvly-..."
                autoComplete="off"
                disabled={
                  savingConfig
                }
              />
            </label>

            <label className="config-field">
              <span>
                OpenAI model
              </span>

              <input
                type="text"
                value={llmModel}
                onChange={
                  (event) =>
                    setLlmModel(
                      event.target.value,
                    )
                }
                placeholder=
                  "Enter the model available to your API account"
                autoComplete="off"
                disabled={
                  savingConfig
                }
              />
            </label>

            {configError && (
              <div className="setup-error">
                {configError}
              </div>
            )}

            <div className="setup-note">
              TrustNet never stores these
              credentials inside the
              installer or Git repository.
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
              {
                savingConfig
                  ? 'Saving and starting TrustNet...'
                  : 'Save and start TrustNet'
              }
            </button>
          </form>
        </main>

        <footer>
          <strong>
            TrustNet
          </strong>

          <span>
            Evidence-first credibility
            analysis using LLMs and
            retrieval-augmented
            generation.
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
            <h1>
              TrustNet
            </h1>

            <p>
              Evidence-first
              credibility analysis
            </p>
          </div>
        </div>

        <div className="topbar-actions">
          {
            configStatus?.source
              === 'secure'
            && (
              <span className="config-indicator">
                Secure config
              </span>
            )
          }

          {
            configStatus?.editable
            && (
              <button
                type="button"
                className=
                  "settings-trigger"
                onClick={
                  openSettings
                }
              >
                Settings
              </button>
            )
          }

          <div
            className={
              `backend-status ${
                backendOnline === true
                  ? 'online'
                  : backendOnline === false
                    ? 'offline'
                    : ''
              }`
            }
          >
            <span className="status-dot" />

            {
              backendOnline === null
              && 'Starting API...'
            }

            {
              backendOnline === true
              && 'API Online'
            }

            {
              backendOnline === false
              && 'API Offline'
            }
          </div>
        </div>
      </header>

      <main className="main-content">
        <section className="hero">
          <div className="hero-copy">
            <span className="eyebrow">
              Intelligent credibility
              analysis
            </span>

            <h2>
              Check a claim against
              <span>
                {' '}
                real-world evidence.
              </span>
            </h2>

            <p>
              TrustNet retrieves
              relevant evidence,
              analyzes factual
              consistency and source
              quality, and produces an
              evidence-first credibility
              assessment.
            </p>
            <div className="desk-caption" aria-label="Analysis features">
              <span>Source retrieval</span><span>Five factors</span><span>Evidence report</span>
            </div>
          </div>

          <form
            className="analysis-card"
            onSubmit={
              handleAnalyze
            }
          >
            <div className="window-bar" aria-hidden="true">CLAIM / NEW ANALYSIS</div>
            <div className="input-heading">
              <div>
                <h3>
                  Analyze a claim
                </h3>

                <p>
                  Enter one factual claim to investigate.
                </p>
              </div>

              <span className="mode-badge">
                Normal mode
              </span>
            </div>

            <textarea
              aria-label="Claim to analyze"
              value={claim}
              onChange={
                (event) =>
                  setClaim(
                    event.target.value,
                  )
              }
              placeholder=
                "Example: The Earth orbits the Sun."
              rows={6}
              disabled={loading}
            />

            <div className="form-footer">
              <span className="character-count">
                {claim.length}
                {' '}
                characters
              </span>

              <button
                type="submit"
                disabled={
                  loading
                  || !claim.trim()
                  || backendOnline
                    !== true
                }
              >
                {
                  loading
                    ? 'Analyzing...'
                    : 'Analyze credibility'
                }
              </button>
            </div>
          </form>
        </section>

        {error && (
          <section
            role="alert"
            className=
              "message error-message"
          >
            <strong>
              Analysis failed
            </strong>

            <span>
              {error}
            </span>
          </section>
        )}

        {loading && (
          <section className="loading-card" role="status" aria-live="polite">
            <div className="loader" />

            <div>
              <h3>
                Analyzing evidence
              </h3>

              <p>
                Retrieving sources and
                evaluating the claim.
                This can take several
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
                className={
                  `score-circle ${
                    getScoreClass(
                      result.overall_score,
                    )
                  }`
                }
              >
                <span className="score-value">
                  {
                    result.overall_score
                    ?? '—'
                  }
                </span>

                <span className="score-total">
                  {
                    result.overall_score
                      === null
                      ? 'N/A'
                      : '/ 100'
                  }
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
                  {
                    result
                      .uncertainty
                      .level
                  }
                </strong>
              </article>

              <article className="summary-card">
                <span>
                  Evidence sources
                </span>

                <strong>
                  {
                    result
                      .evidence
                      .length
                  }
                </strong>
              </article>

              <article className="summary-card">
                <span>
                  Latency
                </span>

                <strong>
                  {
                    (
                      result.latency_ms
                      / 1000
                    ).toFixed(1)
                  }
                  s
                </strong>
              </article>
            </div>

            {
              result
                .uncertainty
                .gate_triggered
              && (
                <div
                  className=
                    "uncertainty-warning"
                >
                  <strong>
                    Insufficient evidence
                    gate triggered
                  </strong>

                  <span>
                    {
                      result
                        .uncertainty
                        .reason
                    }
                  </span>
                </div>
              )
            }

            {
              result.factors
              && (
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
                    {
                      Object.entries(
                        result.factors,
                      ).map(
                        (
                          [
                            key,
                            factor,
                          ],
                        ) => (
                          <article
                            className=
                              "factor-card"
                            key={key}
                          >
                            <div className="factor-top">
                              <strong>
                                {
                                  factorNames[
                                    key
                                  ]
                                  || key
                                }
                              </strong>

                              <span>
                                {
                                  factor.score
                                }
                                /100
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
                              {
                                factor
                                  .justification
                              }
                            </p>
                          </article>
                        ),
                      )
                    }
                  </div>
                </section>
              )
            }

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
                  {
                    result
                      .evidence
                      .length
                  }
                </span>
              </div>

              {
                result.evidence.length
                  === 0
                  ? (
                    <div className="empty-state">
                      No public evidence
                      items were returned.
                    </div>
                  )
                  : (
                    <div className="evidence-list">
                      {
                        result.evidence.map(
                          (
                            item,
                            index,
                          ) => (
                            <article
                              className=
                                "evidence-card"
                              key={
                                `${item.url}-${index}`
                              }
                            >
                              <div className="evidence-top">
                                <div
                                  className=
                                    "evidence-number"
                                >
                                  {
                                    index
                                    + 1
                                  }
                                </div>

                                <div
                                  className=
                                    "evidence-title"
                                >
                                  <a
                                    href={
                                      item.url
                                    }
                                    target="_blank"
                                    rel="noreferrer"
                                  >
                                    {
                                      item
                                        .title
                                    }
                                  </a>

                                  <span>
                                    {
                                      item
                                        .source
                                    }
                                  </span>
                                </div>
                              </div>

                              <div className="badges">
                                <span
                                  className={
                                    getStanceClass(
                                      item.stance,
                                    )
                                  }
                                >
                                  {
                                    item
                                      .stance
                                  }
                                </span>

                                <span
                                  className={
                                    getQualityClass(
                                      item.quality,
                                    )
                                  }
                                >
                                  {
                                    item
                                      .quality
                                  }
                                  {' '}
                                  quality
                                </span>
                              </div>

                              <p>
                                {
                                  item
                                    .summary
                                }
                              </p>
                            </article>
                          ),
                        )
                      }
                    </div>
                  )
              }
            </section>

            {
              result.flags.length
                > 0
              && (
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
                    {
                      result.flags.map(
                        (
                          flag,
                          index,
                        ) => (
                          <span
                            className="flag"
                            key={
                              `${flag}-${index}`
                            }
                          >
                            {flag}
                          </span>
                        ),
                      )
                    }
                  </div>
                </section>
              )
            }
          </section>
        )}
      </main>

      <footer>
        <strong>
          TrustNet
        </strong>

        <span>
          Evidence-first credibility
          analysis using LLMs and
          retrieval-augmented
          generation.
        </span>
      </footer>

      {
        settingsOpen
        && (
          <div
            className=
              "settings-backdrop"
            onMouseDown={
              (event) => {
                if (
                  event.target
                  === event.currentTarget
                ) {
                  closeSettings()
                }
              }
            }
          >
            <section
              className=
                "settings-modal"
              role="dialog"
              aria-modal="true"
              aria-labelledby=
                "settings-title"
            >
              <div className="settings-header">
                <div>
                  <span className="eyebrow">
                    Desktop configuration
                  </span>

                  <h2 id="settings-title">
                    TrustNet settings
                  </h2>

                  <p>
                    Update the credentials
                    or model used by the
                    local TrustNet backend.
                  </p>
                </div>

                <button
                  type="button"
                  className=
                    "settings-close"
                  onClick={
                    closeSettings
                  }
                  disabled={
                    settingsSaving
                  }
                  aria-label=
                    "Close settings"
                >
                  ×
                </button>
              </div>

              <form
                onSubmit={
                  handleSaveSettings
                }
              >
                <label className="config-field">
                  <span>
                    OpenAI API key
                  </span>

                  <input
                    type="password"
                    value={
                      settingsLlmApiKey
                    }
                    onChange={
                      (event) =>
                        setSettingsLlmApiKey(
                          event.target.value,
                        )
                    }
                    placeholder={
                      configStatus
                        ?.hasLlmApiKey
                        ? 'Stored securely — leave blank to keep it'
                        : 'Enter OpenAI API key'
                    }
                    autoComplete="off"
                    disabled={
                      settingsSaving
                    }
                  />
                </label>

                <label className="config-field">
                  <span>
                    Tavily API key
                  </span>

                  <input
                    type="password"
                    value={
                      settingsTavilyApiKey
                    }
                    onChange={
                      (event) =>
                        setSettingsTavilyApiKey(
                          event.target.value,
                        )
                    }
                    placeholder={
                      configStatus
                        ?.hasTavilyApiKey
                        ? 'Stored securely — leave blank to keep it'
                        : 'Enter Tavily API key'
                    }
                    autoComplete="off"
                    disabled={
                      settingsSaving
                    }
                  />
                </label>

                <label className="config-field">
                  <span>
                    OpenAI model
                  </span>

                  <input
                    type="text"
                    value={
                      settingsLlmModel
                    }
                    onChange={
                      (event) =>
                        setSettingsLlmModel(
                          event.target.value,
                        )
                    }
                    placeholder=
                      "OpenAI model"
                    autoComplete="off"
                    disabled={
                      settingsSaving
                    }
                  />
                </label>

                <div className="settings-security">
                  <strong>
                    Existing API keys are
                    never displayed.
                  </strong>

                  <span>
                    Leave a key field blank
                    to keep the encrypted
                    credential already
                    stored on this device.
                  </span>
                </div>

                <div className="settings-restart-note">
                  Saving settings restarts
                  the local TrustNet backend.
                  This does not perform a
                  credibility analysis or
                  consume analysis API calls.
                </div>

                {
                  settingsError
                  && (
                    <div className="setup-error">
                      {
                        settingsError
                      }
                    </div>
                  )
                }

                <div className="settings-actions">
                  <button
                    type="button"
                    className=
                      "secondary-button"
                    onClick={
                      closeSettings
                    }
                    disabled={
                      settingsSaving
                    }
                  >
                    Cancel
                  </button>

                  <button
                    type="submit"
                    disabled={
                      settingsSaving
                      || !settingsLlmModel
                        .trim()
                    }
                  >
                    {
                      settingsSaving
                        ? 'Saving and restarting...'
                        : 'Save settings'
                    }
                  </button>
                </div>
              </form>
            </section>
          </div>
        )
      }
    </div>
  )
}

export default App