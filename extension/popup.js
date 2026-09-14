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

const backendStatus =
  document.getElementById(
    'backendStatus',
  )

const statusText =
  document.getElementById(
    'statusText',
  )

const claimInput =
  document.getElementById(
    'claimInput',
  )

const analyzeButton =
  document.getElementById(
    'analyzeButton',
  )

const characterCount =
  document.getElementById(
    'characterCount',
  )

const selectionMessage =
  document.getElementById(
    'selectionMessage',
  )

const loadingSection =
  document.getElementById(
    'loadingSection',
  )

const errorSection =
  document.getElementById(
    'errorSection',
  )

const errorText =
  document.getElementById(
    'errorText',
  )

const resultSection =
  document.getElementById(
    'resultSection',
  )

const resultLabel =
  document.getElementById(
    'resultLabel',
  )

const resultClaim =
  document.getElementById(
    'resultClaim',
  )

const scoreCircle =
  document.getElementById(
    'scoreCircle',
  )

const scoreValue =
  document.getElementById(
    'scoreValue',
  )

const confidenceValue =
  document.getElementById(
    'confidenceValue',
  )

const sourceCount =
  document.getElementById(
    'sourceCount',
  )

const uncertaintyBox =
  document.getElementById(
    'uncertaintyBox',
  )

const uncertaintyReason =
  document.getElementById(
    'uncertaintyReason',
  )

const factorList =
  document.getElementById(
    'factorList',
  )

const evidenceList =
  document.getElementById(
    'evidenceList',
  )

let backendOnline = false
let analyzing = false


function updateCharacterCount() {
  characterCount.textContent =
    String(
      claimInput.value.length,
    )
}


function updateAnalyzeButton() {
  analyzeButton.disabled =
    analyzing
    || !backendOnline
    || !claimInput.value.trim()
}


function setBackendStatus(
  state,
  text,
) {
  backendStatus.className =
    `status ${state}`

  statusText.textContent =
    text
}


function hideMessages() {
  errorSection.classList.add(
    'hidden',
  )

  resultSection.classList.add(
    'hidden',
  )
}


function showError(
  message,
) {
  loadingSection.classList.add(
    'hidden',
  )

  resultSection.classList.add(
    'hidden',
  )

  errorText.textContent =
    message

  errorSection.classList.remove(
    'hidden',
  )
}


function getScoreClass(
  score,
) {
  if (
    score === null
    || score === undefined
  ) {
    return ''
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


function clearChildren(
  element,
) {
  while (
    element.firstChild
  ) {
    element.removeChild(
      element.firstChild,
    )
  }
}


function renderFactors(
  factors,
) {
  clearChildren(
    factorList,
  )

  if (
    !factors
  ) {
    return
  }

  Object.entries(
    factors,
  ).forEach(
    (
      [
        key,
        factor,
      ],
    ) => {
      const item =
        document.createElement(
          'div',
        )

      item.className =
        'factor'

      const heading =
        document.createElement(
          'div',
        )

      heading.className =
        'factor-heading'

      const name =
        document.createElement(
          'strong',
        )

      name.textContent =
        factorNames[key]
        || key

      const score =
        document.createElement(
          'span',
        )

      score.textContent =
        `${factor.score}/100`

      heading.append(
        name,
        score,
      )

      const track =
        document.createElement(
          'div',
        )

      track.className =
        'factor-track'

      const fill =
        document.createElement(
          'div',
        )

      fill.className =
        'factor-fill'

      fill.style.width =
        `${Math.max(
          0,
          Math.min(
            100,
            Number(
              factor.score,
            ) || 0,
          ),
        )}%`

      track.appendChild(
        fill,
      )

      item.append(
        heading,
        track,
      )

      factorList.appendChild(
        item,
      )
    },
  )
}


function renderEvidence(
  evidence,
) {
  clearChildren(
    evidenceList,
  )

  if (
    !Array.isArray(
      evidence,
    )
    || evidence.length === 0
  ) {
    const empty =
      document.createElement(
        'div',
      )

    empty.className =
      'evidence-item'

    empty.textContent =
      'No evidence sources returned.'

    evidenceList.appendChild(
      empty,
    )

    return
  }

  evidence.forEach(
    (item) => {
      const card =
        document.createElement(
          'div',
        )

      card.className =
        'evidence-item'

      const link =
        document.createElement(
          'a',
        )

      link.href =
        item.url

      link.target =
        '_blank'

      link.rel =
        'noreferrer'

      link.textContent =
        item.title
        || item.url

      const meta =
        document.createElement(
          'div',
        )

      meta.className =
        'evidence-meta'

      meta.textContent =
        [
          item.source,
          item.stance,
          item.quality
            ? `${item.quality} quality`
            : null,
        ]
          .filter(Boolean)
          .join(' • ')

      const summary =
        document.createElement(
          'p',
        )

      summary.className =
        'evidence-summary'

      summary.textContent =
        item.summary
        || 'No summary available.'

      card.append(
        link,
        meta,
        summary,
      )

      evidenceList.appendChild(
        card,
      )
    },
  )
}


function renderResult(
  result,
) {
  resultLabel.textContent =
    result.label
    || 'Unknown'

  resultClaim.textContent =
    result.claim
    || claimInput.value.trim()

  const score =
    result.overall_score

  scoreValue.textContent =
    score === null
    || score === undefined
      ? '—'
      : String(score)

  scoreCircle.className =
    `score-circle ${getScoreClass(
      score,
    )}`

  confidenceValue.textContent =
    result.confidence
    || 'Unknown'

  sourceCount.textContent =
    String(
      result.evidence?.length
      ?? 0,
    )

  if (
    result.uncertainty
      ?.gate_triggered
  ) {
    uncertaintyReason.textContent =
      result.uncertainty.reason
      || ''

    uncertaintyBox.classList.remove(
      'hidden',
    )
  } else {
    uncertaintyBox.classList.add(
      'hidden',
    )
  }

  renderFactors(
    result.factors,
  )

  renderEvidence(
    result.evidence,
  )

  loadingSection.classList.add(
    'hidden',
  )

  errorSection.classList.add(
    'hidden',
  )

  resultSection.classList.remove(
    'hidden',
  )
}


async function loadSelection() {
  try {
    const tabs =
      await chrome.tabs.query({
        active: true,
        currentWindow: true,
      })

    const tab =
      tabs[0]

    if (
      !tab?.id
    ) {
      return
    }

    const results =
      await chrome.scripting.executeScript({
        target: {
          tabId:
            tab.id,
        },

        func: () => (
          window
            .getSelection()
            ?.toString()
            .trim()
          || ''
        ),
      })

    const selection =
      results?.[0]?.result
      || ''

    if (
      selection
    ) {
      claimInput.value =
        selection

      selectionMessage.textContent =
        'Selected text loaded from the current page.'
    } else {
      selectionMessage.textContent =
        'No text is currently selected. You can type or paste a claim manually.'
    }
  } catch {
    selectionMessage.textContent =
      'TrustNet cannot read text from this browser page. You can still paste a claim manually.'
  }

  updateCharacterCount()
  updateAnalyzeButton()
}


async function checkBackend() {
  setBackendStatus(
    'starting',
    'Checking...',
  )

  try {
    const response =
      await chrome.runtime.sendMessage({
        type:
          'trustnet:health',
      })

    if (
      response?.ok
    ) {
      backendOnline = true

      setBackendStatus(
        'online',
        'API Online',
      )
    } else {
      backendOnline = false

      setBackendStatus(
        'offline',
        'API Offline',
      )

      selectionMessage.textContent =
        response?.error
        || 'Start the TrustNet desktop app to enable analysis.'
    }
  } catch {
    backendOnline = false

    setBackendStatus(
      'offline',
      'API Offline',
    )
  }

  updateAnalyzeButton()
}


async function analyzeCurrentClaim() {
  const claim =
    claimInput.value.trim()

  if (
    !claim
    || !backendOnline
    || analyzing
  ) {
    return
  }

  analyzing = true
  updateAnalyzeButton()

  hideMessages()

  loadingSection.classList.remove(
    'hidden',
  )

  try {
    const response =
      await chrome.runtime.sendMessage({
        type:
          'trustnet:analyze',

        claim,
      })

    if (
      !response?.ok
    ) {
      throw new Error(
        response?.error
        || 'TrustNet credibility analysis failed.',
      )
    }

    renderResult(
      response.result,
    )
  } catch (error) {
    showError(
      error.message
      || 'Unable to analyze this claim.',
    )
  } finally {
    analyzing = false
    updateAnalyzeButton()
  }
}


claimInput.addEventListener(
  'input',
  () => {
    updateCharacterCount()
    updateAnalyzeButton()
  },
)

analyzeButton.addEventListener(
  'click',
  analyzeCurrentClaim,
)

document.addEventListener(
  'DOMContentLoaded',
  async () => {
    updateCharacterCount()

    await Promise.all([
      loadSelection(),
      checkBackend(),
    ])
  },
)