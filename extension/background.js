const API_BASE =
  'http://127.0.0.1:8000'

async function checkBackend() {
  const controller =
    new AbortController()

  const timeoutId =
    setTimeout(
      () => {
        controller.abort()
      },
      3000,
    )

  try {
    const response =
      await fetch(
        `${API_BASE}/health`,
        {
          signal:
            controller.signal,
        },
      )

    if (
      !response.ok
    ) {
      return {
        ok: false,
        error:
          'TrustNet backend is unavailable.',
      }
    }

    const data =
      await response.json()

    if (
      data.status !== 'ok'
    ) {
      return {
        ok: false,
        error:
          'TrustNet backend is not ready.',
      }
    }

    return {
      ok: true,
    }
  } catch {
    return {
      ok: false,
      error:
        'TrustNet desktop app is not running or the backend is unavailable.',
    }
  } finally {
    clearTimeout(
      timeoutId,
    )
  }
}

async function analyzeClaim(
  claim,
) {
  const trimmedClaim =
    String(
      claim ?? '',
    ).trim()

  if (
    !trimmedClaim
  ) {
    return {
      ok: false,
      error:
        'A claim is required.',
    }
  }

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

    let data = null

    try {
      data =
        await response.json()
    } catch {
      data = null
    }

    if (
      !response.ok
    ) {
      return {
        ok: false,

        error:
          data?.detail
          || 'TrustNet credibility analysis failed.',
      }
    }

    return {
      ok: true,
      result: data,
    }
  } catch {
    return {
      ok: false,
      error:
        'Unable to connect to the TrustNet backend.',
    }
  }
}

chrome.runtime.onMessage.addListener(
  (
    message,
    _sender,
    sendResponse,
  ) => {
    if (
      message?.type
      === 'trustnet:health'
    ) {
      checkBackend()
        .then(
          sendResponse,
        )

      return true
    }

    if (
      message?.type
      === 'trustnet:analyze'
    ) {
      analyzeClaim(
        message.claim,
      ).then(
        sendResponse,
      )

      return true
    }

    return false
  },
)