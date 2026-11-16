const {
  app,
  BrowserWindow,
  Menu,
  dialog,
  ipcMain,
  safeStorage,
} = require('electron')

const {
  spawn,
  spawnSync,
} = require('child_process')

const http = require('http')
const fs = require('fs')
const path = require('path')


const ROOT_DIR = path.resolve(
  __dirname,
  '..',
  '..',
)

const FRONTEND_DIR = path.join(
  ROOT_DIR,
  'frontend',
)

const FRONTEND_URL =
  'http://127.0.0.1:5173'

const BACKEND_HEALTH_URL =
  'http://127.0.0.1:8000/health'

let viteProcess = null
let backendProcess = null
let backendLogFd = null


function getConfigDirectory() {
  return path.join(
    app.getPath('appData'),
    'TrustNet',
  )
}


function getSecureConfigPath() {
  return path.join(
    getConfigDirectory(),
    'config.json',
  )
}


function getLegacyEnvPath() {
  return path.join(
    getConfigDirectory(),
    '.env',
  )
}


function getBackendLogPath() {
  return path.join(
    getConfigDirectory(),
    'logs',
    'backend.log',
  )
}


function getWindowIconPath() {
  const iconPath =
    app.isPackaged
      ? path.join(
        process.resourcesPath,
        'app-icon.png',
      )
      : path.join(
        FRONTEND_DIR,
        'build',
        'icon.png',
      )

  if (
    fs.existsSync(
      iconPath,
    )
  ) {
    return iconPath
  }

  return undefined
}


function waitForUrl(
  url,
  timeoutMs = 60000,
) {
  const start = Date.now()

  return new Promise(
    (resolve, reject) => {
      function check() {
        const request = http.get(
          url,
          (response) => {
            response.resume()

            if (
              response.statusCode
              && response.statusCode >= 200
              && response.statusCode < 500
            ) {
              resolve()
              return
            }

            retry()
          },
        )

        request.setTimeout(
          1500,
          () => {
            request.destroy()
            retry()
          },
        )

        request.on(
          'error',
          () => {
            retry()
          },
        )
      }

      function retry() {
        if (
          Date.now() - start
          >= timeoutMs
        ) {
          reject(
            new Error(
              `Timed out waiting for ${url}`,
            ),
          )

          return
        }

        setTimeout(
          check,
          500,
        )
      }

      check()
    },
  )
}


function waitForUrlUnavailable(
  url,
  timeoutMs = 15000,
) {
  const start = Date.now()

  return new Promise(
    (resolve, reject) => {
      function check() {
        const request = http.get(
          url,
          (response) => {
            response.resume()
            retry()
          },
        )

        request.setTimeout(
          750,
          () => {
            request.destroy()
            resolve()
          },
        )

        request.on(
          'error',
          () => {
            resolve()
          },
        )
      }

      function retry() {
        if (
          Date.now() - start
          >= timeoutMs
        ) {
          reject(
            new Error(
              `Timed out waiting for ${url} to stop responding.`,
            ),
          )

          return
        }

        setTimeout(
          check,
          250,
        )
      }

      check()
    },
  )
}


function waitForProcessExit(
  childProcess,
  timeoutMs = 15000,
) {
  if (
    !childProcess
    || !childProcess.pid
  ) {
    return Promise.resolve()
  }

  if (
    childProcess.exitCode !== null
    || childProcess.signalCode !== null
  ) {
    return Promise.resolve()
  }

  return new Promise(
    (resolve, reject) => {
      let settled = false

      const timeoutId =
        setTimeout(
          () => {
            if (
              settled
            ) {
              return
            }

            settled = true

            reject(
              new Error(
                'Timed out waiting for the previous TrustNet backend to exit.',
              ),
            )
          },
          timeoutMs,
        )

      childProcess.once(
        'exit',
        () => {
          if (
            settled
          ) {
            return
          }

          settled = true

          clearTimeout(
            timeoutId,
          )

          resolve()
        },
      )
    },
  )
}


function hasSecureConfig() {
  return fs.existsSync(
    getSecureConfigPath(),
  )
}


function hasLegacyConfig() {
  return fs.existsSync(
    getLegacyEnvPath(),
  )
}


function readSecureConfigRecord() {
  return JSON.parse(
    fs.readFileSync(
      getSecureConfigPath(),
      'utf8',
    ),
  )
}


function decryptSecureValue(
  encodedValue,
  fieldName,
) {
  if (
    typeof encodedValue !== 'string'
    || !encodedValue
  ) {
    throw new Error(
      `Stored ${fieldName} is missing.`,
    )
  }

  return safeStorage.decryptString(
    Buffer.from(
      encodedValue,
      'base64',
    ),
  )
}


const SUPPORTED_PROVIDERS = new Set(['gemini', 'openai', 'openrouter'])

function normalizeProvider(value) {
  // Existing secure desktop installs were OpenAI-only.
  const provider = String(value || 'openai').trim().toLowerCase()
  if (!SUPPORTED_PROVIDERS.has(provider)) {
    throw new Error(`Unsupported LLM provider: ${provider}`)
  }
  return provider
}

function readSecureConfig() {
  if (!safeStorage.isEncryptionAvailable()) {
    throw new Error('Secure credential storage is unavailable.')
  }
  const rawConfig = readSecureConfigRecord()
  return {
    llmProvider: normalizeProvider(rawConfig.llmProvider),
    llmApiKey: decryptSecureValue(rawConfig.llmApiKey, 'LLM API key'),
    tavilyApiKey: decryptSecureValue(rawConfig.tavilyApiKey, 'Tavily API key'),
    llmModel: String(rawConfig.llmModel ?? '').trim(),
  }
}

function readSecureConfigMetadata() {
  const rawConfig = readSecureConfigRecord()
  return {
    llmProvider: normalizeProvider(rawConfig.llmProvider),
    llmModel: String(rawConfig.llmModel ?? '').trim(),
    hasLlmApiKey: typeof rawConfig.llmApiKey === 'string' && rawConfig.llmApiKey.length > 0,
    hasTavilyApiKey: typeof rawConfig.tavilyApiKey === 'string' && rawConfig.tavilyApiKey.length > 0,
  }
}

function parseLegacyEnv() {
  const contents = fs.readFileSync(
    getLegacyEnvPath(),
    'utf8',
  )

  const values = {}

  for (
    const rawLine
    of contents.split(/\r?\n/)
  ) {
    const line = rawLine.trim()

    if (
      !line
      || line.startsWith('#')
    ) {
      continue
    }

    const equalsIndex =
      line.indexOf('=')

    if (
      equalsIndex <= 0
    ) {
      continue
    }

    const key = line
      .slice(
        0,
        equalsIndex,
      )
      .trim()

    let value = line
      .slice(
        equalsIndex + 1,
      )
      .trim()

    if (
      value.length >= 2
      && (
        (
          value.startsWith('"')
          && value.endsWith('"')
        )
        || (
          value.startsWith("'")
          && value.endsWith("'")
        )
      )
    ) {
      value = value.slice(
        1,
        -1,
      )
    }

    values[key] = value
  }

  return values
}


function readLegacyConfig() {
  const values = parseLegacyEnv()
  const llmProvider = normalizeProvider(values.LLM_PROVIDER || 'openai')
  const providerVars = {
    gemini: ['GEMINI_API_KEY', 'GEMINI_MODEL'],
    openai: ['OPENAI_API_KEY', 'OPENAI_MODEL'],
    openrouter: ['OPENROUTER_API_KEY', 'OPENROUTER_MODEL'],
  }[llmProvider]
  return {
    llmProvider,
    llmApiKey: String(values[providerVars[0]] ?? values.LLM_API_KEY ?? '').trim(),
    tavilyApiKey: String(values.SEARCH_API_KEY ?? values.TAVILY_API_KEY ?? '').trim(),
    llmModel: String(values[providerVars[1]] ?? values.LLM_MODEL ?? '').trim(),
  }
}

function writeSecureConfig(config) {
  if (!safeStorage.isEncryptionAvailable()) {
    throw new Error('Secure credential storage is unavailable.')
  }
  const configDirectory = getConfigDirectory()
  fs.mkdirSync(configDirectory, { recursive: true })
  const secureConfig = {
    llmProvider: normalizeProvider(config.llmProvider),
    llmApiKey: safeStorage.encryptString(config.llmApiKey).toString('base64'),
    tavilyApiKey: safeStorage.encryptString(config.tavilyApiKey).toString('base64'),
    llmModel: config.llmModel,
  }
  fs.writeFileSync(getSecureConfigPath(), JSON.stringify(secureConfig, null, 2), 'utf8')
}

function getConfigStatus() {
  if (!app.isPackaged) {
    return {
      configured: true, source: 'development', editable: false,
      llmProvider: process.env.LLM_PROVIDER || 'gemini', llmModel: null,
      hasLlmApiKey: false, hasTavilyApiKey: false,
    }
  }
  for (const [exists, source, reader] of [
    [hasSecureConfig, 'secure', readSecureConfigMetadata],
    [hasLegacyConfig, 'legacy', () => {
      const c = readLegacyConfig()
      return {
        llmProvider: c.llmProvider, llmModel: c.llmModel,
        hasLlmApiKey: Boolean(c.llmApiKey), hasTavilyApiKey: Boolean(c.tavilyApiKey),
      }
    }],
  ]) {
    if (!exists()) continue
    try {
      const meta = reader()
      return {
        configured: meta.hasLlmApiKey && meta.hasTavilyApiKey && Boolean(meta.llmModel),
        source, editable: true, ...meta,
      }
    } catch {
      return {
        configured: false, source: `${source}-error`, editable: true,
        llmProvider: 'gemini', llmModel: '', hasLlmApiKey: false,
        hasTavilyApiKey: false,
      }
    }
  }
  return {
    configured: false, source: null, editable: true, llmProvider: 'gemini',
    llmModel: '', hasLlmApiKey: false, hasTavilyApiKey: false,
  }
}

function getExistingConfig() {
  try {
    if (hasSecureConfig()) return readSecureConfig()
    if (hasLegacyConfig()) return readLegacyConfig()
  } catch { /* Invalid old configuration: require fresh credentials. */ }
  return { llmProvider: 'gemini', llmApiKey: '', tavilyApiKey: '', llmModel: '' }
}

function resolveConfigUpdate(config) {
  const existing = getExistingConfig()
  const llmProvider = normalizeProvider(config?.llmProvider || existing.llmProvider)
  const providerChanged = llmProvider !== existing.llmProvider
  const resolvedConfig = {
    llmProvider,
    llmApiKey: String(config?.llmApiKey ?? '').trim()
      || (providerChanged ? '' : existing.llmApiKey),
    tavilyApiKey: String(config?.tavilyApiKey ?? '').trim() || existing.tavilyApiKey,
    llmModel: String(config?.llmModel ?? '').trim()
      || (providerChanged ? '' : existing.llmModel),
  }
  if (!resolvedConfig.llmApiKey || !resolvedConfig.tavilyApiKey || !resolvedConfig.llmModel) {
    throw new Error('LLM API key, Tavily API key, and model are required.')
  }
  if (resolvedConfig.llmApiKey.length > 4096 ||
      resolvedConfig.tavilyApiKey.length > 4096 ||
      resolvedConfig.llmModel.length > 200) {
    throw new Error('One or more configuration values are too long.')
  }
  return resolvedConfig
}

function closeBackendLog() {
  if (
    backendLogFd === null
  ) {
    return
  }

  try {
    fs.closeSync(
      backendLogFd,
    )
  } catch {
    // The descriptor may already be closed.
  }

  backendLogFd = null
}


function startDevelopmentBackend() {
  const pythonPath =
    process.platform === 'win32'
      ? path.join(
        ROOT_DIR,
        '.venv',
        'Scripts',
        'python.exe',
      )
      : path.join(
        ROOT_DIR,
        '.venv',
        'bin',
        'python',
      )

  if (
    !fs.existsSync(
      pythonPath,
    )
  ) {
    throw new Error(
      `TrustNet Python environment not found at ${pythonPath}`,
    )
  }

  backendProcess = spawn(
    pythonPath,
    [
      '-m',
      'uvicorn',
      'backend.app.main:app',
      '--host',
      '127.0.0.1',
      '--port',
      '8000',
    ],
    {
      cwd: ROOT_DIR,
      stdio: 'inherit',
      windowsHide: true,
    },
  )

  backendProcess.on(
    'error',
    (error) => {
      console.error(
        'Failed to start TrustNet development backend:',
        error,
      )
    },
  )
}


function startPackagedBackend() {
  const backendPath = path.join(
    process.resourcesPath,
    'backend',
    'trustnet-backend.exe',
  )

  const embeddingModelPath = path.join(
    process.resourcesPath,
    'models',
    'embedding',
  )

  const rerankerModelPath = path.join(
    process.resourcesPath,
    'models',
    'reranker',
  )

  if (
    !fs.existsSync(
      backendPath,
    )
  ) {
    throw new Error(
      `Packaged TrustNet backend not found at ${backendPath}`,
    )
  }

  if (
    !fs.existsSync(
      embeddingModelPath,
    )
  ) {
    throw new Error(
      `Embedding model not found at ${embeddingModelPath}`,
    )
  }

  if (
    !fs.existsSync(
      rerankerModelPath,
    )
  ) {
    throw new Error(
      `Reranker model not found at ${rerankerModelPath}`,
    )
  }

  const configDirectory =
    getConfigDirectory()

  const logDirectory = path.join(
    configDirectory,
    'logs',
  )

  fs.mkdirSync(
    logDirectory,
    {
      recursive: true,
    },
  )

  const logFilePath =
    getBackendLogPath()

  const childEnvironment = {
    ...process.env,
    EMBEDDING_MODEL:
      embeddingModelPath,
    RERANKER_MODEL:
      rerankerModelPath,
    HF_HUB_OFFLINE:
      '1',
  }

  let config
  let configurationSource
  if (hasSecureConfig()) {
    config = readSecureConfig()
    configurationSource = 'secure'
  } else if (hasLegacyConfig()) {
    config = readLegacyConfig()
    configurationSource = 'legacy'
  } else {
    throw new Error('TrustNet API configuration has not been set up.')
  }
  const providerKeys = {
    gemini: ['GEMINI_API_KEY', 'GEMINI_MODEL'],
    openai: ['OPENAI_API_KEY', 'OPENAI_MODEL'],
    openrouter: ['OPENROUTER_API_KEY', 'OPENROUTER_MODEL'],
  }
  // Do not inherit unrelated provider keys from the parent process.
  for (const keys of Object.values(providerKeys)) {
    for (const key of keys) delete childEnvironment[key]
  }
  delete childEnvironment.TRUSTNET_ENV_FILE
  childEnvironment.LLM_PROVIDER = config.llmProvider
  childEnvironment[providerKeys[config.llmProvider][0]] = config.llmApiKey
  childEnvironment[providerKeys[config.llmProvider][1]] = config.llmModel
  childEnvironment.SEARCH_API_KEY = config.tavilyApiKey

  fs.appendFileSync(
    logFilePath,
    (
      `\n\n===== TrustNet startup ${new Date().toISOString()} =====\n`
      + `Backend: ${backendPath}\n`
      + `Embedding model: ${embeddingModelPath}\n`
      + `Reranker model: ${rerankerModelPath}\n`
      + `Configuration source: ${configurationSource}\n`
    ),
  )

  closeBackendLog()

  backendLogFd = fs.openSync(
    logFilePath,
    'a',
  )

  const spawnedBackend = spawn(
    backendPath,
    [],
    {
      cwd: path.dirname(
        backendPath,
      ),
      stdio: [
        'ignore',
        backendLogFd,
        backendLogFd,
      ],
      windowsHide: true,
      env: childEnvironment,
    },
  )

  backendProcess =
    spawnedBackend

  fs.appendFileSync(
    logFilePath,
    `Spawned backend PID: ${spawnedBackend.pid ?? 'unknown'}\n`,
  )

  spawnedBackend.on(
    'error',
    (error) => {
      fs.appendFileSync(
        logFilePath,
        `Backend spawn error: ${error.stack || error}\n`,
      )
    },
  )

  spawnedBackend.on(
    'exit',
    (code, signal) => {
      fs.appendFileSync(
        logFilePath,
        `Backend exited. Code: ${code}, signal: ${signal}\n`,
      )

      if (
        backendProcess
        === spawnedBackend
      ) {
        backendProcess = null
      }
    },
  )
}


function startVite() {
  if (
    process.platform === 'win32'
  ) {
    const commandProcessor =
      process.env.ComSpec
      || 'C:\\Windows\\System32\\cmd.exe'

    viteProcess = spawn(
      commandProcessor,
      [
        '/d',
        '/s',
        '/c',
        'npm.cmd run dev -- --host 127.0.0.1',
      ],
      {
        cwd: FRONTEND_DIR,
        stdio: 'inherit',
        windowsHide: true,
      },
    )
  } else {
    viteProcess = spawn(
      'npm',
      [
        'run',
        'dev',
        '--',
        '--host',
        '127.0.0.1',
      ],
      {
        cwd: FRONTEND_DIR,
        stdio: 'inherit',
      },
    )
  }

  viteProcess.on(
    'error',
    (error) => {
      console.error(
        'Failed to start Vite:',
        error,
      )
    },
  )
}


function stopProcess(
  childProcess,
) {
  if (
    !childProcess
    || !childProcess.pid
  ) {
    return
  }

  if (
    process.platform === 'win32'
  ) {
    spawnSync(
      'taskkill',
      [
        '/PID',
        String(
          childProcess.pid,
        ),
        '/T',
        '/F',
      ],
      {
        stdio: 'ignore',
      },
    )

    return
  }

  childProcess.kill(
    'SIGTERM',
  )
}


async function stopPackagedBackend() {
  const previousBackend =
    backendProcess

  if (
    !previousBackend
  ) {
    await waitForUrlUnavailable(
      BACKEND_HEALTH_URL,
      5000,
    ).catch(
      () => {},
    )

    return
  }

  backendProcess = null

  const exitPromise =
    waitForProcessExit(
      previousBackend,
      15000,
    )

  stopProcess(
    previousBackend,
  )

  try {
    await exitPromise
  } catch {
    // taskkill is forceful on Windows, so continue
    // to the endpoint-release check below.
  }

  closeBackendLog()

  await waitForUrlUnavailable(
    BACKEND_HEALTH_URL,
    15000,
  )
}


async function restartPackagedBackend() {
  await stopPackagedBackend()

  await new Promise(
    (resolve) => {
      setTimeout(
        resolve,
        500,
      )
    },
  )

  startPackagedBackend()

  await waitForUrl(
    BACKEND_HEALTH_URL,
    180000,
  )
}


function registerIpcHandlers() {
  ipcMain.handle(
    'trustnet:get-config-status',
    () => (
      getConfigStatus()
    ),
  )

  ipcMain.handle(
    'trustnet:save-config',
    async (
      _event,
      config,
    ) => {
      if (
        !app.isPackaged
      ) {
        return {
          ok: false,
          error:
            'Desktop API configuration is only used by packaged builds.',
        }
      }

      try {
        const resolvedConfig =
          resolveConfigUpdate(
            config,
          )

        writeSecureConfig(
          resolvedConfig,
        )

        await restartPackagedBackend()

        return {
          ok: true,
          status:
            getConfigStatus(),
        }
      } catch (error) {
        return {
          ok: false,
          error:
            error instanceof Error
              ? error.message
              : String(error),
        }
      }
    },
  )
}


function createWindow() {
  Menu.setApplicationMenu(
    null,
  )

  const window =
    new BrowserWindow({
      title: 'TrustNet',
      width: 1400,
      height: 900,
      minWidth: 1000,
      minHeight: 700,
      show: false,
      autoHideMenuBar: true,
      backgroundColor: '#071525',
      icon: getWindowIconPath(),
      webPreferences: {
        preload: path.join(
          __dirname,
          'preload.cjs',
        ),
        contextIsolation: true,
        nodeIntegration: false,
      },
    })

  window.once(
    'ready-to-show',
    () => {
      window.show()
    },
  )

  if (
    app.isPackaged
  ) {
    window.loadFile(
      path.join(
        __dirname,
        '..',
        'dist',
        'index.html',
      ),
    )
  } else {
    window.loadURL(
      FRONTEND_URL,
    )
  }
}


async function startDevelopmentApp() {
  startDevelopmentBackend()
  startVite()

  await Promise.all([
    waitForUrl(
      BACKEND_HEALTH_URL,
    ),
    waitForUrl(
      FRONTEND_URL,
    ),
  ])

  createWindow()
}


function startPackagedApp() {
  createWindow()

  const status =
    getConfigStatus()

  if (
    !status.configured
  ) {
    return
  }

  try {
    startPackagedBackend()
  } catch (error) {
    console.error(
      'TrustNet packaged backend failed to start:',
      error,
    )

    dialog.showErrorBox(
      'TrustNet backend failed to start',
      (
        `${String(error)}\n\n`
        + `Backend log:\n${getBackendLogPath()}`
      ),
    )
  }
}


app.whenReady().then(
  async () => {
    registerIpcHandlers()

    try {
      if (
        app.isPackaged
      ) {
        startPackagedApp()
      } else {
        await startDevelopmentApp()
      }
    } catch (error) {
      console.error(
        'TrustNet startup failed:',
        error,
      )

      dialog.showErrorBox(
        'TrustNet startup failed',
        String(
          error,
        ),
      )

      app.quit()
      return
    }

    app.on(
      'activate',
      () => {
        if (
          BrowserWindow
            .getAllWindows()
            .length === 0
        ) {
          createWindow()
        }
      },
    )
  },
)


app.on(
  'before-quit',
  () => {
    stopProcess(
      viteProcess,
    )

    stopProcess(
      backendProcess,
    )

    closeBackendLog()
  },
)


app.on(
  'window-all-closed',
  () => {
    if (
      process.platform !== 'darwin'
    ) {
      app.quit()
    }
  },
)