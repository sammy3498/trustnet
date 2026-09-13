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


function readSecureConfig() {
  if (
    !safeStorage.isEncryptionAvailable()
  ) {
    throw new Error(
      'Secure credential storage is unavailable.',
    )
  }

  const rawConfig =
    readSecureConfigRecord()

  return {
    llmApiKey:
      decryptSecureValue(
        rawConfig.llmApiKey,
        'OpenAI API key',
      ),

    tavilyApiKey:
      decryptSecureValue(
        rawConfig.tavilyApiKey,
        'Tavily API key',
      ),

    llmModel:
      String(
        rawConfig.llmModel ?? '',
      ).trim(),
  }
}


function readSecureConfigMetadata() {
  const rawConfig =
    readSecureConfigRecord()

  const llmModel = String(
    rawConfig.llmModel ?? '',
  ).trim()

  return {
    llmModel,

    hasLlmApiKey:
      typeof rawConfig.llmApiKey
        === 'string'
      && rawConfig.llmApiKey.length > 0,

    hasTavilyApiKey:
      typeof rawConfig.tavilyApiKey
        === 'string'
      && rawConfig.tavilyApiKey.length > 0,
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
  const values =
    parseLegacyEnv()

  return {
    llmApiKey:
      String(
        values.LLM_API_KEY ?? '',
      ).trim(),

    tavilyApiKey:
      String(
        values.TAVILY_API_KEY ?? '',
      ).trim(),

    llmModel:
      String(
        values.LLM_MODEL ?? '',
      ).trim(),
  }
}


function writeSecureConfig(
  config,
) {
  if (
    !safeStorage.isEncryptionAvailable()
  ) {
    throw new Error(
      'Secure credential storage is unavailable.',
    )
  }

  const configDirectory =
    getConfigDirectory()

  fs.mkdirSync(
    configDirectory,
    {
      recursive: true,
    },
  )

  const secureConfig = {
    llmApiKey:
      safeStorage.encryptString(
        config.llmApiKey,
      ).toString(
        'base64',
      ),

    tavilyApiKey:
      safeStorage.encryptString(
        config.tavilyApiKey,
      ).toString(
        'base64',
      ),

    llmModel:
      config.llmModel,
  }

  fs.writeFileSync(
    getSecureConfigPath(),
    JSON.stringify(
      secureConfig,
      null,
      2,
    ),
    'utf8',
  )
}


function getConfigStatus() {
  if (
    !app.isPackaged
  ) {
    return {
      configured: true,
      source: 'development',
      editable: false,
      llmModel: null,
      hasLlmApiKey: false,
      hasTavilyApiKey: false,
    }
  }

  if (
    hasSecureConfig()
  ) {
    try {
      const metadata =
        readSecureConfigMetadata()

      return {
        configured:
          metadata.hasLlmApiKey
          && metadata.hasTavilyApiKey
          && Boolean(
            metadata.llmModel,
          ),

        source: 'secure',
        editable: true,
        llmModel:
          metadata.llmModel,

        hasLlmApiKey:
          metadata.hasLlmApiKey,

        hasTavilyApiKey:
          metadata.hasTavilyApiKey,
      }
    } catch {
      return {
        configured: false,
        source: 'secure-error',
        editable: true,
        llmModel: '',
        hasLlmApiKey: false,
        hasTavilyApiKey: false,
      }
    }
  }

  if (
    hasLegacyConfig()
  ) {
    try {
      const legacyConfig =
        readLegacyConfig()

      return {
        configured:
          Boolean(
            legacyConfig.llmApiKey,
          )
          && Boolean(
            legacyConfig.tavilyApiKey,
          )
          && Boolean(
            legacyConfig.llmModel,
          ),

        source: 'legacy',
        editable: true,
        llmModel:
          legacyConfig.llmModel,

        hasLlmApiKey:
          Boolean(
            legacyConfig.llmApiKey,
          ),

        hasTavilyApiKey:
          Boolean(
            legacyConfig.tavilyApiKey,
          ),
      }
    } catch {
      return {
        configured: false,
        source: 'legacy-error',
        editable: true,
        llmModel: '',
        hasLlmApiKey: false,
        hasTavilyApiKey: false,
      }
    }
  }

  return {
    configured: false,
    source: null,
    editable: true,
    llmModel: '',
    hasLlmApiKey: false,
    hasTavilyApiKey: false,
  }
}


function getExistingConfig() {
  if (
    hasSecureConfig()
  ) {
    try {
      return readSecureConfig()
    } catch {
      return {
        llmApiKey: '',
        tavilyApiKey: '',
        llmModel: '',
      }
    }
  }

  if (
    hasLegacyConfig()
  ) {
    try {
      return readLegacyConfig()
    } catch {
      return {
        llmApiKey: '',
        tavilyApiKey: '',
        llmModel: '',
      }
    }
  }

  return {
    llmApiKey: '',
    tavilyApiKey: '',
    llmModel: '',
  }
}


function resolveConfigUpdate(
  config,
) {
  const existing =
    getExistingConfig()

  const requestedLlmApiKey =
    String(
      config?.llmApiKey ?? '',
    ).trim()

  const requestedTavilyApiKey =
    String(
      config?.tavilyApiKey ?? '',
    ).trim()

  const requestedLlmModel =
    String(
      config?.llmModel ?? '',
    ).trim()

  const resolvedConfig = {
    llmApiKey:
      requestedLlmApiKey
      || existing.llmApiKey,

    tavilyApiKey:
      requestedTavilyApiKey
      || existing.tavilyApiKey,

    llmModel:
      requestedLlmModel
      || existing.llmModel,
  }

  if (
    !resolvedConfig.llmApiKey
    || !resolvedConfig.tavilyApiKey
    || !resolvedConfig.llmModel
  ) {
    throw new Error(
      'OpenAI API key, Tavily API key, and model are required.',
    )
  }

  if (
    resolvedConfig.llmApiKey.length
      > 4096
    || resolvedConfig.tavilyApiKey.length
      > 4096
    || resolvedConfig.llmModel.length
      > 200
  ) {
    throw new Error(
      'One or more configuration values are too long.',
    )
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

  let configurationSource = null

  if (
    hasSecureConfig()
  ) {
    const secureConfig =
      readSecureConfig()

    childEnvironment.LLM_API_KEY =
      secureConfig.llmApiKey

    childEnvironment.TAVILY_API_KEY =
      secureConfig.tavilyApiKey

    childEnvironment.LLM_MODEL =
      secureConfig.llmModel

    delete childEnvironment
      .TRUSTNET_ENV_FILE

    configurationSource =
      'secure'
  } else if (
    hasLegacyConfig()
  ) {
    childEnvironment
      .TRUSTNET_ENV_FILE =
        getLegacyEnvPath()

    configurationSource =
      'legacy'
  } else {
    throw new Error(
      'TrustNet API configuration has not been set up.',
    )
  }

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