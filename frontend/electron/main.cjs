const {
  app,
  BrowserWindow,
  Menu,
  dialog,
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

  const configDirectory = path.join(
    app.getPath('appData'),
    'TrustNet',
  )

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

  const envFilePath = path.join(
    configDirectory,
    '.env',
  )

  const logFilePath = path.join(
    logDirectory,
    'backend.log',
  )

  fs.appendFileSync(
    logFilePath,
    `\n\n===== TrustNet startup ${new Date().toISOString()} =====\n`,
  )

  fs.appendFileSync(
    logFilePath,
    `Backend: ${backendPath}\n`,
  )

  fs.appendFileSync(
    logFilePath,
    `Embedding model: ${embeddingModelPath}\n`,
  )

  fs.appendFileSync(
    logFilePath,
    `Reranker model: ${rerankerModelPath}\n`,
  )

  fs.appendFileSync(
    logFilePath,
    `Environment file exists: ${fs.existsSync(envFilePath)}\n`,
  )

  backendLogFd = fs.openSync(
    logFilePath,
    'a',
  )

  backendProcess = spawn(
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
      env: {
        ...process.env,
        EMBEDDING_MODEL:
          embeddingModelPath,
        RERANKER_MODEL:
          rerankerModelPath,
        HF_HUB_OFFLINE:
          '1',
        TRUSTNET_ENV_FILE:
          envFilePath,
      },
    },
  )

  fs.appendFileSync(
    logFilePath,
    `Spawned backend PID: ${backendProcess.pid ?? 'unknown'}\n`,
  )

  backendProcess.on(
    'error',
    (error) => {
      fs.appendFileSync(
        logFilePath,
        `Backend spawn error: ${error.stack || error}\n`,
      )
    },
  )

  backendProcess.on(
    'exit',
    (code, signal) => {
      fs.appendFileSync(
        logFilePath,
        `Backend exited. Code: ${code}, signal: ${signal}\n`,
      )
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


async function startPackagedApp() {
  startPackagedBackend()

  await waitForUrl(
    BACKEND_HEALTH_URL,
    180000,
  )

  createWindow()
}


app.whenReady().then(
  async () => {
    try {
      if (
        app.isPackaged
      ) {
        await startPackagedApp()
      } else {
        await startDevelopmentApp()
      }
    } catch (error) {
      console.error(
        'TrustNet startup failed:',
        error,
      )

      const logFilePath = path.join(
        app.getPath('appData'),
        'TrustNet',
        'logs',
        'backend.log',
      )

      dialog.showErrorBox(
        'TrustNet startup failed',
        `${String(error)}\n\nBackend log:\n${logFilePath}`,
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

    if (
      backendLogFd !== null
    ) {
      try {
        fs.closeSync(
          backendLogFd,
        )
      } catch {
        // File descriptor may already be closed.
      }

      backendLogFd = null
    }
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