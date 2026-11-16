# TrustNet v0.2.0 — Windows Installer

The Windows desktop app bundles its own FastAPI backend and two offline retrieval
models. After installation, open **TrustNet** from the Start menu or the desktop
shortcut. You do **not** need Command Prompt, PowerShell, a separately started
Vite server, or a separately started Uvicorn server to *run* the installed app.

## Build once (on a Windows 10/11 x64 PC)

Requires Node.js/npm, the existing working TrustNet Python virtual environment
and an internet connection for the first install of build tools. The two local
retrieval models must already be cached by the working Python environment.

From PowerShell:

```powershell
cd "$env:USERPROFILE\Downloads\trustnet-v0.2.0"
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1 -PythonPath "$env:USERPROFILE\Downloads\trustnet-v0.1.0\.venv\Scripts\python.exe"
```

The script runs regression tests, exports the retrieval models, freezes FastAPI
with PyInstaller, builds the React frontend, and builds the Windows x64 NSIS installer.
It may download build dependencies. The installer will be large because it includes
PyTorch and two model weights. It does **not** embed your API credentials.

Output: `frontend\release\TrustNet-Setup-0.2.0.exe`.

If your virtual environment moved, replace `-PythonPath` with its real location.
The command uses a process-scoped PowerShell execution-policy bypass; it does not
change the system policy persistently. You can instead run `& .\scripts\build_windows.ps1 ...`
when scripts are enabled.

## Install and start

1. Exit any TrustNet backend running on `127.0.0.1:8000`, and close the
   development frontend on port 5173 if it's open.
2. Double-click `frontend\release\TrustNet-Setup-0.2.0.exe`.
3. Choose the install directory. The installer creates Start Menu and desktop shortcuts.
4. Launch TrustNet. On first use, choose the same LLM provider that works in your development
   environment (for example, **Google Gemini**), enter its API key, exact model identifier
   (copy the value of `GEMINI_MODEL` from your working `.env`, not a marketing name),
   and the Tavily API key. These are encrypted on the local Windows machine; do not
   commit them to GitHub.
5. Future launches use the shortcut, automatically start the bundled backend, and
   stop it when the app exits. The backend remains local at `127.0.0.1:8000`.

**Note:** This is a local desktop installation, not public cloud deployment.
Search and Gemini/OpenAI/OpenRouter API calls still require internet access.
Models and the Python runtime are bundled; no installed Python or Node.js is
needed for daily use.

### Troubleshooting

If the app says API Offline, give it a few moments on first launch while the
backend warms up. Check `%APPDATA%\TrustNet\logs\backend.log` for failures.
Verify port 8000 isn't already held by a development server. If the first build
fails at `prepare_desktop_models.py`, open the development environment and run
TrustNet once to warm/download the two models, then rebuild. The build must be
performed on Windows: a Linux environment cannot freeze a Windows-native
PyTorch/Python executable.

## GitHub commit / push (without secret files or compiled binaries)

In your existing Git worktree, *after applying the updated source files*:

```powershell
cd "$env:USERPROFILE\Downloads\trustnet-v0.2.0"
git status --short
git add backend/app/analysis/pipeline.py backend/app/api/routes.py backend/app/llm/client.py backend/app/llm/scoring.py backend/app/retrieval/search_pipeline.py backend/app/utils/config.py backend/tests/test_latency_fastpaths.py frontend/electron/main.cjs frontend/src/App.jsx frontend/src/App.css frontend/package.json frontend/package-lock.json scripts/prepare_desktop_models.py scripts/build_windows.ps1 docs/desktop_windows_install.md
git diff --cached --stat
git diff --cached --check
git commit -m "Optimize TrustNet analysis and package Windows desktop installer"
git push origin HEAD
```

Verify `git diff --cached --name-only` before committing; it must not include
`.env`, API keys, model files, `dist/`, `desktop_resources/models/`,
`frontend/release/`, or a `.venv`. If the repository has diverged, fetch and
reconcile changes before pushing.
