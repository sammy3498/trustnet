const {
  contextBridge,
  ipcRenderer,
} = require('electron')


contextBridge.exposeInMainWorld(
  'trustnetDesktop',
  {
    getConfigStatus: () =>
      ipcRenderer.invoke(
        'trustnet:get-config-status',
      ),

    saveConfig: (config) =>
      ipcRenderer.invoke(
        'trustnet:save-config',
        config,
      ),
  },
)