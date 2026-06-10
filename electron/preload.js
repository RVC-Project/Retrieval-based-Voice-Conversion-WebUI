const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("rvcApi", {
  getPort: () => ipcRenderer.invoke("get-port"),
  pickFile: (opts) => ipcRenderer.invoke("pick-file", opts),
});
