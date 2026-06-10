const {
  app,
  BrowserWindow,
  dialog,
  ipcMain,
  systemPreferences,
} = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

const repoRoot = path.resolve(__dirname, "..");
const pythonBin =
  process.env.RVC_PYTHON || path.join(repoRoot, ".venv", "bin", "python");
const serverScript = path.join(repoRoot, "tools", "realtime", "server.py");

let win = null;
let backend = null;
let backendPort = null;
let quitting = false;
const stderrTail = [];

function rememberStderr(chunk) {
  for (const line of chunk.toString().split("\n")) {
    if (line.trim()) stderrTail.push(line);
  }
  while (stderrTail.length > 30) stderrTail.shift();
}

function startBackend() {
  return new Promise((resolve, reject) => {
    if (!fs.existsSync(pythonBin)) {
      reject(
        new Error(`Python not found at ${pythonBin}. Run ./run-realtime.sh first.`)
      );
      return;
    }
    let settled = false;
    backend = spawn(pythonBin, [serverScript], { cwd: repoRoot });
    const timeout = setTimeout(() => {
      if (!settled) {
        settled = true;
        reject(
          new Error(
            "Backend did not start within 180s.\n" + stderrTail.join("\n")
          )
        );
      }
    }, 180000);
    let buffer = "";
    backend.stdout.on("data", (chunk) => {
      buffer += chunk.toString();
      let idx;
      while ((idx = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (line) console.log("[backend]", line);
        const m = line.match(/^REALTIME_SERVER_PORT=(\d+)$/);
        if (m) {
          if (!settled) {
            settled = true;
            clearTimeout(timeout);
            backendPort = parseInt(m[1], 10);
            resolve(backendPort);
          }
        }
      }
    });
    backend.stderr.on("data", (chunk) => {
      rememberStderr(chunk);
      console.error("[backend]", chunk.toString().trimEnd());
    });
    backend.on("exit", (code) => {
      backend = null;
      if (!quitting) {
        dialog.showErrorBox(
          "RVC backend stopped",
          `The Python backend exited (code ${code}).\n\nLast output:\n` +
            stderrTail.join("\n")
        );
        app.quit();
      }
    });
  });
}

function createWindow() {
  win = new BrowserWindow({
    width: 1100,
    height: 880,
    title: "RVC Realtime Voice Conversion",
    backgroundColor: "#0f0d18",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
    },
  });
  win.loadFile(path.join(__dirname, "renderer", "index.html"));
}

app.whenReady().then(async () => {
  if (process.platform === "darwin") {
    await systemPreferences.askForMediaAccess("microphone");
  }
  ipcMain.handle("get-port", () => backendPort);
  ipcMain.handle("pick-file", async (event, opts) => {
    const parent = win && !win.isDestroyed() ? win : undefined;
    const result = await dialog.showOpenDialog(parent, {
      properties: ["openFile"],
      defaultPath: opts.defaultDir
        ? path.join(repoRoot, opts.defaultDir)
        : repoRoot,
      filters: opts.filters || [],
    });
    return result.canceled ? null : result.filePaths[0];
  });
  createWindow();
  try {
    await startBackend();
  } catch (err) {
    dialog.showErrorBox(
      "RVC backend failed to start",
      String((err && err.message) || err)
    );
    app.quit();
  }
});

app.on("before-quit", (event) => {
  if (backend && !quitting) {
    quitting = true;
    // hold the quit until the backend exits, then escalate if it hangs
    event.preventDefault();
    const killTimer = setTimeout(() => {
      if (backend) backend.kill("SIGKILL");
    }, 3000);
    backend.once("exit", () => {
      clearTimeout(killTimer);
      app.quit();
    });
    backend.kill();
  } else {
    quitting = true;
  }
});

app.on("window-all-closed", () => {
  app.quit();
});
