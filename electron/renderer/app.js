const $ = (id) => document.getElementById(id);

const SLIDERS = [
  { id: "threhold", decimals: 0 },
  { id: "pitch", decimals: 0 },
  { id: "formant", decimals: 2 },
  { id: "index_rate", decimals: 2 },
  { id: "rms_mix_rate", decimals: 2 },
  { id: "block_time", decimals: 2 },
  { id: "n_cpu", decimals: 0 },
  { id: "crossfade_length", decimals: 2 },
  { id: "extra_time", decimals: 2 },
];

// Params the backend hot-updates on a running stream (mirrors gui_v1)
const HOT_PARAMS = new Set([
  "threhold",
  "pitch",
  "formant",
  "index_rate",
  "rms_mix_rate",
]);

// Controls that require a stream restart, locked while running
const COLD_CONTROLS = [
  "browse_pth",
  "browse_index",
  "sg_hostapi",
  "sg_wasapi_exclusive",
  "sg_input_device",
  "sg_output_device",
  "reload_devices",
  "sr_model",
  "sr_device",
  "block_time",
  "n_cpu",
  "crossfade_length",
  "extra_time",
];

let ws = null;
let running = false;
let awaitingStart = false;

function setStatus(text) {
  $("conn_status").textContent = text;
}

function showError(text) {
  $("banner_text").textContent = text;
  $("banner").classList.remove("hidden");
}

function send(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(msg));
  }
}

function sliderLabel(id, decimals) {
  $(id + "_val").textContent = Number($(id).value).toFixed(decimals);
}

function fillSelect(sel, options, selected) {
  sel.innerHTML = "";
  for (const name of options) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    if (name === selected) opt.selected = true;
    sel.appendChild(opt);
  }
}

function checkedValue(name) {
  const el = document.querySelector(`input[name="${name}"]:checked`);
  return el ? el.value : null;
}

function setRadio(name, value) {
  const el = document.querySelector(`input[name="${name}"][value="${value}"]`);
  if (el) el.checked = true;
}

function currentConfig() {
  return {
    pth_path: $("pth_path").value,
    index_path: $("index_path").value,
    sg_hostapi: $("sg_hostapi").value,
    sg_wasapi_exclusive: $("sg_wasapi_exclusive").checked,
    sg_input_device: $("sg_input_device").value,
    sg_output_device: $("sg_output_device").value,
    sr_type: $("sr_model").checked ? "sr_model" : "sr_device",
    threhold: parseFloat($("threhold").value),
    pitch: parseFloat($("pitch").value),
    formant: parseFloat($("formant").value),
    index_rate: parseFloat($("index_rate").value),
    rms_mix_rate: parseFloat($("rms_mix_rate").value),
    block_time: parseFloat($("block_time").value),
    crossfade_length: parseFloat($("crossfade_length").value),
    extra_time: parseFloat($("extra_time").value),
    n_cpu: parseFloat($("n_cpu").value),
    f0method: checkedValue("f0method"),
    I_noise_reduce: $("I_noise_reduce").checked,
    O_noise_reduce: $("O_noise_reduce").checked,
    use_pv: $("use_pv").checked,
    use_jit: false,
    function: checkedValue("function"),
  };
}

function applyConfig(cfg) {
  $("pth_path").value = cfg.pth_path || "";
  $("index_path").value = cfg.index_path || "";
  $("sg_wasapi_exclusive").checked = !!cfg.sg_wasapi_exclusive;
  $("sr_model").checked = cfg.sr_type !== "sr_device";
  $("sr_device").checked = cfg.sr_type === "sr_device";
  for (const { id, decimals } of SLIDERS) {
    if (cfg[id] !== undefined) {
      $(id).value = cfg[id];
      sliderLabel(id, decimals);
    }
  }
  if (cfg.f0method) setRadio("f0method", cfg.f0method);
  if (cfg.function) setRadio("function", cfg.function);
  $("I_noise_reduce").checked = !!cfg.I_noise_reduce;
  $("O_noise_reduce").checked = !!cfg.O_noise_reduce;
  $("use_pv").checked = !!cfg.use_pv;
}

function setRunning(value) {
  running = value;
  $("start_vc").disabled = value;
  $("stop_vc").disabled = !value;
  for (const id of COLD_CONTROLS) {
    $(id).disabled = value;
  }
}

function handleMessage(msg) {
  switch (msg.type) {
    case "init":
      fillSelect($("sg_hostapi"), msg.hostapis, msg.config.sg_hostapi);
      fillSelect($("sg_input_device"), msg.input_devices, msg.config.sg_input_device);
      fillSelect($("sg_output_device"), msg.output_devices, msg.config.sg_output_device);
      $("n_cpu").max = msg.n_cpu_max;
      applyConfig(msg.config);
      if (parseFloat($("n_cpu").value) > msg.n_cpu_max) {
        $("n_cpu").value = msg.n_cpu_max;
        sliderLabel("n_cpu", 0);
      }
      setStatus("Ready");
      break;
    case "devices":
      fillSelect($("sg_hostapi"), msg.hostapis, $("sg_hostapi").value);
      fillSelect($("sg_input_device"), msg.input_devices, null);
      fillSelect($("sg_output_device"), msg.output_devices, null);
      break;
    case "started":
      awaitingStart = false;
      setRunning(true);
      $("sr_stream").textContent = msg.samplerate;
      $("delay_time").textContent = msg.delay_ms;
      setStatus("Converting");
      break;
    case "stopped":
      awaitingStart = false;
      setRunning(false);
      setStatus("Ready");
      break;
    case "stats":
      $("infer_time").textContent = msg.infer_time_ms;
      break;
    case "param_updated":
      if (msg.delay_ms !== undefined) {
        $("delay_time").textContent = msg.delay_ms;
      }
      break;
    case "error":
      showError(msg.message);
      if (awaitingStart) {
        // start failed before the stream came up; restore controls
        awaitingStart = false;
        setRunning(false);
      }
      if (!running) setStatus("Ready");
      break;
  }
}

function openSocket(port) {
  ws = new WebSocket(`ws://127.0.0.1:${port}/ws`);
  let opened = false;
  ws.onopen = () => {
    opened = true;
    setStatus("Loading saved settings…");
    send({ type: "get_init" });
  };
  ws.onmessage = (e) => handleMessage(JSON.parse(e.data));
  ws.onclose = () => {
    if (!opened) {
      // uvicorn may not be accepting connections yet; retry
      setTimeout(() => openSocket(port), 1000);
    } else {
      setRunning(false);
      setStatus("Backend disconnected");
      showError("Lost connection to the Python backend.");
    }
  };
}

async function connect() {
  setStatus("Starting Python backend… (first start takes a while)");
  let port = await window.rvcApi.getPort();
  while (!port) {
    await new Promise((r) => setTimeout(r, 500));
    port = await window.rvcApi.getPort();
  }
  openSocket(port);
}

function wireEvents() {
  $("browse_pth").onclick = async () => {
    const file = await window.rvcApi.pickFile({
      defaultDir: "assets/weights",
      filters: [{ name: "RVC model", extensions: ["pth"] }],
    });
    if (file) $("pth_path").value = file;
  };
  $("browse_index").onclick = async () => {
    const file = await window.rvcApi.pickFile({
      defaultDir: "logs",
      filters: [{ name: "Feature index", extensions: ["index"] }],
    });
    if (file) $("index_path").value = file;
  };
  $("sg_hostapi").onchange = () =>
    send({ type: "update_devices", hostapi: $("sg_hostapi").value });
  $("reload_devices").onclick = () =>
    send({ type: "update_devices", hostapi: $("sg_hostapi").value });
  $("start_vc").onclick = () => {
    const cfg = currentConfig();
    if (!cfg.pth_path) {
      showError("Select a .pth model file first.");
      return;
    }
    if (!cfg.index_path) {
      showError("Select an .index file first.");
      return;
    }
    $("banner").classList.add("hidden");
    setStatus("Loading model…");
    send({ type: "start", config: cfg });
    awaitingStart = true;
    setRunning(true);
  };
  $("stop_vc").onclick = () => {
    $("stop_vc").disabled = true;
    send({ type: "stop" });
  };

  for (const { id, decimals } of SLIDERS) {
    $(id).oninput = () => {
      sliderLabel(id, decimals);
      if (running && HOT_PARAMS.has(id)) {
        send({ type: "set_param", key: id, value: parseFloat($(id).value) });
      }
    };
  }
  for (const radio of document.querySelectorAll('input[name="f0method"]')) {
    radio.onchange = () => {
      if (running) send({ type: "set_param", key: "f0method", value: radio.value });
    };
  }
  for (const radio of document.querySelectorAll('input[name="function"]')) {
    radio.onchange = () => {
      if (running) send({ type: "set_param", key: "function", value: radio.value });
    };
  }
  for (const id of ["I_noise_reduce", "O_noise_reduce", "use_pv"]) {
    $(id).onchange = () => {
      if (running) send({ type: "set_param", key: id, value: $(id).checked });
    };
  }
  $("banner_close").onclick = () => $("banner").classList.add("hidden");
}

wireEvents();
setRunning(false);
connect();
