// EcoSort front-end: handles upload, calls the inference API, renders results.

const EMOJI = {
  cardboard: "📦", glass: "🍾", metal: "🥫",
  paper: "📄", plastic: "🧴", trash: "🗑️",
};
const SAMPLES = ["cardboard", "glass", "plastic", "paper"];

const el = (id) => document.getElementById(id);
const dropzone = el("dropzone");
const fileInput = el("file-input");
const cameraInput = el("camera-input");
const preview = el("preview");
const heatmapImg = el("heatmap");

function setStatus(state, text) {
  const pill = el("status-pill");
  pill.className = "pill " + state;
  pill.innerHTML = `<span class="dot"></span> ${text}`;
}

async function checkHealth() {
  try {
    const r = await fetch("/api/health");
    if (r.ok) setStatus("ready", "model ready");
    else setStatus("error", "model unavailable");
  } catch {
    setStatus("error", "offline");
  }
}

function showImage(src) {
  preview.src = src;
  preview.classList.remove("hidden");
  el("dz-empty").classList.add("hidden");
  heatmapImg.classList.add("hidden");
  heatmapImg.classList.remove("show");
}

function showLoading() {
  el("result-placeholder").classList.add("hidden");
  el("result-content").classList.add("hidden");
  el("result-loading").classList.remove("hidden");
}

function renderResult(data) {
  el("result-loading").classList.add("hidden");
  el("result-placeholder").classList.add("hidden");
  const content = el("result-content");
  content.classList.remove("hidden");

  el("bin-name").textContent = data.bin;
  el("bin-name").style.color = data.color;
  el("pred-class").textContent = data.prediction;
  el("conf-badge").textContent = (data.confidence * 100).toFixed(1) + "%";
  el("verdict-emoji").textContent = EMOJI[data.prediction] || "♻️";
  el("tip").textContent = data.tip;

  const bars = el("bars");
  bars.innerHTML = "";
  data.ranking.forEach((item, i) => {
    const row = document.createElement("div");
    row.className = "bar-row" + (i === 0 ? " top" : "");
    row.innerHTML = `
      <span class="bar-name">${item.class}</span>
      <span class="bar-track"><span class="bar-fill"></span></span>
      <span class="bar-val">${(item.probability * 100).toFixed(1)}%</span>`;
    bars.appendChild(row);
    const fill = row.querySelector(".bar-fill");
    fill.style.background = item.color;
    requestAnimationFrame(() => { fill.style.width = (item.probability * 100) + "%"; });
  });

  if (data.heatmap) {
    heatmapImg.src = data.heatmap;
    heatmapImg.classList.remove("hidden");
    requestAnimationFrame(() => heatmapImg.classList.add("show"));
  }

  const compareRow = el("compare-row");
  if (data.classical && data.classical.prediction) {
    compareRow.classList.remove("hidden");
    el("compare-text").textContent =
      `${data.classical.prediction} (${(data.classical.confidence * 100).toFixed(0)}%)`;
  } else {
    compareRow.classList.add("hidden");
  }
}

function renderError(msg) {
  el("result-loading").classList.add("hidden");
  const ph = el("result-placeholder");
  ph.classList.remove("hidden");
  ph.innerHTML = `<div class="ph-icon">⚠️</div><p>${msg}</p>`;
}

async function classify(fileOrBlob) {
  const objectUrl = URL.createObjectURL(fileOrBlob);
  showImage(objectUrl);
  showLoading();

  const form = new FormData();
  form.append("image", fileOrBlob, fileOrBlob.name || "upload.jpg");
  form.append("explain", el("heatmap-toggle").checked ? "true" : "false");
  form.append("compare", "true");

  try {
    const res = await fetch("/api/predict", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) return renderError(data.error || "Prediction failed.");
    renderResult(data);
  } catch (e) {
    renderError("Network error — is the server running?");
  }
}

// --- Wire up events ---
el("browse-btn").addEventListener("click", () => fileInput.click());
el("camera-btn").addEventListener("click", () => cameraInput.click());
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keypress", (e) => { if (e.key === "Enter") fileInput.click(); });

[fileInput, cameraInput].forEach((input) =>
  input.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) classify(e.target.files[0]);
  })
);

["dragover", "dragenter"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); })
);
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files && e.dataTransfer.files[0]) classify(e.dataTransfer.files[0]);
});

el("heatmap-toggle").addEventListener("change", () => {
  if (!el("heatmap-toggle").checked) {
    heatmapImg.classList.remove("show");
  } else if (heatmapImg.src) {
    heatmapImg.classList.add("show");
  }
});

// Sample images
const sampleRow = el("sample-row");
SAMPLES.forEach((name) => {
  const chip = document.createElement("button");
  chip.className = "sample-chip";
  chip.textContent = `${EMOJI[name]} ${name}`;
  chip.addEventListener("click", async () => {
    const resp = await fetch(`/static/samples/${name}.jpg`);
    const blob = await resp.blob();
    blob.name = `${name}.jpg`;
    classify(blob);
  });
  sampleRow.appendChild(chip);
});

checkHealth();
