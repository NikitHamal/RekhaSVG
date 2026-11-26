/**
 * RekhaSVG — Live Interactive Studio Controller (docs/app.js)
 * Wires:
 *   1. 10 Curated Nepal STEM Textbook Presets + Local File/Camera Upload
 *   2. Real-Time Client-Side Conversion via RekhaVectorizerJS (with ONNX Runtime Web hook)
 *   3. Split Before/After Comparison Slider
 *   4. 1-Click Semantic Layer Inspector (Fills, Strokes, Dashed, Arrows, Labels)
 *   5. Exam Quiz / Flashcard Mode for SEE / NEB +2 / IOE / CEE Revision
 *   6. Click-to-Highlight Callout Pointers & Inline Bilingual Label Editing
 *   7. 1-Click Export (.svg, .png, and Interactive Study .html)
 */

import { RekhaVectorizerJS } from "./engine/vectorizer.js";
import { NEPAL_STEM_SAMPLES, renderSampleDegradedPhoto } from "./engine/samples.js";

const state = {
  activeSampleIdx: 0,
  theme: "print",
  splitPct: 38,
  quizMode: false,
  revealedCallouts: new Set(),
  selectedCallout: null,
  layerVisibility: {
    fills: true,
    strokes: true,
    dashed: true,
    arrows: true,
    labels: true,
  },
  customLabels: [],
  lastResult: null,
};

const vectorizer = new RekhaVectorizerJS({ theme: state.theme });

function initStudio() {
  renderSampleButtons();
  bindControls();
  loadSample(0);
}

function renderSampleButtons() {
  const listEl = document.getElementById("sample-list");
  if (!listEl) return;
  listEl.innerHTML = "";
  NEPAL_STEM_SAMPLES.forEach((samp, idx) => {
    const btn = document.createElement("button");
    btn.className = `sample-btn ${idx === state.activeSampleIdx ? "active" : ""}`;
    btn.innerHTML = `
      <div class="sample-btn-title">${samp.title}</div>
      <div class="sample-btn-meta">${samp.level}</div>
    `;
    btn.addEventListener("click", () => loadSample(idx));
    listEl.appendChild(btn);
  });
}

function loadSample(idx) {
  state.activeSampleIdx = idx;
  state.revealedCallouts.clear();
  state.selectedCallout = null;
  const samp = NEPAL_STEM_SAMPLES[idx];
  state.customLabels = samp.labels.map((lb) => ({ ...lb }));
  renderSampleButtons();

  const beforeCanvas = document.getElementById("before-canvas");
  renderSampleDegradedPhoto(beforeCanvas, samp);
  revectorizeActiveSample();
  renderLabelEditor();
}

function revectorizeActiveSample() {
  const samp = NEPAL_STEM_SAMPLES[state.activeSampleIdx];
  vectorizer.cfg.theme = state.theme;
  const res = vectorizer.compileFromChains({
    width: 512,
    height: 512,
    strokeChains: samp.strokeChains,
    dashedCenters: samp.dashedCenters,
    arrowChains: samp.arrowChains,
    fillContours: samp.fillContours,
    labels: state.customLabels,
    metadata: { domain: samp.domain, title: samp.title },
  });
  state.lastResult = res;
  renderSvgToStage(res);
  updateTelemetryTiles(res);
}

function renderSvgToStage(res) {
  const svgContainer = document.getElementById("svg-stage-container");
  svgContainer.innerHTML = res.svg;
  const svgEl = svgContainer.querySelector("svg");
  if (!svgEl) return;

  // Apply Layer Inspector visibility toggles
  Object.entries(state.layerVisibility).forEach(([layerKey, isVisible]) => {
    const g = svgEl.querySelector(`#layer-${layerKey}`);
    if (g) g.style.display = isVisible ? "inline" : "none";
  });

  // Wire Exam Quiz Mode & Click-to-Highlight Callouts
  const labelGroups = svgEl.querySelectorAll(".rekha-label-group");
  labelGroups.forEach((grp, idx) => {
    const cid = grp.getAttribute("data-callout");
    const answer = grp.getAttribute("data-quiz-answer") || "";
    const textEl = grp.querySelector("text");
    const chipEl = grp.querySelector("rect");

    if (state.quizMode && !state.revealedCallouts.has(cid)) {
      if (textEl) textEl.textContent = `[ ? Click to Reveal #${idx + 1} ]`;
      if (chipEl) {
        chipEl.setAttribute("fill", "#FFD8E4");
        chipEl.setAttribute("fill-opacity", "0.96");
      }
    }

    if (state.selectedCallout === cid) {
      grp.classList.add("callout-highlight");
    }

    grp.addEventListener("click", (ev) => {
      ev.stopPropagation();
      if (state.quizMode && !state.revealedCallouts.has(cid)) {
        state.revealedCallouts.add(cid);
      }
      state.selectedCallout = state.selectedCallout === cid ? null : cid;
      renderSvgToStage(state.lastResult);
      updateQuizStatus();
    });
  });

  // Highlight linked pointer arrow when callout is selected
  if (state.selectedCallout) {
    const linkedArrow = svgEl.querySelector(`.rekha-callout-arrow[data-callout="${state.selectedCallout}"]`);
    if (linkedArrow) linkedArrow.classList.add("callout-highlight");
  }

  updateSplitView();
  updateQuizStatus();
}

function updateSplitView() {
  const clipEl = document.getElementById("before-clip-wrapper");
  const divLine = document.getElementById("split-divider-line");
  const pct = state.splitPct;
  if (clipEl) {
    clipEl.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
  }
  if (divLine) {
    divLine.style.left = `${pct}%`;
  }
}

function updateQuizStatus() {
  const badge = document.getElementById("quiz-score-badge");
  if (!badge) return;
  const total = state.customLabels.length;
  if (!state.quizMode) {
    badge.textContent = `Study Mode (${total} Bilingual Callouts)`;
  } else {
    badge.textContent = `Exam Quiz: ${state.revealedCallouts.size} / ${total} Revealed`;
  }
}

function updateTelemetryTiles(res) {
  document.getElementById("metric-latency").textContent = `${res.latency_ms} ms`;
  document.getElementById("metric-reduction").textContent = `${res.node_reduction_ratio}× fewer`;
  document.getElementById("metric-primitives").textContent = `${res.strokes.length + res.dashed.length + res.arrows.length}`;
  document.getElementById("metric-callouts").textContent = `${res.labels.length} linked`;
}

function renderLabelEditor() {
  const container = document.getElementById("label-editor-list");
  if (!container) return;
  container.innerHTML = "";
  state.customLabels.forEach((lb, idx) => {
    const row = document.createElement("div");
    row.className = "label-edit-row";
    const inp = document.createElement("input");
    inp.type = "text";
    inp.value = lb.text;
    inp.setAttribute("aria-label", `Edit label ${idx + 1}`);
    inp.addEventListener("input", () => {
      state.customLabels[idx].text = inp.value;
      revectorizeActiveSample();
    });
    row.appendChild(inp);
    container.appendChild(row);
  });
}

function handleUploadedImage(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    const img = new Image();
    img.onload = () => {
      const beforeCanvas = document.getElementById("before-canvas");
      beforeCanvas.width = 512;
      beforeCanvas.height = 512;
      const ctx = beforeCanvas.getContext("2d");
      ctx.fillStyle = "#FFFFFF";
      ctx.fillRect(0, 0, 512, 512);
      ctx.drawImage(img, 0, 0, 512, 512);
      const imgData = ctx.getImageData(0, 0, 512, 512);

      vectorizer.cfg.theme = state.theme;
      const res = vectorizer.vectorizeRGBA(imgData.data, 512, 512, state.customLabels, {
        domain: "custom",
        title: file.name || "Uploaded Textbook Photo",
      });
      state.lastResult = res;
      renderSvgToStage(res);
      updateTelemetryTiles(res);
    };
    img.src = ev.target.result;
  };
  reader.readAsDataURL(file);
}

function downloadBlob(content, filename, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function exportPng() {
  if (!state.lastResult) return;
  const svgBlob = new Blob([state.lastResult.svg], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(svgBlob);
  const img = new Image();
  img.onload = () => {
    const c = document.createElement("canvas");
    c.width = 1024;
    c.height = 1024;
    const ctx = c.getContext("2d");
    ctx.drawImage(img, 0, 0, 1024, 1024);
    URL.revokeObjectURL(url);
    c.toBlob((pngBlob) => {
      const pngUrl = URL.createObjectURL(pngBlob);
      const a = document.createElement("a");
      a.href = pngUrl;
      a.download = "rekhasvg-diagram-1024px.png";
      a.click();
      URL.revokeObjectURL(pngUrl);
    }, "image/png");
  };
  img.src = url;
}

function exportInteractiveStudyHtml() {
  if (!state.lastResult) return;
  const samp = NEPAL_STEM_SAMPLES[state.activeSampleIdx];
  const htmlDoc = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>${samp.title} — RekhaSVG Interactive Study Sheet</title>
<style>
  body { font-family: 'Poppins', system-ui, sans-serif; background:#FEF7FF; color:#1D1B20; padding:24px; max-width:760px; margin:0 auto; }
  .card { background:#F3EDF7; border:2px solid #CAC4D0; border-radius:18px; padding:20px; }
  .btn { background:#6750A4; color:#FFF; border:none; border-radius:999px; padding:8px 16px; font-weight:700; cursor:pointer; margin-right:8px; }
  svg { width:100%; height:auto; border-radius:12px; border:1px solid #CAC4D0; margin-top:14px; }
</style>
</head>
<body>
  <div class="card">
    <h2>${samp.title}</h2>
    <p>${samp.level} — Click any label in Quiz Mode to reveal the answer!</p>
    <button class="btn" onclick="toggleQuiz()">Toggle Exam Quiz Mode</button>
    <div id="svg-wrap">${state.lastResult.svg}</div>
  </div>
  <script>
    let quiz = false;
    function toggleQuiz() {
      quiz = !quiz;
      document.querySelectorAll('.rekha-label-group').forEach((g, i) => {
        const t = g.querySelector('text');
        const ans = g.getAttribute('data-quiz-answer');
        t.textContent = quiz ? '[ ? Click to Reveal #' + (i+1) + ' ]' : ans;
        g.onclick = () => { t.textContent = ans; };
      });
    }
  </script>
</body>
</html>`;
  downloadBlob(htmlDoc, `${samp.id}-interactive-study.html`, "text/html;charset=utf-8");
}

function bindControls() {
  const slider = document.getElementById("split-slider");
  slider.addEventListener("input", () => {
    state.splitPct = Number(slider.value);
    document.getElementById("split-pct-label").textContent = `${state.splitPct}% Photo / ${100 - state.splitPct}% SVG`;
    updateSplitView();
  });

  // Layer toggles
  document.querySelectorAll("[data-toggle-layer]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const layer = btn.getAttribute("data-toggle-layer");
      state.layerVisibility[layer] = !state.layerVisibility[layer];
      btn.classList.toggle("active", state.layerVisibility[layer]);
      if (state.lastResult) renderSvgToStage(state.lastResult);
    });
  });

  // Theme buttons
  document.querySelectorAll("[data-theme-opt]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.theme = btn.getAttribute("data-theme-opt");
      document.querySelectorAll("[data-theme-opt]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      revectorizeActiveSample();
    });
  });

  // Quiz mode button
  const quizBtn = document.getElementById("btn-quiz-mode");
  quizBtn.addEventListener("click", () => {
    state.quizMode = !state.quizMode;
    state.revealedCallouts.clear();
    quizBtn.classList.toggle("active", state.quizMode);
    quizBtn.textContent = state.quizMode ? "Exit Exam Quiz Mode" : "Exam Quiz / Flashcard Mode";
    if (state.lastResult) renderSvgToStage(state.lastResult);
  });

  // File upload
  const fileInp = document.getElementById("photo-upload-input");
  const dropZone = document.getElementById("upload-drop-zone");
  dropZone.addEventListener("click", () => fileInp.click());
  fileInp.addEventListener("change", (e) => handleUploadedImage(e.target.files[0]));

  // Exports
  document.getElementById("btn-export-svg").addEventListener("click", () => {
    if (state.lastResult) {
      downloadBlob(state.lastResult.svg, "rekhasvg-vector.svg", "image/svg+xml;charset=utf-8");
    }
  });
  document.getElementById("btn-export-png").addEventListener("click", exportPng);
  document.getElementById("btn-export-html").addEventListener("click", exportInteractiveStudyHtml);
}

window.addEventListener("DOMContentLoaded", initStudio);
