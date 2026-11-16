/**
 * GlaucoVision Frontend Logic - CA-Net Multistage Glaucoma System
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const dropZonePrompt = document.getElementById("dropZonePrompt");
  const previewContainer = document.getElementById("previewContainer");
  const imagePreview = document.getElementById("imagePreview");
  const btnClearImage = document.getElementById("btnClearImage");
  const btnAnalyze = document.getElementById("btnAnalyze");
  const analyzeSpinner = document.getElementById("analyzeSpinner");
  const btnText = btnAnalyze.querySelector(".btn-text");

  // Status & Weights
  const statusModelText = document.getElementById("statusModelText");
  const valW1 = document.getElementById("valW1");
  const valW2 = document.getElementById("valW2");
  const valW3 = document.getElementById("valW3");
  const barW1 = document.getElementById("barW1");
  const barW2 = document.getElementById("barW2");
  const barW3 = document.getElementById("barW3");

  // Results elements
  const emptyState = document.getElementById("emptyState");
  const activeResults = document.getElementById("activeResults");
  const verdictBanner = document.getElementById("verdictBanner");
  const verdictBadge = document.getElementById("verdictBadge");
  const verdictConfidence = document.getElementById("verdictConfidence");
  const verdictDesc = document.getElementById("verdictDesc");
  const inferenceLatency = document.getElementById("inferenceLatency");

  // Probability bars
  const pctNormal = document.getElementById("pctNormal");
  const pctEarly = document.getElementById("pctEarly");
  const pctAdv = document.getElementById("pctAdv");
  const fillNormal = document.getElementById("fillNormal");
  const fillEarly = document.getElementById("fillEarly");
  const fillAdv = document.getElementById("fillAdv");

  // Explainability
  const viewTabs = document.querySelectorAll(".tab-btn");
  const singleViewport = document.getElementById("singleViewport");
  const compareViewport = document.getElementById("compareViewport");
  const mainDisplayImg = document.getElementById("mainDisplayImg");
  const viewportLabel = document.getElementById("viewportLabel");
  const compareOrigImg = document.getElementById("compareOrigImg");
  const compareGcamppImg = document.getElementById("compareGcamppImg");
  const opacitySlider = document.getElementById("opacitySlider");
  const opacityValue = document.getElementById("opacityValue");
  const sliderGroup = document.getElementById("sliderGroup");

  // Samples
  const samplesButtons = document.querySelectorAll(".sample-btn");

  // Architecture Modal
  const archModal = document.getElementById("archModal");
  const btnOpenArchModal = document.getElementById("btnOpenArchModal");
  const btnCloseArchModal = document.getElementById("btnCloseArchModal");

  // State
  let currentFile = null;
  let currentBlob = null;
  let explainabilityData = null;
  let activeTab = "gcampp";

  // 1. Fetch system status & weights
  async function fetchStatus() {
    try {
      const res = await fetch("/api/status");
      if (res.ok) {
        const data = await res.json();
        statusModelText.textContent = `${data.backbone} + CAM Active`;
        if (data.tcab_weights) {
          const w1 = data.tcab_weights.w1_channel;
          const w2 = data.tcab_weights.w2_cross_width;
          const w3 = data.tcab_weights.w3_cross_height;
          valW1.textContent = w1.toFixed(3);
          valW2.textContent = w2.toFixed(3);
          valW3.textContent = w3.toFixed(3);
          barW1.style.width = `${Math.min(w1 * 100, 100)}%`;
          barW2.style.width = `${Math.min(w2 * 100, 100)}%`;
          barW3.style.width = `${Math.min(w3 * 100, 100)}%`;
        }
      }
    } catch (e) {
      console.warn("Status fetch failed:", e);
    }
  }
  fetchStatus();

  // 2. Drag and drop file upload
  dropZone.addEventListener("click", () => fileInput.click());

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });

  function handleFile(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please upload an image file (PNG, JPG, BMP).");
      return;
    }
    currentFile = file;
    currentBlob = null;

    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      dropZonePrompt.style.display = "none";
      previewContainer.style.display = "flex";
      btnAnalyze.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  btnClearImage.addEventListener("click", (e) => {
    e.stopPropagation();
    currentFile = null;
    currentBlob = null;
    fileInput.value = "";
    imagePreview.src = "";
    previewContainer.style.display = "none";
    dropZonePrompt.style.display = "block";
    btnAnalyze.disabled = true;
  });

  // 3. Preset sample buttons
  samplesButtons.forEach((btn) => {
    btn.addEventListener("click", async () => {
      const stage = btn.getAttribute("data-stage");
      btn.style.opacity = "0.6";
      try {
        const res = await fetch("/api/samples");
        if (res.ok) {
          const samples = await res.json();
          const match = samples.find((s) => s.stage === stage);
          if (match) {
            imagePreview.src = match.data_url;
            dropZonePrompt.style.display = "none";
            previewContainer.style.display = "flex";
            btnAnalyze.disabled = false;

            // Convert base64 data url to blob
            const fetchRes = await fetch(match.data_url);
            currentBlob = await fetchRes.blob();
            currentFile = new File([currentBlob], `${stage}_sample.png`, { type: "image/png" });
          }
        }
      } catch (err) {
        console.error("Error loading sample:", err);
      } finally {
        btn.style.opacity = "1";
      }
    });
  });

  // 4. Run Analysis
  btnAnalyze.addEventListener("click", async () => {
    if (!currentFile) return;

    btnAnalyze.disabled = true;
    btnText.textContent = "Analyzing Retinal Microstructure...";
    analyzeSpinner.style.display = "inline-block";

    try {
      const formData = new FormData();
      formData.append("file", currentFile);

      // Run prediction and explainability concurrently
      const [predRes, explainRes] = await Promise.all([
        fetch("/api/predict", { method: "POST", body: formData }),
        fetch("/api/explain", { method: "POST", body: formData })
      ]);

      if (!predRes.ok || !explainRes.ok) {
        throw new Error("Analysis failed. Server returned an error.");
      }

      const predData = await predRes.json();
      explainabilityData = await explainRes.json();

      renderResults(predData, explainabilityData);
    } catch (err) {
      alert("Error during analysis: " + err.message);
    } finally {
      btnAnalyze.disabled = false;
      btnText.textContent = "Run CA-Net Analysis";
      analyzeSpinner.style.display = "none";
    }
  });

  // 5. Render Results
  function renderResults(pred, explain) {
    emptyState.style.display = "none";
    activeResults.style.display = "block";

    // Verdict card
    verdictBadge.textContent = pred.prediction;
    verdictConfidence.textContent = `${pred.confidence}%`;
    verdictDesc.textContent = pred.description;
    inferenceLatency.textContent = `Latency: ${pred.latency_ms}ms`;

    verdictBanner.className = "verdict-banner";
    if (pred.prediction === "Early-stage") {
      verdictBanner.classList.add("stage-early");
    } else if (pred.prediction === "Advanced-stage") {
      verdictBanner.classList.add("stage-adv");
    }

    // Probability Bars
    const normalProb = pred.probabilities["Normal"] || 0;
    const earlyProb = pred.probabilities["Early-stage"] || 0;
    const advProb = pred.probabilities["Advanced-stage"] || 0;

    pctNormal.textContent = `${normalProb.toFixed(1)}%`;
    pctEarly.textContent = `${earlyProb.toFixed(1)}%`;
    pctAdv.textContent = `${advProb.toFixed(1)}%`;

    fillNormal.style.width = `${normalProb}%`;
    fillEarly.style.width = `${earlyProb}%`;
    fillAdv.style.width = `${advProb}%`;

    // Render explainability display
    updateExplainabilityView();
  }

  // 6. View tabs
  viewTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      viewTabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      activeTab = tab.getAttribute("data-view");
      updateExplainabilityView();
    });
  });

  function updateExplainabilityView() {
    if (!explainabilityData) return;

    if (activeTab === "compare") {
      singleViewport.style.display = "none";
      compareViewport.style.display = "grid";
      sliderGroup.style.display = "none";

      compareOrigImg.src = explainabilityData.original_image;
      compareGcamppImg.src = explainabilityData.gradcam_pp_overlay;
    } else {
      singleViewport.style.display = "flex";
      compareViewport.style.display = "none";
      sliderGroup.style.display = "flex";

      if (activeTab === "gcampp") {
        mainDisplayImg.src = explainabilityData.gradcam_pp_overlay;
        viewportLabel.textContent = "Grad-CAM++ Overlay (Paper Recommended)";
      } else if (activeTab === "gcam") {
        mainDisplayImg.src = explainabilityData.gradcam_overlay;
        viewportLabel.textContent = "Standard Grad-CAM Overlay";
      } else if (activeTab === "original") {
        mainDisplayImg.src = explainabilityData.original_image;
        viewportLabel.textContent = "Original Preprocessed Fundus";
      }
    }
  }

  // 7. Opacity Slider
  opacitySlider.addEventListener("input", (e) => {
    const val = e.target.value;
    opacityValue.textContent = `${val}%`;
    // Modulate brightness/contrast overlay effect
    if (activeTab === "gcampp" || activeTab === "gcam") {
      const alpha = val / 100.0;
      mainDisplayImg.style.opacity = Math.max(0.3, alpha);
    } else {
      mainDisplayImg.style.opacity = 1.0;
    }
  });

  // 8. Modal handlers
  btnOpenArchModal.addEventListener("click", () => {
    archModal.style.display = "flex";
  });

  btnCloseArchModal.addEventListener("click", () => {
    archModal.style.display = "none";
  });

  archModal.addEventListener("click", (e) => {
    if (e.target === archModal) {
      archModal.style.display = "none";
    }
  });
});
