const fileInput = document.getElementById("file");
const analyzeBtn = document.getElementById("analyze");
const original = document.getElementById("original");
const redacted = document.getElementById("redacted");
const download = document.getElementById("download");

async function loadHealth(){
  const r = await fetch("/api/health");
  const d = await r.json();
  document.getElementById("runtime").textContent = d.execution_provider || "-";
  document.getElementById("target").textContent = d.target_backend || "-";
  document.getElementById("model").textContent = d.model_available ? d.model : "model missing";
  document.getElementById("status").textContent = d.snapdragon_ready ? "QNN ready" : "CPU development mode";
}

fileInput.addEventListener("change", () => {
  analyzeBtn.disabled = !fileInput.files.length;
  if (!fileInput.files.length) return;
  const url = URL.createObjectURL(fileInput.files[0]);
  original.innerHTML = `<img src="${url}" alt="Original upload">`;
  redacted.innerHTML = `<span>Ready to analyze</span>`;
  download.hidden = true;
});

analyzeBtn.addEventListener("click", async () => {
  if (!fileInput.files.length) return;
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Processing...";
  const form = new FormData();
  form.append("image", fileInput.files[0]);
  form.append("policy", document.getElementById("policy").value);
  try {
    const r = await fetch("/api/analyze", {method:"POST", body:form});
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || "Analysis failed");
    redacted.innerHTML = `<img src="${d.redacted}" alt="Redacted result">`;
    document.getElementById("count").textContent = d.redaction_count ?? (d.faces + d.detections.filter(x => x.class !== "person").length);
    document.getElementById("faces").textContent = d.faces;
    document.getElementById("latency").textContent = `${d.latency_ms} ms`;
    document.getElementById("runtime").textContent = d.execution_provider;
    document.getElementById("target").textContent = d.target_backend;
    document.getElementById("status").textContent = d.snapdragon_ready ? "QNN ready" : "CPU development mode";
    download.href = d.redacted;
    download.hidden = false;
  } catch (e) {
    redacted.innerHTML = `<span>${e.message}</span>`;
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze & redact";
  }
});

loadHealth().catch(()=>{document.getElementById("status").textContent="Runtime unavailable"});
