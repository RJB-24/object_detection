const $ = (id) => document.getElementById(id);
let mode = "analyze";
let currentFile = null;
let stream = null;

const ENDPOINTS = {
  analyze: "/api/analyze",
  objects: "/api/detect/objects",
  faces: "/api/detect/faces",
  recognize: "/api/recognize/faces",
};

// Tabs
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const tab = btn.dataset.tab;
    if (tab === "gallery") {
      $("panel-run").hidden = true;
      $("panel-gallery").hidden = false;
      loadIdentities();
    } else {
      $("panel-run").hidden = false;
      $("panel-gallery").hidden = true;
      mode = tab === "faces" ? "recognize" : tab; // faces tab -> recognition
      $("runBtn").textContent = tab === "analyze" ? "▶ Run analysis" : tab === "objects" ? "▶ Detect objects" : "▶ Recognize faces";
    }
  });
});

// Sliders
$("conf").addEventListener("input", (e) => ($("confVal").textContent = (+e.target.value).toFixed(2)));
$("thr").addEventListener("input", (e) => ($("thrVal").textContent = (+e.target.value).toFixed(2)));

// Dropzone
const dz = $("dropzone"), fi = $("fileInput");
dz.addEventListener("click", () => fi.click());
fi.addEventListener("change", () => setFile(fi.files[0]));
["dragover", "dragenter"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

function setFile(f) {
  if (!f) return;
  currentFile = f;
  const url = URL.createObjectURL(f);
  const pv = $("preview");
  pv.src = url;
  pv.style.display = "block";
}

// Webcam
$("webcamBtn").addEventListener("click", async () => {
  const video = $("video");
  if (stream) {
    // capture
    const canvas = $("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      const f = new File([blob], "webcam.jpg", { type: "image/jpeg" });
      setFile(f);
    }, "image/jpeg", 0.92);
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.style.display = "block";
    await video.play();
    $("webcamBtn").textContent = "📸 Capture frame";
  } catch (err) {
    alert("Webcam unavailable: " + err.message);
  }
});

// Sample photo (Wikimedia - group of people + everyday objects scene)
$("sampleBtn").addEventListener("click", async () => {
  const urls = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Fronalpstock_big.jpg/1280px-Fronalpstock_big.jpg",
  ];
  $("sampleBtn").disabled = true;
  try {
    const res = await fetch(urls[0]);
    const blob = await res.blob();
    setFile(new File([blob], "sample.jpg", { type: blob.type || "image/jpeg" }));
  } catch (e) {
    alert("Could not load sample image (network?). Upload your own photo instead.");
  }
  $("sampleBtn").disabled = false;
});

// Run
$("runBtn").addEventListener("click", async () => {
  if (!currentFile) return alert("Choose an image first.");
  const fd = new FormData();
  fd.append("file", currentFile);
  const params = new URLSearchParams({ return_image: "true" });
  if (mode === "analyze" || mode === "objects") {
    params.set("conf", $("conf").value);
    params.set("iou", "0.45");
  }
  if (mode === "analyze" || mode === "recognize") params.set("threshold", $("thr").value);
  setLoading(true);
  try {
    const res = await fetch(`${ENDPOINTS[mode]}?${params}`, { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed");
    render(data);
  } catch (e) {
    alert("Error: " + e.message);
  }
  setLoading(false);
});

function setLoading(on) {
  $("spinner").hidden = !on;
  $("runBtn").disabled = on;
  if (on) { $("emptyResult").style.display = "none"; }
}

function render(data) {
  const img = data.annotated_image || (data.objects && null);
  const annotated = data.annotated_image;
  if (annotated) {
    $("resultImg").src = annotated;
    $("resultImg").style.display = "block";
    $("emptyResult").style.display = "none";
  }
  let objects = [], faces = [], ms = data.inference_ms;
  if (data.objects) { // analyze
    objects = data.objects.detections || [];
    faces = data.faces.faces || [];
    ms = data.inference_ms;
    $("resultTitle").textContent = `Analysis — ${objects.length} objects, ${faces.length} faces`;
  } else if (data.detections) {
    objects = data.detections;
    $("resultTitle").textContent = `Objects — ${objects.length} found`;
  } else {
    faces = data.faces || [];
    $("resultTitle").textContent = `Faces — ${faces.length} found`;
  }
  if (ms != null) {
    $("timePill").hidden = false;
    $("timePill").textContent = `${ms} ms`;
  }
  $("objCount").textContent = objects.length ? `(${objects.length})` : "";
  $("faceCount").textContent = faces.length ? `(${faces.length})` : "";
  $("objList").innerHTML = objects.length
    ? objects.map((d) => `<div class="item"><b>${escapeHtml(d.class_name)}</b><span class="conf">${(d.confidence * 100).toFixed(1)}%</span></div>`).join("")
    : `<span class="muted">—</span>`;
  $("faceList").innerHTML = faces.length
    ? faces.map((f) => {
        const name = f.name ? `${escapeHtml(f.name)}` : "face";
        const pct = f.score != null ? (f.score * 100).toFixed(1) + "%" : (f.confidence * 100).toFixed(1) + "%";
        const mark = f.matched ? "✅" : f.name && f.name !== "Unknown" ? "" : "❔";
        return `<div class="item"><b>${mark} ${name}</b><span class="conf">${pct}</span></div>`;
      }).join("")
    : `<span class="muted">—</span>`;
  $("rawJson").textContent = JSON.stringify(stripImage(data), null, 2);
}

function stripImage(d) {
  const c = JSON.parse(JSON.stringify(d));
  if (c.annotated_image) c.annotated_image = "<base64 jpeg omitted>";
  return c;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]));
}

// Gallery registration
const regDrop = $("regDrop"), regFiles = $("regFiles");
let regList = [];
regDrop.addEventListener("click", () => regFiles.click());
regFiles.addEventListener("change", () => {
  regList = [...regFiles.files];
  $("regCount").textContent = regList.length ? `${regList.length} file(s) selected` : "no files selected";
});
["dragover", "dragenter"].forEach((ev) => regDrop.addEventListener(ev, (e) => { e.preventDefault(); regDrop.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => regDrop.addEventListener(ev, (e) => { e.preventDefault(); regDrop.classList.remove("drag"); }));
regDrop.addEventListener("drop", (e) => {
  regList = [...e.dataTransfer.files];
  $("regCount").textContent = regList.length ? `${regList.length} file(s) selected` : "no files selected";
});

$("registerBtn").addEventListener("click", async () => {
  const name = $("personName").value.trim();
  const msg = $("regMsg");
  msg.className = "msg";
  if (!name) { msg.textContent = "Enter a name first."; msg.classList.add("err"); return; }
  if (!regList.length) { msg.textContent = "Select at least one face photo."; msg.classList.add("err"); return; }
  const fd = new FormData();
  fd.append("name", name);
  regList.slice(0, 10).forEach((f) => fd.append("files", f));
  $("registerBtn").disabled = true;
  try {
    const res = await fetch("/api/faces/register", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Registration failed");
    msg.textContent = `✓ Registered ${data.embeddings_added}/${data.images_received} photo(s) for "${data.name}".`;
    msg.classList.add("ok");
    regList = []; regFiles.value = ""; $("regCount").textContent = "no files selected";
    loadIdentities();
  } catch (e) {
    msg.textContent = "✕ " + e.message;
    msg.classList.add("err");
  }
  $("registerBtn").disabled = false;
});

async function loadIdentities() {
  try {
    const res = await fetch("/api/faces");
    const data = await res.json();
    const box = $("identities");
    if (!data.identities || !data.identities.length) {
      box.innerHTML = `<span class="muted">No identities yet — register someone above.</span>`;
      return;
    }
    box.innerHTML = data.identities.map((n) =>
      `<span class="chip">${escapeHtml(n)} <small>×${data.counts[n] || 0}</small><button data-name="${escapeHtml(n)}" title="Delete">✕</button></span>`
    ).join("");
    box.querySelectorAll("button").forEach((b) =>
      b.addEventListener("click", async () => {
        if (!confirm(`Delete "${b.dataset.name}"?`)) return;
        await fetch(`/api/faces/${encodeURIComponent(b.dataset.name)}`, { method: "DELETE" });
        loadIdentities();
      })
    );
  } catch (e) { /* ignore */ }
}

$("clearBtn").addEventListener("click", async () => {
  if (!confirm("Delete ALL registered identities?")) return;
  await fetch("/api/faces", { method: "DELETE" });
  loadIdentities();
});

// Health
async function pollHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    const yolo = data.models?.yolo, face = data.models?.face;
    const yOk = yolo && !yolo.error, fOk = face && (face.loaded || (face.det_exists && face.rec_exists));
    $("healthText").textContent = `YOLO ${yOk ? "✓" : "…"} · Faces ${fOk ? "✓" : "…"} · ${data.face_identities?.length || 0} identities`;
  } catch (e) {
    $("healthText").textContent = "offline — start server";
  }
}
pollHealth();
setInterval(pollHealth, 8000);
