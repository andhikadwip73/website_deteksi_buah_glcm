const form = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const message = document.getElementById("message");
const previewImage = document.getElementById("previewImage");
const resultLabel = document.getElementById("resultLabel");
const resultStatus = document.getElementById("resultStatus");
const confidenceValue = document.getElementById("confidenceValue");
const probabilityList = document.getElementById("probabilityList");
const hueValue = document.getElementById("hueValue");
const saturationValue = document.getElementById("saturationValue");
const valueValue = document.getElementById("valueValue");
const areaValue = document.getElementById("areaValue");

const allowedExtensions = ["jpg", "jpeg", "png", "webp"];
let previewUrl = "";

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;

  if (!isAllowedImage(file)) {
    fileInput.value = "";
    showMessage("Format file harus JPG, JPEG, PNG, atau WEBP.", "error");
    return;
  }

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }

  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  previewImage.alt = file.name;
  previewImage.style.display = "block";
  showMessage(file.name);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = form.querySelector("button");
  const file = fileInput.files[0];

  if (!file) {
    showMessage("Pilih gambar terlebih dahulu.", "error");
    return;
  }

  if (!isAllowedImage(file)) {
    showMessage("Format file harus JPG, JPEG, PNG, atau WEBP.", "error");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  button.disabled = true;
  button.textContent = "Menganalisis...";
  showMessage("Menganalisis gambar...");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Prediksi gagal.");
    }

    renderResult(data);
    showMessage(data.ready ? "Analisis selesai." : data.message);
  } catch (error) {
    showMessage(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "Deteksi Sekarang";
  }
});

function isAllowedImage(file) {
  const extension = file.name.split(".").pop().toLowerCase();
  return allowedExtensions.includes(extension);
}

function showMessage(text, type = "") {
  message.textContent = text;
  message.className = type ? `message ${type}` : "message";
}

function renderResult(data) {
  previewImage.src = data.image_url;
  previewImage.alt = data.label || "Gambar buah naga";
  previewImage.style.display = "block";

  if (data.ready) {
    resultLabel.textContent = data.label;
    resultStatus.textContent = data.status;
    confidenceValue.textContent = `${data.confidence}%`;
    renderProbabilities(data.probabilities);
  } else {
    resultLabel.textContent = "Model belum dilatih";
    resultStatus.textContent = "Fitur HSV sudah terbaca, tetapi model belum bisa memprediksi sampai training dijalankan.";
    confidenceValue.textContent = "0%";
    probabilityList.innerHTML = "";
  }

  hueValue.textContent = data.hsv.hue;
  saturationValue.textContent = data.hsv.saturation;
  valueValue.textContent = data.hsv.value;
  areaValue.textContent =
    data.hsv.red_magenta_area === "-" ? "-" : `${data.hsv.red_magenta_area}%`;
}

function renderProbabilities(items) {
  probabilityList.innerHTML = "";
  items.forEach((item) => {
    const row = document.createElement("div");
    const label = document.createElement("span");
    const track = document.createElement("div");
    const fill = document.createElement("div");
    const percent = document.createElement("strong");
    const confidence = normalizeConfidence(item.confidence);

    row.className = "bar-row";
    label.textContent = item.label;
    track.className = "bar-track";
    fill.className = "bar-fill";
    fill.style.width = `${confidence}%`;
    percent.textContent = `${formatPercent(confidence)}%`;

    track.appendChild(fill);
    row.append(label, track, percent);
    probabilityList.appendChild(row);
  });
}

function normalizeConfidence(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return 0;
  return Math.max(0, Math.min(100, number));
}

function formatPercent(value) {
  return Number(value.toFixed(2)).toString();
}
