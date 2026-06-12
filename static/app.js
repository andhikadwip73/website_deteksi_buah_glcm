const form = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const message = document.getElementById("message");

const previewImage = document.getElementById("previewImage");
const previewCaption = document.getElementById("previewCaption");

const resultCard = document.querySelector("[data-result-card]");
const resultLabel = document.getElementById("resultLabel");
const resultStatus = document.getElementById("resultStatus");

const scoreRing = document.getElementById("scoreRing");
const confidenceValue = document.getElementById("confidenceValue");

const probabilityList = document.getElementById("probabilityList");

const contrastValue = document.getElementById("contrastValue");
const correlationValue = document.getElementById("correlationValue");
const energyValue = document.getElementById("energyValue");
const homogeneityValue = document.getElementById("homogeneityValue");

const MAX_SIZE = 8 * 1024 * 1024;
const ALLOWED = ["jpg", "jpeg", "png", "webp"];

let previewUrl = null;

/* =====================================
   PREVIEW GAMBAR
===================================== */

fileInput.addEventListener("change", () => {

    const file = fileInput.files[0];

    if (!file) return;

    const ext = file.name.split(".").pop().toLowerCase();

    if (!ALLOWED.includes(ext)) {
        showMessage(
            "Format file harus JPG, JPEG, PNG, atau WEBP.",
            "error"
        );
        fileInput.value = "";
        return;
    }

    if (file.size > MAX_SIZE) {
        showMessage(
            "Ukuran file maksimal 8 MB.",
            "error"
        );
        fileInput.value = "";
        return;
    }

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }

    previewUrl = URL.createObjectURL(file);

    previewImage.src = previewUrl;
    previewImage.alt = file.name;
    previewCaption.textContent = file.name;

    showMessage(file.name);
});

/* =====================================
   SUBMIT
===================================== */

form.addEventListener("submit", async (event) => {

    event.preventDefault();

    const file = fileInput.files[0];

    if (!file) {
        showMessage(
            "Pilih gambar terlebih dahulu.",
            "error"
        );
        return;
    }

    const button = form.querySelector("button");

    const formData = new FormData();
    formData.append("file", file);

    button.disabled = true;
    button.textContent = "Menganalisis...";

    showMessage("Menganalisis gambar...");

    try {

        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.error || "Prediksi gagal."
            );
        }

        updateResult(data);

        showMessage("Analisis selesai.");

    } catch (error) {

        showMessage(
            error.message,
            "error"
        );

    } finally {

        button.disabled = false;
        button.textContent = "Deteksi Sekarang";

    }

});

/* =====================================
   HASIL PREDIKSI
===================================== */

function updateResult(data) {

    previewImage.src = data.image_url;

    if (data.features) {

        contrastValue.textContent =
            data.features.contrast;

        correlationValue.textContent =
            data.features.correlation;

        energyValue.textContent =
            data.features.energy;

        homogeneityValue.textContent =
            data.features.homogeneity;
    }

    if (!data.ready) {

        resultLabel.textContent =
            "Model Belum Dilatih";

        resultStatus.textContent =
            data.message;

        confidenceValue.textContent =
            "0%";

        scoreRing.style.setProperty(
            "--score",
            0
        );

        return;
    }

    resultLabel.textContent =
        data.label;

    resultStatus.textContent =
        data.status;

    confidenceValue.textContent =
        data.confidence + "%";

    scoreRing.style.setProperty(
        "--score",
        data.confidence
    );

    previewCaption.textContent =
        data.label;

    updateCardColor(
        data.label
    );

    updateProbabilities(
        data.probabilities
    );
}

/* =====================================
   WARNA HASIL
===================================== */

function updateCardColor(label) {

    resultCard.className =
        "result-card";

    const text =
        label.toLowerCase();

    if (text.includes("busuk")) {

        resultCard.classList.add(
            "is-rotten"
        );

    } else if (
        text.includes("terlalu")
    ) {

        resultCard.classList.add(
            "is-overripe"
        );

    } else {

        resultCard.classList.add(
            "is-ripe"
        );

    }
}

/* =====================================
   PROBABILITAS
===================================== */

function updateProbabilities(items) {

    probabilityList.innerHTML = "";

    items.forEach(item => {

        const row =
            document.createElement("div");

        row.className =
            "bar-row";

        row.innerHTML = `
            <span>${item.label}</span>

            <div class="bar-track">
                <div
                    class="bar-fill"
                    style="width:${item.confidence}%">
                </div>
            </div>

            <strong>
                ${item.confidence}%
            </strong>
        `;

        probabilityList.appendChild(row);

    });
}

/* =====================================
   PESAN
===================================== */

function showMessage(
    text,
    type = ""
) {

    message.textContent = text;

    message.className =
        type
            ? `message ${type}`
            : "message";
}