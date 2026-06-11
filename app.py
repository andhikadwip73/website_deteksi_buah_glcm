from pathlib import Path
from uuid import uuid4
import json
from math import exp, sqrt

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from PIL import UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

from hsv_utils import calculate_hsv_features, vector_from_features


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "hsv_maturity_model.json"
LABELS_PATH = BASE_DIR / "models" / "class_names.json"
UPLOAD_DIR = BASE_DIR / "uploads"

DISPLAY_LABELS = {
    "Ripe_frames": "Matang",
    "Ripe2_frames": "Matang",
    "Overripe_frames": "Terlalu Matang",
    "Rotten_frames": "Busuk",
}
STATUS_LABELS = {
    "Ripe_frames": "Layak / Matang",
    "Ripe2_frames": "Layak / Matang",
    "Overripe_frames": "Tidak Ideal",
    "Rotten_frames": "Tidak Layak",
}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
UPLOAD_DIR.mkdir(exist_ok=True)

model = None
class_names = None


def get_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            return None
        model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    return model


def get_class_names():
    global class_names
    if class_names is not None:
        return class_names
    if LABELS_PATH.exists():
        class_names = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    else:
        class_names = ["Overripe_frames", "Ripe2_frames", "Ripe_frames", "Rotten_frames"]
    return class_names


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"jpg", "jpeg", "png", "webp"}


def classify_with_hsv_model(features, hsv_model):
    vector = vector_from_features(features)
    distances = {}
    for class_name, centroid in hsv_model["centroids"].items():
        distances[class_name] = sqrt(sum((value - center) ** 2 for value, center in zip(vector, centroid)))

    best_class = min(distances, key=distances.get)
    scores = {name: exp(-distance * 8.0) for name, distance in distances.items()}
    total = sum(scores.values()) or 1.0
    probabilities = {name: score / total for name, score in scores.items()}
    return best_class, probabilities


def aggregate_display_probabilities(probabilities, labels):
    grouped = {}
    order = []

    for class_name in labels:
        label = DISPLAY_LABELS.get(class_name, class_name)
        status = STATUS_LABELS.get(class_name, "Perlu Dicek")
        key = (label, status)

        if key not in grouped:
            grouped[key] = {
                "class": class_name,
                "label": label,
                "status": status,
                "score": 0.0,
            }
            order.append(key)

        grouped[key]["score"] += float(probabilities.get(class_name, 0))

    return [grouped[key] for key in order]


def predict_image(image_path):
    hsv_model = get_model()
    hsv = calculate_hsv_features(image_path)

    if hsv_model is None:
        return {
            "ready": False,
            "message": "Model HSV belum tersedia. Jalankan python train_model.py dulu untuk membuat models/hsv_maturity_model.json.",
            "hsv": hsv,
        }

    raw_label, probabilities = classify_with_hsv_model(hsv, hsv_model)
    labels = hsv_model.get("class_names", get_class_names())
    display_probabilities = aggregate_display_probabilities(probabilities, labels)
    best_display = max(display_probabilities, key=lambda item: item["score"])

    return {
        "ready": True,
        "class": raw_label,
        "raw_label": DISPLAY_LABELS.get(raw_label, raw_label),
        "label": best_display["label"],
        "status": best_display["status"],
        "confidence": round(best_display["score"] * 100, 2),
        "hsv": hsv,
        "probabilities": [
            {
                "class": item["class"],
                "label": item["label"],
                "confidence": round(item["score"] * 100, 2),
            }
            for item in display_probabilities
        ],
    }


@app.route("/")
def index():
    return render_template("index.html", model_ready=MODEL_PATH.exists())


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_error):
    return jsonify({"error": "Ukuran file terlalu besar. Maksimal 8 MB."}), 413


@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "File gambar belum dipilih."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "File gambar belum dipilih."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Format file harus jpg, jpeg, png, atau webp."}), 400

    extension = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid4().hex}.{extension}"
    save_path = UPLOAD_DIR / filename
    file.save(save_path)

    try:
        result = predict_image(save_path)
    except (UnidentifiedImageError, OSError, ValueError):
        save_path.unlink(missing_ok=True)
        return jsonify({"error": "Gambar tidak dapat dibaca. Gunakan file JPG, JPEG, PNG, atau WEBP yang valid."}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    result["image_url"] = url_for("uploaded_file", filename=filename)
    return jsonify(result)


@app.get("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
