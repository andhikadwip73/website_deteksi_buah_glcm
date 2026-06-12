from pathlib import Path
from uuid import uuid4
from functools import lru_cache
from math import exp, sqrt
import json

from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from PIL import UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

from glcm_utils import (
    calculate_glcm_features,
    vector_from_features
)

# ==================================================
# CONFIG
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR /
    "models" /
    "glcm_model.json"
)

LABELS_PATH = (
    BASE_DIR /
    "models" /
    "class_names.json"
)

DATASET_DIR = (
    BASE_DIR /
    "local_datasets" /
    "dataset"
)

UPLOAD_DIR = (
    BASE_DIR /
    "uploads"
)

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}

CLASS_MAPPING = {

    "Ripe_frames":
        ("Matang", "Layak Konsumsi"),

    "Ripe2_frames":
        ("Matang", "Layak Konsumsi"),

    "Overripe_frames":
        ("Terlalu Matang", "Kurang Ideal"),

    "Rotten_frames":
        ("Busuk", "Tidak Layak"),
}

DEFAULT_CLASSES = [
    "Overripe_frames",
    "Ripe2_frames",
    "Ripe_frames",
    "Rotten_frames",
]

PREFERRED_SAMPLE = (
    DATASET_DIR /
    "Ripe_frames" /
    "frame_000015.jpg"
)

UPLOAD_DIR.mkdir(exist_ok=True)

# ==================================================
# FLASK
# ==================================================

app = Flask(__name__)

app.config.update(
    UPLOAD_FOLDER=UPLOAD_DIR,
    MAX_CONTENT_LENGTH=8 * 1024 * 1024,
)

# ==================================================
# HELPER
# ==================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


@lru_cache(maxsize=1)
def get_model():

    if not MODEL_PATH.exists():
        return None

    with open(
        MODEL_PATH,
        encoding="utf-8"
    ) as file:

        return json.load(file)


@lru_cache(maxsize=1)
def get_class_names():

    if LABELS_PATH.exists():

        with open(
            LABELS_PATH,
            encoding="utf-8"
        ) as file:

            return json.load(file)

    return DEFAULT_CLASSES


@lru_cache(maxsize=1)
def get_sample_image_path():

    if PREFERRED_SAMPLE.exists():
        return PREFERRED_SAMPLE

    if DATASET_DIR.exists():

        for image_path in DATASET_DIR.rglob("*"):

            if (
                image_path.is_file()
                and allowed_file(image_path.name)
            ):
                return image_path

    return None

# ==================================================
# CLASSIFIER
# ==================================================

def classify_with_glcm_model(
    features,
    model
):

    vector = vector_from_features(
        features
    )

    distances = {}

    for class_name, centroid in model[
        "centroids"
    ].items():

        distance = sqrt(

            sum(

                (
                    value - center
                ) ** 2

                for value, center

                in zip(
                    vector,
                    centroid
                )
            )
        )

        distances[
            class_name
        ] = distance

    best_class = min(
        distances,
        key=distances.get
    )

    scores = {

        name:
        exp(
            -distance * 8.0
        )

        for name,
        distance

        in distances.items()
    }

    total_score = (
        sum(scores.values())
        or 1.0
    )

    probabilities = {

        name:
        score / total_score

        for name,
        score

        in scores.items()
    }

    return (
        best_class,
        probabilities
    )


def aggregate_display_probabilities(
    probabilities,
    labels
):

    grouped = {}

    for class_name in labels:

        label, status = (
            CLASS_MAPPING.get(
                class_name,
                (
                    class_name,
                    "Perlu Dicek"
                )
            )
        )

        key = (
            label,
            status
        )

        if key not in grouped:

            grouped[key] = {

                "class":
                    class_name,

                "label":
                    label,

                "status":
                    status,

                "score":
                    0.0,
            }

        grouped[key]["score"] += float(
            probabilities.get(
                class_name,
                0
            )
        )

    return list(
        grouped.values()
    )

# ==================================================
# PREDICT
# ==================================================

def predict_image(image_path):

    model = get_model()

    features = (
        calculate_glcm_features(
            image_path
        )
    )

    if model is None:

        return {

            "ready": False,

            "message":
                (
                    "Model GLCM belum tersedia. "
                    "Jalankan train_model.py terlebih dahulu."
                ),

            "features":
                features,
        }

    raw_class, probabilities = (
        classify_with_glcm_model(
            features,
            model
        )
    )

    labels = model.get(
        "class_names",
        get_class_names()
    )

    display_probabilities = (
        aggregate_display_probabilities(
            probabilities,
            labels
        )
    )

    best_display = max(
        display_probabilities,
        key=lambda item:
        item["score"]
    )

    return {

        "ready": True,

        "class":
            raw_class,

        "raw_label":
            CLASS_MAPPING.get(
                raw_class,
                (
                    raw_class,
                    ""
                )
            )[0],

        "label":
            best_display["label"],

        "status":
            best_display["status"],

        "confidence":
            round(
                best_display["score"]
                * 100,
                2
            ),

        "features": {

            "contrast":
                round(
                    features["contrast"],
                    4
                ),

            "correlation":
                round(
                    features["correlation"],
                    4
                ),

            "energy":
                round(
                    features["energy"],
                    4
                ),

            "homogeneity":
                round(
                    features["homogeneity"],
                    4
                )
        },

        "probabilities": [

            {

                "class":
                    item["class"],

                "label":
                    item["label"],

                "confidence":
                    round(
                        item["score"]
                        * 100,
                        2
                    ),
            }

            for item
            in display_probabilities
        ]
    }

# ==================================================
# ROUTES
# ==================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        model_ready=MODEL_PATH.exists()
    )


@app.get("/sample-image")
def sample_image():

    image_path = (
        get_sample_image_path()
    )

    if image_path is None:
        abort(404)

    return send_from_directory(
        image_path.parent,
        image_path.name
    )


@app.post("/predict")
def predict():

    file = request.files.get(
        "file"
    )

    if (
        not file
        or not file.filename
    ):

        return jsonify({

            "error":
                "File gambar belum dipilih."

        }), 400

    if not allowed_file(
        file.filename
    ):

        return jsonify({

            "error":
                "Format file harus JPG, JPEG, PNG atau WEBP."

        }), 400

    extension = (
        file.filename
        .rsplit(".", 1)[1]
        .lower()
    )

    filename = (
        f"{uuid4().hex}.{extension}"
    )

    save_path = (
        UPLOAD_DIR /
        filename
    )

    file.save(save_path)

    try:

        result = predict_image(
            save_path
        )

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ):

        save_path.unlink(
            missing_ok=True
        )

        return jsonify({

            "error":
                "Gambar tidak dapat dibaca."

        }), 400

    except Exception:

        app.logger.exception(
            "Prediction Error"
        )

        return jsonify({

            "error":
                "Terjadi kesalahan pada server."

        }), 500

    result["image_url"] = url_for(
        "uploaded_file",
        filename=filename
    )

    return jsonify(result)


@app.get("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_DIR,
        filename
    )

# ==================================================
# ERROR HANDLER
# ==================================================

@app.errorhandler(
    RequestEntityTooLarge
)
def file_too_large(_):

    return jsonify({

        "error":
            "Ukuran file terlalu besar. Maksimal 8 MB."

    }), 413

# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    app.run(
        debug=False,
        use_reloader=False
    )