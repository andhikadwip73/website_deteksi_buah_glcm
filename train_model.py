from collections import defaultdict
from math import exp, sqrt
from pathlib import Path
import json

from hsv_utils import FEATURE_KEYS, calculate_hsv_features, image_files, vector_from_features


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "local_datasets" / "dataset"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "hsv_maturity_model.json"
LABELS_PATH = MODEL_DIR / "class_names.json"
MAX_IMAGES_PER_CLASS = 250


def nearest_class(vector, centroids):
    distances = {}
    for class_name, centroid in centroids.items():
        distance = sqrt(sum((value - center) ** 2 for value, center in zip(vector, centroid)))
        distances[class_name] = distance

    best_class = min(distances, key=distances.get)
    scores = {name: exp(-distance * 8.0) for name, distance in distances.items()}
    total = sum(scores.values()) or 1.0
    probabilities = {name: score / total for name, score in scores.items()}
    return best_class, probabilities


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATASET_DIR}")

    class_dirs = sorted(path for path in DATASET_DIR.iterdir() if path.is_dir())
    if not class_dirs:
        raise RuntimeError(f"Tidak ada folder kelas di {DATASET_DIR}")

    vectors_by_class = defaultdict(list)
    print("Membaca dataset dan menghitung fitur HSV...")

    for class_dir in class_dirs:
        files = image_files(class_dir)[:MAX_IMAGES_PER_CLASS]
        print(f"- {class_dir.name}: {len(files)} gambar dipakai")
        for file_path in files:
            features = calculate_hsv_features(file_path)
            vectors_by_class[class_dir.name].append(vector_from_features(features))

    centroids = {}
    feature_means = {}
    for class_name, vectors in vectors_by_class.items():
        if not vectors:
            continue

        count = len(vectors)
        centroid = [sum(vector[index] for vector in vectors) / count for index in range(len(FEATURE_KEYS))]
        centroids[class_name] = centroid
        feature_means[class_name] = {
            key: round(value, 4) for key, value in zip(FEATURE_KEYS, centroid)
        }

    correct = 0
    total = 0
    for class_name, vectors in vectors_by_class.items():
        for vector in vectors:
            prediction, _ = nearest_class(vector, centroids)
            correct += int(prediction == class_name)
            total += 1

    accuracy = (correct / total) * 100 if total else 0
    MODEL_DIR.mkdir(exist_ok=True)
    class_names = list(centroids.keys())

    model_data = {
        "model_type": "HSV nearest-centroid classifier",
        "python_compatible": "3.14",
        "feature_keys": FEATURE_KEYS,
        "class_names": class_names,
        "centroids": centroids,
        "feature_means": feature_means,
        "training_samples": {name: len(vectors) for name, vectors in vectors_by_class.items()},
        "training_accuracy_percent": round(accuracy, 2),
    }

    MODEL_PATH.write_text(json.dumps(model_data, ensure_ascii=False, indent=2), encoding="utf-8")
    LABELS_PATH.write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Model tersimpan di: {MODEL_PATH}")
    print(f"Akurasi training sederhana: {accuracy:.2f}%")
    print("Catatan: ini model HSV kompatibel Python 3.14, bukan CNN TensorFlow.")


if __name__ == "__main__":
    main()
