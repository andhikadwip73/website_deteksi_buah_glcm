from collections import defaultdict
from pathlib import Path
import json

from glcm_utils import (
    FEATURE_KEYS,
    calculate_glcm_features,
    image_files,
    vector_from_features
)

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "local_datasets" / "dataset"

MODEL_PATH = (
    BASE_DIR /
    "models" /
    "glcm_model.json"
)


def main():

    vectors_by_class = defaultdict(list)

    print("Menghitung fitur GLCM...")

    for class_dir in DATASET_DIR.iterdir():

        if not class_dir.is_dir():
            continue

        print(f"Kelas : {class_dir.name}")

        for file in image_files(class_dir):

            features = calculate_glcm_features(file)

            vector = vector_from_features(
                features
            )

            vectors_by_class[
                class_dir.name
            ].append(vector)

    centroids = {}

    for class_name, vectors in vectors_by_class.items():

        jumlah = len(vectors)

        centroid = [

            sum(
                vector[i]
                for vector in vectors
            ) / jumlah

            for i in range(
                len(FEATURE_KEYS)
            )
        ]

        centroids[class_name] = centroid

    model = {

        "feature_keys":
            FEATURE_KEYS,

        "centroids":
            centroids,

        "class_names":
            list(
                centroids.keys()
            )
    }

    MODEL_PATH.parent.mkdir(
        exist_ok=True
    )

    with open(
        MODEL_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            model,
            f,
            indent=4
        )

    print(
        "\nModel berhasil dibuat"
    )

    print(
        MODEL_PATH
    )


if __name__ == "__main__":
    main()