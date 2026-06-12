from pathlib import Path
from PIL import Image
import numpy as np

from skimage.feature import graycomatrix, graycoprops

FEATURE_KEYS = [
    "contrast",
    "correlation",
    "energy",
    "homogeneity"
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


def image_files(folder):

    folder = Path(folder)

    return [
        file
        for file in folder.rglob("*")
        if file.suffix.lower() in IMAGE_EXTENSIONS
    ]


def calculate_glcm_features(image_path):

    img = Image.open(image_path)

    gray = img.convert("L")
    gray = gray.resize((128, 128))

    gray = np.array(gray)

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    return {
        "contrast": graycoprops(glcm, "contrast")[0, 0],
        "correlation": graycoprops(glcm, "correlation")[0, 0],
        "energy": graycoprops(glcm, "energy")[0, 0],
        "homogeneity": graycoprops(glcm, "homogeneity")[0, 0]
    }


def vector_from_features(features):

    return [
        features["contrast"],
        features["correlation"],
        features["energy"],
        features["homogeneity"]
    ]