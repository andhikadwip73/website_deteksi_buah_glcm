from colorsys import rgb_to_hsv
from pathlib import Path

from PIL import Image


FEATURE_KEYS = ["hue", "saturation", "value", "red_magenta_area"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def image_files(directory):
    directory = Path(directory)
    return [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def calculate_hsv_features(image_path, max_size=128, pixel_step=3):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_size, max_size))
        pixels = list(img.getdata())[::pixel_step]

    if not pixels:
        raise ValueError("Gambar tidak memiliki pixel yang bisa dibaca.")

    hue_total = 0.0
    saturation_total = 0.0
    value_total = 0.0
    red_magenta_count = 0

    for red, green, blue in pixels:
        hue, saturation, value = rgb_to_hsv(red / 255.0, green / 255.0, blue / 255.0)
        hue_cv = hue * 179.0
        saturation_cv = saturation * 255.0
        value_cv = value * 255.0

        hue_total += hue_cv
        saturation_total += saturation_cv
        value_total += value_cv

        if (hue_cv <= 12 or hue_cv >= 155) and saturation_cv >= 45 and value_cv >= 40:
            red_magenta_count += 1

    total = len(pixels)
    return {
        "hue": round(hue_total / total, 2),
        "saturation": round(saturation_total / total, 2),
        "value": round(value_total / total, 2),
        "red_magenta_area": round((red_magenta_count / total) * 100, 2),
    }


def vector_from_features(features):
    return [
        float(features["hue"]) / 179.0,
        float(features["saturation"]) / 255.0,
        float(features["value"]) / 255.0,
        float(features["red_magenta_area"]) / 100.0,
    ]
