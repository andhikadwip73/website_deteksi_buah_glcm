from colorsys import rgb_to_hsv
from pathlib import Path
from PIL import Image

# Fitur yang digunakan model
FEATURE_KEYS = [
    "hue",
    "saturation",
    "value",
    "red_magenta_area"
]

# Format gambar yang didukung
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


def image_files(folder):
    """
    Mengambil seluruh file gambar dalam folder dataset.
    """

    folder = Path(folder)

    return [
        file
        for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]


def calculate_hsv_features(image_path):
    """
    Menghitung rata-rata HSV dan area merah-magenta.
    """

    with Image.open(image_path) as img:

        # Ubah ke RGB
        img = img.convert("RGB")

        # Perkecil gambar agar lebih cepat diproses
        img.thumbnail((128, 128))

        # Ambil setiap 3 pixel
        pixels = list(img.getdata())[::3]

    if not pixels:
        raise ValueError("Gambar tidak dapat dibaca.")

    hue_total = 0
    saturation_total = 0
    value_total = 0

    red_magenta_count = 0

    for r, g, b in pixels:

        # RGB → HSV
        h, s, v = rgb_to_hsv(
            r / 255,
            g / 255,
            b / 255
        )

        # Skala OpenCV
        h *= 179
        s *= 255
        v *= 255

        hue_total += h
        saturation_total += s
        value_total += v

        # Deteksi area merah-magenta
        if (h <= 12 or h >= 155) and s >= 45 and v >= 40:
            red_magenta_count += 1

    total_pixels = len(pixels)

    return {
        "hue": round(hue_total / total_pixels, 2),

        "saturation": round(
            saturation_total / total_pixels,
            2
        ),

        "value": round(
            value_total / total_pixels,
            2
        ),

        "red_magenta_area": round(
            (red_magenta_count / total_pixels) * 100,
            2
        )
    }


def vector_from_features(features):
    """
    Normalisasi fitur ke rentang 0-1.
    """

    return [
        features["hue"] / 179,
        features["saturation"] / 255,
        features["value"] / 255,
        features["red_magenta_area"] / 100
    ]