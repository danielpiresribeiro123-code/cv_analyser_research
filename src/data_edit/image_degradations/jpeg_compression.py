import cv2
import numpy as np
import random
import os
import json
import tempfile
from pathlib import Path
from typing import Union, List
from PIL import Image
import subprocess

COMPRESSION_STRENGTHS = {
    "jpeg_dct": {
        "low":    {"quality_range": (70, 90)},
        "medium": {"quality_range": (30, 70)},
        "strong": {"quality_range": (5, 30)},
    },

    "jpeg2000": {
        # rate ≈ bits por pixel (mais baixo = mais compressão)
        "low":    {"rate_range": (1.5, 3.0)},
        "medium": {"rate_range": (0.5, 1.5)},
        "strong": {"rate_range": (0.1, 0.5)},
    },

    "jpeg_xl": {
        # distance ≈ perceptual distance (mais alto = mais compressão)
        "low":    {"distance_range": (0.5, 1.5)},
        "medium": {"distance_range": (1.5, 4.0)},
        "strong": {"distance_range": (4.0, 8.0)},
    }
}


# ==================================================
# JPEG
# ==================================================
def jpeg_dct_compress(image, quality_range=(5, 95)):
    quality = random.randint(*quality_range)

    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    pil_img.save(tmp_path, "JPEG", quality=quality)
    compressed = cv2.imread(tmp_path)

    size_bytes = os.path.getsize(tmp_path)
    h, w = image.shape[:2]
    bpp = (size_bytes * 8) / (h * w)

    os.remove(tmp_path)

    return compressed, {
        "codec": "jpeg_dct",
        "quality": quality,
        "bpp": bpp
    }


# ==================================================
# JPEG2000 (via OpenCV ou Kakadu externo)
# ==================================================
def jpeg2000_compress(image, rate_range=(0.1, 3.0)):
    rate = random.uniform(*rate_range)

    with tempfile.NamedTemporaryFile(suffix=".jp2", delete=False) as tmp:
        jp2_path = tmp.name

    # OpenCV usa COMPRESSION_X1000 (quanto maior, mais compressão)
    # Invertendo a lógica para aproximar bitrate:
    compression = int(1000 / max(rate, 1e-3))
    params = [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, compression]

    cv2.imwrite(jp2_path, image, params)
    compressed = cv2.imread(jp2_path)

    size_bytes = os.path.getsize(jp2_path)
    h, w = image.shape[:2]
    bpp = (size_bytes * 8) / (h * w)

    os.remove(jp2_path)

    return compressed, {
        "codec": "jpeg2000",
        "rate": rate,
        "bpp": bpp
    }



def jpeg_xl_compress(image, distance_range=(0.5, 8.0),
                     cjxl_path="cjxl", djxl_path="djxl"):
    distance = random.uniform(*distance_range)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_src:
        src_path = tmp_src.name
    with tempfile.NamedTemporaryFile(suffix=".jxl", delete=False) as tmp_dst:
        jxl_path = tmp_dst.name
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_out:
        out_path = tmp_out.name

    cv2.imwrite(src_path, image)

    # Compressão
    subprocess.run(
        [cjxl_path, src_path, jxl_path, "--distance", str(distance)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Descompressão
    subprocess.run(
        [djxl_path, jxl_path, out_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    compressed = cv2.imread(out_path)

    size_bytes = os.path.getsize(jxl_path)
    h, w = image.shape[:2]
    bpp = (size_bytes * 8) / (h * w)

    os.remove(src_path)
    os.remove(jxl_path)
    os.remove(out_path)

    return compressed, {
        "codec": "jpeg_xl",
        "distance": distance,
        "bpp": bpp
    }



COMPRESSION_TYPES = {
    "jpeg_dct": jpeg_dct_compress,
    "jpeg2000": jpeg2000_compress,
    "jpeg_xl": jpeg_xl_compress
}



def apply_compression(image, compression_type="jpeg_dct", strength="medium"):
    if compression_type not in COMPRESSION_TYPES:
        raise ValueError(f"compression_type inválido: {compression_type}")

    if strength not in COMPRESSION_STRENGTHS[compression_type]:
        raise ValueError(f"strength inválido para {compression_type}")

    func = COMPRESSION_TYPES[compression_type]
    params = COMPRESSION_STRENGTHS[compression_type][strength]

    out, meta = func(image, **params)

    applied = f"{compression_type}_{strength}"
    return out, applied, meta


def apply_random_compression(image):
    compression_type = random.choice(list(COMPRESSION_TYPES.keys()))
    strength = random.choice(["low", "medium", "strong"])
    return apply_compression(image, compression_type, strength)


def process_folder_compression(
    input_dir,
    output_dir,
    valid_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
    save_manifest=True,
    manifest_name="compression_manifest.json",
    use_random=False,
    compression_type="jpeg_dct",  # jpeg_dct, jpeg2000, jpeg_xl
    strength="medium"             # low, medium, strong
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    for img_path in input_dir.iterdir():
        if img_path.suffix.lower() not in valid_exts:
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        if use_random:
            compressed, applied, params = apply_random_compression(image)
        else:
            compressed, applied, params = apply_compression(
                image, compression_type, strength
            )

        tag = applied if applied else "none"
        out_name = f"{img_path.stem}_compression_{tag}{img_path.suffix}"
        cv2.imwrite(str(output_dir / out_name), compressed)

        if save_manifest:
            manifest.append({
                "input": img_path.name,
                "output": out_name,
                "applied_compression": applied,
                "params": params
            })

        print(f"{img_path.name} -> {out_name}")

    if save_manifest:
        with open(output_dir / manifest_name, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
