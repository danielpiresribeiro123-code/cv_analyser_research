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


# ==================================================
# JPEG
# ==================================================

def jpeg_compress(image, quality_range=(5, 95)):
    """
    Simula o:
        imwrite(image, filename, 'Quality', q)
    e mede o bitrate real (bpp).
    """
    quality = random.randint(*quality_range)

    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    pil_img.save(tmp_path, "JPEG", quality=quality)

    # Ler imagem comprimida
    compressed = cv2.imread(tmp_path)

    # Calcular bitrate real
    size_bytes = os.path.getsize(tmp_path)
    h, w = image.shape[:2]
    bpp = (size_bytes * 8) / (h * w)

    os.remove(tmp_path)

    return compressed, {
        "codec": "jpeg",
        "quality": quality,
        "bpp": bpp
    }


# ==================================================
# JPEG2000 (via OpenCV ou Kakadu externo)
# ==================================================

def jpeg2000_compress(image, rate_range=(0.1, 3.0), use_kakadu=False, kakadu_path="kdu_compress"):
    """
    Simula:
        kdu_compress -i src.bmp -o dst.jp2 -rate bitrate

    rate ≈ bits por pixel (como no dataset original)
    """
    rate = random.uniform(*rate_range)

    with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as tmp_src:
        src_path = tmp_src.name
    with tempfile.NamedTemporaryFile(suffix=".jp2", delete=False) as tmp_dst:
        dst_path = tmp_dst.name

    cv2.imwrite(src_path, image)

    if use_kakadu:
        # Chamada real ao Kakadu (se o tiveres instalado)
        cmd = [
            kakadu_path,
            "-i", src_path,
            "-o", dst_path,
            "-rate", str(rate),
            "-no_weights"
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        compressed = cv2.imread(dst_path)

    else:
        # Fallback usando OpenCV (não é Kakadu real, mas aproxima)
        params = [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, int(rate * 1000)]
        cv2.imwrite(dst_path, image, params)
        compressed = cv2.imread(dst_path)

    size_bytes = os.path.getsize(dst_path)
    h, w = image.shape[:2]
    bpp = (size_bytes * 8) / (h * w)

    os.remove(src_path)
    os.remove(dst_path)

    return compressed, {
        "codec": "jpeg2000",
        "rate": rate,
        "bpp": bpp
    }


# ==================================================
# Dispatcher (como no noise.py)
# ==================================================

COMPRESSION_FUNCTIONS = {
    "jpeg": jpeg_compress,
    #"jpeg2000": jpeg2000_compress,
}


def apply_random_compression(
    image,
    compression_types: Union[str, List[str]] = "all",
    strength_ranges: dict | None = None,
    use_kakadu=False
):
    """
    strength_ranges exemplo:
    {
        "jpeg": {"quality_range": (10, 40)},
        "jpeg2000": {"rate_range": (0.2, 1.5)}
    }
    """

    if compression_types == "all":
        available = list(COMPRESSION_FUNCTIONS.keys())
    elif isinstance(compression_types, list):
        available = compression_types
    else:
        raise ValueError("compression_types must be 'all' or a list")

    chosen = random.choice(available)
    compress_func = COMPRESSION_FUNCTIONS[chosen]

    params = {}
    if strength_ranges and chosen in strength_ranges:
        params = strength_ranges[chosen]

    if chosen == "jpeg2000":
        params["use_kakadu"] = use_kakadu

    compressed, meta = compress_func(image, **params)
    meta["type"] = "compression"

    return compressed, chosen, meta


# ==================================================
# Processamento em lote + Manifest
# ==================================================

def process_folder_compression(
    input_dir,
    output_dir,
    compression_types="all",
    strength_ranges=None,
    valid_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
    save_manifest=True,
    manifest_name="compression_manifest.json",
    use_kakadu=False
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
            print(f"Erro ao ler: {img_path.name}")
            continue

        compressed, ctype, meta = apply_random_compression(
            image,
            compression_types=compression_types,
            strength_ranges=strength_ranges,
            use_kakadu=use_kakadu
        )

        output_name = f"{img_path.stem}_{ctype}{img_path.suffix}"
        output_path = output_dir / output_name

        ok = cv2.imwrite(str(output_path), compressed)
        if not ok:
            print(f"Erro ao escrever: {output_name}")
            continue

        if save_manifest:
            manifest.append({
                "input_file": img_path.name,
                "output_file": output_name,
                "compression_type": ctype,
                "params": meta
            })

        print(f"{img_path.name} -> {output_name}")

    if save_manifest:
        manifest_path = output_dir / manifest_name
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"Manifest guardado em: {manifest_path}")
