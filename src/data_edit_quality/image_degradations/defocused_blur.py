import cv2
import os
import random
from pathlib import Path
import numpy as np
import json

BLUR_STRENGTHS = {
    "low": {
        "kernel_range": (5, 11),
        "passes": 1
    },
    "medium": {
        "kernel_range": (13, 25),
        "passes": 1
    },
    "strong": {
        "kernel_range": (25, 61),
        "passes": 2
    }
}

def _random_kernel(kmin, kmax, max_size=None):
    k = random.randint(kmin, kmax)
    if k % 2 == 0:
        k += 1
    if max_size is not None:
        k = min(k, max_size if max_size % 2 == 1 else max_size - 1)
        k = max(1, k)
    return k



def global_defocus_blur(image, strength="medium"):
    cfg = BLUR_STRENGTHS[strength]
    k = _random_kernel(*cfg["kernel_range"])
    passes = cfg["passes"]

    out = image.copy()
    for _ in range(passes):
        out = cv2.GaussianBlur(out, (k, k), 0)

    return out, {
        "type": "global",
        "kernel": k,
        "passes": passes,
        "strength": strength
    }


def partial_random_defocus_blur(image, strength="medium"):
    h, w = image.shape[:2]
    cfg = BLUR_STRENGTHS[strength]

    k = _random_kernel(*cfg["kernel_range"])
    passes = cfg["passes"]

    # tamanho da região a desfocar
    rw = random.randint(w // 4, w // 2)
    rh = random.randint(h // 4, h // 2)

    x = random.randint(0, w - rw)
    y = random.randint(0, h - rh)

    out = image.copy()
    patch = out[y:y+rh, x:x+rw]

    for _ in range(passes):
        patch = cv2.GaussianBlur(patch, (k, k), 0)

    out[y:y+rh, x:x+rw] = patch

    return out, {
        "type": "partial_random",
        "kernel": k,
        "passes": passes,
        "strength": strength,
        "region": {"x": x, "y": y, "w": rw, "h": rh}
    }


def partial_radial_defocus_blur(image, strength="medium"):
    h, w = image.shape[:2]
    cfg = BLUR_STRENGTHS[strength]

    k = _random_kernel(*cfg["kernel_range"])
    passes = cfg["passes"]

    # cria versão totalmente desfocada
    blurred = image.copy()
    for _ in range(passes):
        blurred = cv2.GaussianBlur(blurred, (k, k), 0)

    # máscara radial: centro nítido, bordas desfocadas
    cx, cy = w // 2, h // 2
    max_dist = np.sqrt(cx**2 + cy**2)

    mask = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            d = np.sqrt((x - cx)**2 + (y - cy)**2)
            mask[y, x] = np.clip(d / max_dist, 0, 1)

    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    mask = mask[:, :, None]  # para 3 canais

    out = image.astype(np.float32) * (1 - mask) + blurred.astype(np.float32) * mask
    out = np.clip(out, 0, 255).astype(np.uint8)

    return out, {
        "type": "partial_radial",
        "kernel": k,
        "passes": passes,
        "strength": strength
    }


DEFOCUS_BLUR_TYPES = {
    "global": global_defocus_blur,
    "partial_random": partial_random_defocus_blur,
    "partial_radial": partial_radial_defocus_blur
}


def apply_defocus_blur(image, blur_type="global", strength="medium"):
    if blur_type not in DEFOCUS_BLUR_TYPES:
        raise ValueError(f"blur_type inválido: {blur_type}")

    if strength not in BLUR_STRENGTHS:
        raise ValueError(f"strength inválido: {strength}")

    func = DEFOCUS_BLUR_TYPES[blur_type]
    out, params = func(image, strength=strength)

    applied = f"{blur_type}_{strength}"
    return out, applied, params


def apply_random_defocus_blur(image):
    blur_type = random.choice(list(DEFOCUS_BLUR_TYPES.keys()))
    strength = random.choice(["low", "medium", "strong"])
    return apply_defocus_blur(image, blur_type, strength)




def process_folder_defocus_blur(
    input_dir,
    output_dir,
    valid_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
    save_manifest=True,
    manifest_name="defocus_manifest.json",
    use_random=False,
    blur_type="global",      # global, partial_random, partial_radial
    strength="medium"        # low, medium, strong
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
            blurred, applied, params = apply_random_defocus_blur(image)
        else:
            blurred, applied, params = apply_defocus_blur(image, blur_type, strength)

        tag = applied if applied else "none"
        out_name = f"{img_path.stem}_defocus_{tag}{img_path.suffix}"
        cv2.imwrite(str(output_dir / out_name), blurred)

        if save_manifest:
            manifest.append({
                "input": img_path.name,
                "output": out_name,
                "applied_blur": applied,
                "params": params
            })

        print(f"{img_path.name} -> {out_name}")

    if save_manifest:
        with open(output_dir / manifest_name, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
