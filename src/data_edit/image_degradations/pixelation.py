import cv2
import numpy as np
import random
from pathlib import Path
import json

def pixelate(image, block_size):
    h, w = image.shape[:2]
    temp = cv2.resize(image, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelated


def global_pixelation(image, block_range=(6, 40)):
    block_size = random.randint(*block_range)
    return pixelate(image, block_size), {"block_size": block_size}


def local_pixelation(image, block_range=(6, 40), area_range=(0.2, 0.6)):
    h, w = image.shape[:2]
    block_size = random.randint(*block_range)

    area_ratio = random.uniform(*area_range)
    patch_w = int(w * area_ratio)
    patch_h = int(h * area_ratio)

    x = random.randint(0, w - patch_w)
    y = random.randint(0, h - patch_h)

    patch = image[y:y+patch_h, x:x+patch_w]
    pixel_patch = pixelate(patch, block_size)

    output = image.copy()
    output[y:y+patch_h, x:x+patch_w] = pixel_patch

    return output, {
        "block_size": block_size,
        "x": x, "y": y,
        "patch_w": patch_w, "patch_h": patch_h
    }


def mosaic_pixelation(image, block_range=(8, 50), num_patches_range=(3, 8)):
    h, w = image.shape[:2]
    output = image.copy()

    num_patches = random.randint(*num_patches_range)
    block_size = random.randint(*block_range)

    params = []

    for _ in range(num_patches):
        pw = random.randint(w//10, w//3)
        ph = random.randint(h//10, h//3)

        x = random.randint(0, w - pw)
        y = random.randint(0, h - ph)

        patch = output[y:y+ph, x:x+pw]
        pixel_patch = pixelate(patch, block_size)
        output[y:y+ph, x:x+pw] = pixel_patch

        params.append({
            "x": x, "y": y,
            "w": pw, "h": ph,
            "block_size": block_size
        })

    return output, {
        "num_patches": num_patches,
        "patches": params
    }


PIXELATION_FUNCTIONS = {
    "global": global_pixelation,
    "local": local_pixelation,
    "mosaic": mosaic_pixelation
}

def sample_num_pixelations(distribution: dict):
    """
    distribution: {num_pixelations: probability}
    Ex: {1: 0.5, 2: 0.4, 3: 0.1}
    """
    nums = list(distribution.keys())
    probs = list(distribution.values())
    return random.choices(nums, weights=probs, k=1)[0]
def apply_random_pixelation(
    image,
    pixelation_types="all",
    num_pixelations=1,
    num_pixelations_distribution=None,
    strength_ranges=None
):
    if pixelation_types == "all":
        available = list(PIXELATION_FUNCTIONS.keys())
    elif isinstance(pixelation_types, list):
        available = pixelation_types
    else:
        raise ValueError("pixelation_types must be 'all' or list")

    # Escolher quantas pixelations aplicar
    if num_pixelations_distribution is not None:
        num_pixelations = sample_num_pixelations(num_pixelations_distribution)

    num_pixelations = min(num_pixelations, len(available))

    chosen = random.sample(available, num_pixelations)

    out = image.copy()
    applied = []
    params_used = []

    for ptype in chosen:
        func = PIXELATION_FUNCTIONS[ptype]
        params = strength_ranges.get(ptype, {}) if strength_ranges else {}
        out, p = func(out, **params)
        applied.append(ptype)
        params_used.append(p)

    return out, applied, params_used

def process_folder_pixelation(
    input_dir,
    output_dir,
    num_pixelations=1,
    num_pixelations_distribution=None,
    pixelation_types="all",
    strength_ranges=None,
    valid_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
    save_manifest=True,
    manifest_name="pixelation_manifest.json"
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

        pixelated, applied, params = apply_random_pixelation(
            image,
            pixelation_types=pixelation_types,
            num_pixelations=num_pixelations,
            num_pixelations_distribution=num_pixelations_distribution,
            strength_ranges=strength_ranges
        )

        tag = "_".join(applied) if applied else "none"
        out_name = f"{img_path.stem}_pixel_{tag}{img_path.suffix}"
        cv2.imwrite(str(output_dir / out_name), pixelated)

        if save_manifest:
            manifest.append({
                "input": img_path.name,
                "output": out_name,
                "applied_pixelations": applied,
                "params": params
            })

        print(f"{img_path.name} -> {out_name}")

    if save_manifest:
        with open(output_dir / manifest_name, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
