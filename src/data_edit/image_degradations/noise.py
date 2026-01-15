import cv2
import numpy as np
import random
from typing import List, Union
import cv2
import json
from pathlib import Path

def gaussian_noise(image, mean=0, std_range=(5, 25)):
    std = random.uniform(*std_range)
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8), {"std": std}


def salt_and_pepper_noise(image, amount_range=(0.001, 0.02), salt_vs_pepper=0.5):
    amount = random.uniform(*amount_range)

    noisy = image.copy()
    h, w, c = image.shape
    num_pixels = int(amount * h * w)

    # Salt
    num_salt = int(num_pixels * salt_vs_pepper)
    coords = [np.random.randint(0, i - 1, num_salt) for i in (h, w)]
    noisy[coords[0], coords[1], :] = 255

    # Pepper
    num_pepper = int(num_pixels * (1 - salt_vs_pepper))
    coords = [np.random.randint(0, i - 1, num_pepper) for i in (h, w)]
    noisy[coords[0], coords[1], :] = 0

    return noisy, {"amount": amount}


def speckle_noise(image, std_range=(0.05, 0.3)):
    std = random.uniform(*std_range)
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + image.astype(np.float32) * noise
    return np.clip(noisy, 0, 255).astype(np.uint8), {"std": std}


def poisson_noise(image, scale_range=(0.5, 2.0)):
    scale = random.uniform(*scale_range)
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image.astype(np.float32) * vals * scale) / (vals * scale)
    return np.clip(noisy, 0, 255).astype(np.uint8), {"scale": scale}

NOISE_FUNCTIONS = {
    "gaussian": gaussian_noise,
    "salt_pepper": salt_and_pepper_noise,
    "speckle": speckle_noise,
    "poisson": poisson_noise,
}

def sample_num_noises(distribution: dict):
    """
    distribution: {num_noises: probability}
    Ex: {1: 0.5, 2: 0.4, 3: 0.1}
    """
    nums = list(distribution.keys())
    probs = list(distribution.values())
    return random.choices(nums, weights=probs, k=1)[0]

def apply_random_noise(
    image,
    noise_types: Union[str, List[str]] = "all",
    num_noises: int = 2,
    num_noises_distribution: dict | None = None,
    strength_ranges: dict | None = None
):
    """
    num_noises_distribution:
        Ex:
        {
            1: 0.5,
            2: 0.4,
            3: 0.1
        }
    """

    if noise_types == "all":
        available = list(NOISE_FUNCTIONS.keys())
    elif isinstance(noise_types, list):
        available = noise_types
    else:
        raise ValueError("noise_types must be 'all' or a list of noise names")

    # Escolher quantos noises aplicar
    if num_noises_distribution is not None:
        num_noises = sample_num_noises(num_noises_distribution)

    num_noises = min(num_noises, len(available))

    chosen_noises = random.sample(available, num_noises)

    noisy_image = image.copy()
    applied = []
    applied_params = []

    for noise_name in chosen_noises:
        noise_func = NOISE_FUNCTIONS[noise_name]

        params = {}
        if strength_ranges and noise_name in strength_ranges:
            params = strength_ranges[noise_name]

        noisy_image, used_params = noise_func(noisy_image, **params)
        applied.append(noise_name)
        applied_params.append(used_params)

    return noisy_image, applied, applied_params




def process_folder_noise(
    input_dir,
    output_dir,
    num_noises=2,
    num_noises_distribution=None,
    noise_types="all",
    strength_ranges=None,
    valid_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
    save_manifest=True,
    manifest_name="noise_manifest.json"
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

        noisy, applied, applied_params = apply_random_noise(
            image,
            noise_types=noise_types,
            num_noises=num_noises,
            num_noises_distribution=num_noises_distribution,
            strength_ranges=strength_ranges
        )

        noise_tag = "_".join(applied) if applied else "none"
        output_name = f"{img_path.stem}_noise_{noise_tag}{img_path.suffix}"
        output_path = output_dir / output_name

        ok = cv2.imwrite(str(output_path), noisy)
        if not ok:
            print(f"Erro ao escrever: {output_name}")
            continue

        if save_manifest:
            manifest.append({
                "input_file": img_path.name,
                "output_file": output_name,
                "applied_noises": applied,
                "applied_params": applied_params
            })

        print(f"{img_path.name} -> {output_name}")

    if save_manifest:
        manifest_path = output_dir / manifest_name
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"Manifest guardado em: {manifest_path}")
