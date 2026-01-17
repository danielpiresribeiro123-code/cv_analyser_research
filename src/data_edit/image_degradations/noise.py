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
    coords = [np.random.randint(0, i, num_salt) for i in (h, w)]

    noisy[coords[0], coords[1], :] = 255

    # Pepper
    num_pepper = int(num_pixels * (1 - salt_vs_pepper))
    coords = [np.random.randint(0, i, num_pepper) for i in (h, w)]
    noisy[coords[0], coords[1], :] = 0

    return noisy, {"amount": amount}


def speckle_noise(image, std_range=(0.05, 0.3)):
    std = random.uniform(*std_range)
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + image.astype(np.float32) * noise
    return np.clip(noisy, 0, 255).astype(np.uint8), {"std": std}


def poisson_noise(image, scale_range=(0.5, 5.0)):
    scale = random.uniform(*scale_range)

    image_f = image.astype(np.float32)

    # normalizar para [0,1]
    image_norm = image_f / 255.0

    # aplicar Poisson
    noisy = np.random.poisson(image_norm * scale) / scale

    # voltar para [0,255]
    noisy = noisy * 255.0

    return np.clip(noisy, 0, 255).astype(np.uint8), {"scale": scale}


def _get_luminance(img):
    # img em BGR (OpenCV)
    return 0.114*img[:,:,0] + 0.587*img[:,:,1] + 0.299*img[:,:,2]


def luminance_gaussian_noise(image, std_range=(5, 25)):
    std = random.uniform(*std_range)

    img = image.astype(np.float32)
    lum = _get_luminance(img)

    noise = np.random.normal(0, std, lum.shape).astype(np.float32)
    lum_noisy = lum + noise

    delta = lum_noisy - lum
    img[:,:,0] += delta
    img[:,:,1] += delta
    img[:,:,2] += delta

    return np.clip(img, 0, 255).astype(np.uint8), {
        "std": std,
        "mode": "luminance"
    }


def luminance_poisson_noise(image, scale_range=(0.5, 5.0)):
    scale = random.uniform(*scale_range)

    img = image.astype(np.float32)
    lum = _get_luminance(img) / 255.0

    noisy = np.random.poisson(lum * scale) / scale
    noisy = noisy * 255.0

    delta = noisy - (lum * 255.0)
    img[:,:,0] += delta
    img[:,:,1] += delta
    img[:,:,2] += delta

    return np.clip(img, 0, 255).astype(np.uint8), {
        "scale": scale,
        "mode": "luminance"
    }

def luminance_speckle_noise(image, std_range=(0.05, 0.3)):
    std = random.uniform(*std_range)

    img = image.astype(np.float32)
    lum = _get_luminance(img)

    noise = np.random.normal(0, std, lum.shape).astype(np.float32)
    lum_noisy = lum + lum * noise

    delta = lum_noisy - lum
    img[:,:,0] += delta
    img[:,:,1] += delta
    img[:,:,2] += delta

    return np.clip(img, 0, 255).astype(np.uint8), {
        "std": std,
        "mode": "luminance"
    }

def uniform_noise(image, range_val=(5, 30)):
    a = random.uniform(*range_val)
    noise = np.random.uniform(-a, a, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8), {"range": a}
"""
def banding_noise(image, amplitude_range=(10, 60), frequency_range=(2, 10), orientation=None):
    amp = random.uniform(*amplitude_range)
    freq = random.randint(*frequency_range)

    img = image.astype(np.float32)
    h, w, _ = img.shape

    if orientation is None:
        orientation = random.choice(["horizontal", "vertical"])

    if orientation == "horizontal":
        # várias ondas ao longo da altura
        x = np.linspace(0, 2*np.pi * freq, h)
        band = np.sin(x).reshape(h, 1) * amp
    else:
        # várias ondas ao longo da largura
        x = np.linspace(0, 2*np.pi * freq, w)
        band = np.sin(x).reshape(1, w) * amp

    img[:,:,0] += band
    img[:,:,1] += band
    img[:,:,2] += band

    return np.clip(img, 0, 255).astype(np.uint8), {
        "amplitude": amp,
        "frequency": freq,
        "orientation": orientation
    }
"""


NOISE_FUNCTIONS = {
    # RGB noise (colorido)
    "gaussian": gaussian_noise,
    "salt_pepper": salt_and_pepper_noise,
    "speckle": speckle_noise,
    "poisson": poisson_noise,
    "uniform": uniform_noise,

    # Luminance noise (fotográfico, realista)
    "lum_gaussian": luminance_gaussian_noise,
    "lum_poisson": luminance_poisson_noise,
    "lum_speckle": luminance_speckle_noise,

    # Estrutural / sensor
    #"banding": banding_noise, não usado
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
