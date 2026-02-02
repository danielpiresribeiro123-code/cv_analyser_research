import cv2
import numpy as np
import random
from typing import List, Union
import cv2
import json
from pathlib import Path

def _to_float(img):
    return img.astype(np.float32)

def _clip(img):
    return np.clip(img, 0, 255).astype(np.uint8)


# COLOR NOISE

def color_gaussian_noise(image, mean=0, std_range=(5, 25)):
    std = random.uniform(*std_range)
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = _to_float(image) + noise
    return _clip(noisy), {"std": std, "mode": "color"}


def color_poisson_noise(image, scale_range=(0.5, 5.0)):
    scale = random.uniform(*scale_range)

    img = _to_float(image) / 255.0
    noisy = np.random.poisson(img * scale) / scale
    noisy = noisy * 255.0

    return _clip(noisy), {"scale": scale, "mode": "color"}


def color_speckle_noise(image, std_range=(0.05, 0.3)):
    std = random.uniform(*std_range)
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    noisy = _to_float(image) + _to_float(image) * noise
    return _clip(noisy), {"std": std, "mode": "color"}


def color_salt_pepper_noise(image, amount_range=(0.001, 0.02), salt_vs_pepper=0.5):
    amount = random.uniform(*amount_range)

    noisy = image.copy()
    h, w, _ = image.shape
    num_pixels = int(amount * h * w)

    # Salt
    num_salt = int(num_pixels * salt_vs_pepper)
    coords = [np.random.randint(0, i, num_salt) for i in (h, w)]
    noisy[coords[0], coords[1], :] = 255

    # Pepper
    num_pepper = int(num_pixels * (1 - salt_vs_pepper))
    coords = [np.random.randint(0, i, num_pepper) for i in (h, w)]
    noisy[coords[0], coords[1], :] = 0

    return noisy, {"amount": amount, "mode": "color"}


def color_uniform_noise(image, range_val=(5, 30)):
    a = random.uniform(*range_val)
    noise = np.random.uniform(-a, a, image.shape).astype(np.float32)
    noisy = _to_float(image) + noise
    return _clip(noisy), {"range": a, "mode": "color"}



# GREYSCALE NOISE

def _apply_gray_noise(image, gray_noise):
    noisy = image.astype(np.float32)
    for c in range(3):
        noisy[:, :, c] += gray_noise
    return _clip(noisy)




def gray_gaussian_noise(image, mean=0, std_range=(5, 25)):
    std = random.uniform(*std_range)
    h, w, _ = image.shape
    noise = np.random.normal(mean, std, (h, w)).astype(np.float32)
    noisy = _apply_gray_noise(image, noise)
    return noisy, {"std": std, "mode": "greyscale"}


def gray_poisson_noise(image, scale_range=(0.5, 5.0)):
    scale = random.uniform(*scale_range)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    noisy = np.random.poisson(gray * scale) / scale
    noisy = noisy * 255.0

    delta = noisy - gray * 255.0
    noisy_img = _apply_gray_noise(image, delta)
    return noisy_img, {"scale": scale, "mode": "greyscale"}


def gray_speckle_noise(image, std_range=(0.05, 0.3)):
    std = random.uniform(*std_range)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    noise = np.random.normal(0, std, gray.shape).astype(np.float32)
    delta = gray * noise

    noisy = _apply_gray_noise(image, delta)
    return noisy, {"std": std, "mode": "greyscale"}


def gray_salt_pepper_noise(image, amount_range=(0.001, 0.02), salt_vs_pepper=0.5):
    amount = random.uniform(*amount_range)

    noisy = image.copy()
    h, w, _ = image.shape
    num_pixels = int(amount * h * w)

    num_salt = int(num_pixels * salt_vs_pepper)
    coords = [np.random.randint(0, i, num_salt) for i in (h, w)]
    noisy[coords[0], coords[1], :] = 255

    num_pepper = int(num_pixels * (1 - salt_vs_pepper))
    coords = [np.random.randint(0, i, num_pepper) for i in (h, w)]
    noisy[coords[0], coords[1], :] = 0

    return noisy, {"amount": amount, "mode": "greyscale"}


def gray_uniform_noise(image, range_val=(5, 30)):
    a = random.uniform(*range_val)
    h, w, _ = image.shape
    noise = np.random.uniform(-a, a, (h, w)).astype(np.float32)
    noisy = _apply_gray_noise(image, noise)
    return noisy, {"range": a, "mode": "greyscale"}


# Dicts

NOISE_STRENGTHS = {

    # ===== COLOR =====
    "color_gaussian": {
        "low":    {"std_range": (5, 10)},
        "medium": {"std_range": (10, 20)},
        "strong": {"std_range": (20, 40)},
    },

    "color_poisson": {
        "low":    {"scale_range": (0.5, 1.5)},
        "medium": {"scale_range": (1.5, 3.0)},
        "strong": {"scale_range": (3.0, 6.0)},
    },

    "color_speckle": {
        "low":    {"std_range": (0.05, 0.1)},
        "medium": {"std_range": (0.1, 0.2)},
        "strong": {"std_range": (0.2, 0.4)},
    },

    "color_salt_pepper": {
        "low":    {"amount_range": (0.001, 0.005)},
        "medium": {"amount_range": (0.005, 0.015)},
        "strong": {"amount_range": (0.015, 0.03)},
    },

    "color_uniform": {
        "low":    {"range_val": (5, 10)},
        "medium": {"range_val": (10, 20)},
        "strong": {"range_val": (20, 40)},
    },


    # ===== GREYSCALE =====
    "gray_gaussian": {
        "low":    {"std_range": (5, 10)},
        "medium": {"std_range": (10, 20)},
        "strong": {"std_range": (20, 40)},
    },

    "gray_poisson": {
        "low":    {"scale_range": (0.5, 1.5)},
        "medium": {"scale_range": (1.5, 3.0)},
        "strong": {"scale_range": (3.0, 6.0)},
    },

    "gray_speckle": {
        "low":    {"std_range": (0.05, 0.1)},
        "medium": {"std_range": (0.1, 0.2)},
        "strong": {"std_range": (0.2, 0.4)},
    },

    "gray_salt_pepper": {
        "low":    {"amount_range": (0.001, 0.005)},
        "medium": {"amount_range": (0.005, 0.015)},
        "strong": {"amount_range": (0.015, 0.03)},
    },

    "gray_uniform": {
        "low":    {"range_val": (5, 10)},
        "medium": {"range_val": (10, 20)},
        "strong": {"range_val": (20, 40)},
    },
}



NOISE_TYPES = {
    # Color
    "color_gaussian": color_gaussian_noise,
    "color_poisson": color_poisson_noise,
    "color_speckle": color_speckle_noise,
    "color_salt_pepper": color_salt_pepper_noise,
    "color_uniform": color_uniform_noise,

    # Greyscale
    "gray_gaussian": gray_gaussian_noise,
    "gray_poisson": gray_poisson_noise,
    "gray_speckle": gray_speckle_noise,
    "gray_salt_pepper": gray_salt_pepper_noise,
    "gray_uniform": gray_uniform_noise,
}

# Applicar noise

def apply_noise(image, noise_type: str, strength: str):
    """
    noise_type : 'gaussian', 'salt_pepper', 'speckle', ...
    strength   : 'low', 'medium', 'strong'
    """

    if noise_type not in NOISE_TYPES:
        raise ValueError(f"Tipo de noise inválido: {noise_type}")

    if strength not in NOISE_STRENGTHS[noise_type]:
        raise ValueError(f"Força inválida para {noise_type}: {strength}")

    func = NOISE_TYPES[noise_type]
    params = NOISE_STRENGTHS[noise_type][strength]

    out, p = func(image, **params)

    applied = f"{noise_type}_{strength}"
    return out, applied, p

def apply_random_noise(image):
    noise_type = random.choice(list(NOISE_TYPES.keys()))
    strength = random.choice(["low", "medium", "strong"])
    return apply_noise(image, noise_type, strength)



def process_folder_noise(
    input_dir,
    output_dir,
    valid_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
    save_manifest=True,
    manifest_name="noise_manifest.json",
    use_random: bool = False,
    noise_type="gaussian",   # gaussian, salt_pepper, speckle, ...
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
            noisy, applied, params = apply_random_noise(image)
        else:
            noisy, applied, params = apply_noise(image, noise_type, strength)

        tag = applied if applied else "none"
        out_name = f"{img_path.stem}_noise_{tag}{img_path.suffix}"
        cv2.imwrite(str(output_dir / out_name), noisy)

        if save_manifest:
            manifest.append({
                "input": img_path.name,
                "output": out_name,
                "applied_noise": applied,
                "params": params
            })

        print(f"{img_path.name} -> {out_name}")

    if save_manifest:
        with open(output_dir / manifest_name, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
