import cv2
import numpy as np
import random
import json
from pathlib import Path

CHANNEL_STRENGTHS = {
    "fast_fading_rayleigh": {
        "low":    {"snr_range": (25, 40)},
        "medium": {"snr_range": (12, 25)},
        "strong": {"snr_range": (3, 12)}
    },

    "block_drop": {
        "low":    {"drop_rate": 0.01},
        "medium": {"drop_rate": 0.05},
        "strong": {"drop_rate": 0.15}
    },

    "block_shuffle": {
        "low":    {"num_blocks": 20},
        "medium": {"num_blocks": 60},
        "strong": {"num_blocks": 150}
    },

    "color_channel_swap": {
        "low":    {"prob": 0.2},
        "medium": {"prob": 0.5},
        "strong": {"prob": 1.0}
    },

    "posterize_glitch": {
        "low":    {"levels": 64},
        "medium": {"levels": 16},
        "strong": {"levels": 4}
    },

    "random_bit_flip": {
        "low":    {"num_pixels": 500},
        "medium": {"num_pixels": 3000},
        "strong": {"num_pixels": 10000}
    }
}




def fast_fading_rayleigh(image, snr_range=(5, 30)):
    """
    Simula degradação de transmissão num canal Rayleigh.

    SNR alto  -> poucos erros
    SNR baixo -> muitos erros
    """

    snr = random.uniform(*snr_range)

    # Modelo empírico de BER (quanto menor o SNR, maior o erro)
    ber = np.exp(-snr / 6.0) * 0.1
    ber = np.clip(ber, 1e-4, 0.25)

    degraded = image.copy()
    h, w, _ = image.shape

    # Número de blocos corrompidos
    num_blocks = int(ber * 6000)
    num_blocks = max(1, num_blocks)

    for _ in range(num_blocks):
        block_w = random.randint(4, 32)
        block_h = random.randint(4, 32)

        x = random.randint(0, w - block_w)
        y = random.randint(0, h - block_h)

        mode = random.choice(["noise", "zero", "shift"])

        if mode == "noise":
            noise = np.random.randint(0, 256, (block_h, block_w, 3), dtype=np.uint8)
            degraded[y:y+block_h, x:x+block_w] = noise

        elif mode == "zero":
            degraded[y:y+block_h, x:x+block_w] = 0

        elif mode == "shift":
            patch = degraded[y:y+block_h, x:x+block_w]
            shifted = np.roll(patch, random.randint(1, 5), axis=1)
            degraded[y:y+block_h, x:x+block_w] = shifted

    return degraded, {
        "type": "channel",
        "channel": "fast_fading_rayleigh",
        "snr": snr,
        "ber_estimated": ber,
        "num_corrupted_blocks": num_blocks
    }


def block_drop(image, drop_rate=0.05):
    degraded = image.copy()
    h, w, _ = image.shape

    num_blocks = int(drop_rate * 2000)
    for _ in range(num_blocks):
        bw = min(random.randint(8, 64), w-1)
        bh = min(random.randint(8, 64), h-1)
        x = random.randint(0, w - bw)
        y = random.randint(0, h - bh)
        degraded[y:y+bh, x:x+bw] = 0

    return degraded, {
        "type": "channel",
        "channel": "block_drop",
        "drop_rate": drop_rate,
        "num_blocks": num_blocks
    }


def block_shuffle(image, num_blocks=50):
    degraded = image.copy()
    h, w, _ = image.shape

    for _ in range(num_blocks):
        bw = min(random.randint(8, 64), w-1)
        bh = min(random.randint(8, 64), h-1)

        x1 = random.randint(0, w - bw)
        y1 = random.randint(0, h - bh)
        x2 = random.randint(0, w - bw)
        y2 = random.randint(0, h - bh)

        tmp = degraded[y1:y1+bh, x1:x1+bw].copy()
        degraded[y1:y1+bh, x1:x1+bw] = degraded[y2:y2+bh, x2:x2+bw]
        degraded[y2:y2+bh, x2:x2+bw] = tmp

    return degraded, {
        "type": "channel",
        "channel": "block_shuffle",
        "num_blocks": num_blocks
    }

def color_channel_swap(image, prob=0.5):
    out = image.copy()
    if random.random() < prob:
        out = out[:, :, [1, 2, 0]]  # BGR → GRB
    return out, {
        "type": "channel",
        "channel": "color_channel_swap",
        "prob": prob
    }

def posterize_glitch(image, levels=16):
    levels = max(1, levels)
    out = image.copy().astype(np.float32)
    step = 256 / levels
    out = np.floor(out / step) * step
    return out.astype(np.uint8), {
        "type": "channel",
        "channel": "posterize_glitch",
        "levels": levels
    }

def random_bit_flip(image, num_pixels=2000):
    h, w, _ = image.shape
    num_pixels = min(num_pixels, h*w*3)
    degraded = image.copy()
    
    for _ in range(num_pixels):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        c = random.randint(0, 2)
        bit = 1 << random.randint(0, 7)
        degraded[y, x, c] ^= bit

    return degraded, {
        "type": "channel",
        "channel": "random_bit_flip",
        "num_pixels": num_pixels
    }


CHANNEL_TYPES = {
    "fast_fading_rayleigh": fast_fading_rayleigh,
    "block_drop": block_drop,
    "block_shuffle": block_shuffle,
    "color_channel_swap": color_channel_swap,
    "posterize_glitch": posterize_glitch,
    "random_bit_flip": random_bit_flip
}

def apply_channel(image, channel_type="fast_fading_rayleigh", strength="medium"):
    if channel_type not in CHANNEL_TYPES:
        raise ValueError(f"channel_type inválido: {channel_type}")

    if strength not in CHANNEL_STRENGTHS[channel_type]:
        raise ValueError(f"strength inválido para {channel_type}")

    func = CHANNEL_TYPES[channel_type]
    params = CHANNEL_STRENGTHS[channel_type][strength]

    out, meta = func(image, **params)

    applied = f"{channel_type}_{strength}"
    return out, applied, meta

def apply_random_channel(image):
    channel_type = random.choice(list(CHANNEL_TYPES.keys()))
    strength = random.choice(["low", "medium", "strong"])
    return apply_channel(image, channel_type, strength)


def process_folder_channel(
    input_dir,
    output_dir,
    valid_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
    save_manifest=True,
    manifest_name="channel_manifest.json",
    use_random=False,
    channel_type="fast_fading_rayleigh",
    strength="medium"
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
            degraded, applied, params = apply_random_channel(image)
        else:
            degraded, applied, params = apply_channel(image, channel_type, strength)

        tag = applied if applied else "none"
        out_name = f"{img_path.stem}_channel_{tag}{img_path.suffix}"
        cv2.imwrite(str(output_dir / out_name), degraded)

        if save_manifest:
            manifest.append({
                "input": img_path.name,
                "output": out_name,
                "applied_channel": applied,
                "params": params
            })

        print(f"{img_path.name} -> {out_name}")

    if save_manifest:
        with open(output_dir / manifest_name, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
