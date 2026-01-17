import cv2
import numpy as np
import random
import json
from pathlib import Path


# ==================================================
# Canal Rayleigh + erros de bits (simulado)
# ==================================================

def fast_fading_rayleigh(image, snr_range=(5, 30)):
    """
    Simula a degradação causada por erros de transmissão num canal
    Rayleigh com base no SNR do recetor.

    SNR baixo  -> muitos erros
    SNR alto   -> poucos erros
    """

    snr = random.uniform(*snr_range)

    # Converter SNR (dB) para uma probabilidade de erro aproximada
    # Isto é um modelo empírico, não físico rigoroso.
    ber = np.exp(-snr / 6.0) * 0.1
    ber = np.clip(ber, 1e-4, 0.2)

    degraded = image.copy()
    h, w, c = image.shape

    # Número de blocos corrompidos depende da BER
    num_blocks = int(ber * 5000)
    num_blocks = max(1, num_blocks)

    for _ in range(num_blocks):
        # Blocos pequenos (artefactos típicos do JPEG2000 corrompido)
        block_w = random.randint(4, 32)
        block_h = random.randint(4, 32)

        x = random.randint(0, w - block_w)
        y = random.randint(0, h - block_h)

        # Três tipos de erro típicos:
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


# ==================================================
# Dispatcher (se no futuro adicionares outros canais)
# ==================================================

CHANNEL_FUNCTIONS = {
    "fast_fading_rayleigh": fast_fading_rayleigh
}


def apply_random_channel(
    image,
    channel_types="all",
    strength_ranges=None
):
    if channel_types == "all":
        available = list(CHANNEL_FUNCTIONS.keys())
    elif isinstance(channel_types, list):
        available = channel_types
    else:
        raise ValueError("channel_types must be 'all' ou uma lista")

    chosen = random.choice(available)
    channel_func = CHANNEL_FUNCTIONS[chosen]

    params = {}
    if strength_ranges and chosen in strength_ranges:
        params = strength_ranges[chosen]

    degraded, meta = channel_func(image, **params)
    return degraded, chosen, meta


# ==================================================
# Processamento em pasta + Manifest
# ==================================================

def process_folder_channel(
    input_dir,
    output_dir,
    channel_types="all",
    strength_ranges=None,
    valid_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
    save_manifest=True,
    manifest_name="channel_manifest.json"
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

        degraded, ctype, meta = apply_random_channel(
            image,
            channel_types=channel_types,
            strength_ranges=strength_ranges
        )

        output_name = f"{img_path.stem}_{ctype}{img_path.suffix}"
        output_path = output_dir / output_name

        ok = cv2.imwrite(str(output_path), degraded)
        if not ok:
            print(f"Erro ao escrever: {output_name}")
            continue

        if save_manifest:
            manifest.append({
                "input_file": img_path.name,
                "output_file": output_name,
                "channel_type": ctype,
                "params": meta
            })

        print(f"{img_path.name} -> {output_name}")

    if save_manifest:
        manifest_path = output_dir / manifest_name
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"Manifest guardado em: {manifest_path}")
