import cv2
import os
import random
from pathlib import Path

import cv2
import random

def apply_defocus_blur(image, strength="normal"):
    if strength == "normal":
        kernel_sizes = [5, 7, 9]
    elif strength == "strong":
        kernel_sizes = [11, 15, 21]
    elif strength == "random":
        kernel_sizes = [5, 7, 9, 11, 15, 21]
    elif isinstance(strength, list):
        kernel_sizes = strength
    else:
        raise ValueError("strength must be 'normal', 'strong', 'random' or a list of odd integers")

    # garantir que todos os valores são inteiros ímpares
    kernel_sizes = [int(k) for k in kernel_sizes if int(k) % 2 == 1]

    if not kernel_sizes:
        raise ValueError("kernel_sizes must contain at least one odd integer")

    k = int(random.choice(kernel_sizes))
    blurred = cv2.GaussianBlur(image, (k, k), 0)

    return blurred, k




def process_folder_defocus_blur(
    input_dir,
    output_dir,
    strength="normal",
    valid_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tiff")
):
    """
    Aplica defocused blur a todas as imagens de uma pasta e guarda noutra.

    input_dir: pasta com as imagens originais
    output_dir: pasta onde as imagens desfocadas vão ser guardadas
    strength: "normal", "strong", "random" ou lista de kernel sizes
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in input_dir.iterdir():
        if img_path.suffix.lower() not in valid_exts:
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Erro ao ler: {img_path}")
            continue

        blurred, k = apply_defocus_blur(image, strength=strength)

        # opcional: incluir o kernel no nome do ficheiro
        output_name = f"{img_path.stem}_defocus_k{k}{img_path.suffix}"
        output_path = output_dir / output_name

        cv2.imwrite(str(output_path), blurred)

        print(f"{img_path.name} -> {output_name} (k={k})")
