import cv2
import os
import random
from pathlib import Path

import cv2
import random
def apply_defocus_blur(image, strength="normal"):
    if strength == "low":
        kernel_sizes = [5, 7, 9]
        passes = 1
    elif strength == "normal":
        kernel_sizes = [15, 21, 31]
        passes = 1
    elif strength == "strong":
        kernel_sizes = [31, 41, 61, 81]
        passes = 2
    elif strength == "extreme":
        kernel_sizes = [61, 81, 101, 121, 151]
        passes = 3
    elif strength == "random_low":
        kernel_sizes = [5, 7, 9, 15, 21, 25, 31, 41, 61, 81]
        passes = random.randint(1, 3)
    elif strength == "random_strong":
        kernel_sizes = [23, 25, 31, 35, 41, 51, 61, 81, 101, 121, 151]
        passes = random.randint(1, 3)
    elif isinstance(strength, list):
        kernel_sizes = strength
        passes = 1
    else:
        raise ValueError("strength inválido")

    kernel_sizes = [int(k) for k in kernel_sizes if int(k) % 2 == 1]
    if not kernel_sizes:
        raise ValueError("kernel_sizes precisa de ímpares")

    k = random.choice(kernel_sizes)

    blurred = image.copy()
    for _ in range(passes):
        blurred = cv2.GaussianBlur(blurred, (k, k), 0)

    return blurred, {"kernel": k, "passes": passes}





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

        blurred, info = apply_defocus_blur(image, strength=strength)

        k = info["kernel"]
        passes = info["passes"]

        output_name = f"{img_path.stem}_defocus_k{k}_p{passes}{img_path.suffix}"
        output_path = output_dir / output_name

        ok = cv2.imwrite(str(output_path), blurred)
        if not ok:
            print(f"Erro ao guardar: {output_path}")

        print(f"{img_path.name} -> {output_name} (kernel={k}, passes={passes})")
