from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


# ===== CONFIGURAÇÃO =====
MASKS_DIR = Path(
        "data/image_analysis/detection/yolo/blur_and_sharp_segmentation/"
        "model_sharp-background_blur/25black_masks")

# Segurança: só processa PNG
EXTENSION = ".png"

# =======================


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """
    Inverte uma máscara binária:
    255 -> 0
    0   -> 255
    """
    if mask.ndim != 2:
        raise ValueError("A máscara deve ser grayscale")

    # Garantir binária antes de inverter
    mask_bin = (mask > 0).astype("uint8") * 255

    return 255 - mask_bin


def invert_masks_in_folder(folder: Path):
    """
    Inverte todas as máscaras numa pasta (in-place).
    """
    if not folder.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {folder}")

    mask_files = sorted(folder.glob(f"*{EXTENSION}"))

    print(f"Encontradas {len(mask_files)} máscaras para inverter.")

    for mask_path in tqdm(mask_files, desc="Inverting masks"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"⚠️ Falha ao ler {mask_path.name}")
            continue

        inverted = invert_mask(mask)

        # Substituir a máscara original
        cv2.imwrite(str(mask_path), inverted)

    print("\n✔ Inversão concluída com sucesso.")


if __name__ == "__main__":
    invert_masks_in_folder(MASKS_DIR)
