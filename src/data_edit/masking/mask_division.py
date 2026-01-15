from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


RAW_IMAGES = Path("data/image_analysis/detection/yolo/blur_and_sharp_segmentation/raw_images")
RAW_MASKS = Path("data/image_analysis/detection/yolo/blur_and_sharp_segmentation/raw_masks")

TYPE = "White"
#TYPE = "Black"


# Configuração por tipo
if TYPE == "White":
    PERCEN = 45  # máximo de white permitido
    OUT_MASKS = Path(
        f"data/image_analysis/detection/yolo/blur_and_sharp_segmentation/"
        f"model_blur-background_sharp/{PERCEN}white_masks"
    )
    OUT_IMAGES = Path(
        f"data/image_analysis/detection/yolo/blur_and_sharp_segmentation/"
        f"model_blur-background_sharp/{PERCEN}white_images"
    )

elif TYPE == "Black":
    PERCEN = 25  # máximo de black permitido
    OUT_MASKS = Path(
        f"data/image_analysis/detection/yolo/blur_and_sharp_segmentation/"
        f"model_sharp-background_blur/{PERCEN}black_masks"
    )
    OUT_IMAGES = Path(
        f"data/image_analysis/detection/yolo/blur_and_sharp_segmentation/"
        f"model_sharp-background_blur/{PERCEN}black_images"
    )

else:
    raise ValueError("TYPE inválido! Use 'White' ou 'Black'.")


def compute_white_black_percentage(mask: np.ndarray):
    """
    Calcula percentagem de branco e preto numa máscara binária.
    """
    total_pixels = mask.size
    white_pixels = np.count_nonzero(mask == 255)
    black_pixels = np.count_nonzero(mask == 0)

    white_pct = (white_pixels / total_pixels) * 100
    black_pct = (black_pixels / total_pixels) * 100

    return white_pct, black_pct


def mask_passes_filter(mask: np.ndarray) -> bool:
    """
    Decide se a máscara cumpre os critérios definidos por TYPE e PERCEN.
    """
    white_pct, black_pct = compute_white_black_percentage(mask)

    if TYPE == "White":
        return white_pct <= PERCEN

    if TYPE == "Black":
        return black_pct <= PERCEN

    return False


def divide_masks_and_images():
    """
    Move (copia) as masks e imagens correspondentes
    para as pastas de saída, de acordo com a percentagem
    de branco/preto.
    """
    # Criar diretórios se não existirem
    OUT_MASKS.mkdir(parents=True, exist_ok=True)
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0

    for mask_path in tqdm(sorted(RAW_MASKS.glob("*.png")), desc=f"Processing ({TYPE})"):
        img_path = RAW_IMAGES / f"{mask_path.stem}.jpg"

        if not img_path.exists():
            skipped += 1
            continue

        # Ler mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            skipped += 1
            continue

        # Garantir máscara binária correta
        mask_bin = (mask > 0).astype("uint8") * 255

        if not mask_passes_filter(mask_bin):
            skipped += 1
            continue

        # Copiar mask
        cv2.imwrite(str(OUT_MASKS / mask_path.name), mask_bin)

        # Copiar imagem correspondente
        image = cv2.imread(str(img_path))
        cv2.imwrite(str(OUT_IMAGES / img_path.name), image) #type: ignore

        kept += 1

    print("\nResumo:")
    print(f"  Tipo: {TYPE}")
    print(f"  Percentagem limite: {PERCEN}%")
    print(f"  Mantidas: {kept}")
    print(f"  Ignoradas: {skipped}")


if __name__ == "__main__":
    divide_masks_and_images()
