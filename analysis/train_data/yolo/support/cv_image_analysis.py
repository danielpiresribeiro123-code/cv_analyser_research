from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image


def _colorfulness_hasler_susstrunk(rgb: np.ndarray) -> float:
    """
    Métrica de 'colorfulness' (Hasler & Süsstrunk).
    Entrada: imagem RGB uint8 com shape (H, W, 3).
    """
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)

    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)

    mu_rg, sigma_rg = rg.mean(), rg.std()
    mu_yb, sigma_yb = yb.mean(), yb.std()

    return float(np.sqrt(sigma_rg**2 + sigma_yb**2) + 0.3 * np.sqrt(mu_rg**2 + mu_yb**2))


def _color_variation_rgb(rgb: np.ndarray) -> float:
    """
    Variação global de cor: desvio padrão médio dos canais RGB (normalizado 0..1).
    """
    x = rgb.astype(np.float32) / 255.0
    stds = x.reshape(-1, 3).std(axis=0)  # std por canal
    return float(stds.mean())


def build_images_metadata_df(
    images_dir: str | Path,
    labels_dir: str | Path,
    img_exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
) -> pd.DataFrame:
    """
    Cria um DataFrame com características das imagens e flags relativas aos labels YOLO (.txt).

    Regras:
    - label_path = labels_dir / (image_stem + ".txt")
    - have_to_remove = True se o .txt NÃO existir
    - has_labels = True se o .txt existir e tiver pelo menos 1 linha (1 bbox)
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir não existe: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels_dir não existe: {labels_dir}")

    image_files = [p for p in images_dir.rglob("*") if p.suffix.lower() in img_exts]

    rows: list[dict] = []
    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        label_exists = label_path.exists()

        # has_labels: existe e tem pelo menos 1 bbox (1 linha não vazia)
        has_labels = False
        if label_exists:
            try:
                lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8", errors="ignore").splitlines()]
                has_labels = any(len(ln) > 0 for ln in lines)
            except Exception:
                # se houver algum txt corrompido, marca como "sem labels"
                has_labels = False

        have_to_remove = not label_exists  # sem txt = esqueceste de apagar / fora do dataset

        # Ler imagem e extrair stats
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            rgb = np.array(im)  # (H, W, 3) uint8

        aspect_ratio = (w / h) if h else np.nan
        colorfulness = _colorfulness_hasler_susstrunk(rgb)
        color_variation = _color_variation_rgb(rgb)

        rows.append(
            {
                "filename": img_path.name,
                "path": str(img_path),
                "width": int(w),
                "height": int(h),
                "width_height": f"{w}x{h}",
                "aspect_ratio": float(aspect_ratio),
                "colorfulness": float(colorfulness),
                "color_variation": float(color_variation),
                "has_labels": bool(has_labels),
                "have_to_remove": bool(have_to_remove),
            }
        )

    df = pd.DataFrame(rows).sort_values(["have_to_remove", "has_labels", "filename"]).reset_index(drop=True)
    return df
