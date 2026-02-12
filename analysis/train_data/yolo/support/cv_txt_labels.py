from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image


# ---------------------------------------------------------
# Métricas de cor
# ---------------------------------------------------------
def _colorfulness(rgb: np.ndarray) -> float:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)

    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def _color_variation(rgb: np.ndarray) -> float:
    x = rgb.astype(np.float32) / 255.0
    return float(x.reshape(-1, 3).std(axis=0).mean())


# ---------------------------------------------------------
# Função principal de análise YOLO
# ---------------------------------------------------------
def analyze_cv_yolo_datasets(
    text_photo_images: str | Path,
    text_photo_labels: str | Path,
    skill_images: str | Path,
    skill_labels: str | Path,
    img_exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
) -> pd.DataFrame:
    """
    Analisa conjuntamente:
        - Dataset TEXT-PHOTO
        - Dataset SKILL-TABLE

    Retorna DataFrame com:
        filename, width, height, aspect_ratio
        colorfulness, color_variation
        number_text_block, number_photo, number_skill_table
        x_center_mean, y_center_mean
    """

    text_photo_images = Path(text_photo_images)
    text_photo_labels = Path(text_photo_labels)
    skill_images = Path(skill_images)
    skill_labels = Path(skill_labels)

    rows: list[dict] = []

    # -----------------------------------------------------
    # Conjunto total de imagens (união dos dois datasets)
    # -----------------------------------------------------
    image_files = {
        p.name: p
        for root in [text_photo_images, skill_images]
        for p in Path(root).rglob("*")
        if p.suffix.lower() in img_exts
    }

    # -----------------------------------------------------
    # Iterar imagens
    # -----------------------------------------------------
    for filename, img_path in image_files.items():

        # ---------- Ler imagem ----------
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            rgb = np.array(im)

        aspect_ratio = w / h if h else np.nan
        colorfulness = _colorfulness(rgb)
        color_variation = _color_variation(rgb)

        # ---------- Inicializar contagens ----------
        n_text = 0
        n_photo = 0
        n_skill = 0
        x_centers = []
        y_centers = []

        # =================================================
        # TEXT-PHOTO labels
        # =================================================
        tp_label = text_photo_labels / f"{Path(filename).stem}.txt"

        if tp_label.exists():
            lines = tp_label.read_text(encoding="utf-8", errors="ignore").splitlines()

            for ln in lines:
                parts = ln.strip().split()
                if len(parts) != 5:
                    continue

                class_id, x, y, _, _ = parts
                class_id = int(class_id)

                x_centers.append(float(x))
                y_centers.append(float(y))

                if class_id == 0:
                    n_text += 1
                elif class_id == 1:
                    n_photo += 1

        # =================================================
        # SKILL-TABLE labels
        # =================================================
        sk_label = skill_labels / f"{Path(filename).stem}.txt"

        if sk_label.exists():
            lines = sk_label.read_text(encoding="utf-8", errors="ignore").splitlines()

            for ln in lines:
                parts = ln.strip().split()
                if len(parts) != 5:
                    continue

                _, x, y, _, _ = parts  # class_id = 0 sempre
                n_skill += 1

                x_centers.append(float(x))
                y_centers.append(float(y))

        # ---------- Médias ----------
        x_mean = float(np.mean(x_centers)) if x_centers else np.nan
        y_mean = float(np.mean(y_centers)) if y_centers else np.nan

        rows.append(
            {
                "filename": filename,
                "width": int(w),
                "height": int(h),
                "width_height": f"{w}x{h}",
                "aspect_ratio": float(aspect_ratio),
                "colorfulness": float(colorfulness),
                "color_variation": float(color_variation),
                "number_text_block": int(n_text),
                "number_photo": int(n_photo),
                "number_skill_table": int(n_skill),
                "x_center_mean": x_mean,
                "y_center_mean": y_mean,
            }
        )

    df = pd.DataFrame(rows).sort_values("filename").reset_index(drop=True)
    return df
