from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image


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
        filename
        number_text_block, number_photo, number_skill_table
        x_center_mean, y_center_mean
    """

    text_photo_images = Path(text_photo_images)
    text_photo_labels = Path(text_photo_labels)
    skill_images = Path(skill_images)
    skill_labels = Path(skill_labels)

    rows: list[dict] = []

    # -----------------------------------------------------
    # Conjunto total de imagens (sem colisões de nome)
    # -----------------------------------------------------
    image_files = list(text_photo_images.rglob("*")) + list(skill_images.rglob("*"))
    image_files = [p for p in image_files if p.suffix.lower() in img_exts]

    # remover duplicados reais por caminho absoluto
    image_files = list(dict.fromkeys(image_files))

    # -----------------------------------------------------
    # Iterar imagens
    # -----------------------------------------------------
    for img_path in image_files:

        filename = img_path.name

        # ---------- Proteção contra imagem corrompida ----------
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            continue  # ignora imagem inválida

        # ---------- Inicializar contagens ----------
        n_text = 0
        n_photo = 0
        n_skill = 0
        x_centers = []
        y_centers = []

        # =================================================
        # TEXT-PHOTO labels
        # =================================================
        tp_label = text_photo_labels / f"{img_path.stem}.txt"

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
        sk_label = skill_labels / f"{img_path.stem}.txt"

        if sk_label.exists():
            lines = sk_label.read_text(encoding="utf-8", errors="ignore").splitlines()

            for ln in lines:
                parts = ln.strip().split()
                if len(parts) != 5:
                    continue

                _, x, y, _, _ = parts
                n_skill += 1

                x_centers.append(float(x))
                y_centers.append(float(y))

        # ---------- Médias ----------
        x_mean = float(np.mean(x_centers)) if x_centers else np.nan
        y_mean = float(np.mean(y_centers)) if y_centers else np.nan

        rows.append(
            {
                "filename": filename,
                "number_text_block": int(n_text),
                "number_photo": int(n_photo),
                "number_skill_table": int(n_skill),
                "x_center_mean": x_mean,
                "y_center_mean": y_mean,
            }
        )

    df = pd.DataFrame(rows).sort_values("filename").reset_index(drop=True)
    return df
