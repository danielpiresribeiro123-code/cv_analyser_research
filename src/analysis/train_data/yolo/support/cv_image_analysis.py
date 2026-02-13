from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image


def _colorfulness_hasler_susstrunk(rgb: np.ndarray) -> float:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)

    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)

    mu_rg, sigma_rg = rg.mean(), rg.std()
    mu_yb, sigma_yb = yb.mean(), yb.std()

    return float(np.sqrt(sigma_rg**2 + sigma_yb**2) + 0.3 * np.sqrt(mu_rg**2 + mu_yb**2))


def _color_variation_rgb(rgb: np.ndarray) -> float:
    x = rgb.astype(np.float32) / 255.0
    return float(x.reshape(-1, 3).std(axis=0).mean())


def build_images_metadata_df(
    images_dir: str | Path,
    labels_dir: str | Path,
    img_exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
) -> pd.DataFrame:

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

        # -------------------------
        # has_labels
        # -------------------------
        has_labels = False
        if label_exists:
            try:
                lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                has_labels = any(len(ln.strip()) > 0 for ln in lines)
            except Exception:
                has_labels = False

        have_to_remove = not label_exists

        # -------------------------
        # Ler imagem com proteção
        # -------------------------
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                w, h = im.size
                rgb = np.array(im)

            aspect_ratio = (w / h) if h else np.nan
            num_pixels = w * h
            colorfulness = _colorfulness_hasler_susstrunk(rgb)
            color_variation = _color_variation_rgb(rgb)

            # -------------------------
            # Estatísticas RGB
            # -------------------------
            r = rgb[..., 0].astype(np.float32)
            g = rgb[..., 1].astype(np.float32)
            b = rgb[..., 2].astype(np.float32)

            mean_r, mean_g, mean_b = r.mean(), g.mean(), b.mean()
            std_r, std_g, std_b = r.std(), g.std(), b.std()

        except Exception:
            # imagem inválida/corrompida
            w = h = num_pixels = np.nan
            aspect_ratio = np.nan
            colorfulness = np.nan
            color_variation = np.nan
            mean_r = mean_g = mean_b = np.nan
            std_r = std_g = std_b = np.nan
            have_to_remove = True

        rows.append(
            {
                "filename": img_path.name,
                "path": str(img_path),
                "width": w,
                "height": h,
                "num_pixels": num_pixels,
                "width_height": f"{int(w)}x{int(h)}" if pd.notna(w) else None,
                "aspect_ratio": aspect_ratio,
                "colorfulness": colorfulness,
                "color_variation": color_variation,
                "mean_r": mean_r,
                "mean_g": mean_g,
                "mean_b": mean_b,
                "std_r": std_r,
                "std_g": std_g,
                "std_b": std_b,
                "has_labels": bool(has_labels),
                "have_to_remove": bool(have_to_remove),
            }
        )

    df = (
        pd.DataFrame(rows)
        .sort_values(["have_to_remove", "has_labels", "filename"])
        .reset_index(drop=True)
    )

    return df
