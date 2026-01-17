from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from data_features.features_extractors import (
    extrair_features_qualidade_path,
    extrair_features_clustering_path,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Features que PODEM ser achatadas para CSV
FLATTEN_ALLOW = {
    # clustering
    "c_color_hist_hsv",
    "c_texture_lbp",
    "c_geometry_hu",
    "c_lightness_grid",
    "c_orb_desc_mean",
    "c_orb_desc_std",
}

# Features que devem ser RESUMIDAS (não achatadas)
REDUCE_STATS = {
    "c_geometry_edges",
}


def _flatten_features(prefix: str, feats: dict) -> dict:
    out = {}
    for k, v in feats.items():
        if v is None:
            continue

        key = f"{prefix}{k}" if prefix else k

        if isinstance(v, np.ndarray):
            if key in REDUCE_STATS:
                out[f"{key}_mean"] = float(np.mean(v))
                out[f"{key}_std"] = float(np.std(v))
                out[f"{key}_energy"] = float(np.sum(v ** 2))
                continue  

            if key not in FLATTEN_ALLOW:
                continue  

            v = v.flatten()
            for i, val in enumerate(v):
                out[f"{key}_{i:03d}"] = float(val)

        elif isinstance(v, (list, tuple)):
            for i, val in enumerate(v):
                if isinstance(val, (int, float, np.floating, np.integer)):
                    out[f"{key}_{i:03d}"] = float(val)

        elif isinstance(v, (np.floating, np.integer)):
            out[key] = float(v)

        elif isinstance(v, (int, float)):
            out[key] = v

        else:
            out[key] = v

    return out


def build_dataset(
    images_dir: str | Path,
    output_csv: str | Path | None = None,
    with_label: bool = True,
    class_names: list[str] | None = None,
    feature_set: str = "both",  # "quality" | "clustering" | "both"
    return_df: bool = True,
):
    """
    Modos de funcionamento:

    1) Com classes explícitas:
        build_dataset(..., with_label=True, class_names=["Good", "Noisy", "Blur"])

       Espera:
           images_dir/
               Good/
               Noisy/
               Blur/

    2) Labels automáticas pelo nome da pasta:
        build_dataset(..., with_label=True, class_names=None)

       A label de cada imagem será o nome da pasta onde ela está.

    3) Sem labels:
        build_dataset(..., with_label=False)
    """

    images_dir = Path(images_dir)

    if feature_set not in {"quality", "clustering", "both"}:
        raise ValueError("feature_set tem de ser 'quality', 'clustering' ou 'both'")

    rows = []

    def process_one(img_path: Path, label: str | None):
        feats = {}

        if feature_set in {"quality", "both"}:
            q = extrair_features_qualidade_path(img_path)
            if q is None:
                return
            feats.update(_flatten_features("q_", q))

        if feature_set in {"clustering", "both"}:
            c = extrair_features_clustering_path(img_path)
            if c is None:
                return
            feats.update(_flatten_features("c_", c))

        row = {"image_path": str(img_path)}
        if label is not None:
            row["label"] = label
        row.update(feats)
        rows.append(row)

    # -------------------------------------------------
    # COM LABELS
    # -------------------------------------------------
    if with_label:

        # Caso 1: classes fornecidas explicitamente
        if class_names is not None:
            for class_name in class_names:
                class_path = images_dir / class_name

                if not class_path.exists():
                    raise ValueError(f"Pasta não encontrada: {class_path}")

                for img_path in tqdm(
                    class_path.rglob("*"),
                    desc=f"Processando {class_name}"
                ):
                    if not img_path.is_file():
                        continue
                    if img_path.suffix.lower() not in IMG_EXTS:
                        continue

                    process_one(img_path, class_name)

        # Caso 2: labels automáticas pelo nome da pasta
        else:
            for img_path in tqdm(images_dir.rglob("*"), desc="Processando imagens"):
                if not img_path.is_file():
                    continue
                if img_path.suffix.lower() not in IMG_EXTS:
                    continue

                # label = nome da pasta onde a imagem está
                label = img_path.parent.name
                process_one(img_path, label)

    # -------------------------------------------------
    # SEM LABELS
    # -------------------------------------------------
    else:
        for img_path in tqdm(images_dir.rglob("*"), desc="Processando imagens"):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in IMG_EXTS:
                continue

            process_one(img_path, None)

    df = pd.DataFrame(rows)

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    else:
        print("Aviso: output_csv não foi fornecido, apenas devolvo o DataFrame.")

    if return_df:
        return df

    return None



"""
Exemplos de uso:
Classes explícitas:
build_dataset(
    "data/images",
    "dataset.csv",
    with_label=True,
    class_names=["Good", "Noisy", "Defocused Blur"]
)


Labels automáticas:
build_dataset(
    "data/images",
    "dataset.csv",
    with_label=True,
    class_names=None
)


Sem labels:
build_dataset(
    "data/images",
    "dataset.csv",
    with_label=False
)

"""