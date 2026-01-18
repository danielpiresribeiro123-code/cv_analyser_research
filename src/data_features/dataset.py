import cv2
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from data_features.features_extractors import (extrair_features_common,extrair_features_quality_only,
    extrair_features_clustering_only,extrair_features_qualidade,extrair_features_clustering)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Features que podem ser achatadas para CSV
FLATTEN_ALLOW = {
    # common
    "c_color_hist_hsv",
    "c_texture_lbp",
    "c_geometry_hu",
    "c_lightness_grid",

    # clustering
    "cl_orb_desc_mean",
    "cl_orb_desc_std"}

# Features que devem ser resumidas (não achatadas)
REDUCE_STATS = {"c_geometry_edges"}


def _flatten_features(prefix: str, feats: dict) -> dict:
    out = {}
    for k in sorted(feats.keys()):
        v = feats[k]

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
            # usado apenas para vetores muito pequenos
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
    feature_mode: str = "quality",  
    # "common" | "quality_only" | "clustering_only" |
    # "quality" | "clustering" | "all"
    return_df: bool = True,
):
    """
    Prefixos das features no CSV:
        c_  -> common features
        q_  -> quality_only features
        cl_ -> clustering_only features

    Modos:
        common           -> só common
        quality_only     -> só quality_only
        clustering_only  -> só clustering_only
        quality          -> common + quality_only
        clustering       -> common + clustering_only
        all              -> common + quality_only + clustering_only
    """


    images_dir = Path(images_dir)

    valid_modes = {
        "common",
        "quality_only",
        "clustering_only",
        "quality",
        "clustering",
        "all"
    }
    if feature_mode not in valid_modes:
        raise ValueError(f"feature_mode tem de ser um de {valid_modes}")

    rows = []

    def process_one(img_path: Path, label: str | None):
        feats = {}

        image = cv2.imread(str(img_path))
        if image is None:
            return

        # ---------------------------
        # COMMON
        # ---------------------------
        if feature_mode in {"common", "quality", "clustering", "all"}:
            try:
                c = extrair_features_common(image)
                feats.update(_flatten_features("", c))
            except Exception as e:
                print(f"[ERRO] {img_path} - common features: {e}")

        # ---------------------------
        # QUALITY ONLY
        # ---------------------------
        if feature_mode in {"quality_only", "quality", "all"}:
            try:
                q = extrair_features_quality_only(image)
                feats.update(_flatten_features("", q))
            except Exception as e:
                print(f"[ERRO] {img_path} - quality_only features: {e}")

        # ---------------------------
        # CLUSTERING ONLY
        # ---------------------------
        if feature_mode in {"clustering_only", "clustering", "all"}:
            try:
                cl = extrair_features_clustering_only(image)
                feats.update(_flatten_features("", cl))
            except Exception as e:
                print(f"[ERRO] {img_path} - clustering_only features: {e}")

        if not feats:
            print(f"[AVISO] Nenhuma feature extraída para {img_path}")
            return

        row = {"image_path": str(img_path)}
        if label is not None:
            row["label"] = label

        row.update(feats)
        rows.append(row)


    # -------------------------------------------------
    # COM LABELS
    # -------------------------------------------------
    if with_label:
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

        else:
            for img_path in tqdm(images_dir.rglob("*"), desc="Processando imagens"):
                if not img_path.is_file():
                    continue
                if img_path.suffix.lower() not in IMG_EXTS:
                    continue

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
