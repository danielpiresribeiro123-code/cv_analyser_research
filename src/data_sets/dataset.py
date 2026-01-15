from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from data_extraction.features import (
    extract_quality_features_from_path,
    extract_clustering_features_from_path,
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

    # quality (escalares já vêm bem)
}

# Features que devem ser RESUMIDAS (não achatadas)
REDUCE_STATS = {
    "c_geometry_edges",
}



def _flatten_features(prefix: str, feats: dict) -> dict:
    """
    Converte features que sejam np.ndarray/list em colunas escalares:
    - { "color_hist_hsv": array([..]) } -> { "color_hist_hsv_000": ..., ... }
    Mantém escalares como estão.
    Ignora None.
    """
    out = {}
    for k, v in feats.items():
        if v is None:
            continue

        key = f"{prefix}{k}" if prefix else k

        if isinstance(v, np.ndarray):
            # Caso 1: resumir (ex: geometry_edges)
            if key in REDUCE_STATS:
                out[f"{key}_mean"] = float(np.mean(v))
                out[f"{key}_std"] = float(np.std(v))
                out[f"{key}_energy"] = float(np.sum(v ** 2))
                continue  

            # Caso 2: achatar apenas se permitido
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
    feature_set: str = "both",  # "quality" | "clustering" | "both"
    return_df: bool = True,
):
    """
    Cria um dataset de features a partir de um diretório de imagens.

    Estrutura esperada (with_label=True):

        images_dir/
            Defocused Blur/
                (pode ter imagens diretamente ou subpastas)
            Good/
                kwentar-sharp/
                personal-sharp/
                ...
            Motion Blur/
            Noisy/
            Pixelated/
            Very Noisy/

    As labels são strings:
        "Defocused Blur", "Good", "Motion Blur", "Noisy", "Pixelated", "Very Noisy"

    A função percorre recursivamente todas as subpastas de cada classe.
    """

    images_dir = Path(images_dir)

    if feature_set not in {"quality", "clustering", "both"}:
        raise ValueError("feature_set tem de ser 'quality', 'clustering' ou 'both'")

    rows = []

    def process_one(img_path: Path, label: str | None):
        feats = {}

        if feature_set in {"quality", "both"}:
            q = extract_quality_features_from_path(img_path)
            if q is None:
                return
            feats.update(_flatten_features("q_", q))

        if feature_set in {"clustering", "both"}:
            c = extract_clustering_features_from_path(img_path)
            if c is None:
                return
            feats.update(_flatten_features("c_", c))

        row = {"image_path": str(img_path)}
        if label is not None:
            row["label"] = label
        row.update(feats)
        rows.append(row)

    if with_label:

        class_names = [
            "Defocused Blur",
            "Good",
            "Motion Blur",
            "Noisy",
            "Pixelated",
            "Very Noisy",
        ]

        for class_name in class_names:
            class_path = images_dir / class_name

            if not class_path.exists():
                raise ValueError(f"Pasta não encontrada: {class_path}")

            # Percurso recursivo: aceita imagens em qualquer subpasta
            for img_path in tqdm(class_path.rglob("*"), desc=f"Processando {class_name}"):
                if not img_path.is_file():
                    continue
                if img_path.suffix.lower() not in IMG_EXTS:
                    continue

                process_one(img_path, class_name)

    else:
        # Sem labels: percorre todo o diretório recursivamente
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
        print("Esqueceste-te da variável output_csv na função!")

    if return_df:
        return df

    return None
