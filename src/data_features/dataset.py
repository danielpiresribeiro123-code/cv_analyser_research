import cv2
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from data_features.features_extractors import (
    extrair_features_common,
    extrair_features_quality_only,
    extrair_features_clustering_only,
    extrair_features_quality,
    extrair_features_clustering,
    extrair_features_total
)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# Features que podem ser achatadas diretamente
FLATTEN_ALLOW = {
    "c_color_hist_hsv",
    "c_texture_lbp",
    "c_geometry_hu",
    "c_lightness_grid",
    "cl_orb_desc_mean",
    "cl_orb_desc_std",
}

# Features que devem ser resumidas
REDUCE_STATS = {"c_geometry_edges"}

# Funções extras:
def _flatten_features(feats: dict) -> dict:
    """
    Converte um dicionário de features (escalares, arrays, vetores)
    num formato plano pronto para CSV.
    """
    out = {}

    for key, v in feats.items():

        # Arrays grandes
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

        # Vetores pequenos
        elif isinstance(v, (list, tuple)):
            for i, val in enumerate(v):
                if isinstance(val, (int, float, np.floating, np.integer)):
                    out[f"{key}_{i:03d}"] = float(val)

        # Escalares numpy
        elif isinstance(v, (np.floating, np.integer)):
            out[key] = float(v)

        # Escalares normais
        elif isinstance(v, (int, float)):
            out[key] = float(v)

        # Qualquer outra coisa
        else:
            out[key] = v

    return out


def extract_hierarchical_labels(img_path: Path, root_dir: Path, num_labels: int):
    """
    Extrai labels hierárquicas a partir do caminho da imagem.

    Ex:
    root_dir = .../Random_images
    img_path = .../Random_images/Good/personal/camera/img.jpg
    num_labels = 3

    → ["Random_images", "Good", "personal"]
    """
    rel = img_path.relative_to(root_dir)
    parts = rel.parts[:-1]  # remove o nome do ficheiro

    labels = []
    for i in range(num_labels):
        if i < len(parts):
            labels.append(parts[i])
        else:
            labels.append(None)

    return labels

def short_image_path(img_path: Path, depth: int):
    """
    Reduz o image_path para apenas os últimos `depth` diretórios + nome do ficheiro.

    Ex:
    img_path = C:/.../VSCODE/Data/Random_images/Good/img.jpg
    depth = 2  -> Data/Random_images/img.jpg
    depth = 3  -> VSCODE/Data/Random_images/img.jpg
    """
    parts = img_path.parts
    if depth <= 0:
        return img_path.name

    # evita depth maior do que o caminho disponível
    depth = min(depth, len(parts) - 1)
    return str(Path(*parts[-(depth + 1):]))


def build_dataset(
    images_dir: str | Path,
    output_csv: str | Path | None = None,
    grey_dir: str | Path | None = None,   # None = gray é criada em memória e não é guardada
    num_labels: int = 0, # 0 = sem labels, >0 = labels hierárquicas
    path_depth: int | None = None,
    feature_mode: str = "quality",
    version: int = 1,
    blur_grid_size: int = 4,
    common_color_light_features: bool = True,
    log_sigma: float = 1.0,
    log_quantile: float = 0.999,
    block: int = 32,
    return_df: bool = True,
):
    """
    Constrói um dataset de imagens em CSV com extração automática de features e,
    opcionalmente, criação de labels hierárquicas baseadas na estrutura das pastas.

    Se grey_dir for fornecido:
        - procura primeiro a versão grey da imagem em grey_dir
        - se existir, usa-a
        - se não existir, cria-a e guarda para futuras execuções

    Se grey_dir=None:
        - a imagem grey é criada apenas em memória (comportamento atual)
    """

    if num_labels < 0:
        raise ValueError("num_labels tem de ser >= 0")
    
    if path_depth is not None and path_depth < 0:
        raise ValueError("path_depth tem de ser >= 0 ou None")

    images_dir = Path(images_dir)

    if not images_dir.exists():
        raise ValueError(f"images_dir não existe: {images_dir}")

    if grey_dir is not None:
        grey_dir = Path(grey_dir)
        grey_dir.mkdir(parents=True, exist_ok=True)
    
    if grey_dir is not None and grey_dir.resolve().is_relative_to(images_dir.resolve()):
        raise ValueError("grey_dir não pode estar dentro de images_dir.")

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

    def get_gray_image(img_path: Path, image_bgr):
        """
        Obtém a versão grayscale da imagem:
        - Se grey_dir existir, usa cache em disco
        - Caso contrário, cria apenas em memória
        """
        if grey_dir is None:
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # caminho relativo dentro do dataset
        rel = img_path.relative_to(images_dir)
        grey_subdir = grey_dir / rel.parent
        grey_subdir.mkdir(parents=True, exist_ok=True)

        grey_name = img_path.stem + "_grey.png"
        grey_path = grey_subdir / grey_name

        if grey_path.exists():
            gray = cv2.imread(str(grey_path), cv2.IMREAD_GRAYSCALE)
            if gray is not None:
                return gray
            else:
                print(f"[AVISO] Grey corrompida, recriando: {grey_path}")

        # criar nova
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(grey_path), gray)
        return gray

    def process_one(img_path: Path):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[ERRO] Não foi possível ler {img_path}")
            return

        gray = get_gray_image(img_path, image)

        try:
            if feature_mode == "common":
                feats = extrair_features_common(
                    image, gray,
                    blur_grid_size,
                    common_color_light_features,
                    version
                )

            elif feature_mode == "quality_only":
                feats = extrair_features_quality_only(
                    image, gray,
                    log_sigma, log_quantile, block
                )

            elif feature_mode == "clustering_only":
                feats = extrair_features_clustering_only(
                    image, gray,
                    common_color_light_features
                )

            elif feature_mode == "quality":
                feats = extrair_features_quality(
                    image, gray,
                    log_sigma, log_quantile, block,
                    blur_grid_size,
                    common_color_light_features,
                    version
                )

            elif feature_mode == "clustering":
                feats = extrair_features_clustering(
                    image, gray,
                    blur_grid_size,
                    common_color_light_features,
                    version
                )

            elif feature_mode == "all":
                feats = extrair_features_total(
                    image, gray,
                    log_sigma, log_quantile, block,
                    blur_grid_size,
                    common_color_light_features,
                    version
                )

        except Exception as e:
            print(f"[ERRO] {img_path}: {e}")
            return

        flat = _flatten_features(feats) # type: ignore

        if path_depth is None:
            image_path = str(img_path)
        else:
            image_path = short_image_path(img_path, path_depth)

        # Normalizar separadores para /
        image_path = image_path.replace("\\", "/")

        row = {"image_path": image_path}


        # Labels hierárquicas
        if num_labels > 0:
            labels = extract_hierarchical_labels(img_path, images_dir, num_labels)
            for i, lab in enumerate(labels):
                row[f"label_{i}"] = lab

        row.update(flat)
        rows.append(row)

    # -------------------------------------------------
    # Percorrer imagens
    # -------------------------------------------------
    for img_path in tqdm(images_dir.rglob("*"), desc="Processando imagens"):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        process_one(img_path)

    if not rows:
        raise RuntimeError("Nenhuma imagem válida foi encontrada em images_dir.")

    df = pd.DataFrame(rows)

    # image_path sempre primeiro (se existir)
    base_cols = []
    if "image_path" in df.columns:
        base_cols.append("image_path")

    # Depois os labels (label_0, label_1, ...)
    label_cols = sorted([c for c in df.columns if c.startswith("label_")])

    # Depois tudo o resto por ordem alfabética
    other_cols = sorted([
        c for c in df.columns
        if c not in base_cols and not c.startswith("label_")
    ])

    # Ordem final
    df = df[base_cols + label_cols + other_cols]


    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False, encoding="utf-8")

    else:
        print("Aviso: output_csv não foi fornecido, apenas devolvo o DataFrame.")

    if return_df:
        return df

    return None
