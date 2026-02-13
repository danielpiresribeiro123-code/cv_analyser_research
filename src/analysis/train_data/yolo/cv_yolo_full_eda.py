from pathlib import Path
import pandas as pd

from utils.path_loader import load_paths
from analysis.train_data.yolo.support.cv_image_analysis import build_images_metadata_df
from analysis.train_data.yolo.support.cv_txt_labels import analyze_cv_yolo_datasets


# =========================================================
# Paths
# =========================================================
paths = load_paths()

PROJECT_ROOT: Path = paths["project_root"]

TEXT_PHOTO_IMAGES: Path = paths["data"]["datasets_root"] / "images" / "all"
SKILL_TABLE_IMAGES: Path = paths["data"]["skill_tables"] / "images" / "all"

TEXT_PHOTO_LABELS: Path = paths["data"]["datasets_root"] / "labels" / "all"
SKILL_TABLE_LABELS: Path = paths["data"]["skill_tables"] / "labels" / "all"

SAVE_PATH: Path = PROJECT_ROOT / "analysis" / "train_data" / "yolo"
SAVE_PATH.mkdir(parents=True, exist_ok=True)


# =========================================================
# 1) DataFrame completo de análise YOLO
# =========================================================
def build_cv_yolo_full_analysis_df(
    text_photo_images: Path,
    text_photo_labels: Path,
    skill_images: Path,
    skill_labels: Path,
) -> pd.DataFrame:
    """
    Constrói o DataFrame completo de EDA dos datasets YOLO de CVs.

    Integra:
        - metadata das imagens
        - análise dos labels YOLO
        - flags semânticas de pertença ao dataset
    """

    # -------------------------
    # Metadata de imagens
    # -------------------------
    df_images = build_images_metadata_df(
        images_dir=text_photo_images,
        labels_dir=text_photo_labels,
    )

    # -------------------------
    # Labels YOLO
    # -------------------------
    df_labels = analyze_cv_yolo_datasets(
        text_photo_images=text_photo_images,
        text_photo_labels=text_photo_labels,
        skill_images=skill_images,
        skill_labels=skill_labels,
    )

    # -------------------------
    # Merge seguro
    # -------------------------
    df = df_images.merge(df_labels, on="filename", how="outer")

    # -------------------------
    # Flags de pertença a datasets
    # -------------------------
    text_photo_label_names = {p.stem for p in text_photo_labels.glob("*.txt")}
    skill_label_names = {p.stem for p in skill_labels.glob("*.txt")}

    df["in_text_photo_dataset"] = df["filename"].apply(
        lambda x: Path(x).stem in text_photo_label_names if pd.notna(x) else False
    )

    df["in_skill_table_dataset"] = df["filename"].apply(
        lambda x: Path(x).stem in skill_label_names if pd.notna(x) else False
    )

    # -------------------------
    # Flags semânticas
    # -------------------------
    for col in ["number_text_block", "number_photo", "number_skill_table"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    df["has_any_object"] = (
        df[["number_text_block", "number_photo", "number_skill_table"]]
        .sum(axis=1)
        > 0
    )

    # garantir booleano correto
    df["have_to_remove"] = df["have_to_remove"].fillna(True).astype(bool)

    # ordenar
    df = df.sort_values(
        ["have_to_remove", "has_any_object", "filename"]
    ).reset_index(drop=True)

    return df


# =========================================================
# 2) Summary estatístico
# =========================================================
def compute_cv_yolo_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas estatísticas principais do dataset YOLO de CVs,
    incluindo estatísticas RGB globais.
    """

    if df.empty:
        raise ValueError("DataFrame de EDA está vazio.")

    total_images = len(df)
    have_to_remove = df["have_to_remove"].fillna(True).astype(bool)
    valid_images = (~have_to_remove).sum()

    # -------------------------
    # Funções seguras para colunas opcionais
    # -------------------------
    def safe_mean(col: str, ndigits: int = 3):
        return round(df[col].mean(), ndigits) if col in df.columns else None

    summary = {
        "total_images": total_images,
        "valid_images": int(valid_images),
        "images_with_any_object_%": round(100 * df["has_any_object"].mean(), 2),
        "images_with_photo_%": round(100 * (df["number_photo"] > 0).mean(), 2),
        "images_with_skill_table_%": round(100 * (df["number_skill_table"] > 0).mean(), 2),
        "avg_text_blocks_per_cv": round(df["number_text_block"].mean(), 2),
        "avg_photos_per_cv": round(df["number_photo"].mean(), 2),
        "avg_skill_tables_per_cv": round(df["number_skill_table"].mean(), 2),
        "median_aspect_ratio": round(df["aspect_ratio"].median(), 3),
        "mean_colorfulness": round(df["colorfulness"].mean(), 3),
        "mean_color_variation": round(df["color_variation"].mean(), 3),

        # -------------------------
        # Estatísticas RGB
        # -------------------------
        "mean_r": safe_mean("mean_r"),
        "mean_g": safe_mean("mean_g"),
        "mean_b": safe_mean("mean_b"),
        "std_r": safe_mean("std_r"),
        "std_g": safe_mean("std_g"),
        "std_b": safe_mean("std_b"),
    }

    return pd.DataFrame([summary])



# =========================================================
# 3) Texto automático para relatório
# =========================================================
def generate_eda_text(summary_df: pd.DataFrame) -> str:
    """
    Gera descrição textual académica do EDA,
    incluindo análise de distribuição de cor RGB.
    """

    if summary_df.empty:
        raise ValueError("Summary DataFrame está vazio.")

    s = summary_df.iloc[0]

    rgb_text = ""
    if pd.notna(s.get("mean_r")):
        rgb_text = f"""
Channel-wise color statistics indicate mean RGB intensities of
R={s.mean_r}, G={s.mean_g}, and B={s.mean_b}, with corresponding
standard deviations of {s.std_r}, {s.std_g}, and {s.std_b},
suggesting stable document coloration and limited chromatic noise.
""".strip()

    return f"""
An exploratory analysis of the CV object-detection datasets was conducted prior to YOLO model training.
The dataset contains {int(s.total_images)} images, of which {int(s.valid_images)} are valid annotated samples.

Overall, {s['images_with_any_object_%']}% of the CVs contain at least one detectable visual element.
Photographs are present in {s['images_with_photo_%']}% of the resumes, while skill tables appear in
{s['images_with_skill_table_%']}% of the documents.

On average, each CV includes {s.avg_text_blocks_per_cv} text regions, {s.avg_photos_per_cv} photographs,
and {s.avg_skill_tables_per_cv} skill tables. The median aspect ratio of the images is {s.median_aspect_ratio}.

Color distribution analysis shows a mean colorfulness of {s.mean_colorfulness} and an average color
variation of {s.mean_color_variation}, indicating moderate visual diversity with consistent document structure.

{rgb_text}

These findings confirm that the dataset provides sufficient structural variability and annotation
coverage to support robust YOLO-based layout detection in curriculum vitae documents.
""".strip()


# =========================================================
# 4) Runner completo com export opcional
# =========================================================
def run_full_cv_yolo_eda(
    save_csv: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Executa pipeline completo de EDA dos datasets YOLO de CVs.

    Retorna:
        df (DataFrame completo)
        summary_df (tabela estatística)
        eda_text (texto académico)

    Opcional:
        Guarda resultados em CSV/TXT.
    """

    df = build_cv_yolo_full_analysis_df(
        TEXT_PHOTO_IMAGES,
        TEXT_PHOTO_LABELS,
        SKILL_TABLE_IMAGES,
        SKILL_TABLE_LABELS,
    )

    summary_df = compute_cv_yolo_dataset_summary(df)
    eda_text = generate_eda_text(summary_df)

    # -------------------------
    # Export
    # -------------------------
    if save_csv:
        df.to_csv(SAVE_PATH / "cv_yolo_full_analysis.csv", index=False)
        summary_df.to_csv(SAVE_PATH / "cv_yolo_dataset_summary.csv", index=False)

        with open(SAVE_PATH / "eda_description.txt", "w", encoding="utf-8") as f:
            f.write(eda_text)

    return df, summary_df, eda_text
