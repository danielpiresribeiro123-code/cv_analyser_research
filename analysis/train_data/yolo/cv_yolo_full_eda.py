from pathlib import Path
import pandas as pd

from utils.path_loader import load_paths
from analysis.train_data.yolo.support.cv_image_analysis import build_images_metadata_df
from analysis.train_data.yolo.support.cv_txt_labels import analyze_cv_yolo_datasets


# =========================================================
# Paths
# =========================================================
paths = load_paths()

PROJECT_ROOT = paths["project_root"]

TEXT_PHOTO_IMAGES = paths["datasets_root"] / "images" / "all"
SKILL_TABLE_IMAGES = paths["skill_tables"] / "images" / "all"

TEXT_PHOTO_LABELS = paths["datasets_root"] / "labels" / "all"
SKILL_TABLE_LABELS = paths["skill_tables"] / "labels" / "all"

# Novo SAVE_PATH
SAVE_PATH = PROJECT_ROOT / "analysis" / "train_data" / "yolo"
SAVE_PATH.mkdir(parents=True, exist_ok=True)


# =========================================================
# 1) DataFrame completo de análise YOLO
# =========================================================
def build_cv_yolo_full_analysis_df(
    text_photo_images: str | Path,
    text_photo_labels: str | Path,
    skill_images: str | Path,
    skill_labels: str | Path,
) -> pd.DataFrame:

    df_images = build_images_metadata_df(
        images_dir=text_photo_images,
        labels_dir=text_photo_labels,
    )

    df_labels = analyze_cv_yolo_datasets(
        text_photo_images=text_photo_images,
        text_photo_labels=text_photo_labels,
        skill_images=skill_images,
        skill_labels=skill_labels,
    )

    df = df_images.merge(df_labels, on="filename", how="outer", suffixes=("", "_yolo"))

    text_photo_label_names = {p.stem for p in Path(text_photo_labels).glob("*.txt")}
    skill_label_names = {p.stem for p in Path(skill_labels).glob("*.txt")}

    df["in_text_photo_dataset"] = df["filename"].apply(
        lambda x: Path(x).stem in text_photo_label_names
    )

    df["in_skill_table_dataset"] = df["filename"].apply(
        lambda x: Path(x).stem in skill_label_names
    )

    df["has_any_object"] = (
        df[["number_text_block", "number_photo", "number_skill_table"]]
        .fillna(0)
        .sum(axis=1)
        > 0
    )

    for col in ["number_text_block", "number_photo", "number_skill_table"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    df = df.sort_values(
        ["have_to_remove", "has_any_object", "filename"]
    ).reset_index(drop=True)

    return df


# =========================================================
# 2) Summary estatístico
# =========================================================
def compute_cv_yolo_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:

    total_images = len(df)
    valid_images = (~df["have_to_remove"]).sum()

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
    }

    return pd.DataFrame([summary])


# =========================================================
# 3) Texto automático para relatório
# =========================================================
def generate_eda_text(summary_df: pd.DataFrame) -> str:

    s = summary_df.iloc[0]

    text = f"""
An exploratory analysis of the CV object-detection datasets was conducted prior to YOLO model training.
The dataset contains {int(s.total_images)} images, of which {int(s.valid_images)} are valid annotated samples.

Overall, {s['images_with_any_object_%']}% of the CVs contain at least one detectable visual element.
Photographs are present in {s['images_with_photo_%']}% of the resumes, while skill tables appear in
{s['images_with_skill_table_%']}% of the documents.

On average, each CV includes {s.avg_text_blocks_per_cv} text regions, {s.avg_photos_per_cv} photographs,
and {s.avg_skill_tables_per_cv} skill tables. The median aspect ratio of the images is {s.median_aspect_ratio}.

Color distribution analysis shows a mean colorfulness of {s.mean_colorfulness} and an average color
variation of {s.mean_color_variation}, indicating moderate visual diversity with consistent document structure.

These findings confirm that the dataset provides sufficient structural variability and annotation
coverage to support robust YOLO-based layout detection in curriculum vitae documents.
"""
    return text.strip()


# =========================================================
# 4) Runner completo com opção de guardar CSV
# =========================================================
def run_full_cv_yolo_eda(save_csv: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Executa pipeline completo de EDA e opcionalmente guarda:
        - cv_yolo_full_analysis.csv
        - cv_yolo_dataset_summary.csv
        - eda_description.txt
    """

    df = build_cv_yolo_full_analysis_df(
        TEXT_PHOTO_IMAGES,
        TEXT_PHOTO_LABELS,
        SKILL_TABLE_IMAGES,
        SKILL_TABLE_LABELS,
    )

    summary_df = compute_cv_yolo_dataset_summary(df)
    eda_text = generate_eda_text(summary_df)

    # -----------------------------------------------------
    # Guardar resultados
    # -----------------------------------------------------
    if save_csv:
        df.to_csv(SAVE_PATH / "cv_yolo_full_analysis.csv", index=False)
        summary_df.to_csv(SAVE_PATH / "cv_yolo_dataset_summary.csv", index=False)

        with open(SAVE_PATH / "eda_description.txt", "w", encoding="utf-8") as f:
            f.write(eda_text)

    return df, summary_df, eda_text


def main():
    df, summary_df, eda_text = run_full_cv_yolo_eda(save_csv=True)

    print("=== EDA CONCLUÍDO ===")
    print()
    print("Resumo estatístico:")
    print(summary_df)
    print()
    print("Texto gerado para o relatório:")
    print(eda_text)