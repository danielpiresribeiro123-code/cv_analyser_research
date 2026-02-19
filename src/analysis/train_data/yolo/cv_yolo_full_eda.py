from pathlib import Path
import pandas as pd

from utils.path_loader import load_paths
from analysis.train_data.yolo.support.cv_image_analysis import build_images_metadata_df
from analysis.train_data.yolo.support.cv_txt_labels import analyze_cv_yolo_datasets


paths = load_paths()

PROJECT_ROOT: Path = paths["project_root"]

TEXT_PHOTO_IMAGES: Path = paths["data"]["datasets_root"] / "images" / "all"
SKILL_TABLE_IMAGES: Path = paths["data"]["skill_tables"] / "images" / "all"

TEXT_PHOTO_LABELS: Path = paths["data"]["datasets_root"] / "labels" / "all"
SKILL_TABLE_LABELS: Path = paths["data"]["skill_tables"] / "labels" / "all"

SAVE_PATH: Path = PROJECT_ROOT / "src" / "analysis" / "train_data" / "yolo"
SAVE_PATH.mkdir(parents=True, exist_ok=True)


def build_cv_yolo_full_analysis_df(
    text_photo_images: Path,
    text_photo_labels: Path,
    skill_images: Path,
    skill_labels: Path,
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

    df = df_images.merge(df_labels, on="filename", how="outer")

    text_photo_label_names = {p.stem for p in text_photo_labels.glob("*.txt")}
    skill_label_names = {p.stem for p in skill_labels.glob("*.txt")}

    df["in_text_photo_dataset"] = df["filename"].apply(
        lambda x: Path(x).stem in text_photo_label_names if pd.notna(x) else False
    )

    df["in_skill_table_dataset"] = df["filename"].apply(
        lambda x: Path(x).stem in skill_label_names if pd.notna(x) else False
    )

    for col in ["number_text_block", "number_photo", "number_skill_table"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    df["has_any_object"] = (
        df[["number_text_block", "number_photo", "number_skill_table"]]
        .sum(axis=1)
        > 0
    )

    df["have_to_remove"] = df["have_to_remove"].fillna(True).astype(bool)

    df = df.sort_values(
        ["have_to_remove", "has_any_object", "filename"]
    ).reset_index(drop=True)

    return df


def compute_cv_yolo_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        raise ValueError("DataFrame de EDA está vazio.")

    total_images = len(df)
    have_to_remove = df["have_to_remove"].fillna(True).astype(bool)
    valid_images = (~have_to_remove).sum()

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

        # RGB
        "mean_red": safe_mean("mean_r"),
        "mean_green": safe_mean("mean_g"),
        "mean_blue": safe_mean("mean_b"),
        "std_red": safe_mean("std_r"),
        "std_green": safe_mean("std_g"),
        "std_blue": safe_mean("std_b"),

        # HSV
        "mean_hue": safe_mean("mean_h"),
        "mean_saturation": safe_mean("mean_s"),
        "mean_value": safe_mean("mean_v"),
        "std_hue": safe_mean("std_h"),
        "std_saturation": safe_mean("std_s"),
        "std_value": safe_mean("std_v"),
    }

    return pd.DataFrame([summary])


def generate_eda_text(summary_df: pd.DataFrame) -> str:

    if summary_df.empty:
        raise ValueError("Summary DataFrame está vazio.")

    s = summary_df.iloc[0]

    rgb_text = ""
    if pd.notna(s.get("mean_r")):
        rgb_text = f"""
Channel-wise RGB statistics show mean intensities of R={s.mean_r}, G={s.mean_g}, and B={s.mean_b},
with corresponding standard deviations of {s.std_r}, {s.std_g}, and {s.std_b}, indicating
stable document coloration and limited chromatic dispersion.
""".strip()

    hsv_text = ""
    if pd.notna(s.get("mean_h")):
        hsv_text = f"""
In the HSV color space, the dataset presents mean values of H={s.mean_h}, S={s.mean_s}, and V={s.mean_v},
with standard deviations of {s.std_h}, {s.std_s}, and {s.std_v}. These statistics suggest
predominantly low-saturation, high-brightness documents consistent with typical monochromatic CV layouts,
while still preserving moderate hue variability introduced by graphical elements and photographs.
""".strip()

    return f"""
An exploratory analysis of the CV object-detection datasets was conducted prior to YOLO model training.
The dataset contains {int(s.total_images)} images, of which {int(s.valid_images)} are valid annotated samples.

Overall, {s['images_with_any_object_%']}% of the CVs contain at least one detectable visual element.
Photographs are present in {s['images_with_photo_%']}% of the resumes, while skill tables appear in
{s['images_with_skill_table_%']}% of the documents.

On average, each CV includes {s.avg_text_blocks_per_cv} text regions, {s.avg_photos_per_cv} photographs,
and {s.avg_skill_tables_per_cv} skill tables. The median aspect ratio of the images is {s.median_aspect_ratio}.

Color analysis indicates a mean colorfulness of {s.mean_colorfulness} and an average color variation of
{s.mean_color_variation}, reflecting moderate visual diversity with consistent structural layout.

{rgb_text}

{hsv_text}

These findings confirm that the dataset provides sufficient structural variability and annotation
coverage to support robust YOLO-based layout detection in curriculum vitae documents.
""".strip()


def run_full_cv_yolo_eda(save_csv: bool = True):

    df = build_cv_yolo_full_analysis_df(
        TEXT_PHOTO_IMAGES,
        TEXT_PHOTO_LABELS,
        SKILL_TABLE_IMAGES,
        SKILL_TABLE_LABELS,
    )

    summary_df = compute_cv_yolo_dataset_summary(df)
    eda_text = generate_eda_text(summary_df)

    if save_csv:
        df.to_csv(SAVE_PATH / "cv_yolo_full_analysis.csv", index=False)
        summary_df.to_csv(SAVE_PATH / "cv_yolo_dataset_summary.csv", index=False)

        with open(SAVE_PATH / "eda_description.txt", "w", encoding="utf-8") as f:
            f.write(eda_text)

    return df, summary_df, eda_text



if __name__ == "__main__":
    run_full_cv_yolo_eda(save_csv =True)