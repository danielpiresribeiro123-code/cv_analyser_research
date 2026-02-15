import os
import cv2
import pandas as pd
from pathlib import Path
from collections import Counter
from paddleocr import PaddleOCR
from utils.path_loader import load_paths


# =========================================================
# Windows PaddleOCR path + PIR fix (important for stability)
# =========================================================
safe_dir = r"C:\\paddle_cache"
os.environ["USERPROFILE"] = safe_dir
os.environ["HOME"] = safe_dir
os.environ["FLAGS_enable_pir_api"] = "0"


# =========================================================
# Paths
# =========================================================
paths = load_paths()

PROJECT_ROOT = paths["project_root"]

LABELS_DIR = paths["data"]["datasets_root"] / "labels" / "train"
IMAGES_DIR = paths["data"]["datasets_root"] / "images" / "train"

OUTPUT_DIR = PROJECT_ROOT / "cropped_yolo"
TEXT_DIR = OUTPUT_DIR / "text_block"
PHOTO_DIR = OUTPUT_DIR / "photo"

TEXT_DIR.mkdir(parents=True, exist_ok=True)
PHOTO_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Supported image extensions
# =========================================================
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")


# =========================================================
# Helper: YOLO bbox → pixel coords
# =========================================================
def yolo_to_xyxy(bbox, img_w, img_h):
    """
    bbox: (x_center, y_center, width, height) normalized [0,1]
    returns: (x1, y1, x2, y2) in pixels
    """
    x_c, y_c, w, h = bbox

    x1 = int((x_c - w / 2) * img_w)
    y1 = int((y_c - h / 2) * img_h)
    x2 = int((x_c + w / 2) * img_w)
    y2 = int((y_c + h / 2) * img_h)

    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


# =========================================================
# Main cropping function
# =========================================================
def crop_yolo_dataset(images_dir: Path, labels_dir: Path, output_dir: Path):
    """
    Crops YOLO bounding boxes into class folders:
        0 → text_block
        1 → photo

    Rules:
    - Skip if image has no matching txt
    - Skip if txt has no matching image
    - Skip duplicated names
    - Accept multiple image extensions
    """

    image_files = []
    for ext in IMAGE_EXTS:
        image_files.extend(images_dir.glob(f"*{ext}"))
        image_files.extend(images_dir.glob(f"*{ext.upper()}"))

    label_files = list(labels_dir.glob("*.txt"))

    image_stems = [p.stem for p in image_files]
    label_stems = [p.stem for p in label_files]

    dup_images = {k for k, v in Counter(image_stems).items() if v > 1}
    dup_labels = {k for k, v in Counter(label_stems).items() if v > 1}

    valid_stems = set(image_stems) & set(label_stems)
    valid_stems -= dup_images
    valid_stems -= dup_labels

    print(f"Total valid pairs: {len(valid_stems)}")

    for stem in valid_stems:
        img_candidates = []
        for ext in IMAGE_EXTS:
            img_candidates.extend(images_dir.glob(stem + ext))
            img_candidates.extend(images_dir.glob(stem + ext.upper()))

        if len(img_candidates) != 1:
            continue

        img_path = img_candidates[0]
        txt_path = labels_dir / f"{stem}.txt"

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        h, w = image.shape[:2]

        with open(txt_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 5:
                continue

            cls = int(parts[0])
            bbox = list(map(float, parts[1:]))

            x1, y1, x2, y2 = yolo_to_xyxy(bbox, w, h)
            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            if cls == 0:
                save_dir = TEXT_DIR
            elif cls == 1:
                save_dir = PHOTO_DIR
            else:
                continue

            save_path = save_dir / f"{stem}_{i}.jpg"
            cv2.imwrite(str(save_path), crop)

    print("Cropping finished.")


# =========================================================
# PaddleOCR initialization (PP‑OCRv3 auto‑downloaded)
# =========================================================
ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    enable_mkldnn=False,
    show_log=False,
)


# =========================================================
# OCR dataset generation for text blocks
# =========================================================
def build_ocr_text_dataset(cropped_root: Path, output_root: Path):
    """
    Runs PaddleOCR over cropped text_block images organized by class folders
    and saves a CSV per class with columns: image_path, text, label.
    """

    output_root.mkdir(parents=True, exist_ok=True)

    class_dirs = [p for p in cropped_root.iterdir() if p.is_dir()]

    for class_dir in class_dirs:
        label = class_dir.name
        rows = []

        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue

            result = ocr_engine.ocr(str(img_path), cls=True)

            if not result or not result[0]:
                continue

            texts = [line[1][0] for line in result[0] if line[1][0].strip()]
            text = " ".join(texts).strip()

            if not text:
                continue

            rows.append(
                {
                    "image_path": str(img_path),
                    "text": text,
                    "label": label,
                }
            )

        if not rows:
            continue

        df = pd.DataFrame(rows)
        save_path = output_root / f"{label}.csv"
        df.to_csv(save_path, index=False)

        print(f"Saved OCR dataset for class '{label}' → {save_path}")


# =========================================================
# Run full pipeline
# =========================================================
CROPPED_TEXTBLOCK_ROOT = paths["data"]["datasets_root"] / "cropped_yolo" / "text_block"
OCR_OUTPUT_ROOT = paths["data"]["datasets_root"] / "cropped_yolo" / "ocr_text_block"


# =========================================================
# OCR RAW IMAGES → TXT FILES
# =========================================================
def ocr_images_to_txt(images_dir: Path, text_dir: Path):
    """
    Runs PaddleOCR over all images in IMAGES_DIR
    and saves extracted text into TEXT_DIR as .txt files.
    """

    text_dir.mkdir(parents=True, exist_ok=True)

    image_files = [
        p for p in images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    ]

    print(f"Images found for OCR: {len(image_files)}")

    saved = 0
    skipped = 0

    for img_path in image_files:

        result = ocr_engine.ocr(str(img_path), cls=True)

        if not result or not result[0]:
            skipped += 1
            continue

        texts = [line[1][0] for line in result[0] if line[1][0].strip()]
        text = "\n".join(texts).strip()

        if not text:
            skipped += 1
            continue

        txt_path = text_dir / f"{img_path.stem}.txt"

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        saved += 1

    print("OCR finished.")
    print(f"TXT saved: {saved}")
    print(f"Skipped (no text): {skipped}")

if __name__ == "__main__":
    # 3) OCR raw full images → TXT
    RAW_TEXT_DIR = PROJECT_ROOT / "ocr_full_images"
    ocr_images_to_txt(IMAGES_DIR, RAW_TEXT_DIR)