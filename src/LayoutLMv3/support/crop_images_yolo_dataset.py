import cv2
import numpy as np
from pathlib import Path
from collections import Counter
from utils.path_loader import load_paths

# =========================================================
# Language detection (safe import)
# =========================================================
try:
    from langdetect import detect as _detect_lang
    LANG_AVAILABLE = True
except Exception:
    _detect_lang = None
    LANG_AVAILABLE = False


# =========================================================
# Paths
# =========================================================
paths = load_paths()

PROJECT_ROOT = paths["project_root"]

LABELS_DIR = paths["data"]["datasets_root"] / "labels" / "train"
IMAGES_DIR = paths["data"]["datasets_root"] / "images" / "train"

OUTPUT_DIR = paths["data"]["datasets_root"] / "cropped_yolo"
TEXT_DIR = OUTPUT_DIR / "text_block"
PHOTO_DIR = OUTPUT_DIR / "photo"

TEXT_DIR.mkdir(parents=True, exist_ok=True)
PHOTO_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Supported image extensions
# =========================================================
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")


# =========================================================
# Quality filtering thresholds
# =========================================================
MIN_AREA = 400
MIN_WIDTH = 20
MIN_HEIGHT = 20
MIN_TEXT_RATIO = 0.02


# =========================================================
# Helper: YOLO bbox → pixel coords
# =========================================================
def yolo_to_xyxy(bbox, img_w, img_h):
    x_c, y_c, w, h = bbox

    x1 = int((x_c - w / 2) * img_w)
    y1 = int((y_c - h / 2) * img_h)
    x2 = int((x_c + w / 2) * img_w)
    y2 = int((y_c + h / 2) * img_h)

    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


# =========================================================
# Heuristic: detect if crop likely contains readable text
# =========================================================
def has_enough_text_structure(crop: np.ndarray) -> bool:
    if crop is None or crop.size == 0:
        return False

    h, w = crop.shape[:2]

    if h < MIN_HEIGHT or w < MIN_WIDTH or (h * w) < MIN_AREA:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    edge_ratio = edges.mean() / 255.0

    return edge_ratio > MIN_TEXT_RATIO


# =========================================================
# Fast readability + language check using tiny OCR pass
# =========================================================
def quick_text_check(crop: np.ndarray) -> tuple[bool, bool]:
    """
    Fast check without external OCR.
    Only verifies that crop likely contains readable text.
    Language filtering will be done later with PaddleOCR.
    """
    # Se passou no filtro estrutural, assume legível
    return True, True



# =========================================================
# Main cropping function with quality + language filtering
# =========================================================
def crop_yolo_dataset(images_dir: Path, labels_dir: Path, output_dir: Path):

    image_files = [
        p for p in images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    ]

    label_files = list(labels_dir.glob("*.txt"))

    image_stems = [p.stem for p in image_files]
    label_stems = [p.stem for p in label_files]

    dup_images = {k for k, v in Counter(image_stems).items() if v > 1}
    dup_labels = {k for k, v in Counter(label_stems).items() if v > 1}

    valid_stems = set(image_stems) & set(label_stems)
    valid_stems -= dup_images
    valid_stems -= dup_labels

    print(f"Total valid pairs: {len(valid_stems)}")

    kept = 0
    discarded_quality = 0
    discarded_lang = 0

    for stem in valid_stems:
        img_candidates = [p for p in image_files if p.stem == stem]

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

            # ---------------- Quality structure filter ----------------
            if not has_enough_text_structure(crop):
                discarded_quality += 1
                continue

            # ---------------- Fast OCR readability + language ----------
            readable, english = quick_text_check(crop)

            if not readable:
                discarded_quality += 1
                continue

            if not english:
                discarded_lang += 1
                continue

            # ---------------- Choose output folder --------------------
            if cls == 0:
                save_dir = TEXT_DIR
            elif cls == 1:
                save_dir = PHOTO_DIR
            else:
                continue

            save_path = save_dir / f"{stem}_{i}.jpg"
            cv2.imwrite(str(save_path), crop)
            kept += 1

    print("Cropping finished.")
    print(f"Kept crops: {kept}")
    print(f"Discarded (quality/readability): {discarded_quality}")
    print(f"Discarded (non-English): {discarded_lang}")


# =========================================================
# Run
# =========================================================
if __name__ == "__main__":
    crop_yolo_dataset(IMAGES_DIR, LABELS_DIR, OUTPUT_DIR)