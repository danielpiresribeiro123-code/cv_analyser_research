import os
import json
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR

from PIL import Image
from tqdm import tqdm
from utils.path_loader import load_paths

# Supported image extensions
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")


# =============================================================================
# PADDLE OCR + LAYOUTLMV3 DATASET GENERATOR
# =============================================================================


# Fix paddle cache path (opt)
safe_dir = r"C:\\paddle_cache"
os.environ["USERPROFILE"] = safe_dir
os.environ["HOME"] = safe_dir
os.environ["FLAGS_enable_pir_api"] = "0"


# Initialize PaddleOCR (English)
ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    enable_mkldnn=False,
    show_log=False,
)

# Paths
paths = load_paths()
CROPPED_ROOT = paths["data"]["datasets_root"] / "cropped_yolo" / "text_block"
OUTPUT_JSONL = paths["data"]["datasets_root"] / "cropped_yolo" / "layoutlmv3_dataset.jsonl"

# Remove old output
if OUTPUT_JSONL.exists():
    OUTPUT_JSONL.unlink()

# =============================================================================
# FUNC: normalize bounding box to 0-1000
# =============================================================================
def normalize_bbox(box, image_width, image_height):
    """
    Convert pixel bbox [x1,y1,x2,y2] to 0..1000 normalized bounding box
    """
    x1, y1, x2, y2 = box
    return [
        int((x1 / image_width) * 1000),
        int((y1 / image_height) * 1000),
        int((x2 / image_width) * 1000),
        int((y2 / image_height) * 1000),
    ]


# =============================================================================
# MAIN LOOP
# =============================================================================
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:

    for class_dir in tqdm(list(CROPPED_ROOT.iterdir())):
        if not class_dir.is_dir():
            continue

        label = class_dir.name

        for img_path in class_dir.glob("*"):
            # Only process image files
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue

            # Read image
            pil_image = Image.open(img_path).convert("RGB")
            w, h = pil_image.size

            # PaddleOCR
            ocr_results = ocr_engine.ocr(str(img_path), cls=True)

            words = []
            boxes = []

            # iterate words detected
            if ocr_results and ocr_results[0]:
                for line in ocr_results[0]:
                    bbox_points = line[0]  # quadrilateral points
                    text = line[1][0].strip()

                    if not text:
                        continue

                    # convert paddle quadrilateral to [x1,y1,x2,y2]
                    pts = np.array(bbox_points)
                    x1 = int(min(pts[:, 0]))
                    y1 = int(min(pts[:, 1]))
                    x2 = int(max(pts[:, 0]))
                    y2 = int(max(pts[:, 1]))

                    # normalize
                    norm_box = normalize_bbox([x1, y1, x2, y2], w, h)

                    words.append(text)
                    boxes.append(norm_box)

            if not words:
                continue

            # write line JSON
            record = {
                "image_file": str(img_path),
                "words": words,
                "boxes": boxes,
                "label": label,
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Saved LayoutLMv3 dataset → {OUTPUT_JSONL}")
