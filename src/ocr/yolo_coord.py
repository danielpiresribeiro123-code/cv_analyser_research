import sys
import os

from ultralytics import YOLO
from pathlib import Path

from utils.path_loader import load_paths
paths = load_paths()

# Caminho base do projeto
PROJECT_ROOT = paths["project_root"]

MOLDEL_PATH = PROJECT_ROOT / "models" / "trained" / "yolov8s-text-photo-5.pt"

import cv2
from PIL import Image
import json


def detect_and_segment_cv(
    image_path,
    model_path,
    conf_thres=0.25
):
    """
    Detecta text_blocks e photos num CV,
    segmenta as regiões e ordena de cima para baixo e esquerda para direita.
    """

    model = YOLO(model_path)

    # Ler imagem
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    results = model.predict(
        source=img_rgb,
        conf=conf_thres,
        device="cpu",
        verbose=False
    )

    detections = []

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls.item())
            cls_name = model.names[cls_id]
            conf = float(box.conf.item())

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Garantir limites válidos
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = Image.fromarray(img_rgb[y1:y2, x1:x2])

            detections.append({
                "class": cls_name,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "y": y1,
                "x": x1,
                "crop": crop
            })

    # Ordenação: cima → baixo, esquerda → direita
    detections.sort(key=lambda d: (d["y"], d["x"]))

    return detections


if __name__ == "__main__":
    # Example run (adjust `image_path` as needed)
    image_path = r"C:\Users\Tomás\Documents\GitHub\CapStone\src\ocr\WhatsApp Image 2026-01-28 at 18.16.19.jpeg"

    detections = detect_and_segment_cv(
        image_path=image_path,
        model_path=MOLDEL_PATH)

    bbox_data = []
    for i, d in enumerate(detections):
        print(i, d["class"], d["bbox"], d["confidence"])
        bbox_data.append({
            "image_path": os.path.relpath(image_path, PROJECT_ROOT),
            "index": i,
            "class": d["class"],
            "confidence": d["confidence"],
            "bbox": list(d["bbox"])  # [x1, y1, x2, y2]
        })

    base_name = os.path.basename(image_path)
    name, _ = os.path.splitext(base_name)

    # Save JSON to `src/ocr/bounding_boxes/` directory
    bb_dir = PROJECT_ROOT / "src" / "ocr" / "bounding_boxes"
    bb_dir.mkdir(parents=True, exist_ok=True)

    out_path = bb_dir / f"{name}_bbox.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bbox_data, f, indent=2, ensure_ascii=False)
