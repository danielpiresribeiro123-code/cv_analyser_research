import sys
import os
from typing import Union, List, Dict, Any
import numpy as np

from ultralytics import YOLO
from pathlib import Path

from utils.path_loader import load_paths
paths = load_paths()

# Caminho base do projeto
PROJECT_ROOT = paths["project_root"]

MODEL_PATH = PROJECT_ROOT / "models" / "trained" / "yolo11s-text-photo-150e4b.pt"

import cv2
from PIL import Image
import json


def detect_and_segment_cv(
    image_data: Union[np.ndarray, bytes],
    model_path: Union[str, Path],
    conf_thres: float = 0.25,
    return_crops: bool = True
) -> List[Dict[str, Any]]:
    """
    Detecta text_blocks e photos num CV,
    segmenta as regiões e ordena de cima para baixo e esquerda para direita.
    
    Args:
        image_data: Either a numpy array (BGR) or bytes representing the image.
        model_path: Path to the YOLO model.
        conf_thres: Confidence threshold for detections.
        return_crops: Whether to include PIL Image crops in the output.
    
    Returns:
        List of detections with class, confidence, bbox, and optionally crops.
    """

    model = YOLO(model_path)

    # Convert bytes to numpy array if needed
    if isinstance(image_data, bytes):
        nparr = np.frombuffer(image_data, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image from bytes")
    elif isinstance(image_data, np.ndarray):
        img_bgr = image_data
    else:
        raise TypeError(f"image_data must be bytes or numpy array, got {type(image_data)}")

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

            detection = {
                "class": cls_name,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "y": y1,
                "x": x1,
            }
            
            # Optionally include crop as PIL Image
            if return_crops:
                crop = Image.fromarray(img_rgb[y1:y2, x1:x2])
                detection["crop"] = crop

            detections.append(detection)

    # Ordenação: cima → baixo, esquerda → direita
    detections.sort(key=lambda d: (d["y"], d["x"]))
    
    # Remove bboxes fully contained within another bbox of the same class
    filtered_detections = []
    for i, det_i in enumerate(detections):
        x1_i, y1_i, x2_i, y2_i = det_i["bbox"]
        is_contained = False
        
        for j, det_j in enumerate(detections):
            if i == j or det_i["class"] != det_j["class"]:
                continue
            
            x1_j, y1_j, x2_j, y2_j = det_j["bbox"]
            
            # Check if det_i is fully contained in det_j
            if (x1_i >= x1_j and y1_i >= y1_j and 
                x2_i <= x2_j and y2_i <= y2_j and
                (x1_i, y1_i, x2_i, y2_i) != (x1_j, y1_j, x2_j, y2_j)):
                is_contained = True
                break
        
        if not is_contained:
            filtered_detections.append(det_i)
    
    return filtered_detections


def detections_to_dict(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert detections list to a format suitable for JSON serialization.
    Removes PIL Image crops and converts tuples to lists.
    
    Returns a list of detection dicts with bbox as list and no crop data.
    """
    serializable = []
    for det in detections:
        serializable.append({
            "class": det["class"],
            "confidence": det["confidence"],
            "bbox": list(det["bbox"])  # Convert tuple to list for JSON
        })
    return serializable


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        # Read image file into memory as bytes
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        # Process in memory
        detections = detect_and_segment_cv(
            image_data=image_bytes,
            model_path=MODEL_PATH,
            return_crops=False  # Don't include crops for this example
        )
        
        # Convert to serializable format
        bbox_data = detections_to_dict(detections)
        
        # Print results
        for i, d in enumerate(bbox_data):
            print(i, d["class"], d["bbox"], d["confidence"])
        
        # Optionally save to file (still can if needed)
        bb_dir = PROJECT_ROOT / "src" / "ocr" / "bounding_boxes"
        bb_dir.mkdir(parents=True, exist_ok=True)

        base_name = Path(image_path).stem
        out_path = bb_dir / f"{base_name}_bbox.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(bbox_data, f, indent=2, ensure_ascii=False)
        print(f"Saved: {out_path}")
