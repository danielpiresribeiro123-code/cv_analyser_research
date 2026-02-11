import os
import json
from pathlib import Path
from PIL import Image

# --- CRITICAL FIX FOR SPECIAL CHARACTERS IN USERNAME ---
# We force Paddle to look for models in a simple path without accents.
# This must be set BEFORE importing paddleocr.
safe_dir = r"C:\paddle_cache"
os.environ["USERPROFILE"] = safe_dir
os.environ["HOME"] = safe_dir 
# -------------------------------------------------------

from paddleocr import PaddleOCR

os.environ['FLAGS_enable_pir_api'] = '0'

import paddle
import sys
from types import ModuleType

# LangChain Compatibility
try:
    from langchain_core.documents import Document
    m = ModuleType("langchain.docstore.document")
    m.Document = Document
    sys.modules["langchain.docstore.document"] = m
    import langchain_text_splitters as ts
    sys.modules["langchain.text_splitter"] = ts
except ImportError:
    pass

# OCR Initialization
from paddleocr import PaddleOCR
from utils.path_loader import load_paths

paths = load_paths()
PROJECT_ROOT = paths["project_root"]

# Conservative buffer to capture clipped text without including unrelated content
BBOX_BUFFER = 5  # pixels

ocr = PaddleOCR(use_angle_cls=True, lang='en', enable_mkldnn=False)

print("OCR initialized with PaddleOCR. Ready to process images.")


def extract_text_from_bboxes(image_path, bbox_json_path):
    """
    Extract text from bounding boxes using OCR.
    
    Args:
        image_path: Path to the image file
        bbox_json_path: Path to the bounding box JSON (from yolo_coord)
    
    Returns:
        List of dicts with structure:
        {
            "index": int,
            "class": str,
            "confidence": float,
            "bbox": [x1, y1, x2, y2],
            "text": str
        }
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    
    # Load bounding box data
    with open(bbox_json_path, "r", encoding="utf-8") as f:
        bbox_data = json.load(f)
    
    results = []
    
    print(f"\n Processing {len(bbox_data)} bounding boxes from {Path(image_path).name}...\n")
    
    for item in bbox_data:
        # Skip photo class bounding boxes
        if item["class"] == "photo":
            continue
        
        x1, y1, x2, y2 = item["bbox"]
        
        # Apply buffer with bounds checking
        x1 = max(0, x1 - BBOX_BUFFER)
        y1 = max(0, y1 - BBOX_BUFFER)
        x2 = min(img_w, x2 + BBOX_BUFFER)
        y2 = min(img_h, y2 + BBOX_BUFFER)
        
        crop = img.crop((x1, y1, x2, y2))
        
        # Convert PIL Image to numpy array for OCR
        import numpy as np
        crop_array = np.array(crop)
        
        # Run OCR on crop
        ocr_result = ocr.predict(crop_array)
        
        # Extract text from OCR result
        text_lines = []
        if ocr_result and len(ocr_result) > 0:
            for res in ocr_result:
                for line in res['rec_texts']: 
                    text = line
                    text_lines.append(text)
        
        extracted_text = " ".join(text_lines)
        
        results.append({
            "index": item["index"],
            "class": item["class"],
            "confidence": item["confidence"],
            "bbox": item["bbox"],
            "text": extracted_text
        })
        
        print(f"[{item['index']}] {item['class']} (conf={item['confidence']:.2f}): {extracted_text[:60]}...")
    
    return results


def process_image_with_bbox(image_path, bbox_json_path=None):
    """
    Full pipeline: load image + bbox JSON, extract text, save to extracted_text folder as JSON.
    Concatenates all text in order (top to bottom, left to right).
    
    Args:
        image_path: Path to image
        bbox_json_path: Path to bbox JSON. If None, infer from image name in bounding_boxes/
    """
    image_path = Path(image_path)
    
    # Auto-find bbox JSON if not provided
    if bbox_json_path is None:
        bbox_json_path = PROJECT_ROOT / "src" / "ocr" / "bounding_boxes" / f"{image_path.stem}_bbox.json"
    
    bbox_json_path = Path(bbox_json_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not bbox_json_path.exists():
        raise FileNotFoundError(f"Bbox JSON not found: {bbox_json_path}")
    
    # Extract text from bboxes
    ocr_results = extract_text_from_bboxes(str(image_path), str(bbox_json_path))
    
    # Concatenate all text in order
    concatenated_text = "\n".join([item["text"] for item in ocr_results])
    
    # Create output directory
    out_dir = PROJECT_ROOT / "src" / "ocr" / "extracted_text"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON file
    out_path = out_dir / f"{image_path.stem}_text.json"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(concatenated_text)
    
    
    return concatenated_text


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        bbox_json_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        process_image_with_bbox(image_path, bbox_json_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)