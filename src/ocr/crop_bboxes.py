"""
Crop bounding boxes from image and save to crops folder.
"""

import sys
import json
from pathlib import Path
from PIL import Image

from utils.path_loader import load_paths

paths = load_paths()
PROJECT_ROOT = paths["project_root"]

# Conservative buffer to capture clipped text without including unrelated content
BBOX_BUFFER = 5  # pixels


def crop_bboxes(image_path, bbox_json_path=None):
    """
    Crop bounding boxes from image and save to crops folder.
    
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
    
    # Load image
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    
    # Load bbox data
    with open(bbox_json_path, "r", encoding="utf-8") as f:
        bbox_data = json.load(f)
    
    # Create crops directory
    crops_dir = PROJECT_ROOT / "src" / "ocr" / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCropping {len(bbox_data)} bounding boxes from {image_path.name}...\n")
    
    for item in bbox_data:
        x1, y1, x2, y2 = item["bbox"]
        
        # Apply buffer with bounds checking
        x1 = max(0, x1 - BBOX_BUFFER)
        y1 = max(0, y1 - BBOX_BUFFER)
        x2 = min(img_w, x2 + BBOX_BUFFER)
        y2 = min(img_h, y2 + BBOX_BUFFER)
        
        crop = img.crop((x1, y1, x2, y2))
        
        # Save crop with naming: {image_stem}_{index}_{class}.png
        crop_filename = f"{image_path.stem}_{item['index']:02d}_{item['class']}.png"
        crop_path = crops_dir / crop_filename
        crop.save(crop_path)
        
        print(f"Saved: {crop_filename}")
    
    print(f"\nCropped images saved to: {crops_dir}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        bbox_json_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        crop_bboxes(image_path, bbox_json_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
