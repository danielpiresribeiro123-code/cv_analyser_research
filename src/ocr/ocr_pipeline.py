"""
OCR Pipeline: Process a folder of images
1. Detect bounding boxes using YOLO
2. Extract text from each bounding box using PaddleOCR
"""

import sys
import json
import os
from pathlib import Path

from utils.path_loader import load_paths
from yolo_coord import detect_and_segment_cv, MODEL_PATH, PROJECT_ROOT
from ocr_image import process_image_with_bbox

# Image extensions to process
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def process_image_folder(folder_path):
    """
    Process all images in a folder:
    1. Detect bounding boxes (YOLO)
    2. Extract text from boxes (OCR)
    
    Args:
        folder_path: Path to folder containing images
    """
    folder = Path(folder_path)
    
    if not folder.is_dir():
        raise ValueError(f"Folder not found: {folder_path}")
    
    # Find all images
    image_files = [
        f for f in folder.glob("*")
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return
    
    print(f"\nFound {len(image_files)} images to process\n")
    
    # Create output directories
    bbox_dir = PROJECT_ROOT / "src" / "ocr" / "bounding_boxes"
    text_dir = PROJECT_ROOT / "src" / "ocr" / "extracted_text"
    bbox_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(image_files)}] Processing: {image_path.name}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Detect bounding boxes
            print(f"\nStep 1: Detecting bounding boxes...")
            detections = detect_and_segment_cv(
                image_path=str(image_path),
                model_path=MODEL_PATH
            )
            print(f"Found {len(detections)} bounding boxes")
            
            # Save bbox JSON
            bbox_data = []
            for i, d in enumerate(detections):
                bbox_data.append({
                    "image_path": os.path.relpath(image_path, PROJECT_ROOT),
                    "index": i,
                    "class": d["class"],
                    "confidence": d["confidence"],
                    "bbox": list(d["bbox"])
                })
            
            bbox_json_path = bbox_dir / f"{image_path.stem}_bbox.json"
            with open(bbox_json_path, "w", encoding="utf-8") as f:
                json.dump(bbox_data, f, indent=2, ensure_ascii=False)
            print(f"Saved bbox JSON: {bbox_json_path.name}")
            
            # Step 2: Extract text from bboxes
            print(f"\nStep 2: Extracting text from bounding boxes...")
            process_image_with_bbox(str(image_path), str(bbox_json_path))
            print(f"Text extraction complete")
            
            success_count += 1
            
        except Exception as e:
            print(f"\nError processing {image_path.name}: {e}")
            fail_count += 1
    
    # Summary
    print(f"\n\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Processed: {success_count}/{len(image_files)}")
    if fail_count > 0:
        print(f"Failed: {fail_count}/{len(image_files)}")
    print(f"\nOutput directories:")
    print(f"  - Bounding boxes: {bbox_dir}")
    print(f"  - Extracted text: {text_dir}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        # Default example folder
        folder_path = PROJECT_ROOT / "src" / "ocr"
    
    try:
        process_image_folder(folder_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
