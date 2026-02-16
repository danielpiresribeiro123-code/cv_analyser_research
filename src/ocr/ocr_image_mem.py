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

from utils.path_loader import load_paths

paths = load_paths()
PROJECT_ROOT = paths["project_root"]

# Conservative buffer to capture clipped text without including unrelated content
BBOX_BUFFER = 18  # pixels, increased to avoid left-edge clipping

# Lazy initialize PaddleOCR to avoid noisy prints at import time
_ocr = None
def get_ocr():
    """Return a singleton PaddleOCR instance, initializing it lazily and quietly."""
    global _ocr
    if _ocr is None:
        # Import and create PaddleOCR lazily while silencing noisy native prints
        import io as _io
        from contextlib import redirect_stdout, redirect_stderr
        _buf = _io.StringIO()
        with redirect_stdout(_buf), redirect_stderr(_buf):
            from paddleocr import PaddleOCR
            _ocr = PaddleOCR(use_angle_cls=True, lang='en', enable_mkldnn=False)
    return _ocr


def extract_text_from_bboxes(image_data, detections, verbose: bool = False):
    """
    Extract text from bounding boxes using OCR.
    
    Args:
        image_data: PIL Image or numpy array (RGB/BGR)
        detections: List of detection dicts from detect_and_segment_cv() with crops
    
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
    import numpy as np
    from PIL import Image
    
    # Convert to PIL Image if needed
    if isinstance(image_data, np.ndarray):
        if image_data.shape[2] == 3:
            # Check if BGR or RGB by trying to convert
            image_data = Image.fromarray(image_data if image_data.max() <= 1 else image_data.astype(np.uint8))
    
    if not isinstance(image_data, Image.Image):
        raise TypeError("image_data must be PIL Image or numpy array")
    
    img_w, img_h = image_data.size
    
    results = []
    if verbose:
        print(f"\n Processing {len(detections)} bounding boxes...\n")
    
    for index, item in enumerate(detections):
        # Skip photo class detections
        if item["class"] == "photo":
            continue
        
        # Get crop - it's already in memory from detect_and_segment_cv
        if "crop" in item:
            crop = item["crop"]
        else:
            # Fallback: extract from bbox if crop not available
            x1, y1, x2, y2 = item["bbox"]
            
            # Apply buffer with bounds checking
            x1 = max(0, x1 - BBOX_BUFFER)
            y1 = max(0, y1 - BBOX_BUFFER)
            x2 = min(img_w, x2 + BBOX_BUFFER)
            y2 = min(img_h, y2 + BBOX_BUFFER)
            
            crop = image_data.crop((x1, y1, x2, y2))
        
        # Preprocess crop: sharpen only
        arr = np.array(crop.convert('RGB'))
        try:
            import cv2
            kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
            arr_proc = cv2.filter2D(arr, -1, kernel)
        except Exception:
            from PIL import ImageFilter
            crop_sharp = crop.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            arr_proc = np.array(crop_sharp.convert('RGB'))
        
        # Run OCR once on preprocessed crop
        ocr_inst = get_ocr()
        try:
            if hasattr(ocr_inst, 'predict'):
                res = ocr_inst.predict(arr_proc)
                lines = []
                if isinstance(res, list):
                    for r in res:
                        if isinstance(r, dict) and 'rec_texts' in r:
                            lines.extend(r['rec_texts'])
                        elif isinstance(r, list):
                            for item in r:
                                if isinstance(item, (list, tuple)) and len(item) >= 2:
                                    txt = item[1][0] if isinstance(item[1], (list, tuple)) else item[1]
                                    lines.append(txt)
                extracted_text = '\n'.join([l for l in lines if l])
            else:
                res = ocr_inst.ocr(arr_proc, cls=True)
                lines = []
                for item in res:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        txt = item[1][0] if isinstance(item[1], (list, tuple)) else item[1]
                        lines.append(txt)
                extracted_text = '\n'.join([l for l in lines if l])
        except Exception:
            extracted_text = ''
        
        bbox = item["bbox"]
        confidence = item["confidence"]
        class_name = item["class"]
        
        results.append({
            "index": index,
            "class": class_name,
            "confidence": confidence,
            "bbox": list(bbox) if isinstance(bbox, tuple) else bbox,
            "text": extracted_text
        })
        if verbose:
            print(f"[{index}] {class_name} (conf={confidence:.2f}): {extracted_text[:60]}...")
    
    return results


def process_image_with_bbox(image_data, detections, output_path=None, verbose: bool = False):
    """
    Full pipeline: extract text from detections in memory, optionally save results.
    Concatenates all text in order (top to bottom, left to right).
    
    Args:
        image_data: PIL Image or numpy array (RGB/BGR) or bytes
        detections: List of detection dicts from detect_and_segment_cv() with crops
        output_path: Optional path to save results as JSON. If None, no file is written.
    
    Returns:
        Dict with structure:
        {
            "text": str (concatenated text),
            "results": list (detailed OCR results with bbox,confidence, etc)
        }
    """
    import numpy as np
    from PIL import Image
    
    # Handle bytes input
    if isinstance(image_data, bytes):
        nparr = np.frombuffer(image_data, np.uint8)
        from cv2 import imdecode, COLOR_BGR2RGB, cvtColor
        img_bgr = imdecode(nparr, 1)
        if img_bgr is None:
            raise ValueError("Could not decode image from bytes")
        image_data = Image.fromarray(cvtColor(img_bgr, COLOR_BGR2RGB))
    
    # Convert numpy BGR to RGB PIL Image if needed
    if isinstance(image_data, np.ndarray):
        from cv2 import COLOR_BGR2RGB, cvtColor
        if image_data.ndim == 3 and image_data.shape[2] == 3:
            # Assume BGR from cv2
            image_data = cvtColor(image_data, COLOR_BGR2RGB)
        image_data = Image.fromarray(image_data.astype(np.uint8))
    
    if not isinstance(image_data, Image.Image):
        raise TypeError("image_data must be PIL Image, numpy array, or bytes")
    
    # Extract text from bboxes
    ocr_results = extract_text_from_bboxes(image_data, detections, verbose=verbose)
    
    # Concatenate all text in order (separate bounding boxes with a double newline)
    concatenated_text = "\n\n".join([item["text"] for item in ocr_results])
    
    result = {
        "text": concatenated_text,
        "results": ocr_results
    }
    
    # Optionally save to file
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"text": concatenated_text}, f, ensure_ascii=False, indent=2)
        if verbose:
            print(f"\nText saved to: {output_path}")
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        # Example: Process image from file path
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        # Import yolo detection function
        from yolo_coord_mem import detect_and_segment_cv, MODEL_PATH
        
        # Get detections from YOLO
        detections = detect_and_segment_cv(
            image_data=image_bytes,
            model_path=MODEL_PATH,
            return_crops=True  # Include crops for OCR
        )
        
        # Process with OCR (in memory)
        result = process_image_with_bbox(
            image_data=image_bytes,
            detections=detections,
            output_path=None  # Set to path if you want to save
        )
        
        print(f"\nExtracted text:\n{result['text']}")
    else:
        print("Usage: python ocr_image_mem.py <image_path>")