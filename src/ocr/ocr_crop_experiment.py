import sys
from pathlib import Path
import json

def main(argv):
    if len(argv) < 2:
        print("Usage: python ocr_crop_experiment.py <pdf_path> [page_index] [det_index]")
        return 1

    pdf_path = Path(argv[1])
    page_idx = int(argv[2]) if len(argv) > 2 else 0
    det_idx = int(argv[3]) if len(argv) > 3 else 0

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return 1

    # Ensure outputs folder (now in src/ocr/ocr_experiments)
    repo_root = Path(__file__).resolve().parents[2]
    outputs_dir = Path(__file__).resolve().parent / 'ocr_experiments'
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Ensure repo `src` is importable
    repo_src = repo_root / 'src'
    import sys
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))
    # Some modules rely on environment paths loaded by `utils.path_loader`.
    # Provide safe fallbacks so the experiment can run standalone.
    import os
    defaults = {
        "PROJECT_ROOT": str(repo_root),
        "DATASETS_ROOT": str(repo_root / 'data' / 'datasets'),
        "DATASETS_SKILL_TABLE": str(repo_root / 'data' / 'skill_tables'),
        "MODELS_IN_ROOT": str(repo_root / 'models'),
        "MODELS_OUT_ROOT": str(repo_root / 'models' / 'out'),
        "ENV": "dev",
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)

    # Ensure minimal dirs exist to satisfy path checks
    (repo_root / 'data' / 'datasets').mkdir(parents=True, exist_ok=True)
    (repo_root / 'data' / 'skill_tables').mkdir(parents=True, exist_ok=True)
    (repo_root / 'models').mkdir(parents=True, exist_ok=True)

    try:
        from input_processing.pdf_to_image import convert_pdf_to_images
        from yolo_coord_mem import detect_and_segment_cv, MODEL_PATH
        from ocr_image_mem import get_ocr
    except Exception as e:
        # Try absolute imports if relative fail
        try:
            from input_processing.pdf_to_image import convert_pdf_to_images
            from ocr.yolo_coord_mem import detect_and_segment_cv, MODEL_PATH
            from ocr.ocr_image_mem import get_ocr
        except Exception as e2:
            print("Missing imports for experiment:", e2)
            raise

    # Convert PDF to images
    images = convert_pdf_to_images(pdf_path.read_bytes())
    if not images or page_idx >= len(images):
        print("No images/pages available")
        return 1

    img = images[page_idx]

    # Get detections with crops — ensure image passed is bytes or numpy array
    import io as _io
    img_buf = _io.BytesIO()
    img.save(img_buf, format='PNG')
    img_bytes_for_det = img_buf.getvalue()

    detections = detect_and_segment_cv(image_data=img_bytes_for_det, model_path=MODEL_PATH, return_crops=True)
    if not detections or det_idx >= len(detections):
        print("No detections on the page")
        return 1

    crop = detections[det_idx].get('crop')
    if crop is None:
        # Fallback: crop bbox from PIL image
        import numpy as _np
        bbox = detections[det_idx]['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        crop = img.crop((x1, y1, x2, y2))

    # Save raw crop
    raw_path = outputs_dir / 'crop_raw.png'
    crop.save(raw_path)

    # Prepare preprocessing variants
    variants = {}
    try:
        import cv2
        import numpy as np
        has_cv2 = True
    except Exception:
        has_cv2 = False
        from PIL import Image, ImageFilter, ImageOps
        import numpy as np

    def pil_to_np(im):
        return np.array(im.convert('RGB'))

    def save_np(name, arr):
        p = outputs_dir / f"{name}.png"
        from PIL import Image
        Image.fromarray(arr).save(p)
        return str(p)

    arr_raw = pil_to_np(crop)
    variants['raw'] = arr_raw

    if has_cv2:
        # Upscale
        up = cv2.resize(arr_raw, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        variants['upscale'] = up

        # CLAHE on L-channel
        lab = cv2.cvtColor(up, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        clahe_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        variants['clahe'] = clahe_rgb

        # Sharpen
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        sharp = cv2.filter2D(clahe_rgb, -1, kernel)
        variants['sharpen'] = sharp

        # Binarize (adaptive)
        gray = cv2.cvtColor(sharp, cv2.COLOR_RGB2GRAY)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 15)
        variants['binarize'] = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)

        # Deskew attempt
        try:
            coords = np.column_stack(np.where(gray > 0))
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = gray.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            desk = cv2.warpAffine(sharp, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            variants['deskew'] = desk
        except Exception:
            pass
    else:
        # PIL-based simple variants
        from PIL import ImageFilter, ImageOps
        up_p = crop.resize((crop.width*2, crop.height*2), Image.BICUBIC)
        variants['upscale'] = pil_to_np(up_p)
        clahe_p = ImageOps.autocontrast(up_p)
        variants['clahe'] = pil_to_np(clahe_p)
        sharp_p = up_p.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        variants['sharpen'] = pil_to_np(sharp_p)

    # Run OCR on each variant
    ocr = get_ocr()
    results = []
    for name, arr in variants.items():
        try:
            res = ocr.ocr(arr) if hasattr(ocr, 'ocr') else ocr.predict(arr)
        except Exception as e:
            res = [{"error": str(e)}]

        out_img = save_np(name, arr)
        # Extract concatenated text if available
        texts = []
        if isinstance(res, list):
            for r in res:
                # PaddleOCR predict format may vary
                if isinstance(r, dict) and 'rec_texts' in r:
                    texts.extend(r['rec_texts'])
                elif isinstance(r, list):
                    for item in r:
                        if isinstance(item, dict):
                            texts.extend(item.get('rec_texts', []))
                        elif len(item) >= 2:
                            # older paddle format: [box, (text, conf)]
                            if isinstance(item[1], tuple) or isinstance(item[1], list):
                                texts.append(item[1][0])
                            else:
                                texts.append(str(item[1]))
                elif isinstance(r, dict) and 'text' in r:
                    texts.append(r['text'])

        text_join = "\n".join(texts)
        results.append({"variant": name, "image": out_img, "text": text_join})

    # Save results JSON and CSV
    results_json = outputs_dir / 'ocr_experiment_results.json'
    with open(results_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # CSV
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        csv_path = outputs_dir / 'ocr_experiment_results.csv'
        df.to_csv(csv_path, index=False, sep=';')
    except Exception:
        pass

    print(f"Saved variants and OCR outputs to: {outputs_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
