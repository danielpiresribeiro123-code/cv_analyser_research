import os

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

ocr = PaddleOCR(use_angle_cls=True, lang='en', enable_mkldnn=False)

print("✅ Initialization successful!")

def run_ocr_base(path: str, save: bool = False):
  result = ocr.predict(path)

  text_doc = ''
  for res in result:
    for text_line in res['rec_texts']:
      text_doc += (text_line + '\n')

  print("\n--- RAW EXTRACTED TEXT ---\n")
  print(text_doc)
  print("\n--------------------------\n")

  if save:
    base_name = os.path.basename(path)
    name, _ = os.path.splitext(base_name)
    out_path = os.path.join(os.path.dirname(path), f"{name}_ocr.json")
    with open(out_path, 'w', encoding='utf-8') as f:
      f.write(text_doc)
    print(f"✅ Text saved to: {out_path}")

img_path = r"C:\Users\Tomás\Documents\GitHub\CapStone\src\ocr\WhatsApp Image 2026-01-28 at 18.16.19.jpeg"

if os.path.exists(img_path):
    print(f"✅ File found: {img_path}")
else:
    print(f"❌ File NOT found. Check the name and path.")
    exit() 

run_ocr_base(img_path, save=True)