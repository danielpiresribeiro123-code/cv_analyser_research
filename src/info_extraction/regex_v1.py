import re
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from utils.path_loader import load_paths

paths = load_paths()
PROJECT_ROOT = Path(paths["project_root"])

TEXT_TRAIN_DATA = PROJECT_ROOT / "data" / "raw" / "Regex_test"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "processed_regex"
OUTPUT_FILE_NAME = "regex_output.csv"

EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_REGEX = re.compile(r"\b(?:\+?\d{1,3}[\s\-]?)?\(?\d{2,4}\)?[\s\-]?\d{3,5}[\s\-]?\d{3,5}\b")
ADDRESS_BLOCK_REGEX = re.compile(
    r"(?:Address|Morada|Location)\s*[:\-]\s*(.*?)(?=E-mail|Email|Phone|Telefone|Objective|Experience|Personal|Education|$)",
    re.IGNORECASE | re.DOTALL
)

def clean_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r'[\r\n\t\uf0b7\u2022*]+', ' ', text)
    return re.sub(r'\s{2,}', ' ', text).strip()

def get_regex_dataframe(limit: Optional[int] = None) -> pd.DataFrame:
    files = list(TEXT_TRAIN_DATA.glob("*.json")) + list(TEXT_TRAIN_DATA.glob("*.txt"))
    if limit:
        files = files[:limit]

    rows = []
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = json.load(f) if file_path.suffix == ".json" else {"text": f.read()}
            
            text = content.get("text", "")
            
            emails = [clean_text(m.group(0)) for m in EMAIL_REGEX.finditer(text)]
            phones = [clean_text(m.group(0)) for m in PHONE_REGEX.finditer(text)]
            
            addr_match = ADDRESS_BLOCK_REGEX.search(text)
            address = clean_text(addr_match.group(1)) if addr_match else None

            rows.append({
                "file": file_path.name,
                "email": emails[0] if emails else None,
                "phone": phones[0] if phones else None,
                "address": address
            })
        except Exception as e:
            rows.append({"file": file_path.name, "error": str(e)})

    return pd.DataFrame(rows)

if __name__ == "__main__":
    df_regex = get_regex_dataframe()
    
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    output_path = OUTPUT_FOLDER / OUTPUT_FILE_NAME
    
    df_regex.to_csv(output_path, index=False, sep=';', encoding='utf-8-sig')
    
    print(f"Sucesso! Ficheiro guardado em: {output_path}")