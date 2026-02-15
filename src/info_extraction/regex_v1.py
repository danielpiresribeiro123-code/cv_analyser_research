import re
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union

from utils.path_loader import load_paths

# ==========================================================
# CONFIGURAÇÕES DE INPUT E OUTPUT
# ==========================================================
paths = load_paths()
PROJECT_ROOT = paths["project_root"]

TEXT_TRAIN_DATA = PROJECT_ROOT / "data" / "raw" / "Regex_test"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "processed_regex"
OUTPUT_FILE_NAME = "extracao.csv"

# ==========================================================
# PADRÕES REGEX (CALIBRADOS)
# ==========================================================

# Email: Ajustado para lidar com espaços acidentais antes do @ (comum em OCR/JSON)
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

PHONE_REGEX = re.compile(r"\b(?:\+?\d{1,3}[\s\-]?)?\(?\d{2,4}\)?[\s\-]?\d{3,5}[\s\-]?\d{3,5}\b")

# Morada: Captura tudo após "Address :" até encontrar um delimitador de outro campo
# O (.*?) com re.DOTALL é o que permite apanhar as várias linhas do teu exemplo
ADDRESS_BLOCK_REGEX = re.compile(
    r"(?:Address|Morada|Location)\s*[:\-]\s*(.*?)(?=E-mail|Email|Phone|Telefone|Objective|Experience|Personal|Education|$)",
    re.IGNORECASE | re.DOTALL
)



# ==========================================================
# FUNÇÕES DE APOIO
# ==========================================================

def clean_extracted_text(text: str) -> str:
    """Limpa quebras de linha, espaços duplos e caracteres estranhos."""
    if not text: return ""
    # Substitui quebras de linha e tabs por espaços
    text = re.sub(r'[\r\n\t]+', ' ', text)
    # Remove caracteres especiais de listas (bullets)
    text = re.sub(r'[\uf0b7\u2022*]', '', text)
    # Remove espaços múltiplos
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def extract_address(text: str) -> Optional[str]:
    match = ADDRESS_BLOCK_REGEX.search(text)
    if match:
        addr = match.group(1)
        # Filtro de segurança para não capturar o currículo inteiro se o delimitador falhar
        if len(addr.strip()) > 5 and len(addr) < 300:
            return clean_extracted_text(addr)
    return None

def extract_generic(pattern: re.Pattern, text: str) -> List[str]:
    return [clean_extracted_text(m.group(0)) for m in pattern.finditer(text)]

# ==========================================================
# PROCESSAMENTO PRINCIPAL
# ==========================================================

def get_extracted_dataframe(limit: Optional[int] = None) -> pd.DataFrame:
    if not TEXT_TRAIN_DATA.exists():
        raise FileNotFoundError(f"Pasta de entrada não encontrada: {TEXT_TRAIN_DATA}")

    files = list(TEXT_TRAIN_DATA.glob("*.json")) + list(TEXT_TRAIN_DATA.glob("*.txt"))
    if limit:
        files = files[:limit]

    rows = []
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = json.load(f) if file_path.suffix == ".json" else {"text": f.read()}
            
            text = content.get("text", "")
            
            rows.append({
                "file": file_path.name,
                "emails": ", ".join(extract_generic(EMAIL_REGEX, text)) or None,
                "phones": ", ".join(extract_generic(PHONE_REGEX, text)) or None,
                "address": extract_address(text),
            })
        except Exception as e:
            rows.append({"file": file_path.name, "error": str(e)})

    df = pd.DataFrame(rows)

    # Gravação do Output
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FOLDER / OUTPUT_FILE_NAME, index=False, encoding='utf-8-sig')
    
    return df

if __name__ == "__main__":
    df_final = get_extracted_dataframe()
    
    # Configuração para veres a morada completa no print do terminal
    pd.set_option('display.max_colwidth', None)
    
    print(f"Sucesso! Ficheiro guardado em: {OUTPUT_FOLDER / OUTPUT_FILE_NAME}")
    print(df_final.head())