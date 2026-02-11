import re
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union

from utils.path_loader import load_paths

# ==========================================================
# Paths
# ==========================================================

paths = load_paths()
PROJECT_ROOT = paths["project_root"]

TEXT_TRAIN_DATA = PROJECT_ROOT / "data" / "raw" / "Json"


# ==========================================================
# Regex patterns
# ==========================================================

EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

PHONE_REGEX = re.compile(
    r"""
    (?:
        \+?\d{1,3}[\s\-]?
    )?
    (?:\(?\d{2,3}\)?[\s\-]?)?
    \d{3,4}[\s\-]?\d{3,4}
    """,
    re.VERBOSE,)

LINKEDIN_REGEX = re.compile(
    r"(https?://)?(www\.)?linkedin\.com/[^\s]+",
    re.IGNORECASE,
)

GITHUB_REGEX = re.compile(
    r"(https?://)?(www\.)?github\.com/[^\s]+",
    re.IGNORECASE,
)

# ==========================================================
# Additional regex patterns
# ==========================================================

ADDRESS_REGEX = re.compile(
    r"""
    \b
    \d{1,5}                                  # street number
    \s+
    [A-Za-z0-9\s.,'-]+                        # street name
    (?:Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Lane|Ln|Drive|Dr)?
    [^\n,]*                                   # rest of line
    (?:,\s*[A-Za-z\s]+){1,3}                  # city/state
    (?:\s*\d{4,6})?                           # postal code
    (?:,\s*[A-Za-z\s]+)?                      # country
    """,
    re.VERBOSE | re.IGNORECASE,
)


EXPERIENCE_REGEX = re.compile(
    r"""
    (?:
        experience|exp\.?|yrs?|years?
    )
    [^\d]{0,20}                # até ~3 palavras no meio
    (\d{1,2})\+?               # número de anos (capturado)
    |
    (\d{1,2})\+?               # número antes
    [^\n]{0,20}
    (?:
        years?|yrs?|experience|exp\.?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ==========================================================
# Text loading
# ==========================================================

def load_text_from_file(file_path: Path) -> str:
    """
    Carrega texto de ficheiro JSON ou TXT.
    Espera JSON no formato: {"text": "..."}.
    """
    if file_path.suffix.lower() == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("text", "")

    if file_path.suffix.lower() == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError(f"Formato não suportado: {file_path.suffix}")


# ==========================================================
# Extraction helpers
# ==========================================================

def extract_first(pattern: re.Pattern, text: str) -> Optional[str]:
    """Retorna a primeira ocorrência de um padrão."""
    match = pattern.search(text)
    return match.group(0).strip() if match else None


def extract_all(pattern: re.Pattern, text: str) -> List[str]:
    """Retorna todas as ocorrências únicas de um padrão."""
    matches = pattern.findall(text)
    cleaned = []

    for m in matches:
        if isinstance(m, tuple):
            m = "".join(m)

        m = m.strip()
        if m and m not in cleaned:
            cleaned.append(m)

    return cleaned

def extract_address(text: str) -> Optional[str]:
    """Extrai a primeira morada encontrada."""
    match = ADDRESS_REGEX.search(text)
    return match.group(0).strip() if match else None


def extract_experience_years(text: str) -> Optional[int]:
    """
    Extrai anos de experiência a partir de múltiplos formatos de CV.
    Retorna inteiro ou None.
    """
    matches = EXPERIENCE_REGEX.findall(text)

    for m in matches:
        # regex tem dois grupos possíveis
        for g in m:
            if g.isdigit():
                return int(g)

    return None


# ==========================================================
# Core extraction
# ==========================================================

def extract_contact_info(text: str) -> Dict[str, Union[str, List[str], int, None]]:
    """
    Extrai informação de contacto e metadados de um texto OCR de CV.
    """

    emails = extract_all(EMAIL_REGEX, text)
    phones = extract_all(PHONE_REGEX, text)
    linkedin = extract_first(LINKEDIN_REGEX, text)
    github = extract_first(GITHUB_REGEX, text)
    address = extract_address(text)
    experience_years = extract_experience_years(text)

    return {
        "email": emails[0] if emails else None,
        "all_emails": emails,
        "phone": phones[0] if phones else None,
        "all_phones": phones,
        "linkedin": linkedin,
        "github": github,
        "address": address,
        "experience_years": experience_years,
    }


# ==========================================================
# Dataset processing
# ==========================================================

def process_folder(
    folder: Path = TEXT_TRAIN_DATA,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Processa todos os ficheiros JSON/TXT numa pasta
    e retorna DataFrame com contactos extraídos.
    """

    if not folder.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {folder}")

    files = list(folder.glob("*.json")) + list(folder.glob("*.txt"))

    if limit:
        files = files[:limit]

    rows = []

    for file_path in files:
        try:
            text = load_text_from_file(file_path)
            info = extract_contact_info(text)

            rows.append(
                {
                    "file": file_path.name,
                    **info,
                }
            )

        except Exception as e:
            rows.append(
                {
                    "file": file_path.name,
                    "error": str(e),
                }
            )

    return pd.DataFrame(rows)


# ==========================================================
# CLI execution
# ==========================================================

if __name__ == "__main__":
    df = process_folder(limit=10)
    print(df.head())
