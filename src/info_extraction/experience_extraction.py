import re
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

from utils.path_loader import load_paths

# ==========================================================
# Paths
# ==========================================================

paths = load_paths()
PROJECT_ROOT = paths["project_root"]

TEXT_TRAIN_DATA = PROJECT_ROOT / "data" / "raw" / "Json"
JOB_TITLES_PATH = PROJECT_ROOT / "src" / "info_extraction"/ "lexicons" / "job_titles.json"


# ==========================================================
# Date regex patterns
# ==========================================================

YEAR_RANGE_REGEX = re.compile(
    r"""
    (?P<start>19\d{2}|20\d{2})
    \s*[-–]\s*
    (?P<end>Present|present|19\d{2}|20\d{2})
    """,
    re.VERBOSE,
)

MONTH_YEAR_RANGE_REGEX = re.compile(
    r"""
    (?:
        Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec
    )
    \s*
    (?P<start>19\d{2}|20\d{2})
    \s*[-–]\s*
    (?:
        Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec
    )?
    \s*
    (?P<end>Present|present|19\d{2}|20\d{2})
    """,
    re.VERBOSE | re.IGNORECASE,
)


CURRENT_YEAR = datetime.now().year


# ==========================================================
# Load job titles
# ==========================================================

def load_job_titles(path: Path = JOB_TITLES_PATH) -> List[str]:
    """Carrega lista de job titles do JSON."""
    if not path.exists():
        raise FileNotFoundError(f"job_titles.json não encontrado: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # aceita lista simples ou dict por categorias
    if isinstance(data, dict):
        titles = []
        for values in data.values():
            titles.extend(values)
        return [t.lower() for t in titles]

    return [t.lower() for t in data]


# ==========================================================
# Text loading
# ==========================================================

def load_text_from_file(file_path: Path) -> str:
    """Carrega texto de JSON ou TXT."""
    if file_path.suffix.lower() == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("text", "")

    if file_path.suffix.lower() == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError(f"Formato não suportado: {file_path.suffix}")


# ==========================================================
# Date extraction
# ==========================================================

def parse_year(value: str) -> int:
    """Converte 'Present' → ano atual."""
    if value.lower() == "present":
        return CURRENT_YEAR
    return int(value)


def extract_date_ranges(text: str) -> List[Dict[str, int]]:
    """Extrai todos os intervalos de datas encontrados."""
    ranges = []

    for pattern in (YEAR_RANGE_REGEX, MONTH_YEAR_RANGE_REGEX):
        for match in pattern.finditer(text):
            start = parse_year(match.group("start"))
            end = parse_year(match.group("end"))

            if end >= start:
                ranges.append({"start": start, "end": end})

    return ranges


# ==========================================================
# Job title proximity detection
# ==========================================================

def has_job_title_nearby(text_block: str, job_titles: List[str]) -> bool:
    """Verifica se um bloco de texto contém um job title conhecido."""
    text_block = text_block.lower()
    return any(title in text_block for title in job_titles)


def split_into_blocks(text: str) -> List[str]:
    """
    Divide CV em blocos aproximados de experiência.
    Estratégia simples baseada em linhas vazias.
    """
    blocks = re.split(r"\n\s*\n", text)
    return [b.strip() for b in blocks if b.strip()]


# ==========================================================
# Experience inference
# ==========================================================

def merge_overlapping_ranges(ranges: List[Dict[str, int]]) -> List[Dict[str, int]]:
    """Une intervalos sobrepostos."""
    if not ranges:
        return []

    ranges = sorted(ranges, key=lambda x: x["start"])
    merged = [ranges[0]]

    for current in ranges[1:]:
        last = merged[-1]

        if current["start"] <= last["end"]:
            last["end"] = max(last["end"], current["end"])
        else:
            merged.append(current)

    return merged


def compute_total_years(ranges: List[Dict[str, int]]) -> int:
    """Soma anos totais de intervalos não sobrepostos."""
    return sum(r["end"] - r["start"] for r in ranges)


def infer_experience_years(text: str, job_titles: List[str]) -> int:
    """
    Inferência principal de anos de experiência profissional.
    """

    blocks = split_into_blocks(text)
    valid_ranges: List[Dict[str, int]] = []

    for block in blocks:
        if not has_job_title_nearby(block, job_titles):
            continue

        ranges = extract_date_ranges(block)
        valid_ranges.extend(ranges)

    merged = merge_overlapping_ranges(valid_ranges)
    return compute_total_years(merged)


# ==========================================================
# Folder processing
# ==========================================================

def process_folder(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Calcula anos de experiência para todos os CVs numa pasta.
    """

    job_titles = load_job_titles()

    files = list(TEXT_TRAIN_DATA.glob("*.json")) + list(TEXT_TRAIN_DATA.glob("*.txt"))
    if limit:
        files = files[:limit]

    rows = []

    for file_path in files:
        try:
            text = load_text_from_file(file_path)
            years = infer_experience_years(text, job_titles)

            rows.append(
                {
                    "file": file_path.name,
                    "experience_years": years,
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
# CLI
# ==========================================================

if __name__ == "__main__":
    df = process_folder(limit=10)
    print(df)
