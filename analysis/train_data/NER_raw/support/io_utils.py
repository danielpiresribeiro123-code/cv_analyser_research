from pathlib import Path
import json
from typing import List, Dict


def load_raw_ner_json(path: Path) -> Dict:
    """Carrega um único ficheiro JSON NER."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_raw_ner(folder: Path) -> List[Dict]:
    """Carrega todos os JSON do dataset raw."""
    data = []

    for json_file in sorted(folder.glob("*.json")):
        data.append(load_raw_ner_json(json_file))

    return data
