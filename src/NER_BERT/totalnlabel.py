"""Resultado deste programa:

🔍 Ficheiros .json encontrados (recursivo): 5029

📊 RELATÓRIO GLOBAL NER
✔️ Distribuição de num_labels (BIO):
  - 3: 4971 ficheiros
🏷️ Entidades únicas globais (1): ['SKILL']
🔥 Top 10 entidades mais frequentes (nº de ficheiros onde aparecem):
  - SKILL: 4971
🔝 Máximo de tipos de entidade num ficheiro: 1
   Exemplo: cv (1)_annotated.json -> ['SKILL']
⚠️ Ficheiros sem annotations: 58 (ex.: cv (1038)_annotated.json)"""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

print("Torch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())

from utils.path_loader import load_paths
paths = load_paths()

# Caminho base do projeto
PROJECT_ROOT = paths["project_root"]

import json
from pathlib import Path
from collections import Counter, defaultdict

def analyze_ner_labels_in_folder(folder_path: str | Path):
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"O caminho não existe: {folder.resolve()}")

    json_files = list(folder.rglob("*.json"))  # <-- recursivo
    print(f"🔍 Ficheiros .json encontrados (recursivo): {len(json_files)}")

    if not json_files:
        print("⚠️ A pasta não tem ficheiros .json")
        return None

    # por ficheiro
    file_entities = {}
    file_num_labels = {}

    # global
    global_entity_counter = Counter()
    num_labels_counter = Counter()
    files_with_errors = []
    files_with_no_annotations = []

    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))

            anns = data.get("annotations", [])
            if not anns:
                files_with_no_annotations.append(jf)
                continue

            # entidades do tipo "SKILL: python" -> "SKILL"
            ents = set()
            for ann in anns:
                if len(ann) < 3:
                    continue
                raw = str(ann[2])
                ent = raw.split(":", 1)[0].strip().upper()  # normaliza
                if ent:
                    ents.add(ent)

            file_entities[jf] = ents
            global_entity_counter.update(ents)

            # BIO: O + B/I por entidade
            num_labels = 1 + 2 * len(ents)
            file_num_labels[jf] = num_labels
            num_labels_counter[num_labels] += 1

        except Exception as e:
            files_with_errors.append((jf, str(e)))

    print("\n📊 RELATÓRIO GLOBAL NER\n")

    # 1) Distribuição de num_labels
    if num_labels_counter:
        print("✔️ Distribuição de num_labels (BIO):")
        for k in sorted(num_labels_counter):
            print(f"  - {k}: {num_labels_counter[k]} ficheiros")
    else:
        print("⚠️ Não foi possível calcular num_labels (talvez sem annotations).")

    # 2) Labels globais encontradas
    all_labels = sorted(global_entity_counter.keys())
    print(f"\n🏷️ Entidades únicas globais ({len(all_labels)}): {all_labels}")

    # 3) Top labels mais comuns (por ficheiro, não por ocorrência)
    if global_entity_counter:
        print("\n🔥 Top 10 entidades mais frequentes (nº de ficheiros onde aparecem):")
        for lab, c in global_entity_counter.most_common(10):
            print(f"  - {lab}: {c}")

    # 4) Máximo de entidades num ficheiro
    if file_entities:
        jf_max = max(file_entities, key=lambda p: len(file_entities[p]))
        print(f"\n🔝 Máximo de tipos de entidade num ficheiro: {len(file_entities[jf_max])}")
        print(f"   Exemplo: {jf_max.name} -> {sorted(file_entities[jf_max])}")

    # 5) Ficheiros sem annotations / com erro
    if files_with_no_annotations:
        print(f"\n⚠️ Ficheiros sem annotations: {len(files_with_no_annotations)} (ex.: {files_with_no_annotations[0].name})")
    if files_with_errors:
        print(f"\n❌ Ficheiros com erro: {len(files_with_errors)}")
        print(f"   Ex.: {files_with_errors[0][0].name} -> {files_with_errors[0][1]}")

    return {
        "file_entities": {p.name: sorted(v) for p, v in file_entities.items()},
        "num_labels_distribution": dict(num_labels_counter),
        "global_entities": all_labels,
        "top_entities": global_entity_counter.most_common(10),
        "no_annotations": [p.name for p in files_with_no_annotations],
        "errors": [(p.name, err) for p, err in files_with_errors],
    }


if __name__ == "__main__":
    path = PROJECT_ROOT / "data" /"ResumesJsonAnnotated"
    report = analyze_ner_labels_in_folder(path)