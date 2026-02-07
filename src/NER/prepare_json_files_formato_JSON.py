import json
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer
import os
import sys

# Não sei porquê não reconhecia a pasta "utils"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.path_loader import load_paths
paths = load_paths()

# Caminho base do projeto
PROJECT_ROOT = paths["project_root"]

def normalize_text_field(text):
    """
    Normaliza o campo 'text' para uma string segura para o tokenizer.
    Aceita str, list (mesmo aninhada). Caso contrário devolve None.
    """

    def flatten(x):
        for el in x:
            if isinstance(el, list):
                yield from flatten(el)
            else:
                yield el

    if isinstance(text, str):
        text = text.strip()
        return text if text else None

    if isinstance(text, list):
        flat = list(flatten(text))
        parts = [str(t).strip() for t in flat if isinstance(t, (str, int, float)) and str(t).strip()]
        return "\n".join(parts) if parts else None

    # qualquer outro tipo (dict, int sozinho, None, etc.)
    return None



def json_to_bert_ner_features(
    text: str,
    annotations: list,
    tokenizer,
    label2id: dict,
    max_length: int = 512
):
    """
    Converte um CV (texto + spans char) para input BERT NER (BIO).

    Returns:
        dict com input_ids, attention_mask e labels
    """

    # 1) manter apenas spans SKILL
    spans = []
    for ann in annotations:
        if len(ann) < 3:
            continue
        start, end, label = ann
        if str(label).upper().startswith("SKILL"):
            spans.append((int(start), int(end)))

    spans.sort(key=lambda x: x[0])

    # 2) tokenizar com offsets
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length
    )

    offsets = encoding["offset_mapping"]
    labels = ["O"] * len(offsets)

    # 3) alinhar spans → tokens
    for (span_start, span_end) in spans:
        token_idxs = []

        for i, (tok_start, tok_end) in enumerate(offsets):
            # tokens especiais ([CLS], [SEP])
            if tok_start == tok_end:
                continue

            # interseção token ↔ span
            if tok_end <= span_start:
                continue
            if tok_start >= span_end:
                continue

            token_idxs.append(i)

        if not token_idxs:
            continue

        # BIO (com proteção contra sobreposição)
        if labels[token_idxs[0]] == "O":
            labels[token_idxs[0]] = "B-SKILL"

        for idx in token_idxs[1:]:
            if labels[idx] == "O":
                labels[idx] = "I-SKILL"

    # 4) converter para ids + ignorar tokens especiais
    label_ids = []
    for (tok_start, tok_end), lab in zip(offsets, labels):
        if tok_start == tok_end:
            label_ids.append(-100)
        else:
            label_ids.append(label2id[lab])

    encoding.pop("offset_mapping")
    encoding["labels"] = label_ids

    return encoding

def build_bert_ner_dataset(
    input_path,
    tokenizer,
    label2id: dict,
    output_dir=None,
    max_length: int = 512
):
    input_path = Path(input_path).resolve()

    records = []
    invalid_texts = 0
    no_annotations = 0

    for json_file in iter_json_files(input_path):
        data = json.loads(json_file.read_text(encoding="utf-8"))

        annotations = data.get("annotations", [])
        if not annotations:
            no_annotations += 1
            continue

        raw_text = data.get("text")
        text = normalize_text_field(raw_text)

        if text is None:
            invalid_texts += 1
            continue

        try:
            features = json_to_bert_ner_features(
                text=text,
                annotations=annotations,
                tokenizer=tokenizer,
                label2id=label2id,
                max_length=max_length
            )
        except Exception as e:
            print(f"❌ Erro ao tokenizar: {json_file.name}")
            print(e)
            invalid_texts += 1
            continue

        records.append(features)

    dataset = Dataset.from_list(records)

    print(f"✅ Dataset criado com {len(dataset)} CVs anotados")
    print(f"⚠️ Sem annotations: {no_annotations}")
    print(f"⚠️ Texto inválido / erro: {invalid_texts}")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "process_Json"
        dataset.to_json(output_file)
        
        print(f"💾 Dataset guardado em: {output_file.resolve()}")
    return dataset



def iter_json_files(input_path: Path):
    if input_path.is_file() and input_path.suffix == ".json":
        yield input_path
    elif input_path.is_dir():
        yield from input_path.rglob("*.json")
    else:
        raise ValueError(
            f"Input inválido: {input_path}\n"
            f"Existe: {input_path.exists()}\n"
            f"É ficheiro: {input_path.is_file()}\n"
            f"É pasta: {input_path.is_dir()}"
        )




def debug_tokens(sample, tokenizer, id2label):
    tokens = tokenizer.convert_ids_to_tokens(sample["input_ids"])
    for t, l in zip(tokens, sample["labels"]):
        if l == -100:
            continue
        print(f"{t:15} -> {id2label[l]}")



tokenizer = AutoTokenizer.from_pretrained(
    PROJECT_ROOT / "models" / "BERT" / "bert-base-cased",
    use_fast=True
)

label2id = {
    "O": 0,
    "B-SKILL": 1,
    "I-SKILL": 2
}

dataset = build_bert_ner_dataset(
    input_path=PROJECT_ROOT / "data" / "raw" /"Json",  # ou ficheiro .json
    tokenizer=tokenizer,
    label2id=label2id,
    output_dir=PROJECT_ROOT / "data" / "process_Json"
)

print(dataset)
print(dataset[0].keys())


id2label = {v: k for k, v in label2id.items()}
debug_tokens(dataset[0], tokenizer, id2label)
