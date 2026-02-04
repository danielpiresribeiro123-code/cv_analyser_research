import json
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer


from utils.path_loader import load_paths
paths = load_paths()

# Caminho base do projeto
PROJECT_ROOT = paths["project_root"]


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
    json_folder: Path,
    tokenizer,
    label2id: dict,
    max_length: int = 512
):
    """
    Converte todos os JSON anotados num Dataset HuggingFace.
    Ignora ficheiros sem annotations.
    """

    records = []

    for json_file in json_folder.rglob("*.json"):
        data = json.loads(json_file.read_text(encoding="utf-8"))

        annotations = data.get("annotations", [])
        if not annotations:
            continue  # ignora os 58 ficheiros vazios

        text = data["text"]

        features = json_to_bert_ner_features(
            text=text,
            annotations=annotations,
            tokenizer=tokenizer,
            label2id=label2id,
            max_length=max_length
        )

        records.append(features)

    print(f"✅ Dataset criado com {len(records)} CVs anotados")
    return Dataset.from_list(records)



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
    json_folder=PROJECT_ROOT / "data" / "ResumesJsonAnnotated",
    tokenizer=tokenizer,
    label2id=label2id
)

print(dataset)
print(dataset[0].keys())


id2label = {v: k for k, v in label2id.items()}
debug_tokens(dataset[0], tokenizer, id2label)
