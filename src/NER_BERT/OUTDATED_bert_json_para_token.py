import json
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer
import os

def normalize_text_field(text):
    def flatten(x):
        if isinstance(x, list):
            for el in x:
                yield from flatten(el)
        else:
            yield x
    if text is None: return None
    if isinstance(text, (str, int, float)):
        s = str(text).strip()
        return s if s else None
    if isinstance(text, list):
        flat = list(flatten(text))
        parts = [str(t).strip() for t in flat if t is not None and str(t).strip()]
        return "\n".join(parts) if parts else None
    return None

def json_to_bert_ner_features(text, annotations, tokenizer, label2id, max_length=512):
    spans = []
    for ann in annotations:
        if len(ann) < 3: continue
        start, end, label = ann
        # Mudança: Verifica se "SKILL" está contido na label (mais flexível)
        if "SKILL" in str(label).upper():
            spans.append((int(start), int(end)))
    
    spans.sort(key=lambda x: x[0])

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )

    offsets = encoding["offset_mapping"]
    labels = ["O"] * len(offsets)

    for (span_start, span_end) in spans:
        token_idxs = [
            i for i, (ts, te) in enumerate(offsets)
            if ts != te and not (te <= span_start or ts >= span_end)
        ]

        if token_idxs:
            if labels[token_idxs[0]] == "O":
                labels[token_idxs[0]] = "B-SKILL"
            for idx in token_idxs[1:]:
                if labels[idx] == "O":
                    labels[idx] = "I-SKILL"

    label_ids = [label2id[lab] if ts != te else -100 for (ts, te), lab in zip(offsets, labels)]
    encoding.pop("offset_mapping")
    encoding["labels"] = label_ids
    return encoding

def build_bert_ner_dataset(input_path, tokenizer, label2id, output_file):
    input_path = Path(input_path).resolve()
    print(f"🔍 A procurar ficheiros em: {input_path}")
    
    records = []
    files = list(input_path.rglob("*.json")) if input_path.is_dir() else ([input_path] if input_path.exists() else [])
    
    print(f"📄 Ficheiros encontrados: {len(files)}")

    for json_file in files:
        try:
            content = json_file.read_text(encoding="utf-8").strip()
            if not content: continue
            
            data = json.loads(content)
            text = normalize_text_field(data.get("text"))
            annotations = data.get("annotations", [])

            if text is None or not annotations:
                continue

            features = json_to_bert_ner_features(text, annotations, tokenizer, label2id)
            records.append(features)
        except Exception as e:
            print(f"❌ Erro em {json_file.name}: {e}")

    if not records:
        print("⚠️ Nada processado. Verifica se as labels nos JSONs são 'SKILL' e se o texto não está vazio.")
        return None

    dataset = Dataset.from_list(records)
    dataset.to_json(output_file)
    print(f"✅ Sucesso! {len(records)} CVs processados.")
    print(f"💾 Guardado em: {Path(output_file).resolve()}")
    return dataset

# --- EXECUÇÃO ---
# Certifica-te que este caminho está correto para a tua máquina
PASTA_DATA = Path("./data/raw/Json") 

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
label2id = {"O": 0, "B-SKILL": 1, "I-SKILL": 2}

build_bert_ner_dataset(
    input_path=PASTA_DATA,
    tokenizer=tokenizer,
    label2id=label2id,
    output_file="data/processed_Json/dataset_processado.json"
)