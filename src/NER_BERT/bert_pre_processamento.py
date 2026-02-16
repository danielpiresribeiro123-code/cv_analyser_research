import json
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer
import os
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

DATA_RAW = Path("data/raw/Json")
DATA_PROCESS = Path("data/process_Json")

# --- CONFIGURAÇÃO DA WHITELIST (DOMÍNIOS RELEVANTES) ---
DOMINIOS = {
    "programacao": ["python", "java", "c++", "c", "sql", "pyspark", "javascript", "scala", "r"],
    "ia_ml": ["machine learning", "deep learning", "nlp", "computer vision", "algorithms", "neural networks", "yolo", "cnn", "ocr"],
    "ferramentas": ["kubeflow", "docker", "kubernetes", "aws", "azure", "git", "tableau", "spark", "hadoop"],
    "analise": ["data analysis", "object detection", "pattern recognition", "data science", "statistics"]
}

# Criar um conjunto único com todas as palavras permitidas
WHITELIST = set([item for sublist in DOMINIOS.values() for item in sublist])

def normalize_text_field(text):
    if text is None: return None
    if isinstance(text, str):
        cleaned = text.strip().lower()
        return cleaned if cleaned else None
    if isinstance(text, list):
        parts = [str(t).strip().lower() for t in text if t is not None]
        return "\n".join(parts) if parts else None
    return None

def json_to_bert_ner_features(text, annotations, tokenizer, label2id, max_length=512):
    if not isinstance(text, str): return None

    spans = []
    for ann in annotations:
        if len(ann) < 3: continue
        start, end, label = ann
        
        # Extrair o texto exato da skill anotada
        skill_text = text[int(start):int(end)].strip()
        
        # LÓGICA DE FILTRAGEM POR DOMÍNIO
        if str(label).upper().startswith("SKILL"):
            # Só aceita se a skill estiver na nossa lista de domínios relevantes
            if skill_text in WHITELIST:
                spans.append((int(start), int(end)))
            else:
                # Opcional: print para ver o que está a ser descartado (ex: 'team', 'science')
                # print(f"🚫 Descartado (fora do domínio): {skill_text}")
                pass

    spans.sort(key=lambda x: x[0])

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length
    )

    offsets = encoding["offset_mapping"]
    labels = ["O"] * len(offsets)

    for (span_start, span_end) in spans:
        token_idxs = [i for i, (ts, te) in enumerate(offsets) if ts != te and te > span_start and ts < span_end]
        if not token_idxs: continue
        labels[token_idxs[0]] = "B-SKILL"
        for idx in token_idxs[1:]:
            labels[idx] = "I-SKILL"

    label_ids = [label2id[lab] if ts != te else -100 for (ts, te), lab in zip(offsets, labels)]
    
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": label_ids
    }

def build_dataset():
    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", do_lower_case=True, local_files_only=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", do_lower_case=True)
        
    label2id = {"O": 0, "B-SKILL": 1, "I-SKILL": 2}
    records = []

    print(f"🔍 Filtragem ativa para {len(WHITELIST)} competências específicas.")

    for json_file in DATA_RAW.rglob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            raw_text = data.get("text") or data.get("content")
            text = normalize_text_field(raw_text)
            if text is None: continue
            
            features = json_to_bert_ner_features(text, data.get("annotations", []), tokenizer, label2id)
            if features:
                records.append(features)
        except Exception as e:
            print(f"❌ Erro em {json_file.name}: {e}")

    if not records:
        print("❌ Nenhum dado restou após a filtragem.")
        return

    dataset = Dataset.from_list(records)
    DATA_PROCESS.mkdir(parents=True, exist_ok=True)
    dataset.to_json(DATA_PROCESS / "dataset.json")
    print(f"✅ Sucesso! Dataset filtrado criado com {len(dataset)} exemplos.")

if __name__ == '__main__':
    build_dataset()