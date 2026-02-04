import os
import json
from transformers import AutoTokenizer
from utils.path_loader import load_paths
paths = load_paths()

# Caminho base do projeto
PROJECT_ROOT = paths["project_root"]
input_path = PROJECT_ROOT / 'data' / 'raw' / "Json"
output_path = PROJECT_ROOT / 'data' / 'processed' / "output"
model_checkpoint = "bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def process_batch():
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for filename in os.listdir(input_path):
        if filename.endswith('.json'):
            with open(os.path.join(input_path, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            text = data.get('text', '')
            annotations = data.get('annotations', [])
            
            char_labels = ["0"] * len(text)
            for start, end, _ in annotations:
                if start < len(char_labels):
                    char_labels[start] = "B-Skill"
                    for i in range(start + 1, min(end, len(char_labels))):
                        char_labels[i] = "I-Skill"

            tokenized = tokenizer(text, truncation=True, return_offsets_mapping=True)
            labels = []
            
            for start, end in tokenized['offset_mapping']:
                if start == 0 and end == 0:
                    labels.append(-100)
                else:
                    labels.append(char_labels[start])

            output_data = {
                "tokens": tokenizer.convert_ids_to_tokens(tokenized['input_ids']),
                "input_ids": tokenized['input_ids'],
                "labels": labels
            }

            with open(os.path.join(output_path, f"processed_{filename}"), 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    process_batch()