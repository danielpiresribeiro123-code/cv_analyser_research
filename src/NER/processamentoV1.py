import os
import json
from transformers import AutoTokenizer
from utils.path_loader import load_paths
paths = load_paths()
PROJECT_ROOT = paths["project_root"]


input_path = PROJECT_ROOT / 'data' / 'raw' / "Json"
output_path = PROJECT_ROOT / 'data' / 'processed' / "output"
model_checkpoint = "bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def process_batch():
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Pasta de output criada: {output_path}")

    if not os.path.exists(input_path):
        print(f"ERRO: A pasta de entrada não existe: {input_path}")
        return

    files = [f for f in os.listdir(input_path) if f.lower().endswith('.json')]
    print(f"A processar {len(files)} ficheiros...")

    for filename in files:
        try:
            with open(os.path.join(input_path, filename), 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # Lógica para lidar com o teu formato específico (fullContent)
            if 'fullContent' in raw_data:
                data = raw_data['fullContent']
            else:
                data = raw_data
            
            text = data.get('text', '')
            annotations = data.get('annotations', [])

            if not text:
                print(f"Aviso: Ficheiro {filename} não tem texto. A saltar...")
                continue

            # Mapeamento BIO
            char_labels = ["0"] * len(text)
            for start, end, _ in annotations:
                if start < len(char_labels):
                    char_labels[start] = "B-Skill"
                    for i in range(start + 1, min(end, len(char_labels))):
                        char_labels[i] = "I-Skill"

            # Tokenização
            tokenized = tokenizer(text, truncation=True, return_offsets_mapping=True)
            labels = []
            
            for start, end in tokenized['offset_mapping']:
                if start == 0 and end == 0:
                    labels.append(-100)
                else:
                    labels.append(char_labels[start])

            # Output final
            output_data = {
                "tokens": tokenizer.convert_ids_to_tokens(tokenized['input_ids']),
                "input_ids": tokenized['input_ids'],
                "labels": labels
            }

            output_file = os.path.join(output_path, f"processed_{filename}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4)
            
            print(f"Sucesso: {filename} -> processed_{filename}")

        except Exception as e:
            print(f"Falha ao processar {filename}: {str(e)}")

if __name__ == "__main__":
    process_batch()