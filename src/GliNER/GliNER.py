import json
import os
import random
from gliner import GLiNER

PATH_JSON_DATA = r"C:\\Users\\duart\\OneDrive\\Ambiente de Trabalho\\Faculdade\\Samsung\\Capstone\\data\\processed_GliNER\\gliner_train_data.json"
PASTA_MODELO_FINAL = r"C:\\Users\\duart\\OneDrive\\Ambiente de Trabalho\\Faculdade\\Samsung\\Capstone\\models\\GliNER\\"

def convert_to_token_level(data):
    converted_data = []
    for entry in data:
        text = entry["text"]
        ner_spans = entry["ner"]
        
        # O GLiNER moderno prefere texto já tokenizado
        tokens = text.split()
        
        # Criar mapeamento de caracteres para tokens
        # Nota: Esta é uma aproximação simples. Se o seu dataset for complexo,
        # use o tokenizer do modelo para maior precisão.
        new_ner = []
        words = text.split()
        
        for start_char, end_char, label in ner_spans:
            # Encontrar os índices dos tokens correspondentes
            current_char = 0
            start_token = -1
            end_token = -1
            
            for i, word in enumerate(words):
                word_start = text.find(word, current_char)
                word_end = word_start + len(word)
                current_char = word_end
                
                if word_start <= start_char < word_end:
                    start_token = i
                if word_start < end_char <= word_end:
                    end_token = i
                    break
            
            if start_token != -1 and end_token != -1:
                new_ner.append([start_token, end_token, label])
        
        converted_data.append({
            "tokenized_text": words,
            "ner": new_ner
        })
    return converted_data

def run_pipeline():
    if not os.path.exists(PASTA_MODELO_FINAL):
        os.makedirs(PASTA_MODELO_FINAL, exist_ok=True)

    with open(PATH_JSON_DATA, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Converte de nível de caractere para nível de token
    formatted_data = convert_to_token_level(data)
    
    random.seed(42)
    random.shuffle(formatted_data)
    
    n = len(formatted_data)
    train_dataset = formatted_data[:int(n * 0.8)]
    val_dataset = formatted_data[int(n * 0.8):int(n * 0.9)]

    model = GLiNER.from_pretrained("gliner-community/gliner_small-v2.5")

    model.train_model(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        output_dir=PASTA_MODELO_FINAL,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        learning_rate=5e-5
    )

    model.save_pretrained(PASTA_MODELO_FINAL)

if __name__ == "__main__":
    run_pipeline()