import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification
from torch.utils.data import DataLoader

def gerar_matriz_confusao():
    # Usar Path().resolve() para garantir um caminho absoluto válido no Windows
    model_path = str(Path("models/BERT/modelo_final_ner_lowercase").resolve())
    data_path = str(Path("data/process_Json/dataset.json").resolve())
    
    if not Path(model_path).exists():
        print(f"❌ Erro: Pasta do modelo não encontrada em: {model_path}")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=True)
    model.eval()
    
    dataset = load_dataset('json', data_files=data_path, split='train')
    split = dataset.train_test_split(test_size=0.2, seed=42)
    test_data = split['test']
    
    label_list = ["O", "B-SKILL", "I-SKILL"]
    collator = DataCollatorForTokenClassification(tokenizer)
    dataloader = DataLoader(test_data, batch_size=8, collate_fn=collator)
    
    y_true = []
    y_pred = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for batch in tqdm(dataloader, desc="A avaliar"):
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
        for i in range(labels.shape[0]):
            mask = labels[i] != -100
            y_true.extend(labels[i][mask].cpu().numpy())
            y_pred.extend(predictions[i][mask].cpu().numpy())
            
    cm = confusion_matrix(y_true, y_pred)
    
    fp = cm[0, 1] + cm[0, 2] # Erro Tipo 1
    fn = cm[1, 0] + cm[2, 0] # Erro Tipo 2
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_list, yticklabels=label_list)
    plt.title('Matriz de Confusao (Nivel de Token)')
    plt.ylabel('Realidade')
    plt.xlabel('Previsao')
    
    plt.figtext(0.15, 0.02, f"Erro Tipo 1 (Falsos Positivos): {fp}", fontsize=11, color='red')
    plt.figtext(0.15, -0.02, f"Erro Tipo 2 (Falsos Negativos): {fn}", fontsize=11, color='orange')
    
    plt.tight_layout()
    plt.savefig("matriz_confusao_final.png", bbox_inches='tight')
    plt.close()
    
    print(f"✅ Sucesso! Matriz guardada em matriz_confusao_final.png")

if __name__ == "__main__":
    gerar_matriz_confusao()