import json
import torch
import numpy as np
import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification
)

# 1. Configurações de Labels e Modelo
model_nm = "bert-base-cased"
# Adiciona aqui todas as variantes de labels que aparecem no teu JSON
label_list = ["O", "B-SKILL", "I-SKILL"] 
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for i, l in enumerate(label_list)}

# Mapeamento para converter as strings do teu JSON em números
# Se o teu JSON tem "SKILL: Building", ele será mapeado para 1 (B-SKILL)
tag_to_id_map = {
    "O": 0,
    "SKILL: Building": 1,
    "B-SKILL": 1,
    "I-SKILL": 2
}

# 2. Carregar Tokenizer e Modelo
tokenizer = AutoTokenizer.from_pretrained(model_nm)
model = AutoModelForTokenClassification.from_pretrained(
    model_nm, 
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# 3. Carregamento e Tratamento Manual do JSON
path_json = 'C:\\Users\\duart\\OneDrive\\Ambiente de Trabalho\\Faculdade\\Samsung\\CapStone\\Repository\\CapStone\\Data\\raw\\ResumesJsonAnnotated\\cv (1)_annotated.json'

with open(path_json, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Se o JSON tiver uma chave "data", usa: data = data['data']
# Se for uma lista direta, ignora.

tokens_list = []
labels_list = []

for item in data:
    tokens_list.append(item['tokens'])
    # Converte as strings das anotações em números usando o nosso mapa
    # Se não encontrar a tag, assume 0 ("O")
    numeric_tags = [tag_to_id_map.get(tag, 0) for tag in item['annotations']]
    labels_list.append(numeric_tags)

# Criar o Dataset formatado
raw_datasets = DatasetDict({
    "train": Dataset.from_dict({"tokens": tokens_list, "ner_tags": labels_list})
})
# Divide em 80% treino e 20% teste
raw_datasets = raw_datasets['train'].train_test_split(test_size=0.2)

# 4. Função de Alinhamento (Lida com sub-tokens do BERT)
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True)

# 5. Métricas
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# 6. Treino
training_args = TrainingArguments(
    output_dir="./bert-skill-results",
    learning_rate=2e-5,
    per_device_train_batch_size=8, # Reduzido para poupar memória
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none" # Evita avisos de falta de login em plataformas de log
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics
)

# 7. Execução
print("A iniciar o fine-tuning...")
trainer.train()

# 8. Guardar resultados
model.save_pretrained("./meu_modelo_skills")
tokenizer.save_pretrained("./meu_modelo_skills")
print("Modelo guardado em ./meu_modelo_skills")