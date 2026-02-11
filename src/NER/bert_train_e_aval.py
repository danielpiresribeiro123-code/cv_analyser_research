import os
<<<<<<< HEAD
import torch
import numpy as np
import evaluate
=======
import sys
import torch
import numpy as np
import evaluate
from pathlib import Path
>>>>>>> cd3a734ce14f1af11a3974521153526cf036b158
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification,
    AutoTokenizer
)

<<<<<<< HEAD
# Função para calcular o sucesso do modelo durante o treino
def compute_metrics(p, label_list):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove os tokens especiais (-100) para calcular métricas reais
=======
label2id = {"O": 0, "B-SKILL": 1, "I-SKILL": 2}
id2label = {v: k for k, v in label2id.items()}
label_list = ["O", "B-SKILL", "I-SKILL"]

full_dataset = load_dataset('json', data_files='data/process_Json/dataset.json', split='train')
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

split_dataset = full_dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

>>>>>>> cd3a734ce14f1af11a3974521153526cf036b158
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

<<<<<<< HEAD
    metric = evaluate.load("seqeval")
=======
>>>>>>> cd3a734ce14f1af11a3974521153526cf036b158
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

<<<<<<< HEAD
def main():
    # Verifica se a placa gráfica está pronta
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Treino iniciado em: {device.upper()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    label_list = ["O", "B-SKILL", "I-SKILL"]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}

    # 1. Carregar o dataset gerado pelo Script A
    # O Script A já fez o trabalho de colocar tudo em minúsculas
    dataset = load_dataset('json', data_files='data/process_Json/dataset.json', split='train')
    
    # Divide em 80% treino e 20% teste
    split_dataset = dataset.train_test_split(test_size=0.2)
    train_data = split_dataset["train"]
    eval_data = split_dataset["test"]

    # 2. Carregar Tokenizer e Modelo (do_lower_case=True é essencial aqui também)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", do_lower_case=True)
    model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-cased",
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    # 3. Configurações de Treino Otimizadas para a tua GTX 1050 Ti (4GB VRAM)
    training_args = TrainingArguments(
        output_dir="data/processed/results_ner_lowercase",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8, 
        per_device_eval_batch_size=8,
        num_train_epochs=5,            
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(), 
        gradient_checkpointing=True,    
        dataloader_num_workers=0,       
        report_to="none"
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # 4. Inicializar o Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, label_list),
    )

    # 5. Começar o Treino
    print("✨ A ajustar os pesos do modelo...")
    trainer.train()

    # 6. Guardar o resultado final
    output_dir = "./modelo_final_ner_lowercase"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Sucesso! O teu modelo 'lowercase' está guardado em: {output_dir}")

if __name__ == '__main__':
    main()
=======
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="./results_bert_ner",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("🚀 A iniciar o treino...")
trainer.train()

model.save_pretrained("./modelo_final_ner")
tokenizer.save_pretrained("./modelo_final_ner")
print("✅ Sucesso!")
>>>>>>> cd3a734ce14f1af11a3974521153526cf036b158
