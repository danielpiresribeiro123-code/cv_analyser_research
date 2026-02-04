import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

print("Torch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=3)

print("Modelo carregado com sucesso")