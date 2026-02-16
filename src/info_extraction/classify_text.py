from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "ahmedheakl/bert-resume-classification"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

id2label = model.config.id2label


def classify_text(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    score, pred_id = torch.max(probs, dim=-1)

    return {
        "label": id2label[pred_id.item()],
        "score": float(score.item())
    }
