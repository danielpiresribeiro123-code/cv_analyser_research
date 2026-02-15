from datasets import load_dataset
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from PIL import Image
from sklearn.preprocessing import LabelEncoder


# =========================================================
# 1. Load dataset
# =========================================================
dataset = load_dataset(
    "json",
    data_files="layoutlmv3_dataset.jsonl",
)["train"]

print(dataset[0])


# =========================================================
# 2. Encode labels → integers
# =========================================================
label_encoder = LabelEncoder()

encoded_labels = label_encoder.fit_transform(list(dataset["label"]))
dataset = dataset.add_column("label_id", encoded_labels.tolist())

num_labels = len(label_encoder.classes_)
print("Classes:", list(label_encoder.classes_))


# =========================================================
# 3. Train / validation split
# =========================================================
dataset = dataset.train_test_split(test_size=0.1)


# =========================================================
# 4. Load processor + model
# =========================================================
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")

model = LayoutLMv3ForSequenceClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=num_labels,
)


# =========================================================
# 5. Encoding function
# =========================================================
def encode_examples(examples):
    images = [Image.open(p).convert("RGB") for p in examples["image_file"]]

    encoding = processor(
        images,
        examples["words"],
        boxes=examples["boxes"],
        padding="max_length",
        truncation=True,
    )

    encoding["labels"] = examples["label_id"]
    return encoding


dataset = dataset.map(encode_examples, batched=True)


# =========================================================
# 6. Training setup
# =========================================================
training_args = TrainingArguments(
    output_dir="layoutlmv3_cv_sections",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    load_best_model_at_end=True,
    report_to="none",
)


# =========================================================
# 7. Trainer
# =========================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)


# =========================================================
# 8. Train
# =========================================================
trainer.train()


# =========================================================
# 9. Save model
# =========================================================
trainer.save_model("layoutlmv3_cv_sections_model")
processor.save_pretrained("layoutlmv3_cv_sections_model")
