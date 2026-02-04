from datasets import load_from_disk
from utils.path_loader import load_paths

paths = load_paths()
PROJECT_ROOT = paths["project_root"]

dataset = load_from_disk(
    PROJECT_ROOT / "data" / "bert_ner_dataset"
)

dataset = dataset.train_test_split(
    test_size=0.1,
    seed=42
)

train_ds = dataset["train"]
eval_ds  = dataset["test"]

print(len(train_ds), len(eval_ds))


train_path = PROJECT_ROOT / "data" / "bert_ner_dataset_train"
eval_path  = PROJECT_ROOT / "data" / "bert_ner_dataset_eval"

train_ds.save_to_disk(train_path)
eval_ds.save_to_disk(eval_path)

print(f"✅ Train dataset guardado em: {train_path}")
print(f"✅ Eval dataset guardado em:  {eval_path}")
