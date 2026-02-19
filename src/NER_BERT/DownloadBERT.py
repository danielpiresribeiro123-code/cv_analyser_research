from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig

from utils.path_loader import load_paths
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification
paths = load_paths()

# Caminho base do projeto
PROJECT_ROOT = paths["project_root"]

def download_bert_model(
    model_name: str,
    output_dir: Path,
    label2id: dict
):
    """
    Download do tokenizer e modelo BERT preparado para NER.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    id2label = {v: k for k, v in label2id.items()}

    print(f"Download do modelo: {model_name}")
    print(f"A guardar em: {output_dir.resolve()}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label
    )
    model.save_pretrained(output_dir)

    print("Modelo BERT NER pronto.")


label2id = {
    "O": 0,
    "B-SKILL": 1,
    "I-SKILL": 2
}


download_bert_model(model_name="bert-base-cased",output_dir=PROJECT_ROOT / "models" / "BERT" / "bert-base-cased",label2id=label2id)

cfg = AutoConfig.from_pretrained(PROJECT_ROOT / "models" / "BERT" / "bert-base-cased")
print("\nConfirmar que o modelo ficou bem configurado:")
print(cfg.id2label)