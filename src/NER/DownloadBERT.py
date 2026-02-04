from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification

from utils.path_loader import load_paths
paths = load_paths()

# Caminho base do projeto
PROJECT_ROOT = paths["project_root"]

def download_bert_model(
    model_name: str = "bert-base-cased",
    output_dir: str = "models/bert-base-cased",
    num_labels: int = 3
):
    """
    Faz download do tokenizer e do modelo BERT para um diretório específico.

    Args:
        model_name (str): Nome do modelo HuggingFace (ex: bert-base-cased)
        output_dir (str): Path onde o modelo será guardado
        num_labels (int): Número de labels para TokenClassification (NER)
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 A fazer download de '{model_name}'...")
    print(f"📁 Diretório: {output_dir.resolve()}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)

    # Modelo (NER)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    model.save_pretrained(output_dir)

    print("✅ Download concluído com sucesso!")


if __name__ == "__main__":
    download_bert_model(
        model_name="bert-base-cased",
        output_dir=PROJECT_ROOT / "models" /"BERT" /"bert-base-cased",
        num_labels=3
    )