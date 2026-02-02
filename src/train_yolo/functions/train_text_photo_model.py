from ultralytics import YOLO  # type: ignore
from pathlib import Path

from utils.path_loader import load_paths
paths = load_paths()

# Caminho base do projeto
PROJECT_ROOT = paths["project_root"]

# Caminho REAL do data.yaml (dataset)
DATASET_YAML = Path("C:/Users/danie/Desktop/labellmg/Yolo-CV-Datasets/data-text-photo.yaml")

def train_text_photo_model():
    """
    Treina um modelo YOLO para detetar:
    - text_block
    - photo
    em CVs / Resumes.
    """

    BASE_MODEL = PROJECT_ROOT / "models" / "downloaded" / "yolo11s.pt"

    SAVE_DIR = PROJECT_ROOT / "src" / "train_yolo" / "yolo11s-text-photo"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # 🔎 Validações rápidas (muito recomendadas)
    if not DATASET_YAML.exists():
        raise FileNotFoundError(f"data.yaml não encontrado: {DATASET_YAML}")

    if not BASE_MODEL.exists():
        raise FileNotFoundError(f"Modelo base não encontrado: {BASE_MODEL}")

    print("📄 Dataset YAML:", DATASET_YAML)
    print("📦 Modelo base:", BASE_MODEL)
    print("💾 Resultados em:", SAVE_DIR)

    model = YOLO(BASE_MODEL)

    model.train(
        data=str(DATASET_YAML),   # ← garante compatibilidade
        epochs=5,
        imgsz=640,
        batch=4,
        device="cpu",                 # usa "cpu" se não tiveres GPU (0 para gpu)
        workers=4,
        seed=42,
        project=str(SAVE_DIR),
        name="train",
        exist_ok=True)


if __name__ == "__main__":
    train_text_photo_model()
