from ultralytics import YOLO # type: ignore
from pathlib import Path

from utils.path_loader import load_paths
paths = load_paths()

# Caminho base do projeto
PROJECT_ROOT = paths["project_root"]



def train_tables_model():
    """
    Treina um modelo YOLOv8s para detetar apenas tabelas (class: table)
    em CVs / Resumes.
    """

    # Dataset (data.yaml)
    DATASET_YAML = Path(
        r"C:\Users\danie\Desktop\My_Things\Hobbies\VSCODE\Data\CVs_Resumes_Datasets"
        r"\CVs_detection_layout_yolo\Resume.v1i.yolov8_roboflow_tables\data.yaml"
    )

    # Modelo base YOLO
    BASE_MODEL = PROJECT_ROOT / "models" / "downloaded" / "yolov8s.pt"

    # Onde vão ficar os resultados
    SAVE_DIR = PROJECT_ROOT / "src" /"train_yolo" / "yolov8s-tables"

    # Garante que a pasta existe
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("📁 Dataset:", DATASET_YAML)
    print("📦 Modelo base:", BASE_MODEL)
    print("💾 Resultados em:", SAVE_DIR)

    # Carrega o modelo base
    model = YOLO(BASE_MODEL)

    # Treino
    model.train(
        data=str(DATASET_YAML),
        epochs=50,
        imgsz=512,
        batch=4,
        name="yolov8s-tables_v1",
        project=str(SAVE_DIR),
        device="cpu",        # usa GPU se for 0, usa CPU se for 'cpu'
        workers=2,
        patience=8,    # early stopping
        optimizer="AdamW",
        cos_lr=True
    )


if __name__ == "__main__":
    train_tables_model()
