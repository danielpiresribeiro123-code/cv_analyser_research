import random

import time
from ultralytics import YOLO # type: ignore

import pandas as pd

from pathlib import Path
from utils.path_loader import load_paths

paths = load_paths()
PROJECT_ROOT = paths["project_root"]
MODELS_DATA_PATH = PROJECT_ROOT / "src" / "train_yolo"


TEXT_PHOTO_TEST = Path(r"C:\Users\danie\Desktop\labellmg\Yolo-CV-Datasets\labels\test")

SKILL_TABLE_TEST = Path(r"C:\Users\danie\Desktop\labellmg\Yolo-CV-Datasets\Skill_tables\labels\test")



def get_test_images_from_labels(labels_path: Path, num_images: int = 150):
    """
    A partir de labels/test encontra as imagens correspondentes em images/test.
    Seleciona aleatoriamente até num_images.
    """

    labels_path = Path(labels_path)
    images_path = labels_path.parent.parent / "images" / "test"

    image_paths = []

    for label_file in labels_path.glob("*.txt"):
        img = images_path / (label_file.stem + ".jpg")
        if img.exists():
            image_paths.append(img)

    if len(image_paths) == 0:
        raise RuntimeError(f"Nenhuma imagem encontrada para {labels_path}")

    random.shuffle(image_paths)

    return image_paths[:num_images]



def measure_model_speed(model_path: Path, images, device="cpu"):
    """
    Mede ms/img e FPS de um modelo YOLO.
    """

    model = YOLO(model_path)

    # warmup (muito importante para medir corretamente)
    for img in images[:5]:
        _ = model.predict(source=img, device=device, verbose=False)

    start = time.time()

    for img in images:
        _ = model.predict(source=img, device=device, verbose=False)

    total_time = time.time() - start
    n = len(images)

    ms_per_image = (total_time / n) * 1000
    fps = 1000 / ms_per_image

    return ms_per_image, fps




def collect_models_speed(
    models_root: Path = MODELS_DATA_PATH,
    num_images: int = 150,
    device: str = "cpu",
):
    """
    Mede velocidade de inferência de todos os best.pt.
    Retorna DataFrame com ms/img e FPS.
    """

    # preparar imagens de teste
    text_photo_imgs = get_test_images_from_labels(TEXT_PHOTO_TEST, num_images)
    skill_table_imgs = get_test_images_from_labels(SKILL_TABLE_TEST, num_images)

    rows = []

    for model_dir in models_root.iterdir():
        if not model_dir.is_dir():
            continue

        # descobrir task pelo nome da pasta
        # ex: yolo11m-text-photo
        parts = model_dir.name.split("-", 1)
        if len(parts) != 2:
            continue

        yolo_version, task = parts

        # escolher imagens corretas
        if task == "text-photo":
            images = text_photo_imgs
        elif task == "skill_table":
            images = skill_table_imgs
        else:
            continue

        for train_dir in model_dir.iterdir():
            best_path = train_dir / "weights" / "best.pt"

            if not best_path.exists():
                continue

            print(f"Benchmarking {best_path.name} ({task})...")

            ms, fps = measure_model_speed(best_path, images, device=device)

            rows.append(
                {
                    "yolo_version": yolo_version,
                    "task": task,
                    "train_run": train_dir.name,
                    "ms_per_image": ms,
                    "fps": fps,
                    "num_images": len(images),
                    "device": device,
                    "model_path": str(best_path),
                }
            )

    return pd.DataFrame(rows).sort_values("ms_per_image")
