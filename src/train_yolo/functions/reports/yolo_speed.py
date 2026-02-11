import random
import time
from ultralytics import YOLO  # type: ignore
import pandas as pd
from pathlib import Path

from utils.path_loader import load_paths

paths = load_paths()
PROJECT_ROOT = paths["project_root"]
MODELS_DATA_PATH = PROJECT_ROOT / "src" / "train_yolo"

TEXT_PHOTO_TEST = Path(r"C:\Users\danie\Desktop\labellmg\Yolo-CV-Datasets\labels\test")
SKILL_TABLE_TEST = Path(r"C:\Users\danie\Desktop\labellmg\Yolo-CV-Datasets\Skill_tables\labels\test")


# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------

def to_relative(path: Path) -> str:
    """Converte path absoluto para relativo ao PROJECT_ROOT."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def get_test_images_from_labels(labels_path: Path, num_images: int = 150):
    """
    A partir de labels/test encontra as imagens correspondentes em images/test,
    suportando múltiplas extensões.

    Seleciona aleatoriamente até num_images imagens existentes.
    """

    labels_path = Path(labels_path)
    images_path = labels_path.parent.parent / "images" / "test"

    # extensões comuns em datasets YOLO
    valid_exts = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]

    image_paths = []

    for label_file in labels_path.glob("*.txt"):
        stem = label_file.stem

        # procurar qualquer extensão válida
        for ext in valid_exts:
            img = images_path / f"{stem}{ext}"
            if img.exists():
                image_paths.append(img)
                break  # encontrou imagem correspondente → parar extensões

    if len(image_paths) == 0:
        raise RuntimeError(f"Nenhuma imagem encontrada para {labels_path}")

    # embaralhar para amostragem aleatória
    random.shuffle(image_paths)

    # garantir no máximo num_images
    selected = image_paths[:num_images]

    print(
        f"[DATA] {labels_path.name}: "
        f"{len(image_paths)} imagens válidas encontradas | "
        f"usando {len(selected)} para benchmark"
    )

    return selected



def measure_model_speed(model_path: Path, images, device: str | int = "cpu"):
    """
    Mede ms/img e FPS de um modelo YOLO.
    """
    model = YOLO(model_path)

    # warmup (importante)
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


# ---------------------------------------------------------
# Main benchmarking
# ---------------------------------------------------------

def collect_models_speed(
    models_root: Path = MODELS_DATA_PATH,
    num_images: int = 150,
    device: str | int = "cpu",
):
    """
    Mede velocidade de inferência de todos os best.pt.
    Retorna DataFrame com ms/img e FPS.
    """

    print("\n=== YOLO SPEED BENCHMARK ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Device: {device}")
    print(f"Images per test: {num_images}\n")

    # preparar imagens de teste
    text_photo_imgs = get_test_images_from_labels(TEXT_PHOTO_TEST, num_images)
    skill_table_imgs = get_test_images_from_labels(SKILL_TABLE_TEST, num_images)

    rows = []
    model_jobs = []

    # descobrir todos os modelos primeiro (para mostrar progresso)
    for model_dir in models_root.iterdir():
        if not model_dir.is_dir():
            continue

        parts = model_dir.name.split("-", 1)
        if len(parts) != 2:
            continue

        yolo_version, task = parts

        if task not in {"text-photo", "skill_table"}:
            continue

        for train_dir in model_dir.iterdir():
            best_path = train_dir / "weights" / "best.pt"
            if best_path.exists():
                model_jobs.append((yolo_version, task, train_dir, best_path))

    print(f"Número de trains encontrados: {len(model_jobs)}\n")

    # benchmarking
    t0 = time.time()

    for i, (yolo_version, task, train_dir, best_path) in enumerate(model_jobs, 1):

        images = text_photo_imgs if task == "text-photo" else skill_table_imgs

        print(
            f"[{i:02d}/{len(model_jobs)}] "
            f"{yolo_version.upper():>7} | "
            f"TASK={task:<10} | "
            f"TRAIN={train_dir.name:<12} | "
            f"WEIGHTS={best_path.name}"
        )


        ms, fps = measure_model_speed(best_path, images, device=device)

        print(
            f"    → {ms:.2f} ms/img | {fps:.2f} FPS\n"
        )

        rows.append({
                "yolo_version": yolo_version,
                "task": task,
                "train_run": train_dir.name,
                "ms_per_image": ms,
                "fps": fps,
                "num_images": len(images),
                "device": device,
                "model_path": to_relative(best_path),})

    total_time = time.time() - t0

    print("=== BENCHMARK COMPLETO ===")
    print(f"Tempo total: {total_time:.2f} s")
    print("==========================\n")

    return pd.DataFrame(rows).sort_values("ms_per_image")
