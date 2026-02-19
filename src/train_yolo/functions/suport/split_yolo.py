from pathlib import Path
import random
import shutil


def split_yolo_from_all(
    dataset_root,
    split=(0.7, 0.2, 0.1),
    seed=42,
    img_exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")
):
    """
    Divide dataset YOLO a partir de:

        images/all
        labels/all

    Cria:
        images/train|val|test
        labels/train|val|test

    Regras:
    - Ignora imagens sem label correspondente
    - Não altera a pasta 'all'
    """

    random.seed(seed)
    dataset_root = Path(dataset_root)

    images_all = dataset_root / "images" / "all"
    labels_all = dataset_root / "labels" / "all"

    assert images_all.exists(), "images/all não existe"
    assert labels_all.exists(), "labels/all não existe"
    assert abs(sum(split) - 1.0) < 1e-6, "As percentagens devem somar 1.0"

    valid_images = [
        img for img in images_all.iterdir()
        if img.is_file()
        and img.suffix.lower() in img_exts
        and (labels_all / f"{img.stem}.txt").exists()
    ]

    print(f"Pares imagem↔label válidos: {len(valid_images)}")

    if not valid_images:
        print("Nenhum par válido encontrado.")
        return

    random.shuffle(valid_images)

    n = len(valid_images)
    n_train = int(n * split[0])
    n_val = int(n * split[1])

    splits = {
        "train": valid_images[:n_train],
        "val": valid_images[n_train:n_train + n_val],
        "test": valid_images[n_train + n_val:]
    }

    for split_name, imgs in splits.items():

        img_dst = dataset_root / "images" / split_name
        lbl_dst = dataset_root / "labels" / split_name

        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for img_path in imgs:
            lbl_path = labels_all / f"{img_path.stem}.txt"

            shutil.copy2(img_path, img_dst / img_path.name)
            shutil.copy2(lbl_path, lbl_dst / lbl_path.name)

    print("\nSplit concluído:")
    for k, v in splits.items():
        print(f"  {k}: {len(v)} imagens")


split_yolo_from_all(
    dataset_root="C:/Users/danie/Desktop/labellmg/Yolo-CV-Datasets/Skill_tables",
    split=(0.8, 0.1, 0.1)
)
