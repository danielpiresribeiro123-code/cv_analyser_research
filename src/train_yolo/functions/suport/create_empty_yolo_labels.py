from pathlib import Path
from collections import Counter


def create_empty_yolo_labels(
    images_path,
    labels_path,
    image_extensions=(".jpg", ".jpeg", ".png", ".bmp", ".webp")
):
    """
    Cria labels YOLO vazias para imagens sem label.

    Regras:
    - Se já existir .txt → não cria nada
    - Se houver imagens com o mesmo nome → ignora todas essas imagens
    """

    images_path = Path(images_path)
    labels_path = Path(labels_path)
    labels_path.mkdir(parents=True, exist_ok=True)

    # ---------- listar imagens ----------
    image_files = [
        img for img in images_path.iterdir()
        if img.suffix.lower() in image_extensions
    ]

    # ---------- detetar nomes duplicados ----------
    name_counts = Counter(img.stem for img in image_files)
    duplicated_names = {name for name, count in name_counts.items() if count > 1}

    created = []
    skipped_existing = []
    skipped_duplicates = []

    for img in image_files:
        stem = img.stem

        # ---------- ignorar duplicados ----------
        if stem in duplicated_names:
            skipped_duplicates.append(img.name)
            continue

        label_file = labels_path / f"{stem}.txt"

        # ---------- já existe label ----------
        if label_file.exists():
            skipped_existing.append(img.name)
            continue

        # ---------- criar label vazia ----------
        label_file.write_text("", encoding="utf-8")
        created.append(img.name)

    # ================= RESUMO =================
    print("\n========== RESUMO ==========")
    print(f"Total imagens: {len(image_files)}")
    print(f"Labels vazias criadas: {len(created)}")
    print(f"Ignoradas (já tinham label): {len(skipped_existing)}")
    print(f"Ignoradas (nomes duplicados): {len(skipped_duplicates)}")

    if skipped_duplicates:
        print("\nImagens ignoradas por nome duplicado:")
        print("\n".join(skipped_duplicates))



create_empty_yolo_labels(
    images_path="C:\\Users\\danie\\Desktop\\labellmg\\Yolo-CV-Datasets\\Skill_tables\\images\\all",
    labels_path="C:\\Users\\danie\\Desktop\\labellmg\\Yolo-CV-Datasets\\Skill_tables\\labels"
)
