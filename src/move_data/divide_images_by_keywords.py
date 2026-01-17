from pathlib import Path
import shutil

from utils.path_loader import load_paths
paths = load_paths()

TRAIN_IMAGES_PERSONAL_ROOT = paths["train_personal_images"]

def split_images_by_keywords(
    input_dir,
    output_dir,
    keywords,
    valid_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
    move_files=False
):
    """
    Divide imagens em pastas de acordo com palavras-chave no nome do ficheiro.

    input_dir   : pasta onde estão todas as imagens
    output_dir  : pasta base onde vão ser criadas as subpastas
    keywords    : lista/tuple de palavras (ex: ["sharp", "blur", "partial_blur"])
    valid_exts  : extensões aceites
    move_files  : se True move os ficheiros, se False copia
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # criar as subpastas
    for kw in keywords:
        (output_dir / kw).mkdir(parents=True, exist_ok=True)

    for img_path in input_dir.iterdir():
        if img_path.suffix.lower() not in valid_exts:
            continue

        name_lower = img_path.name.lower()
        matched = False

        for kw in keywords:
            if kw.lower() in name_lower:
                dest = output_dir / kw / img_path.name

                if move_files:
                    shutil.move(str(img_path), str(dest))
                else:
                    shutil.copy2(str(img_path), str(dest))

                print(f"{img_path.name} -> {kw}/")
                matched = True
                break

        if not matched:
            print(f"{img_path.name} -> nenhuma keyword encontrada")


keywords = ["-sharp", "-motion_blur", "-partial_blur"]

split_images_by_keywords(
    input_dir=TRAIN_IMAGES_PERSONAL_ROOT / "2",
    output_dir=TRAIN_IMAGES_PERSONAL_ROOT / "2_1",
    keywords=keywords,
    move_files=False   # False = copia, True = move
)
