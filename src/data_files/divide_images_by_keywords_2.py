from pathlib import Path
import shutil

from utils.path_loader import load_paths
paths = load_paths()

TRAIN_IMAGES_PERSONAL_ROOT = paths["train_personal_images"]
from pathlib import Path
import shutil

def extract_keyword_from_name(filename: str, divide_str="_"):
    """
    Extrai o texto entre o primeiro "_" e o último "_".
    Ex:
    7_NIKON-D3400-35MM_S.jpg -> NIKON-D3400-35MM
    """
    name = Path(filename).stem  # remove extensão
    parts = name.split(divide_str)

    if len(parts) < 3:
        return None  # não segue o padrão esperado

    # junta tudo entre o primeiro e o último _
    return "_".join(parts[1:-1])


def split_images_by_pattern(
    input_dir,
    output_dir,
    valid_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
    move_files=False,
    divide_str="_"
):
    """
    Divide imagens criando automaticamente as pastas com base
    no padrão do nome do ficheiro:
    <qualquer>_<KEYWORD>_<qualquer>.<ext>
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in input_dir.iterdir():
        if img_path.suffix.lower() not in valid_exts:
            continue

        keyword = extract_keyword_from_name(img_path.name, divide_str)

        if keyword is None:
            print(f"{img_path.name} -> padrão inválido")
            continue

        # cria pasta automaticamente se ainda não existir
        dest_dir = output_dir / keyword
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest = dest_dir / img_path.name

        if move_files:
            shutil.move(str(img_path), str(dest))
        else:
            shutil.copy2(str(img_path), str(dest))

        print(f"{img_path.name} -> {keyword}/")

split_images_by_pattern(
    input_dir=TRAIN_IMAGES_PERSONAL_ROOT / "inside",
    output_dir=TRAIN_IMAGES_PERSONAL_ROOT / "inside_divided",
    move_files=False,   # False = copia, True = move
    divide_str="_"
)
