from pathlib import Path

from utils.path_loader import load_paths
paths = load_paths()

TRAIN_IMAGES_PERSONAL_ROOT = paths["train_personal_images"]


def replace_in_filenames(
    folder,
    new_str,
    change_str="-", 
    valid_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tiff")
):
    """
    Substitui parte do nome do ficheiro.
    """

    folder = Path(folder)

    for img_path in folder.iterdir():
        if img_path.suffix.lower() not in valid_exts:
            continue

        old_name = img_path.name
        name = img_path.stem
        ext = img_path.suffix

        if change_str in name:
            new_name = name.replace(change_str, f"{new_str}", 1)


            new_path = img_path.with_name(new_name + ext)
            img_path.rename(new_path)
            print(f"{old_name} -> {new_path.name}")
        else:
            print(f"Nenhuma mudança em {old_name}.")
"""
replace_in_filenames(
    folder=TRAIN_IMAGES_PERSONAL_ROOT / "market",
    new_str="-dark",
    change_str="_GalaxyS21-5G-SM-G991BDS_dark"
)
"""

from pathlib import Path

def add_suffix_to_filenames(
    folder,
    suffix,
    valid_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tiff")
):
    """
    Adiciona uma palavra/sufixo ao final do nome dos ficheiros (antes da extensão).

    Ex:
      imagem.jpg -> imagem_sharp.jpg
    """

    folder = Path(folder)

    for img_path in folder.iterdir():
        if img_path.suffix.lower() not in valid_exts:
            continue

        old_name = img_path.name
        name = img_path.stem
        ext = img_path.suffix

        # evita duplicar o sufixo se já existir
        if name.endswith(suffix):
            print(f"Já tem sufixo: {old_name}")
            continue

        new_name = f"{name}{suffix}"
        new_path = img_path.with_name(new_name + ext)

        img_path.rename(new_path)
        print(f"{old_name} -> {new_path.name}")

"""
add_suffix_to_filenames(folder=TRAIN_IMAGES_PERSONAL_ROOT / "market",
                        suffix="_market")

add_suffix_to_filenames(folder=TRAIN_IMAGES_PERSONAL_ROOT / "inside",
                        suffix="_inside")

add_suffix_to_filenames(folder=TRAIN_IMAGES_PERSONAL_ROOT / "outside",
                        suffix="_outside")"""