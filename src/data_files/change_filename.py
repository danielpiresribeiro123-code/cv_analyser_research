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



from pathlib import Path


def adicionar_sufixo_a_folders(root_path, sufixo):
    """
    Adiciona um sufixo ao nome de todos os folders dentro de root_path.
    Não altera o conteúdo interno dos folders.

    Exemplo:
        pasta1  -> pasta1_sufixo
        dados   -> dados_sufixo
    """
    root = Path(root_path)

    if not root.exists() or not root.is_dir():
        raise ValueError(f"O caminho '{root_path}' não é um diretório válido.")

    for folder in root.iterdir():
        if folder.is_dir():
            # Evita duplicar o sufixo se já existir
            if folder.name.endswith(sufixo):
                print(f"Ignorado (já tem sufixo): {folder.name}")
                continue

            novo_nome = folder.name + sufixo
            novo_caminho = folder.with_name(novo_nome)

            folder.rename(novo_caminho)
            print(f"{folder.name} -> {novo_nome}")

"""
adicionar_sufixo_a_folders(
    root_path="C:/Users/danie/Downloads/CVs_Resumes_Datasets/roboflow",
    sufixo="_roboflow"
)
"""
