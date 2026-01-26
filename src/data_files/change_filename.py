from pathlib import Path

from utils.path_loader import load_paths
paths = load_paths()

TRAIN_IMAGES_PERSONAL_ROOT = paths["train_personal_images"]


def replace_in_filenames(
    folder,
    new_str,
    change_str="-", 
    valid_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".xml")
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

replace_in_filenames(
    folder="C:\\Users\\danie\\Desktop\\labellmg\\Yolo-CV-Datasets",
    new_str="-bing_images-accountant_resumes",
    change_str="-bing_images-Accountant resumes"
)


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


from pathlib import Path

def adicionar_sufixo_a_docs(root_path, sufixo, exts=None):
    """
    Adiciona um sufixo ao nome de todos os ficheiros dentro de root_path.
    Não altera a extensão nem o conteúdo.

    Exemplo:
        Image_2.png -> Image_2-bing_images-Accountant resumes.png
    """
    root = Path(root_path)

    if not root.exists() or not root.is_dir():
        raise ValueError(f"O caminho '{root_path}' não é um diretório válido.")

    for file in root.iterdir():
        if not file.is_file():
            continue

        # Se quiseres limitar por extensões (opcional)
        if exts is not None and file.suffix.lower() not in exts:
            continue

        # Evita duplicar o sufixo
        if file.stem.endswith(sufixo):
            print(f"Ignorado (já tem sufixo): {file.name}")
            continue

        novo_nome = f"{file.stem}{sufixo}{file.suffix}"
        novo_caminho = file.with_name(novo_nome)

        file.rename(novo_caminho)
        print(f"{file.name} -> {novo_nome}")




#adicionar_sufixo_a_docs(root_path="C:\\Users\\danie\\Desktop\\labellmg\\Yolo-CV-Datasets",sufixo="-bing_images-Accountant resumes")



from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def rename_images_with_folder_names(root_path):
    root_path = Path(root_path)

    for img_path in root_path.rglob("*"):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        # Nome original
        stem = img_path.stem
        ext = img_path.suffix

        # Dois últimos folders
        parent = img_path.parent
        folder_1 = parent.name
        folder_2 = parent.parent.name if parent.parent else ""

        # Normalizar para evitar espaços estranhos
        folder_1_clean = folder_1.replace(" ", "_").lower()
        folder_2_clean = folder_2.replace(" ", "_").lower()

        # Novo nome
        new_name = f"{stem}-{folder_2_clean}-{folder_1_clean}{ext}"
        new_path = img_path.with_name(new_name)

        # Evita renomear duas vezes o mesmo ficheiro
        if new_path.exists():
            print(f"Já existe: {new_path.name}")
            continue

        img_path.rename(new_path)
        print(f"{img_path.name} -> {new_path.name}")


#rename_images_with_folder_names(
#    r"C:\Users\danie\Desktop\My_Things\Hobbies\VSCODE\Data\CVs_Resumes_Datasets\CVs_em_imagem\youssefkhalil_kaggle\Resumes Datasets")
