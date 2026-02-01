from pathlib import Path
import shutil

from pathlib import Path
import shutil


def copy_images_with_matching_docs_recursive(
    image_input_path,
    doc_input_path,
    output_path,
    doc_type="xml",
    image_exts=(".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"),
):
    """
    Copia imagens (mesmo em subfolders) cujo nome base coincide
    com documentos de um dado tipo.

    Exemplo:
        image_input_path/.../folder_A/image_1.png
        doc_input_path/image_1.xml
        -> copia image_1.png para output_path
    """

    image_input_path = Path(image_input_path)
    doc_input_path = Path(doc_input_path)
    output_path = Path(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    # nomes base dos docs (ex: image_1)
    doc_names = {
        doc.stem
        for doc in doc_input_path.iterdir()
        if doc.is_file() and doc.suffix.lower() == f".{doc_type.lower()}"
    }

    copied = 0
    seen = set()  # evita copiar a mesma imagem duas vezes

    # percorre TODOS os subfolders
    for img in image_input_path.rglob("*"):
        if not img.is_file():
            continue

        if img.suffix.lower() not in image_exts:
            continue

        if img.stem in doc_names:
            if img.name in seen:
                continue

            shutil.copy2(img, output_path / img.name)
            seen.add(img.name)
            copied += 1

    print(f"Copiadas {copied} imagens para {output_path}")

copy_images_with_matching_docs_recursive(
    image_input_path=r"C:\Users\danie\Desktop\My_Things\Hobbies\VSCODE\Data\CVs_Resumes_Datasets\CVs_em_imagem\youssefkhalil_kaggle\Resumes Datasets\Scrapped_Resumes",
    doc_input_path=r"C:\Users\danie\Desktop\labellmg\Yolo-CV-Datasets\Scraped_images_xml",
    output_path=r"C:\Users\danie\Desktop\labellmg\Yolo-CV-Datasets\Scraped_images",
    doc_type="xml"
)
