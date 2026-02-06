from pathlib import Path
import shutil


def move_all_images(input_path, output_path):
    """
    Move todos os ficheiros de imagem existentes em input_path
    (incluindo subpastas) para output_path.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    # Extensões de imagem suportadas
    image_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"
    }

    # Criar pasta de destino se não existir
    output_path.mkdir(parents=True, exist_ok=True)

    moved_files = 0

    # Percorrer todos os ficheiros recursivamente
    for file in input_path.rglob("*"):
        if file.suffix.lower() in image_extensions:
            destination = output_path / file.name

            # Evitar sobrescrever ficheiros com o mesmo nome
            counter = 1
            while destination.exists():
                destination = output_path / f"{file.stem}_{counter}{file.suffix}"
                counter += 1

            shutil.move(str(file), str(destination))
            moved_files += 1

    print(f"Total de imagens movidas: {moved_files}")


move_all_images(
    input_path=r"C:\Users\danie\Desktop\labellmg\Yolo-CV-Datasets\Skill_tables\Resume.v1i.voc",
    output_path=r"C:\Users\danie\Desktop\labellmg\Yolo-CV-Datasets\Skill_tables\images"
)
