from pathlib import Path
import xml.etree.ElementTree as ET


def rename_all_classes_to_text_block(xml_folder):
    """
    Altera o nome de TODAS as classes em ficheiros .xml para 'text_block',
    sem modificar as bounding boxes nem qualquer outra informação.

    Args:
        xml_folder (str ou Path): caminho para a pasta com os ficheiros .xml
    """
    xml_folder = Path(xml_folder)

    if not xml_folder.exists() or not xml_folder.is_dir():
        raise ValueError(f"O caminho '{xml_folder}' não é uma pasta válida.")

    xml_files = list(xml_folder.glob("*.xml"))

    if not xml_files:
        print("Nenhum ficheiro .xml encontrado.")
        return

    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        changed = False

        # Percorre todos os objetos anotados
        for obj in root.findall("object"):
            name_tag = obj.find("name")
            if name_tag is not None:
                if name_tag.text != "text_block":
                    name_tag.text = "text_block"
                    changed = True

        if changed:
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
            print(f"Atualizado: {xml_path.name}")
        else:
            print(f"Já estava correto: {xml_path.name}")


rename_all_classes_to_text_block(
    r"C:\Users\danie\Desktop\labellmg\Yolo-CV-Datasets\smartcv-resume-parser_v1i_xml"
)
