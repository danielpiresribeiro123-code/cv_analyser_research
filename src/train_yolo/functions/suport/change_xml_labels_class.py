from pathlib import Path
import xml.etree.ElementTree as ET


def rename_xml_classes(input_folder, old_name="0", new_name="skill_table"):
    """
    Percorre todos os ficheiros XML numa pasta (recursivamente)
    e substitui o nome da classe old_name por new_name.
    """

    input_folder = Path(input_folder)
    xml_files = list(input_folder.rglob("*.xml"))

    changed_files = 0
    changed_labels = 0

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        file_changed = False

        # Procurar todas as tags <name> dentro de <object>
        for obj in root.findall("object"):
            name_tag = obj.find("name")

            if name_tag is not None and name_tag.text == old_name:
                name_tag.text = new_name
                file_changed = True
                changed_labels += 1

        if file_changed:
            tree.write(xml_file, encoding="utf-8")
            changed_files += 1

    print(f"Ficheiros alterados: {changed_files}")
    print(f"Labels alteradas: {changed_labels}")


# EXEMPLO DE USO
rename_xml_classes(
    input_folder=r"C:\Users\danie\Desktop\labellmg\Yolo-CV-Datasets\Skill_tables\xml",
    old_name="0",
    new_name="skill_table")
