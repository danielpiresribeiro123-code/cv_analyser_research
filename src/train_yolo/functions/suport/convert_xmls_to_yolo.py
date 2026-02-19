from pathlib import Path
import xml.etree.ElementTree as ET


def convert_xmls_to_yolo(
    xml_input_path,
    output_path,
    class_map,
    recursive=True
):
    """
    Converte ficheiros XML (Pascal VOC) para formato YOLO (.txt)
    e guarda em output_path.

    Também reporta XMLs ignorados e estatísticas finais.
    """

    xml_input_path = Path(xml_input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    xml_files = list(
        xml_input_path.rglob("*.xml") if recursive else xml_input_path.glob("*.xml")
    )

    ignored_invalid_size = []
    ignored_unknown_class = []
    ignored_empty = []
    converted = []

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        size = root.find("size")
        img_w = float(size.find("width").text)
        img_h = float(size.find("height").text)

        if img_w == 0 or img_h == 0:
            ignored_invalid_size.append(xml_file.name)
            continue

        yolo_lines = []
        unknown_class_found = False

        for obj in root.findall("object"):
            class_name = obj.find("name").text.strip()

            if class_name not in class_map:
                unknown_class_found = True
                continue

            class_id = class_map[class_name]

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            x_center = ((xmin + xmax) / 2) / img_w
            y_center = ((ymin + ymax) / 2) / img_h
            width = (xmax - xmin) / img_w
            height = (ymax - ymin) / img_h

            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        if not yolo_lines:
            if unknown_class_found:
                ignored_unknown_class.append(xml_file.name)
            else:
                ignored_empty.append(xml_file.name)
            continue

        out_txt = output_path / f"{xml_file.stem}.txt"
        out_txt.write_text("\n".join(yolo_lines), encoding="utf-8")

        converted.append(xml_file.name)
        print(f"[OK] {xml_file.name} -> {out_txt.name}")

    print("\n========== RESUMO ==========")
    print(f"Total XML encontrados: {len(xml_files)}")
    print(f"Convertidos com sucesso: {len(converted)}")
    print(f"Ignorados (tamanho inválido): {len(ignored_invalid_size)}")
    print(f"Ignorados (classe fora do class_map): {len(ignored_unknown_class)}")
    print(f"Ignorados (sem objetos válidos): {len(ignored_empty)}")

    # Mostrar nomes (útil para debug dataset)
    if ignored_invalid_size:
        print("\nXML com tamanho inválido:")
        print("\n".join(ignored_invalid_size))

    if ignored_unknown_class:
        print("\nXML com classes desconhecidas:")
        print("\n".join(ignored_unknown_class))

    if ignored_empty:
        print("\nXML sem bounding boxes válidas:")
        print("\n".join(ignored_empty))


CLASS_MAP = {"skill_table": 0}

convert_xmls_to_yolo(
    xml_input_path="C:\\Users\\danie\\Desktop\\labellmg\\Yolo-CV-Datasets\\Skill_tables\\xml",
    output_path="C:\\Users\\danie\\Desktop\\labellmg\\Yolo-CV-Datasets\\Skill_tables\\labels",
    class_map=CLASS_MAP
)
