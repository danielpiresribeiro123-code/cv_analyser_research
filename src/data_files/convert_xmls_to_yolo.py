from pathlib import Path
import xml.etree.ElementTree as ET


def convert_xmls_to_yolo(
    xml_input_path,
    output_path,
    class_map,
    recursive=True
):
    """
    Converte ficheiros XML (Pascal VOC) para formato YOLOv11 (.txt)
    e copia para output_path.

    Args:
        xml_input_path (str | Path): Pasta com XMLs
        output_path (str | Path): Pasta destino dos .txt YOLO
        class_map (dict): {"class_name": class_id}
        recursive (bool): Percorrer subpastas
    """

    xml_input_path = Path(xml_input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    xml_files = xml_input_path.rglob("*.xml") if recursive else xml_input_path.glob("*.xml")

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        size = root.find("size")
        img_w = float(size.find("width").text)
        img_h = float(size.find("height").text)

        if img_w == 0 or img_h == 0:
            print(f"[IGNORADO] XML com width/height inválidos: {xml_file.name}")
            continue


        yolo_lines = []

        for obj in root.findall("object"):
            class_name = obj.find("name").text.strip()

            if class_name not in class_map:
                print(f"[IGNORADO] Classe '{class_name}' não está no class_map ({xml_file.name})")
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

        out_txt = output_path / f"{xml_file.stem}.txt"
        out_txt.write_text("\n".join(yolo_lines), encoding="utf-8")

        print(f"[OK] {xml_file.name} -> {out_txt.name}")



CLASS_MAP = {
    "text_block": 0,
    "photo": 1}

convert_xmls_to_yolo(
    xml_input_path="C:\\Users\\danie\\Desktop\\labellmg\\Yolo-CV-Datasets\\labels_xml",
    output_path="C:\\Users\\danie\\Desktop\\labellmg\\Yolo-CV-Datasets\\labels_yolo",
    class_map=CLASS_MAP
)
