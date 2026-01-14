from __future__ import annotations

from pathlib import Path
from typing import Iterable, Set

import cv2
import numpy as np
from ultralytics import YOLO #type:ignore


# COCO class ids (Ultralytics / COCO)
# person=0, bicycle=1, car=2, motorcycle=3, bus=5, truck=7, cat=15, dog=16
DEFAULT_DYNAMIC_CLASS_IDS: Set[int] = {0, 1, 2, 3, 5, 7, 15, 16}


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------

def iter_images(
    input_dir: Path,
    exts: Iterable[str] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
):
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def postprocess_mask(
    mask_01: np.ndarray,
    dilate_px: int = 6,
    close_px: int = 6,
    min_area_px: int = 150,
) -> np.ndarray:
    """
    mask_01: uint8 array com valores {0,1}
    retorna: uint8 array com valores {0,255}
    """
    m = (mask_01 > 0).astype(np.uint8)

    if dilate_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px, dilate_px)
        )
        m = cv2.dilate(m, k, iterations=1)

    if close_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_px, close_px)
        )
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

    if min_area_px > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            m, connectivity=8
        )
        out = np.zeros_like(m)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
                out[labels == i] = 1
        m = out

    return (m * 255).astype(np.uint8)


def build_dynamic_mask_from_result(
    result,
    img_shape_hw: tuple[int, int],
    dynamic_class_ids: Set[int],
    conf_thres: float,
) -> np.ndarray:
    """
    Cria uma máscara binária (0/1) combinando todas as instâncias dinâmicas.
    """
    h, w = img_shape_hw
    combined = np.zeros((h, w), dtype=np.uint8)

    if result.masks is None or result.boxes is None:
        return combined

    masks = result.masks.data.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy().astype(float)

    for i in range(len(cls_ids)):
        if cls_ids[i] not in dynamic_class_ids:
            continue
        if confs[i] < conf_thres:
            continue

        mi01 = (masks[i] > 0.5).astype(np.uint8)

        if mi01.shape != (h, w):
            mi01 = cv2.resize(
                mi01, (w, h), interpolation=cv2.INTER_NEAREST
            )

        combined = np.maximum(combined, mi01)

    return combined


# ---------------------------------------------------------------------
# API principal (o que vais usar no projeto)
# ---------------------------------------------------------------------

def gen_dynamic_masks(
    model_path: Path,
    input_dir: Path,
    output_dir: Path,
    dynamic_class_ids: Set[int] = DEFAULT_DYNAMIC_CLASS_IDS,
    conf: float = 0.35,
    imgsz: int = 960,
    device: str | None = None,
    dilate_px: int = 6,
    close_px: int = 6,
    min_area_px: int = 150,
    preview: bool = False,
) -> None:
    """
    Gera máscaras binárias de ruído dinâmico a partir de imagens,
    utilizando um modelo YOLO com segmentação (YOLO-seg).

    Cada máscara identifica regiões da imagem que correspondem a
    objetos considerados "ruído" (ex.: pessoas, veículos, animais),
    permitindo que essas áreas sejam posteriormente ignoradas ou
    removidas em pipelines de reconstrução 3D (fotogrametria).

    Parâmetros
    ----------
    model_path : Path
        Caminho para o ficheiro do modelo YOLO de segmentação
        (ex.: 'yolo11l-seg.pt'). Este modelo é usado apenas para
        inferência, não para treino.

    input_dir : Path
        Diretório que contém as imagens ou frames de entrada
        (ex.: frames extraídos de vídeo). As imagens nesta pasta
        não são modificadas.

    output_dir : Path
        Diretório onde serão guardadas as máscaras binárias geradas.
        Para cada imagem de entrada será criada uma máscara com o
        mesmo nome base e o sufixo '_mask.png'.

    dynamic_class_ids : Set[int], opcional
        Conjunto de IDs de classes (COCO) que devem ser consideradas
        como ruído dinâmico. Apenas instâncias destas classes serão
        incluídas na máscara final.
        Por defeito inclui pessoas, veículos e alguns animais comuns.

    conf : float, opcional
        Threshold de confiança mínima para aceitar uma instância
        detetada pelo YOLO. Valores mais baixos geram máscaras mais
        abrangentes (mais ruído removido), valores mais altos são
        mais conservadores.

    imgsz : int, opcional
        Tamanho da imagem utilizado na inferência do YOLO.
        Valores maiores podem aumentar a qualidade das máscaras,
        mas tornam a inferência mais lenta.

    device : str | None, opcional
        Dispositivo onde a inferência será executada.
        Exemplos:
            - None  → seleção automática
            - "cpu" → força CPU
            - "0"   → GPU 0
            - "0,1" → múltiplas GPUs

    dilate_px : int, opcional
        Tamanho (em pixels) do kernel de dilatação aplicado à máscara.
        Serve para garantir que os contornos dos objetos dinâmicos
        ficam totalmente cobertos. Usar 0 para desativar.

    close_px : int, opcional
        Tamanho (em pixels) do kernel de fecho morfológico aplicado
        à máscara. Ajuda a preencher pequenos buracos dentro das
        regiões mascaradas. Usar 0 para desativar.

    min_area_px : int, opcional
        Área mínima (em pixels) que uma região conectada da máscara
        deve ter para ser mantida. Regiões mais pequenas são removidas,
        reduzindo falsos positivos e ruído residual.

    preview : bool, opcional
        Se True, mostra uma visualização em tempo real da imagem
        original lado a lado com a aplicação da máscara, para fins
        de validação visual. Premir 'q' para sair.

    Retorna
    -------
    None
        As máscaras são guardadas diretamente em 'output_dir'.
    """


    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Pasta input não encontrada: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))

    for img_path in iter_images(input_dir):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Não consegui ler: {img_path}")
            continue

        h, w = img.shape[:2]

        results = model.predict(
            source=img,
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
        )

        combined_01 = build_dynamic_mask_from_result(
            results[0],
            img_shape_hw=(h, w),
            dynamic_class_ids=dynamic_class_ids,
            conf_thres=conf,
        )

        mask_255 = postprocess_mask(
            combined_01,
            dilate_px=dilate_px,
            close_px=close_px,
            min_area_px=min_area_px,
        )

        out_path = output_dir / f"{img_path.stem}_mask.png"
        cv2.imwrite(str(out_path), mask_255)

        if preview:
            overlay = img.copy()
            overlay[mask_255 > 0] = (0, 0, 0)
            show = np.hstack([img, overlay])
            cv2.imshow("Original | Mascara aplicada", show)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        print(f"[OK] {img_path.name} -> {out_path.name}")

    if preview:
        cv2.destroyAllWindows()
