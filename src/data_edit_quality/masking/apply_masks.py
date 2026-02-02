from __future__ import annotations

"""
Módulo: apply_masks.py

Responsabilidade:
    - Aplicar máscaras (YOLO-seg) APENAS aos frames dinâmicos
    - Produzir 2 saídas por vídeo:
        1) with_dynamic
            → apenas os objetos dinâmicos (background removido)
        2) without_dynamic
            → frame limpo, com objetos dinâmicos removidos

Estrutura esperada:
    images_dir (entrada):
        data/raw/frames/<video_id>/dynamic/

    masks_dir (entrada):
        data/masks/dynamic/<video_id>/
            <frame_stem>_mask.png

    output_base_dir (saída):
        data/processed/
            with_dynamic/<video_id>/
            without_dynamic/<video_id>/

Notas importantes:
    - NÃO processa frames static nem discarded
    - NÃO percorre diretórios recursivamente
    - Usa operações OpenCV (bitwise) em vez de boolean indexing
      para evitar erros de dimensão (H,W) vs (H,W,1)
"""

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


# --------------------------------------------------
# Utilitário: iterar imagens diretamente numa pasta
# --------------------------------------------------
def iter_images(
    input_dir: Path,
    exts: Iterable[str] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
):
    """
    Itera apenas sobre ficheiros de imagem diretamente contidos em input_dir.
    Evita rglob() para não entrar em subpastas indevidas.
    """
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


# --------------------------------------------------
# Utilitário: garantir imagem BGR (3 canais)
# --------------------------------------------------
def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    """
    Garante que a imagem está em formato BGR (H, W, 3).

    Aceita:
        - grayscale (H, W)
        - BGRA (H, W, 4)
        - BGR (H, W, 3)
    """
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


# --------------------------------------------------
# Utilitário: carregar máscara binária robusta
# --------------------------------------------------
def _load_mask_u8(mask_path: Path) -> np.ndarray | None:
    """
    Lê uma máscara e garante:
        - formato 2D (H, W)
        - dtype uint8
        - valores binários {0, 255}

    Se a máscara vier como (H,W,3), (H,W,4) ou (H,W,1),
    reduz para um único canal.
    """
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None

    if m.ndim == 3:
        m = m[:, :, 0]

    if m.dtype != np.uint8:
        m = m.astype(np.uint8)

    _, m_bin = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY)
    return m_bin


# --------------------------------------------------
# Função principal
# --------------------------------------------------
def apply_dynamic_masks(
    images_dir: Path,
    masks_dir: Path,
    output_base_dir: Path,
    video_id: str,
    mask_suffix: str = "_mask",
    fill_value: tuple[int, int, int] = (0, 0, 0),
    preview: bool = False,
    copy_if_no_mask: bool = False,
) -> None:
    """
    Aplica máscaras binárias aos frames dinâmicos de um vídeo.

    Produz:
        processed/with_dynamic/<video_id>/
        processed/without_dynamic/<video_id>/

    Args:
        images_dir:
            Pasta com frames dinâmicos
            (ex: data/raw/frames/<video_id>/dynamic)

        masks_dir:
            Pasta com máscaras YOLO-seg
            (ex: data/masks/dynamic/<video_id>)

        output_base_dir:
            Pasta base processed/
            (ex: data/processed)

        video_id:
            Identificador do vídeo

        mask_suffix:
            Sufixo do ficheiro de máscara
            Ex: frame_001.jpg → frame_001_mask.png

        fill_value:
            Cor BGR usada para preencher regiões removidas

        preview:
            Se True, mostra preview (with_dynamic | without_dynamic)

        copy_if_no_mask:
            - True: copia imagem sem máscara para ambas as pastas
            - False: ignora frames sem máscara (recomendado)
    """

    if not images_dir.exists():
        raise FileNotFoundError(f"Frames dinâmicos não encontrados: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Máscaras não encontradas: {masks_dir}")

    out_with = output_base_dir / "with_dynamic" / video_id
    out_without = output_base_dir / "without_dynamic" / video_id
    out_with.mkdir(parents=True, exist_ok=True)
    out_without.mkdir(parents=True, exist_ok=True)

    for img_path in iter_images(images_dir):

        # -----------------------------
        # Ler imagem
        # -----------------------------
        img_raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img_raw is None:
            print(f"[WARN] Não consegui ler imagem: {img_path.name}")
            continue

        img = _ensure_bgr(img_raw)

        # -----------------------------
        # Localizar máscara
        # -----------------------------
        mask_path = masks_dir / f"{img_path.stem}{mask_suffix}.png"
        if not mask_path.exists():
            if copy_if_no_mask:
                cv2.imwrite(str(out_with / img_path.name), img)
                cv2.imwrite(str(out_without / img_path.name), img)
                print(f"[COPY] Sem máscara: {img_path.name}")
            else:
                print(f"[SKIP] Sem máscara: {img_path.name}")
            continue

        # -----------------------------
        # Ler máscara
        # -----------------------------
        mask = _load_mask_u8(mask_path)
        if mask is None:
            print(f"[WARN] Não consegui ler máscara: {mask_path.name}")
            continue

        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(
                mask,
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # --------------------------------------------------
        # WITH_DYNAMIC → apenas objetos dinâmicos
        # --------------------------------------------------
        with_dynamic = cv2.bitwise_and(img, img, mask=mask)

        # --------------------------------------------------
        # WITHOUT_DYNAMIC → remover objetos dinâmicos
        # --------------------------------------------------
        inv_mask = cv2.bitwise_not(mask)

        fill_img = np.full_like(img, fill_value, dtype=np.uint8)

        background = cv2.bitwise_and(img, img, mask=inv_mask)
        removed = cv2.bitwise_and(fill_img, fill_img, mask=mask)

        without_dynamic = cv2.add(background, removed)

        # Guardar resultados
        cv2.imwrite(str(out_with / img_path.name), with_dynamic)
        cv2.imwrite(str(out_without / img_path.name), without_dynamic)

        # Preview opcional
        if preview:
            show = np.hstack([with_dynamic, without_dynamic])
            cv2.imshow("With Dynamic | Without Dynamic", show)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        print(
            f"[OK] {img_path.name} → "
            f"with_dynamic/{video_id}/ | without_dynamic/{video_id}/"
        )

    if preview:
        cv2.destroyAllWindows()
