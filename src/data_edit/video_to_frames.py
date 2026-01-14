from __future__ import annotations

import cv2
from pathlib import Path
from typing import Iterable, Optional


# --------------------------------------------------
# Utilitário: remover barras pretas (simples e robusto)
# --------------------------------------------------
def _crop_black_borders(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        return frame

    x, y, w, h = cv2.boundingRect(coords)
    return frame[y:y + h, x:x + w]


# --------------------------------------------------
# Função principal
# --------------------------------------------------
def video_to_frames(
    video_path: Path | str,
    output_dir: Path | str,
    *,
    prefix: Optional[str] = None,
    start_index: int = 0,
    every_n_frames: int = 1,
    max_frames: Optional[int] = None,
    cut_black_sides: bool = False,
    overwrite: bool = False,
    image_ext: str = ".png",
) -> Path:
    """
    Extrai frames de um vídeo para imagens.

    Args:
        video_path:
            Caminho para o vídeo.

        output_dir:
            Diretório base onde os frames serão guardados.

        prefix:
            Prefixo dos nomes dos frames.
            Default: nome do vídeo.

        start_index:
            Índice inicial dos frames.

        every_n_frames:
            Guarda apenas 1 em cada N frames.

        max_frames:
            Limite máximo de frames extraídos (None = sem limite).

        cut_black_sides:
            Remove automaticamente bordas pretas.

        overwrite:
            Se True, apaga frames existentes.

        image_ext:
            Extensão das imagens (".png", ".jpg").

    Returns:
        Path da pasta onde os frames foram guardados.
    """

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

    if image_ext not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("image_ext deve ser .png ou .jpg")

    video_name = video_path.stem
    prefix = prefix or video_name

    frames_dir = output_dir / video_name

    if frames_dir.exists() and overwrite:
        for f in frames_dir.glob("*"):
            f.unlink()

    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\n🎞️ Extração de frames")
    print(f"   Vídeo: {video_path.name}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total frames: {total_frames}")
    print(f"   Pasta destino: {frames_dir}")

    frame_idx = 0
    saved_idx = start_index
    extracted = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n_frames != 0:
            frame_idx += 1
            continue

        if cut_black_sides:
            frame = _crop_black_borders(frame)

        frame_name = f"{prefix}_{saved_idx:06d}{image_ext}"
        frame_path = frames_dir / frame_name

        cv2.imwrite(str(frame_path), frame)

        extracted += 1
        saved_idx += 1
        frame_idx += 1

        if max_frames is not None and extracted >= max_frames:
            break

    cap.release()

    print(f"✔ Frames extraídos: {extracted}\n")

    return frames_dir
