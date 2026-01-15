from __future__ import annotations

import cv2
import time
import json
from datetime import datetime
from typing import Optional
import numpy as np
from pathlib import Path

from capture.camera import Camera, SKIP_FRAME
from capture.config_cam import escolher_camera_inicial


def record_video(
    output_name: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    preview: bool = True,
    square: bool = False,
    square_shift: float = 0.0,
    target_fps: Optional[float] = None,
    auto_camera: bool = False,
    camera_type: int | str = "1",
    usb_source: int = 0,
    ip_source: str = "",
    videos_dir: Path | str = "data/raw/videos",
) -> Path:
    """
    Grava um vídeo a partir de uma câmara configurada.

    Args:
        output_name:
            Nome base do ficheiro (sem extensão).
            Se None, é gerado automaticamente com timestamp.

        duration_seconds:
            Duração máxima da gravação em segundos.
            Se None:
                - preview=True  → grava até tecla 'q'
                - preview=False → grava até Ctrl+C

        preview:
            Se True, mostra preview em tempo real.

        square:
            Se True, aplica recorte quadrado aos frames.

        square_shift:
            Deslocamento horizontal do quadrado (-1.0 a +1.0).

        target_fps:
            Limita o FPS gravado (None = FPS da fonte).

        auto_camera:
            Se True, usa seleção automática de câmara.

        camera_type:
            Tipo de câmara em modo automático:
                "1" → USB
                "2" → IP

        usb_source:
            ID da webcam (modo automático).

        ip_source:
            URL RTSP (modo automático).

        videos_dir:
            Diretório onde o vídeo e metadata serão guardados.

    Returns:
        Path do vídeo gravado.
    """

    # --------------------------------------------------
    # Preparar diretório de saída
    # --------------------------------------------------
    videos_dir = Path(videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Seleção da câmara
    # --------------------------------------------------
    cam_cfg = escolher_camera_inicial(
        auto=auto_camera,
        tipo=camera_type,
        usb_source=usb_source,
        ip_source=ip_source,
    )

    if cam_cfg is None:
        raise RuntimeError("❌ Falha na seleção da câmara.")

    # --------------------------------------------------
    # Nome do ficheiro
    # --------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = output_name or f"recording_{timestamp}"

    video_path = videos_dir / f"{base_name}.mp4"
    meta_path = videos_dir / f"{base_name}.json"

    # --------------------------------------------------
    # Inicializar câmara
    # --------------------------------------------------
    with Camera(
        source=cam_cfg,
        square=square,
        square_shift=square_shift,
        target_fps=target_fps,
    ) as cam:

        # Ler primeiro frame válido
        first_frame: np.ndarray | None = None

        while first_frame is None:
            frame = cam.read()
            if frame is SKIP_FRAME:
                continue
            first_frame = frame

        height, width = first_frame.shape[:2]

        source_fps = cam.get_source_fps()
        fps = target_fps or (source_fps if source_fps > 0 else 30.0)

        # --------------------------------------------------
        # Inicializar VideoWriter
        # --------------------------------------------------
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            fps,
            (width, height),
        )

        print(f"\n🎥 A gravar vídeo → {video_path}")
        print(f"   Resolução: {width}x{height}")
        print(f"   FPS: {fps:.2f}")
        print("   Prima 'q' para terminar.\n")

        start_time = time.time()
        frames_written = 0
        interrupted = False

        writer.write(first_frame)
        frames_written += 1

        # --------------------------------------------------
        # Loop principal
        # --------------------------------------------------
        try:
            for frame in cam.frames():
                writer.write(frame)
                frames_written += 1

                if preview:
                    cam.show(frame, "Recording")

                if duration_seconds is not None:
                    if time.time() - start_time >= duration_seconds:
                        print("⏹ Duração máxima atingida.")
                        break

                if preview and (cv2.waitKey(1) & 0xFF == ord("q")):
                    print("⏹ Gravação terminada pelo utilizador.")
                    break

        except KeyboardInterrupt:
            interrupted = True
            print("\n⏹ Gravação interrompida pelo utilizador (Ctrl+C).")

        finally:
            writer.release()

    # --------------------------------------------------
    # Guardar metadata
    # --------------------------------------------------
    metadata = {
        "video_file": video_path.name,
        "timestamp": timestamp,
        "resolution": [width, height],
        "fps": fps,
        "frames_written": frames_written,
        "duration_seconds": frames_written / fps if fps > 0 else None,
        "camera_type": cam_cfg["type"],
        "camera_source": cam_cfg["source"],
        "square": square,
        "square_shift": square_shift,
        "target_fps": target_fps,
        "interrupted": interrupted,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"\n✔ Vídeo gravado com sucesso.")
    print(f"✔ Metadata guardada → {meta_path}\n")

    return video_path
