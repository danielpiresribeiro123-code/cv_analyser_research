import cv2
import time
from typing import Union, List, Iterator, Dict, Any, overload, Literal
import numpy as np

# Objeto sentinela usado para indicar que um frame foi ignorado
# por controlo de FPS, sem ser um erro real.
SKIP_FRAME = object()


class Camera:
    """
    Classe para captura de vídeo a partir de webcams, ficheiros
    de vídeo ou streams RTSP.

    Suporta:
    - configuração direta (int / str)
    - configuração normalizada (dict do config_cam.py)
    - controlo de FPS por software
    - recorte opcional para formato quadrado
    - iteração contínua de frames
    """

    def __init__(
        self,
        source: Union[int, str, List[Union[int, str]], Dict[str, Any]] | None = 0,
        square: bool = False,
        square_shift: float = 0.0,
        target_fps: float | None = None,
        backend: int | None = cv2.CAP_DSHOW,  # recomendado em Windows
    ):
        """
        Args:
            source:
                - int → webcam (0, 1, ...)
                - str → ficheiro de vídeo ou RTSP
                - list → lista de fontes (fallback automático)
                - dict → configuração devolvida por escolher_camera_inicial()
                - None → usa webcam (0)

            square:
                Se True, recorta o frame para um quadrado.

            square_shift:
                Deslocamento horizontal do quadrado (-1.0 a +1.0).

            target_fps:
                Limite de FPS por software (None = ilimitado).

            backend:
                Backend OpenCV (CAP_DSHOW recomendado no Windows).
        """

        self.square = square
        self.square_shift = max(-1.0, min(1.0, square_shift))
        self.target_fps = target_fps
        self._last_frame_time = 0.0

        # --------------------------------------------------
        # Normalização da fonte
        # --------------------------------------------------
        if source is None:
            source = 0

        # Se vier um dict (config_cam.py), extrair a fonte real
        if isinstance(source, dict):
            if "source" not in source:
                raise ValueError("Configuração de câmara inválida: falta chave 'source'")
            source = source["source"]

        # --------------------------------------------------
        # Abrir fonte de vídeo
        # --------------------------------------------------
        if isinstance(source, list):
            self.cap = self._abrir_primeira_fonte_valida(source, backend)
        else:
            self.cap = (
                cv2.VideoCapture(source, backend)
                if backend is not None
                else cv2.VideoCapture(source)
            )

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"❌ Não foi possível abrir a fonte de vídeo: {source}")

    # ------------------------------------------------------------
    def _abrir_primeira_fonte_valida(self, lista_fontes, backend):
        """
        Tenta abrir sequencialmente uma lista de fontes
        e devolve a primeira que funcionar.
        """
        for src in lista_fontes:
            cap = cv2.VideoCapture(src, backend) if backend else cv2.VideoCapture(src)
            if cap.isOpened():
                print(f"✔ Fonte encontrada: {src}")
                return cap
            cap.release()

        raise RuntimeError(f"❌ Nenhuma das fontes funcionou: {lista_fontes}")

    # ------------------------------------------------------------
    def _recortar_quadrado(self, frame):
        """
        Recorta o frame para um quadrado com deslocamento horizontal opcional.
        """
        h, w = frame.shape[:2]
        lado = min(h, w)

        x_base = (w - lado) // 2
        y_base = (h - lado) // 2

        max_shift = x_base
        shift_px = int(max_shift * self.square_shift)

        x_inicio = x_base + shift_px

        return frame[y_base:y_base + lado, x_inicio:x_inicio + lado]

    # ------------------------------------------------------------
    def read(self):
        """
        Lê um frame da câmara.

        Returns:
            - np.ndarray → frame válido
            - SKIP_FRAME → frame ignorado (FPS)
            - None → erro real ou stream terminado
        """
        # Controlo de FPS por software
        if self.target_fps:
            now = time.time()
            min_interval = 1.0 / self.target_fps

            if now - self._last_frame_time < min_interval:
                return SKIP_FRAME

            self._last_frame_time = now

        ok, frame = self.cap.read()
        if not ok:
            self.release()
            return None

        if self.square:
            frame = self._recortar_quadrado(frame)

        return frame

    # ------------------------------------------------------------
    def frames(self) -> Iterator:
        """
        Gerador contínuo de frames.

        Uso típico:
            for frame in camera.frames():
                ...
        """
        while True:
            frame = self.read()

            if frame is SKIP_FRAME:
                continue

            if frame is None:
                break

            yield frame

    # ------------------------------------------------------------
    def show(self, frame, window_name: str = "Video"):
        """Mostra um frame numa janela OpenCV."""
        cv2.imshow(window_name, frame)

    # ------------------------------------------------------------
    def get_source_fps(self) -> float:
        """Devolve o FPS reportado pela fonte de vídeo."""
        return float(self.cap.get(cv2.CAP_PROP_FPS))

    # ------------------------------------------------------------
    def release(self):
        """Liberta a câmara e fecha janelas OpenCV."""
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    # ------------------------------------------------------------
    # Suporte a: with Camera(...) as cam:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
