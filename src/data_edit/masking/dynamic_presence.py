from __future__ import annotations

from pathlib import Path
from typing import Set, Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO #type:ignore


# ======================================================
# Classes dinâmicas consideradas (IDs COCO)
# ======================================================
# 0: person
# 1: bicycle
# 2: car
# 3: motorcycle
# 5: bus
# 7: truck
# 15: cat
# 16: dog
DEFAULT_DYNAMIC_CLASS_IDS = {0, 1, 2, 3, 5, 7, 15, 16}


class DynamicPresenceDetector:
    """
    Detector leve de objetos dinâmicos baseado em YOLO (modo detecção).

    Este detector NÃO faz segmentação.
    É usado apenas para:
        - saber se um frame tem objetos dinâmicos
        - identificar que objetos são (classe + quantidade)

    Ideal para:
        - filtrar frames
        - classificar frames antes de gerar máscaras
        - criar relatórios semânticos
    """

    def __init__(
        self,
        model_path: Path,
        dynamic_class_ids: Set[int] = DEFAULT_DYNAMIC_CLASS_IDS,
        conf_threshold: float = 0.25,
        imgsz: int = 640,
        device: str | None = None,
    ):
        """
        Args:
            model_path:
                Caminho para modelo YOLO de detecção
                (ex: yolo11l.pt, yolo11m.pt).

            dynamic_class_ids:
                Conjunto de IDs COCO considerados "dinâmicos".

            conf_threshold:
                Confiança mínima para aceitar uma deteção.

            imgsz:
                Tamanho da imagem para inferência YOLO.
                Menor = mais rápido | Maior = mais preciso.

            device:
                Dispositivo de inferência:
                    - "cpu"
                    - "0" (GPU 0)
                    - None (auto)
        """
        self.model = YOLO(str(model_path))
        self.dynamic_class_ids = dynamic_class_ids
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device = device

        # Mapeamento id → nome da classe (COCO)
        self.class_names = self.model.names

    # --------------------------------------------------
    def _infer(
        self, frame: np.ndarray
    ) -> List[Tuple[str, float]]:
        """
        Executa inferência YOLO num frame e devolve deteções relevantes.

        Returns:
            Lista de tuplos:
                [(class_name, confidence), ...]
            apenas para classes dinâmicas e acima do threshold.
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        r = results[0]
        detections: List[Tuple[str, float]] = []

        if r.boxes is None:
            return detections

        for cls_id, conf in zip(r.boxes.cls, r.boxes.conf):
            cls_id = int(cls_id)
            conf = float(conf)

            # Ignorar classes não dinâmicas
            if cls_id not in self.dynamic_class_ids:
                continue

            if conf < self.conf_threshold:
                continue

            class_name = self.class_names.get(cls_id, f"class_{cls_id}")
            detections.append((class_name, conf))

        return detections

    # --------------------------------------------------
    def detect_dynamic_objects(
        self, frame: np.ndarray
    ) -> Dict[str, int]:
        """
        Deteta objetos dinâmicos num frame OpenCV.

        Args:
            frame:
                Imagem OpenCV (BGR).

        Returns:
            Dicionário:
                { "person": 2, "cow": 1 }

            - vazio {} → nenhum objeto dinâmico
            - cada frame aparece apenas num grupo semântico
        """
        if frame is None or frame.size == 0:
            return {}

        detections = self._infer(frame)

        counts: Dict[str, int] = {}
        for class_name, _ in detections:
            counts[class_name] = counts.get(class_name, 0) + 1

        return counts

    # --------------------------------------------------
    def has_dynamic_objects_from_frame(self, frame: np.ndarray) -> bool:
        """
        Verifica rapidamente se um frame contém objetos dinâmicos.

        Útil para:
            - filtros rápidos
            - decisões booleanas

        Returns:
            True se existir pelo menos um objeto dinâmico.
        """
        if frame is None or frame.size == 0:
            return False

        detections = self._infer(frame)
        return len(detections) > 0

    # --------------------------------------------------
    def has_dynamic_objects(self, image_path: Path) -> bool:
        """
        Verifica se uma imagem no disco contém objetos dinâmicos.

        Args:
            image_path:
                Caminho para a imagem.

        Returns:
            True se existir pelo menos um objeto dinâmico.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return False

        return self.has_dynamic_objects_from_frame(img)
