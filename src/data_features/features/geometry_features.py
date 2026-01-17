import cv2
import numpy as np


# Geometria Global
def geometry_edges_signature(gray, size=64):
    edges = cv2.Canny(gray, 100, 200)
    small = cv2.resize(edges, (size, size))

    sig = small.flatten().astype(np.float32)
    return sig / 255.0


def shape_hu_moments(gray):
    # Suavização leve para evitar ruído no threshold
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, th = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    m = cv2.moments(th)
    hu = cv2.HuMoments(m).flatten()

    # Log scale padrão (invariância a escala)
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu