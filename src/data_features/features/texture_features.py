import cv2
import numpy as np
from skimage.feature import local_binary_pattern


# Textura
def texture_lbp(gray, radius=2, points=16):
    lbp = local_binary_pattern(gray, points, radius, method="uniform")
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=points + 2,
        range=(0, points + 2),
        density=True
    )
    return hist