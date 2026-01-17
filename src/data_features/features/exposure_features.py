import cv2
import numpy as np

def exposure_clipping(gray, low=5, high=250):
    total = gray.size
    clipped = np.sum((gray <= low) | (gray >= high))
    return clipped / total


def intensity_dynamic_range(gray):
    return np.percentile(gray, 99) - np.percentile(gray, 1)

