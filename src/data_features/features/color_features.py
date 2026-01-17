import cv2
import numpy as np


# Color
def color_hist_hsv(image, bins=(4, 4, 4)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def color_stats(image):
    mean = image.mean(axis=(0, 1))
    std = image.std(axis=(0, 1))
    return np.concatenate([mean, std])


# Iluminação, Saturação e Cor perceptual
def lightness_stats(image):
    """
    Estatísticas de lightness perceptual (canal L* do Lab).
    Útil para agrupar imagens com iluminação semelhante.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)

    return {
        "lightness_mean": float(np.mean(L)),
        "lightness_std": float(np.std(L)),
        "lightness_skew": float(np.mean((L - np.mean(L))**3) / (np.std(L)**3 + 1e-6))
    }

def saturation_stats(image):
    """
    Estatísticas do canal Saturation (HSV).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1].astype(np.float32)

    return {
        "saturation_mean": float(np.mean(S)),
        "saturation_std": float(np.std(S)),
        "saturation_low_frac": float(np.mean(S < 20))
    }

def brightness_stats(image):
    """
    Estatísticas do canal Value (HSV).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2].astype(np.float32)

    return {
        "brightness_mean": float(np.mean(V)),
        "brightness_std": float(np.std(V)),
        "brightness_clipped_frac": float(np.mean(V > 245))
    }

def colorfulness(image):
    """
    Métrica de colorfulness perceptual (Hasler & Süsstrunk).
    """
    B, G, R = cv2.split(image.astype(np.float32))

    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)

    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)

    return float(
        np.sqrt(std_rg**2 + std_yb**2) +
        0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
    )

def lightness_grid(image, grid=4):
    """
    Distribuição espacial da lightness (Lab L*) numa grelha.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)

    h, w = L.shape
    feats = []

    for gy in range(grid):
        for gx in range(grid):
            y1 = int(gy * h / grid)
            y2 = int((gy + 1) * h / grid)
            x1 = int(gx * w / grid)
            x2 = int((gx + 1) * w / grid)
            feats.append(np.mean(L[y1:y2, x1:x2]))

    return np.array(feats, dtype=np.float32) / 255.0
