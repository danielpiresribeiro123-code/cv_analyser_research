import cv2
import numpy as np
# Noise
def noise_residual_std(gray, ksize=3):
    blur = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    residual = gray.astype(np.float32) - blur.astype(np.float32)
    return np.std(residual)


def noise_flat_regions(gray, block=16, grad_thresh=5):
    h, w = gray.shape
    noises = []

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    grad = np.sqrt(gx**2 + gy**2)

    for y in range(0, h - block, block):
        for x in range(0, w - block, block):
            g_patch = grad[y:y+block, x:x+block]
            if np.mean(g_patch) < grad_thresh:
                patch = gray[y:y+block, x:x+block]
                noises.append(np.std(patch))

    return np.mean(noises) if noises else 0.0


def noise_fft(gray, size=10):
    fft = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    h, w = gray.shape
    mask = np.ones_like(gray, dtype=bool)
    cy, cx = h // 2, w // 2
    mask[cy-size:cy+size, cx-size:cx+size] = False
    return np.mean(np.abs(fft[mask]))


def jpeg_blockiness(gray, block=8):
    # Vertical edges
    cols1 = gray[:, block::block]
    cols2 = gray[:, block-1::block]

    min_cols = min(cols1.shape[1], cols2.shape[1])
    if min_cols > 0:
        v_edges = np.abs(cols1[:, :min_cols] - cols2[:, :min_cols])
        v_mean = np.mean(v_edges)
    else:
        v_mean = 0.0

    # Horizontal edges
    rows1 = gray[block::block, :]
    rows2 = gray[block-1::block, :]

    min_rows = min(rows1.shape[0], rows2.shape[0])
    if min_rows > 0:
        h_edges = np.abs(rows1[:min_rows, :] - rows2[:min_rows, :])
        h_mean = np.mean(h_edges)
    else:
        h_mean = 0.0

    return v_mean + h_mean
