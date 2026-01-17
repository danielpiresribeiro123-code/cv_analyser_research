import cv2
import numpy as np

# Geometria Local
def geometry_orb(
    gray,
    max_kp=800,
    fast_threshold=30
):
    """
    Extrai descritores ORB para posterior agregação estatística.
    NÃO deve ser usado directamente para clustering.

    Returns:
        np.ndarray ou None (Nx32 uint8)
    """
    orb = cv2.ORB_create(nfeatures=max_kp,fastThreshold=fast_threshold) #type: ignore

    kps, desc = orb.detectAndCompute(gray, None)

    if desc is None or len(kps) == 0:
        return None

    return desc


def orb_descriptor_stats(desc):
    """
    Converte descritores ORB (binários) em estatísticas globais
    adequadas para clustering.
    """
    if desc is None or len(desc) == 0:
        return {
            "orb_desc_mean": np.zeros(32, dtype=np.float32),
            "orb_desc_std": np.zeros(32, dtype=np.float32),
        }

    desc = desc.astype(np.float32)

    return {
        "orb_desc_mean": np.mean(desc, axis=0),
        "orb_desc_std": np.std(desc, axis=0),
    }



# ORB — estatísticas espaciais
def orb_kp_stats(gray, max_kp=800, grid=4, fast_threshold=30):
    """
    Estatísticas espaciais e de qualidade dos keypoints ORB.

    Returns:
        dict com:
        - kp_count
        - kp_entropy
        - orb_response_mean
        - orb_size_mean
    """
    orb = cv2.ORB_create(nfeatures=max_kp,fastThreshold=fast_threshold) #type: ignore

    kps = orb.detect(gray, None)
    n = len(kps)

    # -------------------------
    # Distribuição espacial
    # -------------------------
    h, w = gray.shape
    counts = np.zeros((grid, grid), dtype=np.float32)

    for kp in kps:
        x, y = kp.pt
        gx = min(grid - 1, int(x / w * grid))
        gy = min(grid - 1, int(y / h * grid))
        counts[gy, gx] += 1

    counts = counts.flatten()
    if counts.sum() > 0:
        counts /= counts.sum()

    kp_entropy = -np.sum(counts * np.log(counts + 1e-9))

    # -------------------------
    # Estatísticas dos keypoints
    # -------------------------
    if n > 0:
        responses = np.array([kp.response for kp in kps], dtype=np.float32)
        sizes = np.array([kp.size for kp in kps], dtype=np.float32)

        response_mean = float(np.mean(responses))
        size_mean = float(np.mean(sizes))
    else:
        response_mean = 0.0
        size_mean = 0.0

    return {
        "orb_kp_count": float(n),
        "orb_kp_entropy": float(kp_entropy),
        "orb_response_mean": response_mean,
        "orb_size_mean": size_mean,
    }
