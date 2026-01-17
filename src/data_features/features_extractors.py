import cv2
from data_features.roi_analysis import analisar_roi, get_center_crop

from data_features.features.color_features import color_hist_hsv,color_stats,lightness_stats,saturation_stats,brightness_stats,colorfulness,lightness_grid
from data_features.features.texture_features import texture_lbp
from data_features.features.geometry_features import geometry_edges_signature,shape_hu_moments
from data_features.features.orb_features import geometry_orb,orb_descriptor_stats,orb_kp_stats


def extrair_features_qualidade(
    image,
    log_sigma=1.0,
    log_quantile=0.999
):
    """
    Features escalares para:
    - decidir se a imagem é utilizável
    - rankear imagens semelhantes
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = {}

    # -------------------------
    # Qualidade global
    # -------------------------
    quality = analisar_roi(gray, log_sigma, log_quantile)
    for k, v in quality.items():
        features[f"quality_{k}"] = float(v)

    # -------------------------
    # Centro (foco / subject bias)
    # -------------------------
    roi = get_center_crop(gray, 0.3)
    roi_q = analisar_roi(roi, log_sigma, log_quantile)

    features["center_laplacian_ratio"] = (
        roi_q["laplacian_var"] /
        (quality["laplacian_var"] + 1e-6)
    )

    # -------------------------
    # Capacidade para reconstrução 3D (ORB)
    # -------------------------
    orb_stats = orb_kp_stats(gray)

    features["orb_kp_count"] = orb_stats["orb_kp_count"]
    features["orb_kp_entropy"] = orb_stats["orb_kp_entropy"]
    features["orb_response_mean"] = orb_stats["orb_response_mean"]
    features["orb_size_mean"] = orb_stats["orb_size_mean"]

    return features



def extrair_features_clustering(image):
    """
    Features para clustering e comparação entre imagens
    do mesmo objecto (robustas a blur e noise).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = {}

    # -------------------------
    # Cor (distribuição e estatísticas)
    # -------------------------
    features["color_hist_hsv"] = color_hist_hsv(image)
    features["color_stats"] = color_stats(image)

    # -------------------------
    # Iluminação / saturação
    # -------------------------
    features.update(lightness_stats(image))
    features.update(saturation_stats(image))
    features.update(brightness_stats(image))

    features["colorfulness"] = colorfulness(image)
    features["lightness_grid"] = lightness_grid(image)

    # -------------------------
    # Textura
    # -------------------------
    features["texture_lbp"] = texture_lbp(gray)

    # -------------------------
    # Geometria global
    # -------------------------
    features["geometry_edges"] = geometry_edges_signature(gray)
    features["geometry_hu"] = shape_hu_moments(gray)

    # -------------------------
    # Geometria local — ORB (estatísticas)
    # -------------------------
    features.update(
        orb_kp_stats(gray)
    )

    # -------------------------
    # Geometria local — ORB (descritores agregados)
    # -------------------------
    desc = geometry_orb(gray)
    features.update(
        orb_descriptor_stats(desc)
    )

    return features





def extrair_features_qualidade_path(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    return extrair_features_qualidade(image)


def extrair_features_clustering_path(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    return extrair_features_clustering(image)



