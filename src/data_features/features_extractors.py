import cv2
from data_features.features.blur_features import blur_grid_features
from data_features.features.color_features import (color_hist_hsv,color_stats,lightness_stats,
    saturation_stats,brightness_stats,colorfulness,lightness_grid)
from data_features.features.texture_features import texture_lbp
from data_features.features.geometry_features import geometry_edges_signature,shape_hu_moments
from data_features.features.orb_features import geometry_orb,orb_descriptor_stats,orb_kp_stats
from data_features.roi_analysis import analisar_roi

def extrair_features_common(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = {}

    # Cor e iluminação
    features["c_color_hist_hsv"] = color_hist_hsv(image)
    features["c_color_stats"] = color_stats(image)
    for k, v in lightness_stats(image).items():
        features[f"c_{k}"] = v
    for k, v in saturation_stats(image).items():
        features[f"c_{k}"] = v
    for k, v in brightness_stats(image).items():
        features[f"c_{k}"] = v
    features["c_colorfulness"] = colorfulness(image)
    features["c_lightness_grid"] = lightness_grid(image)

    # Textura
    features["c_texture_lbp"] = texture_lbp(gray)

    # Geometria global
    features["c_geometry_edges"] = geometry_edges_signature(gray)
    features["c_geometry_hu"] = shape_hu_moments(gray)

    # ORB (estatísticas básicas)
    orb_stats = orb_kp_stats(gray)
    for k, v in orb_stats.items():
        features[f"c_{k}"] = float(v)

    # Blur grid
    blur_grid = blur_grid_features(gray, grid=4)

    # Common
    features["c_blur_lap_mean"]  = blur_grid["blur_lap_mean"]
    features["c_blur_lap_ratio"] = blur_grid["blur_lap_ratio"]
    features["c_blur_lap_ratio_max"] = blur_grid["blur_lap_ratio"]
    features["c_blur_ten_ratio"] = blur_grid["blur_ten_ratio"]
    features["c_blur_ten_ratio_max"] = blur_grid["blur_ten_ratio"]

    # Blur grid completo, rico, só para qualidade
    features["q_blur_lap_std"]   = blur_grid["blur_lap_std"]
    features["q_blur_lap_p10"]   = blur_grid["blur_lap_p10"]
    features["q_blur_lap_p90"]   = blur_grid["blur_lap_p90"]
    features["q_blur_lap_max"]   = blur_grid["blur_lap_max"]

    features["q_blur_ten_mean"]  = blur_grid["blur_ten_mean"]
    features["q_blur_ten_std"]   = blur_grid["blur_ten_std"]
    features["q_blur_ten_p10"]   = blur_grid["blur_ten_p10"]
    features["q_blur_ten_p90"]   = blur_grid["blur_ten_p90"]
    features["q_blur_ten_max"]   = blur_grid["blur_ten_max"]

    features["q_motion_mean"]  = blur_grid["motion_mean"]
    features["q_motion_std"]   = blur_grid["motion_std"]
    features["q_motion_p90"]   = blur_grid["motion_p90"]
    features["q_motion_max"]   = blur_grid["motion_max"]
    features["q_motion_ratio"]   = blur_grid["motion_ratio"]
    features["q_motion_ratio_max"]   = blur_grid["motion_ratio"]


    return features


def extrair_features_quality_only(image, log_sigma=1.0, log_quantile=0.999, blur_grid_size=4):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = {}

    # Qualidade global clássica
    quality = analisar_roi(gray, log_sigma, log_quantile)
    for k, v in quality.items():
        features[f"q_{k}"] = float(v)

    return features


def extrair_features_clustering_only(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = {}

    # ORB descritores agregados
    desc = orb_descriptor_stats(geometry_orb(gray))
    for k, v in desc.items():
        features[f"cl_{k}"] = float(v)

    return features





def extrair_features_qualidade(image):
    features = {}
    features.update(extrair_features_common(image))
    features.update(extrair_features_quality_only(image))
    return features


def extrair_features_clustering(image):
    features = {}
    features.update(extrair_features_common(image))
    features.update(extrair_features_clustering_only(image))
    return features
