# Dependências:
import cv2

from data_features.features.blur_features import (blur_laplacian, blur_fft, blur_motion, blur_local,
    blur_contrast, blur_tenengrad, blur_log_quantile,blur_grid_features, blur_all_features)
from data_features.features.noise_features import noise_residual_std, noise_flat_regions, noise_fft, jpeg_blockiness
from data_features.features.exposure_features import exposure_clipping, intensity_dynamic_range

from data_features.features.color_features import (color_hist_hsv,color_stats,lightness_stats,
    saturation_stats,brightness_stats,colorfulness,lightness_grid)
from data_features.features.texture_features import texture_lbp

from data_features.features.geometry_features import geometry_edges_signature,shape_hu_moments
from data_features.features.orb_features import geometry_orb,orb_descriptor_stats,orb_kp_stats

# Extras:
def analisar_quality_features(gray, log_sigma=1.0, log_quantile=0.999, block=32):
    """
    Analisa quality local/global de uma imagem.
    Retorna apenas features escalares.
    """
    local_mean, local_std = blur_local(gray, block)

    return {
        # Nitidez / blur (global)
        "blur_fft": blur_fft(gray),
        "blur_local_laplacian_mean": local_mean,
        "blur_local_laplacian_std": local_std,
        "blur_contrast": blur_contrast(gray),
        "blur_log_quantile": blur_log_quantile(gray, log_sigma, log_quantile),

        # Ruído / compressão
        "noise_residual": noise_residual_std(gray),
        "noise_flat": noise_flat_regions(gray),
        "noise_fft": noise_fft(gray),
        "jpeg_blockiness": jpeg_blockiness(gray),

        # Exposição / gama dinâmica
        "exposure_clipping": exposure_clipping(gray),
        "dynamic_range": intensity_dynamic_range(gray),
    }

def analisar_cor_iluminacao(image, features: dict, type = "c"):
    """
    Analisa a cor e a iluminação de uma imagem.
    Retorna apenas features escalares.
    """
    features[f"{type}_color_hist_hsv"] = color_hist_hsv(image)
    features[f"{type}_color_stats"] = color_stats(image)
    for k, v in lightness_stats(image).items():
        features[f"{type}_{k}"] = v
    for k, v in saturation_stats(image).items():
        features[f"{type}_{k}"] = v
    for k, v in brightness_stats(image).items():
        features[f"{type}_{k}"] = v
    features[f"{type}_colorfulness"] = colorfulness(image)
    features[f"{type}_lightness_grid"] = lightness_grid(image)

    return features

def gray_image(image, gray = None):
    if gray is None:
        try:
            gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            raise ValueError(f"[ERRO] Não foi possível converter a imagem para grayscale: {e}")
    return gray




# Função common version 1:

def extrair_features_common_v1(image, gray = None, blur_grid_size=4, common_color_light_features:bool=True):
    gray=gray_image(image, gray)
    features = {}

    # Cor e iluminação
    if common_color_light_features:
        features = analisar_cor_iluminacao(image, features, type = "c")

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
    blur_grid = blur_grid_features(gray, blur_grid_size)

    # Common
    features["c_grid_blur_lap_mean"]  = blur_grid["grid_blur_lap_mean"]
    features["c_grid_blur_lap_ratio"] = blur_grid["grid_blur_lap_ratio"]
    features["c_grid_blur_lap_ratio_max"] = blur_grid["grid_blur_lap_ratio_max"]

    features["c_grid_blur_ten_ratio"] = blur_grid["grid_blur_ten_ratio"]
    features["c_grid_blur_ten_ratio_max"] = blur_grid["grid_blur_ten_ratio_max"]

    features["c_grid_motion_ratio"]   = blur_grid["grid_motion_ratio"]
    features["c_grid_motion_ratio_max"]   = blur_grid["grid_motion_ratio_max"]

    features["c_global_blur_laplacian"] = blur_laplacian(gray)
    features["c_global_blur_tenengrad"] = blur_tenengrad(gray)
    features["c_global_blur_motion"] = blur_motion(gray)

    # Blur só para quality
    features["q_grid_blur_lap_std"]  = blur_grid["grid_blur_lap_std"]
    features["q_grid_blur_lap_p10"]  = blur_grid["grid_blur_lap_p10"]
    features["q_grid_blur_lap_p90"]  = blur_grid["grid_blur_lap_p90"]
    features["q_grid_blur_lap_max"]  = blur_grid["grid_blur_lap_max"]

    features["q_grid_blur_ten_mean"]  = blur_grid["grid_blur_ten_mean"]
    features["q_grid_blur_ten_std"]   = blur_grid["grid_blur_ten_std"]
    features["q_grid_blur_ten_p10"]   = blur_grid["grid_blur_ten_p10"]
    features["q_grid_blur_ten_p90"]   = blur_grid["grid_blur_ten_p90"]
    features["q_grid_blur_ten_max"]   = blur_grid["grid_blur_ten_max"]

    features["q_grid_motion_mean"]  = blur_grid["grid_motion_mean"]
    features["q_grid_motion_std"]   = blur_grid["grid_motion_std"]
    features["q_grid_motion_p10"]   = blur_grid["grid_motion_p10"]
    features["q_grid_motion_p90"]   = blur_grid["grid_motion_p90"]
    features["q_grid_motion_max"]   = blur_grid["grid_motion_max"]

    return features




# Função common version 2:

def extrair_features_common_v2(image, gray=None,blur_grid_size=4, common_color_light_features:bool=True):
    gray=gray_image(image, gray)
    features = {}

    # Cor e iluminação
    if common_color_light_features:
        features = analisar_cor_iluminacao(image, features, type = "c")

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
    blur_all = blur_all_features(gray, blur_grid_size)

    # Blur common
    features["c_grid_blur_lap_mean"]  = blur_all["grid_blur_lap_mean"]
    features["c_grid_blur_lap_ratio"] = blur_all["grid_blur_lap_ratio"]
    features["c_grid_blur_lap_ratio_max"] = blur_all["grid_blur_lap_ratio_max"]

    features["c_grid_blur_ten_ratio"] = blur_all["grid_blur_ten_ratio"]
    features["c_grid_blur_ten_ratio_max"] = blur_all["grid_blur_ten_ratio_max"]

    features["c_grid_motion_ratio"]   = blur_all["grid_motion_ratio"]
    features["c_grid_motion_ratio_max"]   = blur_all["grid_motion_ratio_max"]

    features["c_global_blur_laplacian"] =blur_all["global_blur_laplacian"]
    features["c_global_blur_tenengrad"] =blur_all["global_blur_tenengrad"]
    features["c_global_blur_motion"] = blur_all["global_blur_motion"]

    # Blur só para quality
    features["q_grid_blur_lap_std"]  = blur_all["grid_blur_lap_std"]
    features["q_grid_blur_lap_p10"]  = blur_all["grid_blur_lap_p10"]
    features["q_grid_blur_lap_p90"]  = blur_all["grid_blur_lap_p90"]
    features["q_grid_blur_lap_max"]  = blur_all["grid_blur_lap_max"]

    features["q_grid_blur_ten_mean"]  = blur_all["grid_blur_ten_mean"]
    features["q_grid_blur_ten_std"]   = blur_all["grid_blur_ten_std"]
    features["q_grid_blur_ten_p10"]   = blur_all["grid_blur_ten_p10"]
    features["q_grid_blur_ten_p90"]   = blur_all["grid_blur_ten_p90"]
    features["q_grid_blur_ten_max"]   = blur_all["grid_blur_ten_max"]

    features["q_grid_motion_mean"]  = blur_all["grid_motion_mean"]
    features["q_grid_motion_std"]   = blur_all["grid_motion_std"]
    features["q_grid_motion_p10"]   = blur_all["grid_motion_p10"]
    features["q_grid_motion_p90"]   = blur_all["grid_motion_p90"]
    features["q_grid_motion_max"]   = blur_all["grid_motion_max"]


    # Relações global (são common/ambas)
    features["c_global_blur_laplacian_minus_grid_mean"] = blur_all["lap_global_minus_grid_mean"]
    features["c_global_blur_laplacian_ratio_grid_mean"] = blur_all["lap_global_ratio_grid_mean"]
    
    features["c_global_blur_tenengrad_minus_grid_mean"] = blur_all["ten_global_minus_grid_mean"]
    features["c_global_blur_tenengrad_ratio_grid_mean"] = blur_all["ten_global_ratio_grid_mean"]

    features["c_global_blur_motion_minus_grid_mean"] = blur_all["motion_global_minus_grid_mean"]
    features["c_global_blur_motion_ratio_grid_mean"] = blur_all["motion_global_ratio_grid_mean"]


    return features





# Função common version 3:

def extrair_features_common_v3(image, gray=None,common_color_light_features:bool=True):
    gray=gray_image(image, gray)
    features = {}

    # Cor e iluminação
    if common_color_light_features:
        features = analisar_cor_iluminacao(image, features, type = "c")

    # Textura
    features["c_texture_lbp"] = texture_lbp(gray)

    # Geometria global
    features["c_geometry_edges"] = geometry_edges_signature(gray)
    features["c_geometry_hu"] = shape_hu_moments(gray)

    # ORB (estatísticas básicas)
    orb_stats = orb_kp_stats(gray)
    for k, v in orb_stats.items():
        features[f"c_{k}"] = float(v)

    # Blur
    features["c_global_blur_laplacian"] = blur_laplacian(gray)
    features["c_global_blur_tenengrad"] = blur_tenengrad(gray)
    features["c_global_blur_motion"] = blur_motion(gray)

    return features





# Funções que inglobam todas as versões

def extrair_features_common(image, gray=None,blur_grid_size=4, common_color_light_features:bool=True, version = 1):
    if version == 1:
        return extrair_features_common_v1(image, gray,blur_grid_size, common_color_light_features)
    if version == 2:
        return extrair_features_common_v2(image, gray,blur_grid_size, common_color_light_features)
    if version == 3:
        return extrair_features_common_v3(image, gray, common_color_light_features) # Sem blur_grid_size
    else:
        raise ValueError("Selecione entre as versões 1, 2 e 3.")


def extrair_features_quality_only(image, gray=None,log_sigma=1.0, log_quantile=0.999, block=32):
    gray=gray_image(image, gray)
    features = {}

    # quality global clássica
    quality = analisar_quality_features(gray, log_sigma, log_quantile, block)
    for k, v in quality.items():
        features[f"q_{k}"] = float(v)

    return features


def extrair_features_clustering_only(image, gray=None, common_color_light_features:bool=True):
    gray=gray_image(image, gray)
    features = {}
    # Cor e iluminação
    if common_color_light_features == False:
        features = analisar_cor_iluminacao(image, features, type = "cl")

    # ORB descritores agregados
    desc = orb_descriptor_stats(geometry_orb(gray))
    for k, v in desc.items():
        features[f"cl_{k}"] = float(v)

    return features


def extrair_features_quality(image, gray=None, log_sigma=1.0, log_quantile=0.999, block=32, blur_grid_size=4, common_color_light_features:bool=True, version=1):
    features = {}
    features.update(extrair_features_common(image, gray,blur_grid_size, common_color_light_features, version))
    features.update(extrair_features_quality_only(image,gray,log_sigma, log_quantile, block))

    return features


def extrair_features_clustering(image, gray=None, blur_grid_size=4, common_color_light_features:bool=True, version=1):
    features = {}
    features.update(extrair_features_common(image, gray,blur_grid_size, common_color_light_features, version))
    features.update(extrair_features_clustering_only(image,gray,common_color_light_features))

    return features

def extrair_features_total(image, gray=None, log_sigma=1.0, log_quantile=0.999, block=32, blur_grid_size=4, common_color_light_features:bool=True, version=1):
    features = {}
    features.update(extrair_features_common(image, gray,blur_grid_size, common_color_light_features, version))
    features.update(extrair_features_quality_only(image,gray,log_sigma, log_quantile, block))
    features.update(extrair_features_clustering_only(image,gray,common_color_light_features))

    return features