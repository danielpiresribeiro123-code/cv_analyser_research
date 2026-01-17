from data_features.features.blur_features import blur_laplacian, blur_fft, blur_motion, blur_local, blur_contrast, blur_tenengrad, blur_log_quantile
from data_features.features.noise_features import noise_residual_std, noise_flat_regions, noise_fft, jpeg_blockiness
from data_features.features.exposure_features import exposure_clipping, intensity_dynamic_range


def get_center_crop(gray, frac):
    """
    Extrai um crop centrado da imagem.

    Args:
        gray (np.ndarray): Imagem em grayscale.
        frac (float): Fração do tamanho (ex: 0.3 = 30%).

    Returns:
        np.ndarray: Região central da imagem.
    """
    h, w = gray.shape
    ch, cw = int(h * frac), int(w * frac)

    y1 = (h - ch) // 2
    x1 = (w - cw) // 2

    return gray[y1:y1 + ch, x1:x1 + cw]



def analisar_roi(gray, log_sigma=1.0, log_quantile=0.999):
    """
    Analisa qualidade local/global de uma ROI ou da imagem inteira.
    Retorna apenas features escalares.
    """
    local_mean, local_std = blur_local(gray)

    return {
        # Nitidez / blur
        "laplacian_var": blur_laplacian(gray),
        "fft_energy": blur_fft(gray),
        "motion_dominance": blur_motion(gray),
        "local_laplacian_mean": local_mean,
        "local_laplacian_std": local_std,
        "contrast_energy": blur_contrast(gray),
        "tenengrad_energy": blur_tenengrad(gray),
        "log_quantile": blur_log_quantile(gray, log_sigma, log_quantile),

        # Ruído / compressão
        "noise_residual": noise_residual_std(gray),
        "noise_flat": noise_flat_regions(gray),
        "noise_fft": noise_fft(gray),
        "jpeg_blockiness": jpeg_blockiness(gray),

        # Exposição / gama dinâmica
        "exposure_clipping": exposure_clipping(gray),
        "dynamic_range": intensity_dynamic_range(gray),
    }