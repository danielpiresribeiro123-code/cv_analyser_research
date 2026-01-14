import cv2
import numpy as np
from skimage.feature import local_binary_pattern


# Blur
def blur_laplacian(gray):
    """
    Mede blur global (defocus blur) usando a variância do Laplaciano.

    A ideia base é:
    - Imagens nítidas têm muitas bordas → Laplaciano com grande variação
    - Imagens desfocadas têm poucas bordas → baixa variância

    Args:
        gray (numpy.ndarray): Imagem em escala de cinzentos.

    Returns:
        float: Variância do Laplaciano (quanto maior, mais nítida).
    """
    # Aplica o operador Laplaciano para realçar bordas
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Calcula a variância do Laplaciano
    return laplacian.var()

def blur_fft(gray, size=30):
    """
    Deteta blur através da análise de frequências altas (FFT).

    O blur remove altas frequências da imagem.
    Ao eliminar o centro do espectro (baixas frequências),
    medimos quanta energia resta.

    Args:
        gray (numpy.ndarray): Imagem em escala de cinzentos.
        size (int): Tamanho da região central removida no espectro.

    Returns:
        float: Energia média das altas frequências.
    """
    # Dimensões da imagem
    h, w = gray.shape

    # FFT 2D da imagem
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)

    # Coordenadas do centro
    cx, cy = w // 2, h // 2

    # Remove baixas frequências (centro)
    fft_shift[cy-size:cy+size, cx-size:cx+size] = 0

    # Volta ao domínio espacial
    fft_back = np.fft.ifft2(np.fft.ifftshift(fft_shift))

    # Energia média das altas frequências
    return np.mean(np.abs(fft_back))

def blur_motion(gray):
    """
    Deteta motion blur analisando a direção dominante dos gradientes.

    Motion blur cria bordas fortes numa direção específica.
    Se uma direção dominar claramente, há forte indício de motion blur.

    Args:
        gray (numpy.ndarray): Imagem em escala de cinzentos.

    Returns:
        float: Grau de dominância direcional (valores altos → motion blur).
    """
    # Gradientes horizontal e vertical
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Magnitude e ângulo dos gradientes
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)

    # Histograma das direções ponderado pela magnitude
    hist, _ = np.histogram(
        angle,
        bins=36,
        range=(-np.pi, np.pi),
        weights=magnitude
    )

    # Dominância: pico máximo vs média
    return hist.max() / (hist.mean() + 1e-6)

def blur_local(gray, block=32):
    """
    Deteta blur parcial (ex: fundo desfocado) analisando blocos da imagem.

    Divide a imagem em blocos e calcula o Laplaciano em cada um.
    Permite perceber variações locais de nitidez.

    Args:
        gray (numpy.ndarray): Imagem em escala de cinzentos.
        block (int): Tamanho do bloco em pixels.

    Returns:
        tuple:
            - média da variância do Laplaciano
            - desvio padrão (indica heterogeneidade de blur)
    """
    h, w = gray.shape
    variances = []

    # Percorre a imagem por blocos
    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            patch = gray[y:y+block, x:x+block]

            # Variância do Laplaciano por bloco
            variances.append(
                cv2.Laplacian(patch, cv2.CV_64F).var()
            )

    return np.mean(variances), np.std(variances)

def blur_contrast(gray):
    """
    Mede perda de contraste através da energia dos gradientes.

    Imagens com pouco contraste ou neblina
    tendem a ter gradientes fracos.

    Args:
        gray (numpy.ndarray): Imagem em escala de cinzentos.

    Returns:
        float: Energia média do gradiente.
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    # Energia média do gradiente
    return np.mean(np.sqrt(gx**2 + gy**2))

def blur_tenengrad(gray):
    """
    Deteta blur artificial (suavização, compressão, AI).

    O método Tenengrad usa a energia do Sobel,
    sendo mais robusto que o Laplaciano em certos cenários.

    Args:
        gray (numpy.ndarray): Imagem em escala de cinzentos.

    Returns:
        float: Energia Tenengrad (quanto maior, mais nítida).
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    return np.mean(gx**2 + gy**2)

def blur_log_quantile(gray, sigma=1.0, quantile=0.999):
    """
    Mede nitidez usando o quantil alto do Laplacian of Gaussian (LoG).

    Este método foca-se nas bordas mais fortes da imagem,
    sendo menos sensível à média global.

    Args:
        gray (numpy.ndarray): Imagem em escala de cinzentos.
        sigma (float): Sigma do Gaussian (controlo de suavização).
        quantile (float): Quantil alto (ex: 0.999 = 99.9%).

    Returns:
        float: Valor do quantil do LoG (quanto maior, mais nítida).
    """

    # Normalizar contraste (reduz influência de iluminação)
    gray_eq = cv2.equalizeHist(gray)

    # Suavização Gaussiana
    blurred = cv2.GaussianBlur(gray_eq, (0, 0), sigmaX=sigma)

    # Laplaciano (LoG aproximado)
    log = cv2.Laplacian(blurred, cv2.CV_64F)

    # Valor absoluto (força das bordas)
    log_abs = np.abs(log)

    # Quantil alto (robusto a ruído)
    return np.quantile(log_abs, quantile)

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



def exposure_clipping(gray, low=5, high=250):
    total = gray.size
    clipped = np.sum((gray <= low) | (gray >= high))
    return clipped / total


def intensity_dynamic_range(gray):
    return np.percentile(gray, 99) - np.percentile(gray, 1)



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

#---------------------
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


#----------------------
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
    orb = cv2.ORB_create(
        nfeatures=max_kp,
        fastThreshold=fast_threshold
    )

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
    orb = cv2.ORB_create(
        nfeatures=max_kp,
        fastThreshold=fast_threshold
    )

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


def extrair_features_filtragem_qualidade(
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







def extract_quality_features_from_path(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    return extrair_features_filtragem_qualidade(image)


def extract_clustering_features_from_path(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    return extrair_features_clustering(image)



