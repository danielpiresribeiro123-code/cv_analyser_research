import cv2
import numpy as np


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






def split_into_grid(gray, grid=4):
    """
    Divide a imagem em grid x grid blocos.
    Returns: lista de patches (sub-imagens)
    """
    h, w = gray.shape
    patches = []

    for gy in range(grid):
        for gx in range(grid):
            y1 = int(gy * h / grid)
            y2 = int((gy + 1) * h / grid)
            x1 = int(gx * w / grid)
            x2 = int((gx + 1) * w / grid)

            patch = gray[y1:y2, x1:x2]
            patches.append(patch)

    return patches




def blur_grid_features(gray, grid=4):
    """
    Extrai métricas de blur por blocos e agrega estatisticamente.
    Serve para detetar:
      - defocus parcial
      - motion blur parcial
      - zonas localmente nítidas

    Returns: dict com métricas robustas.
    """
    patches = split_into_grid(gray, grid=grid)

    lap_vals = []
    ten_vals = []
    motion_vals = []

    for p in patches:
        if p.size == 0:
            continue

        lap_vals.append(blur_laplacian(p))
        ten_vals.append(blur_tenengrad(p))
        motion_vals.append(blur_motion(p))

    lap_vals = np.array(lap_vals, dtype=np.float32)
    ten_vals = np.array(ten_vals, dtype=np.float32)
    motion_vals = np.array(motion_vals, dtype=np.float32)

    def stats(x):
        p10 = np.percentile(x, 10)
        p90 = np.percentile(x, 90)
        maxv = np.max(x)

        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "p10": float(p10),
            "p90": float(p90),
            "max": float(maxv),
            "ratio_p90_p10": float(p90 / (p10 + 1e-6)),
            "ratio_max_p10": float(maxv / (p10 + 1e-6))}

    lap_stats = stats(lap_vals)
    ten_stats = stats(ten_vals)
    motion_stats = stats(motion_vals)

    return {
        # Laplacian (defocused blur)
        "blur_lap_mean": lap_stats["mean"],
        "blur_lap_std": lap_stats["std"],
        "blur_lap_p10": lap_stats["p10"],
        "blur_lap_p90": lap_stats["p90"],
        "blur_lap_max": lap_stats["max"],
        "blur_lap_ratio": lap_stats["ratio_p90_p10"],
        "blur_lap_ratio_max": lap_stats["ratio_max_p10"],

        # Tenengrad
        "blur_ten_mean": ten_stats["mean"],
        "blur_ten_std": ten_stats["std"],
        "blur_ten_p10": ten_stats["p10"],
        "blur_ten_p90": ten_stats["p90"],
        "blur_ten_max": ten_stats["max"],
        "blur_ten_ratio": ten_stats["ratio_p90_p10"],
        "blur_ten_ratio_max": ten_stats["ratio_max_p10"],

        # Motion
        "motion_mean": motion_stats["mean"],
        "motion_std": motion_stats["std"],
        "motion_p90": motion_stats["p90"],
        "motion_max": motion_stats["max"],
        "motion_ratio": motion_stats["ratio_p90_p10"],
        "motion_ratio_max": motion_stats["ratio_max_p10"]}



# Opcional
def blur_all_features(gray, grid=4):
    """
    Extrai:
      - blur global clássico
      - blur por grid
      - relações entre global e grid

    Serve apenas para experimentação / ablação.
    Não usar por defeito no pipeline.
    """
    feats = {}

    # Global
    global_lap = blur_laplacian(gray)
    global_ten = blur_tenengrad(gray)
    global_motion = blur_motion(gray)

    feats["global_blur_laplacian"] = float(global_lap)
    feats["global_blur_tenengrad"] = float(global_ten)
    feats["global_blur_motion"] = float(global_motion)

    # Grid
    grid_feats = blur_grid_features(gray, grid=grid)
    for k, v in grid_feats.items():
        feats[f"grid_{k}"] = float(v)

    # Relações global vs grid (experimentais)
    feats["lap_global_minus_grid_mean"] = float(
        global_lap - grid_feats["blur_lap_mean"]
    )
    feats["lap_global_ratio_grid_mean"] = float(
        global_lap / (grid_feats["blur_lap_mean"] + 1e-6)
    )

    feats["ten_global_minus_grid_mean"] = float(
        global_ten - grid_feats["blur_ten_mean"]
    )
    feats["ten_global_ratio_grid_mean"] = float(
        global_ten / (grid_feats["blur_ten_mean"] + 1e-6)
    )

    return feats
