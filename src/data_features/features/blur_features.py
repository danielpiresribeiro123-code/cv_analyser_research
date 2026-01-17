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

