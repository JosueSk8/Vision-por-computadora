import cv2
import numpy as np


def aplicar_filtro_fft(imagen, tipo_filtro="Butterworth", modo="Pasa-bajas (Suavizar)", cutoff=0.15, orden=2):
    """
    Aplica filtros en el dominio de la frecuencia usando la FFT.
    """
    # 1. Asegurar que trabajamos en escala de grises
    if len(imagen.shape) == 3:
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        gris = imagen.copy()

    # 2. Calcular la FFT y centrarla
    f = np.fft.fft2(gris)
    fshift = np.fft.fftshift(f)

    # 3. Preparar el lienzo para la máscara
    filas, columnas = gris.shape
    centro_fila, centro_columna = filas // 2, columnas // 2
    mascara = np.zeros((filas, columnas), np.float32)

    # El radio de corte real basado en el tamaño de la imagen
    D0 = cutoff * min(filas, columnas) / 2.0

    # 4. Crear la máscara matemática
    for i in range(filas):
        for j in range(columnas):
            # Distancia Euclidiana desde el centro
            D = np.sqrt((i - centro_fila) ** 2 + (j - centro_columna) ** 2)

            # Evitar división por cero
            if D == 0:
                D = 0.0001

            if tipo_filtro == "Ideal":
                mascara[i, j] = 1 if D <= D0 else 0

            elif tipo_filtro == "Gaussiano":
                mascara[i, j] = np.exp(-(D ** 2) / (2 * (D0 ** 2)))

            elif tipo_filtro == "Butterworth":
                mascara[i, j] = 1 / (1 + (D / D0) ** (2 * orden))

    # Si es pasa-altas, invertimos la máscara
    if modo == "Pasa-altas (Bordes)":
        mascara = 1 - mascara

    # 5. Aplicar la máscara al espectro
    fshift_filtrado = fshift * mascara

    # 6. Transformada Inversa (IFFT) para regresar al dominio espacial
    f_ishift = np.fft.ifftshift(fshift_filtrado)
    img_reconstruida = np.fft.ifft2(f_ishift)
    img_reconstruida = np.abs(img_reconstruida)

    # Normalizar la imagen final para que los valores queden entre 0 y 255
    img_final = cv2.normalize(img_reconstruida, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return img_final


def aplicar_compresion_dct(imagen, q_factor=1.0):
    """
    Aplica la compresión DCT por bloques de 8x8 y calcula el PSNR.
    """
    if len(imagen.shape) == 3:
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        gris = imagen.copy()

    # Asegurar que las dimensiones sean múltiplos de 8
    h, w = gris.shape
    h_pad = (h // 8) * 8
    w_pad = (w // 8) * 8
    gris = gris[:h_pad, :w_pad]

    # Convertir a float32 y centrar valores (restar 128)
    img_float = np.float32(gris) - 128.0

    # Matriz estándar de cuantización de luminancia JPEG
    Q_luminancia = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float32)

    # Aplicar el factor de calidad a la matriz
    Q_escalada = Q_luminancia * q_factor
    Q_escalada[Q_escalada < 1] = 1  # Evitar divisiones por cero

    imagen_comprimida = np.zeros_like(img_float)

    # Procesar bloque por bloque (8x8)
    for i in range(0, h_pad, 8):
        for j in range(0, w_pad, 8):
            bloque = img_float[i:i + 8, j:j + 8]

            # Transformada Discreta del Coseno (DCT)
            dct_bloque = cv2.dct(bloque)

            # Cuantización (Aquí se pierde información)
            dct_cuantizada = np.round(dct_bloque / Q_escalada)

            # Descuantización
            dct_descuantizada = dct_cuantizada * Q_escalada

            # Transformada Inversa (IDCT)
            idct_bloque = cv2.idct(dct_descuantizada)

            imagen_comprimida[i:i + 8, j:j + 8] = idct_bloque

    # Des-centrar valores (sumar 128) y recortar a rango válido 0-255
    img_final = np.clip(imagen_comprimida + 128.0, 0, 255).astype(np.uint8)

    # Calcular el PSNR
    mse = np.mean((np.float32(gris) - np.float32(img_final)) ** 2)
    if mse == 0:
        psnr = float('inf')  # Son idénticas
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return img_final, psnr