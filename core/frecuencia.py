import cv2
import numpy as np


def aplicar_filtro_fft(imagen, tipo_filtro="Butterworth", modo="Pasa-bajas (Suavizar)", cutoff=0.15, orden=2):
    # 1. Escala de grises seguro (soporta imágenes RGBA y RGB)
    if len(imagen.shape) == 3:
        if imagen.shape[2] == 4:
            gris = cv2.cvtColor(imagen, cv2.COLOR_BGRA2GRAY)
        else:
            gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        gris = imagen.copy()

    # 2. FFT y centrado
    f = np.fft.fft2(gris)
    fshift = np.fft.fftshift(f)

    # 3. EXTRAER LOS ESPECTROS (Conversión segura anti-crasheo)
    magnitud = 20 * np.log(np.abs(fshift) + 1)
    mag_visual = cv2.normalize(magnitud, None, 0, 255, cv2.NORM_MINMAX)
    mag_visual = np.uint8(mag_visual)  # Forzado a 8-bits manualmente

    fase = np.angle(fshift)
    fase_visual = cv2.normalize(fase, None, 0, 255, cv2.NORM_MINMAX)
    fase_visual = np.uint8(fase_visual)  # Forzado a 8-bits manualmente

    # 4. Dimensiones y centro
    filas, columnas = gris.shape
    centro_fila, centro_columna = filas // 2, columnas // 2
    D0 = cutoff * min(filas, columnas) / 2.0
    if D0 <= 0: D0 = 0.0001

    # 5. MÁSCARA VECTORIZADA (Evita que la pantalla se congele)
    Y, X = np.ogrid[:filas, :columnas]
    D = np.sqrt((X - centro_columna) ** 2 + (Y - centro_fila) ** 2)
    D[D == 0] = 0.0001

    if tipo_filtro == "Ideal":
        mascara = np.where(D <= D0, 1.0, 0.0).astype(np.float32)
    elif tipo_filtro == "Gaussiano":
        mascara = np.exp(-(D ** 2) / (2 * (D0 ** 2))).astype(np.float32)
    elif tipo_filtro == "Butterworth":
        mascara = (1.0 / (1.0 + (D / D0) ** (2 * orden))).astype(np.float32)

    if modo == "Pasa-altas (Bordes)":
        mascara = 1.0 - mascara

    # 6. Aplicar filtro y regresar al dominio espacial
    fshift_filtrado = fshift * mascara
    f_ishift = np.fft.ifftshift(fshift_filtrado)
    img_reconstruida = np.abs(np.fft.ifft2(f_ishift))

    img_final = cv2.normalize(img_reconstruida, None, 0, 255, cv2.NORM_MINMAX)
    img_final = np.uint8(img_final)

    return img_final, mag_visual, fase_visual

def aplicar_compresion_dct(imagen, q_factor=1.0):
    # La compresión DCT se queda exactamente igual
    if len(imagen.shape) == 3:
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        gris = imagen.copy()

    h, w = gris.shape
    h_pad = (h // 8) * 8
    w_pad = (w // 8) * 8
    gris = gris[:h_pad, :w_pad]

    img_float = np.float32(gris) - 128.0

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

    Q_escalada = Q_luminancia * q_factor
    Q_escalada[Q_escalada < 1] = 1

    imagen_comprimida = np.zeros_like(img_float)

    for i in range(0, h_pad, 8):
        for j in range(0, w_pad, 8):
            bloque = img_float[i:i + 8, j:j + 8]
            dct_bloque = cv2.dct(bloque)
            dct_cuantizada = np.round(dct_bloque / Q_escalada)
            dct_descuantizada = dct_cuantizada * Q_escalada
            idct_bloque = cv2.idct(dct_descuantizada)
            imagen_comprimida[i:i + 8, j:j + 8] = idct_bloque

    img_final = np.clip(imagen_comprimida + 128.0, 0, 255).astype(np.uint8)

    mse = np.mean((np.float32(gris) - np.float32(img_final)) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return img_final, psnr