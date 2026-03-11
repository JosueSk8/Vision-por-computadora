import cv2
import numpy as np


def modelo_hsv(matriz_rgb):
    """Convierte RGB a HSV usando OpenCV"""
    return cv2.cvtColor(matriz_rgb, cv2.COLOR_RGB2HSV)


def modelo_cmy(matriz_rgb):
    """Inverso aditivo del RGB (Cian, Magenta, Amarillo)"""
    return 255 - matriz_rgb


def modelo_yiq(matriz_rgb):
    """Conversión matemática estándar NTSC a YIQ"""
    rgb = matriz_rgb.astype(np.float32) / 255.0
    matriz_yiq = np.array([[0.299, 0.587, 0.114],
                           [0.596, -0.274, -0.322],
                           [0.211, -0.523, 0.312]])
    yiq = np.dot(rgb, matriz_yiq.T)
    # Normalizamos los canales I y Q para poder verlos en pantalla (0 a 1)
    yiq[:, :, 1] = (yiq[:, :, 1] + 0.596) / (2 * 0.596)
    yiq[:, :, 2] = (yiq[:, :, 2] + 0.523) / (2 * 0.523)
    return (np.clip(yiq, 0, 1) * 255).astype(np.uint8)


def modelo_hsi(matriz_rgb):
    """Conversión matemática pura a Tono, Saturación e Intensidad"""
    rgb = matriz_rgb.astype(np.float32) / 255.0
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
    # Sumamos 1e-6 para evitar dividir entre cero
    theta = np.arccos(num / (den + 1e-6))

    H = theta
    H[B > G] = 2 * np.pi - H[B > G]
    H = H / (2 * np.pi)  # Normalizar de 0 a 1

    suma = R + G + B
    S = 1 - 3 * np.minimum(np.minimum(R, G), B) / (suma + 1e-6)
    I = suma / 3.0

    hsi = np.stack([H, S, I], axis=-1)
    return (np.clip(hsi, 0, 1) * 255).astype(np.uint8)