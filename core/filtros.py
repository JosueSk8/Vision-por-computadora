import cv2
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def filtro_grises(matriz_rgb):
    gris = cv2.cvtColor(matriz_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2RGB)


def aplicar_mapa_cv2(matriz_rgb, tipo_mapa):
    gris = cv2.cvtColor(matriz_rgb, cv2.COLOR_RGB2GRAY)
    mapa = cv2.applyColorMap(gris, tipo_mapa)
    return cv2.cvtColor(mapa, cv2.COLOR_BGR2RGB)


def aplicar_mapa_personalizado(matriz_rgb, colores):
    gris = cv2.cvtColor(matriz_rgb, cv2.COLOR_RGB2GRAY)
    mapa_custom = LinearSegmentedColormap.from_list("CustomMap", colores, N=256)
    imagen_mapeada = mapa_custom(gris / 255.0)
    imagen_rgb = (imagen_mapeada[:, :, :3] * 255).astype(np.uint8)
    return imagen_rgb


def mapa_pastel(matriz_rgb):
    colores = [(1.0, 0.8, 0.9), (0.8, 1.0, 0.8), (0.8, 0.9, 1.0), (1.0, 1.0, 0.8), (0.9, 0.8, 1.0)]
    return aplicar_mapa_personalizado(matriz_rgb, colores)


def mapa_tierra(matriz_rgb):
    colores = [(0.6, 0.4, 0.2), (0.8, 0.7, 0.5), (0.9, 0.8, 0.6), (0.7, 0.5, 0.3), (0.5, 0.3, 0.1)]
    return aplicar_mapa_personalizado(matriz_rgb, colores)


def mapa_neon_termico(matriz_rgb):
    colores = [(0.0, 0.0, 0.2), (0.0, 0.5, 0.8), (0.8, 0.0, 0.8), (1.0, 0.8, 0.0), (1.0, 1.0, 1.0)]
    return aplicar_mapa_personalizado(matriz_rgb, colores)


def extraer_rostro(matriz_rgb):
    gris = cv2.cvtColor(matriz_rgb, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    rostros = face_cascade.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(rostros) > 0:
        x, y, w, h = rostros[0]
        # SOLUCIÓN: El .copy() asegura que la memoria sea contigua y no crashee PyQt5
        recorte_rgb = matriz_rgb[y:y + h, x:x + w].copy()
        return recorte_rgb

    return None