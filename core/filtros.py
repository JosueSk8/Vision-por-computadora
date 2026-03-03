import cv2
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def filtro_grises(matriz_rgb):
    gris = cv2.cvtColor(matriz_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2RGB)

def aplicar_mapa_cv2(matriz_rgb, tipo_mapa):
    """Aplica mapas predeterminados de OpenCV (JET, HOT, etc.)"""
    gris = cv2.cvtColor(matriz_rgb, cv2.COLOR_RGB2GRAY)
    mapa = cv2.applyColorMap(gris, tipo_mapa)
    return cv2.cvtColor(mapa, cv2.COLOR_BGR2RGB)

def aplicar_mapa_personalizado(matriz_rgb, colores):
    """Aplica mapas personalizados usando Matplotlib y lo convierte a formato de imagen"""
    gris = cv2.cvtColor(matriz_rgb, cv2.COLOR_RGB2GRAY)

    # Creamos el mapa con los 256 niveles que pide la práctica
    mapa_custom = LinearSegmentedColormap.from_list("CustomMap", colores, N=256)

    # Matplotlib aplica el mapa y devuelve valores de 0.0 a 1.0 (RGBA)
    imagen_mapeada = mapa_custom(gris / 255.0)

    # Quitamos la transparencia y convertimos los valores a 0-255 para PyQt5
    imagen_rgb = (imagen_mapeada[:, :, :3] * 255).astype(np.uint8)
    return imagen_rgb


# FILTROS PERSONALIZADOS DE LA PRÁCTICA

def mapa_pastel(matriz_rgb):
    # Valores exactos de la práctica
    colores = [(1.0, 0.8, 0.9), (0.8, 1.0, 0.8), (0.8, 0.9, 1.0), (1.0, 1.0, 0.8), (0.9, 0.8, 1.0)]
    return aplicar_mapa_personalizado(matriz_rgb, colores)

def mapa_tierra(matriz_rgb):
    # Valores exactos de la práctica
    colores = [(0.6, 0.4, 0.2), (0.8, 0.7, 0.5), (0.9, 0.8, 0.6), (0.7, 0.5, 0.3), (0.5, 0.3, 0.1)]
    return aplicar_mapa_personalizado(matriz_rgb, colores)

def mapa_neon_termico(matriz_rgb):
    # Tu mapa de calor facial único
    colores = [(0.0, 0.0, 0.2), (0.0, 0.5, 0.8), (0.8, 0.0, 0.8), (1.0, 0.8, 0.0), (1.0, 1.0, 1.0)]
    return aplicar_mapa_personalizado(matriz_rgb, colores)