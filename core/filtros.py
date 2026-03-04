import cv2
import numpy as np
import io
import matplotlib.pyplot as plt
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
        recorte_rgb = matriz_rgb[y:y + h, x:x + w].copy()
        return recorte_rgb

    return None


def canal_rojo(matriz_rgb):
    r = matriz_rgb.copy()
    r[:, :, 1] = 0
    r[:, :, 2] = 0
    return r


def canal_verde(matriz_rgb):
    g = matriz_rgb.copy()
    g[:, :, 0] = 0
    g[:, :, 2] = 0
    return g


def canal_azul(matriz_rgb):
    b = matriz_rgb.copy()
    b[:, :, 0] = 0
    b[:, :, 1] = 0
    return b


def binarizar_imagen(matriz_rgb, umbral=128):
    gris = cv2.cvtColor(matriz_rgb, cv2.COLOR_RGB2GRAY)
    _, binaria = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(binaria, cv2.COLOR_GRAY2RGB)


def modelo_hsv(matriz_rgb):
    return cv2.cvtColor(matriz_rgb, cv2.COLOR_RGB2HSV)


def generar_histograma(matriz_rgb):
    fig, ax = plt.subplots(figsize=(6, 4))
    colores = ('r', 'g', 'b')

    for i, color in enumerate(colores):
        # Calculamos la estadística
        hist = cv2.calcHist([matriz_rgb], [i], None, [256], [0, 256])
        ax.bar(range(256), hist.flatten(), color=color, alpha=0.6, width=1.0)

    ax.set_xlim([0, 256])

    ax.set_title("Histograma de Intensidad RGB")
    ax.set_xlabel("Nivel de Intensidad (0-255)")
    ax.set_ylabel("Cantidad de Píxeles")

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    buf_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img_hist = cv2.imdecode(buf_arr, cv2.IMREAD_COLOR)
    plt.close(fig)

    return cv2.cvtColor(img_hist, cv2.COLOR_BGR2RGB)