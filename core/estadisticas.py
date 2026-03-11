import cv2
import numpy as np
import io
import matplotlib.pyplot as plt

def generar_histograma(matriz_rgb):
    fig, ax = plt.subplots(figsize=(6, 4))
    colores = ('r', 'g', 'b')
    for i, color in enumerate(colores):
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

def calcular_estadisticas(matriz_rgb):
    gris = cv2.cvtColor(matriz_rgb, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gris], [0], None, [256], [0, 256])
    p = hist.flatten() / gris.size
    i = np.arange(256)
    media = np.sum(i * p)
    varianza = np.sum(((i - media) ** 2) * p)
    desviacion = np.sqrt(varianza)
    asimetria = np.sum(((i - media) ** 3) * p) / (desviacion ** 3) if desviacion > 0 else 0
    energia = np.sum(p ** 2)
    p_non_zero = p[p > 0]
    entropia = -np.sum(p_non_zero * np.log2(p_non_zero)) if len(p_non_zero) > 0 else 0
    return media, varianza, entropia, asimetria, energia