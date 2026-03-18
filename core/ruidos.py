import cv2
import numpy as np


def agregar_ruido_sal_pimienta(imagen_gris, cantidad=0.02):
    imagen_ruido = imagen_gris.copy()
    filas, columnas = imagen_gris.shape
    num_ruido = int(cantidad * filas * columnas)

    for _ in range(num_ruido):
        i = np.random.randint(0, columnas)
        j = np.random.randint(0, filas)
        if np.random.rand() < 0.5:
            imagen_ruido[j, i] = 0  # Pimienta (negro)
        else:
            imagen_ruido[j, i] = 255  # Sal (blanco)

    return imagen_ruido


def agregar_ruido_gaussiano(imagen_gris, media=0, sigma=20):
    gauss = np.random.normal(media, sigma, imagen_gris.shape).astype(np.int16)
    imagen_ruido = imagen_gris.astype(np.int16) + gauss
    return np.clip(imagen_ruido, 0, 255).astype(np.uint8)