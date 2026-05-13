# Archivo: core/morfologia.py
import cv2
import numpy as np


def obtener_kernel(forma, tamano):
    if tamano % 2 == 0:
        tamano += 1

    if forma == "Rectángulo":
        tipo = cv2.MORPH_RECT
    elif forma == "Cruz":
        tipo = cv2.MORPH_CROSS
    elif forma == "Elipse":
        tipo = cv2.MORPH_ELLIPSE
    else:
        tipo = cv2.MORPH_RECT

    return cv2.getStructuringElement(tipo, (tamano, tamano))


def aplicar_morfologia(imagen, operacion, forma_kernel="Rectángulo", tam_kernel=3):
    kernel = obtener_kernel(forma_kernel, tam_kernel)

    # Convertir a grises si es a color, ya que la morfología trabaja mejor así
    if len(imagen.shape) == 3:
        img_proc = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        img_proc = imagen.copy()

    # Operaciones Básicas
    if operacion == "Erosión":
        return cv2.erode(img_proc, kernel, iterations=1)
    elif operacion == "Dilatación":
        return cv2.dilate(img_proc, kernel, iterations=1)
    elif operacion == "Apertura":
        return cv2.morphologyEx(img_proc, cv2.MORPH_OPEN, kernel)
    elif operacion == "Cierre":
        return cv2.morphologyEx(img_proc, cv2.MORPH_CLOSE, kernel)

    # Gradientes Morfológicos
    elif operacion == "Gradiente Simétrico":
        return cv2.morphologyEx(img_proc, cv2.MORPH_GRADIENT, kernel)
    elif operacion == "Gradiente por Erosión (Interno)":
        erosionado = cv2.erode(img_proc, kernel)
        return cv2.subtract(img_proc, erosionado)
    elif operacion == "Gradiente por Dilatación (Externo)":
        dilatado = cv2.dilate(img_proc, kernel)
        return cv2.subtract(dilatado, img_proc)

    # Sombreros (Iluminación)
    elif operacion == "Top Hat":
        return cv2.morphologyEx(img_proc, cv2.MORPH_TOPHAT, kernel)
    elif operacion == "Black Hat (Bot Hat)":
        return cv2.morphologyEx(img_proc, cv2.MORPH_BLACKHAT, kernel)

    # Binarias avanzadas
    elif operacion == "Hit-or-Miss":
        _, binaria = cv2.threshold(img_proc, 127, 255, cv2.THRESH_BINARY)
        return cv2.morphologyEx(binaria, cv2.MORPH_HITMISS, kernel)

    elif operacion == "Esqueleto (Adelgazamiento)":
        _, binaria = cv2.threshold(img_proc, 127, 255, cv2.THRESH_BINARY)
        try:
            return cv2.ximgproc.thinning(binaria)
        except AttributeError:
            skel = np.zeros(binaria.shape, np.uint8)
            img_temp = binaria.copy()
            while True:
                eroded = cv2.erode(img_temp, kernel)
                temp = cv2.dilate(eroded, kernel)
                temp = cv2.subtract(img_temp, temp)
                skel = cv2.bitwise_or(skel, temp)
                img_temp = eroded.copy()
                if cv2.countNonZero(img_temp) == 0:
                    break
            return skel

    return img_proc