import cv2
import numpy as np


# --- 1. FILTRO LIMPIADOR OFICIAL ---
def suavizar_ruido(matriz_rgb):
    """Aplica el Filtro Gaussiano indicado por la profesora"""
    return cv2.GaussianBlur(matriz_rgb, (5, 5), 0)


# --- 2. OPERACIONES ARITMÉTICAS ---
def operacion_suma(matriz_rgb, valor_escalar=50):
    return cv2.add(matriz_rgb, (valor_escalar, valor_escalar, valor_escalar, 0))


def operacion_resta(matriz_rgb, valor_escalar=50):
    return cv2.subtract(matriz_rgb, (valor_escalar, valor_escalar, valor_escalar, 0))


def operacion_multiplicacion(matriz_rgb, valor_escalar=1.5):
    img_float = matriz_rgb.astype(np.float32) * valor_escalar
    return np.clip(img_float, 0, 255).astype(np.uint8)


# --- 3. OPERACIONES RELACIONALES (NUEVO) ---
def operacion_mayor(matriz_rgb, umbral):
    """Segmenta dejando en blanco los píxeles MAYORES al umbral"""
    gris = cv2.cvtColor(matriz_rgb, cv2.COLOR_RGB2GRAY)
    mask = np.where(gris > umbral, 255, 0).astype(np.uint8)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


def operacion_menor(matriz_rgb, umbral):
    """Segmenta dejando en blanco los píxeles MENORES al umbral"""
    gris = cv2.cvtColor(matriz_rgb, cv2.COLOR_RGB2GRAY)
    mask = np.where(gris < umbral, 255, 0).astype(np.uint8)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


def operacion_igual(matriz_rgb, umbral):
    """Segmenta dejando en blanco los píxeles IGUALES al umbral"""
    gris = cv2.cvtColor(matriz_rgb, cv2.COLOR_RGB2GRAY)
    mask = np.where(gris == umbral, 255, 0).astype(np.uint8)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


# --- 4. OPERACIONES LÓGICAS ---
def operacion_not(matriz_rgb):
    return cv2.bitwise_not(matriz_rgb)


def operacion_and(img1, img2):
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return cv2.bitwise_and(img1, img2_resized)


def operacion_or(img1, img2):
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return cv2.bitwise_or(img1, img2_resized)


def operacion_xor(img1, img2):
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return cv2.bitwise_xor(img1, img2_resized)


# --- 5. CONTEO DE OBJETOS ---
def contar_objetos(matriz_rgb, vecindad=8, valor_umbral=127):
    gris = cv2.cvtColor(matriz_rgb, cv2.COLOR_RGB2GRAY)
    _, binaria = cv2.threshold(gris, valor_umbral, 255, cv2.THRESH_BINARY_INV)

    num_labels, labels = cv2.connectedComponents(binaria, connectivity=vecindad)
    imagen_color = cv2.cvtColor(binaria, cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        cv2.drawContours(imagen_color, [contour], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(imagen_color, str(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return imagen_color, num_labels - 1