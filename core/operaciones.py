import cv2
import numpy as np


# --- 1. FILTRO LIMPIADOR OFICIAL ---
def suavizar_ruido(matriz_rgb):
    return cv2.GaussianBlur(matriz_rgb, (5, 5), 0)


# --- 2. OPERACIONES ARITMÉTICAS ---
def operacion_suma(matriz_rgb, valor_escalar=50):
    return cv2.add(matriz_rgb, (valor_escalar, valor_escalar, valor_escalar, 0))


def operacion_resta(matriz_rgb, valor_escalar=50):
    return cv2.subtract(matriz_rgb, (valor_escalar, valor_escalar, valor_escalar, 0))


def operacion_multiplicacion(matriz_rgb, valor_escalar=1.5):
    img_float = matriz_rgb.astype(np.float32) * valor_escalar
    return np.clip(img_float, 0, 255).astype(np.uint8)


# --- 3. OPERACIONES RELACIONALES ---
def operacion_mayor(matriz_rgb, umbral):
    gris = cv2.cvtColor(matriz_rgb, cv2.COLOR_RGB2GRAY)
    mask = np.where(gris > umbral, 255, 0).astype(np.uint8)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


def operacion_menor(matriz_rgb, umbral):
    gris = cv2.cvtColor(matriz_rgb, cv2.COLOR_RGB2GRAY)
    mask = np.where(gris < umbral, 255, 0).astype(np.uint8)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


def operacion_igual(matriz_rgb, umbral):
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


# --- 5. CONTEO Y CÓDIGO DE CADENA (Práctica 3-c) ---
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


def obtener_codigo_cadena(matriz_rgb, valor_umbral=127):
    """Extrae el Código de Cadena (Chain Code) del objeto más grande"""
    gris = cv2.cvtColor(matriz_rgb, cv2.COLOR_RGB2GRAY)
    _, binaria = cv2.threshold(gris, valor_umbral, 255, cv2.THRESH_BINARY_INV)

    # Usamos CHAIN_APPROX_NONE para obtener TODOS los píxeles del borde sin comprimir
    contours, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return matriz_rgb, "No se detectaron objetos para analizar."

    # Tomar el objeto más grande para el análisis
    contorno_principal = max(contours, key=cv2.contourArea)

    # Diccionario de direcciones de Freeman (8-vecinos) para coordenadas de imagen
    direcciones = {
        (1, 0): '0', (1, -1): '1', (0, -1): '2', (-1, -1): '3',
        (-1, 0): '4', (-1, 1): '5', (0, 1): '6', (1, 1): '7'
    }

    codigo = []
    # Recorrer píxel por píxel el contorno calculando hacia dónde se movió
    for i in range(len(contorno_principal) - 1):
        pt1 = contorno_principal[i][0]
        pt2 = contorno_principal[i + 1][0]
        dx = max(-1, min(1, pt2[0] - pt1[0]))  # Limitamos a -1, 0, 1
        dy = max(-1, min(1, pt2[1] - pt1[1]))

        if (dx, dy) in direcciones:
            codigo.append(direcciones[(dx, dy)])

    # Preparar la imagen para mostrar qué analizamos
    imagen_color = cv2.cvtColor(binaria, cv2.COLOR_GRAY2RGB)
    # Dibujamos el contorno analizado en azul
    cv2.drawContours(imagen_color, [contorno_principal], -1, (0, 0, 255), 2)
    # Pintamos un punto verde brillante donde empezó el análisis
    cv2.circle(imagen_color, tuple(contorno_principal[0][0]), 5, (0, 255, 0), -1)

    codigo_str = "".join(codigo)
    # Como el código puede tener miles de números, lo recortamos para la alerta en pantalla
    if len(codigo_str) > 100:
        codigo_str = codigo_str[:100] + f"... \n\n(Mostrando 100 de {len(codigo)} pasos totales)"

    return imagen_color, codigo_str