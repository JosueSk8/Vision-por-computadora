import cv2

def cargar_imagen(ruta):
    imagen_bgr = cv2.imread(ruta)
    if imagen_bgr is not None:
        return cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    return None