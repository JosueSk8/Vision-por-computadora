from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


class VisorImagen(QWidget):
    def __init__(self, matriz_imagen):
        super().__init__()
        self.imagen_original = matriz_imagen
        self.imagen_actual = matriz_imagen

        # --- NUEVA MEMORIA PARA EL SWITCH ---
        self.filtro_actual = "Original"
        self.recorte_activo = False

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.label_imagen = QLabel()
        self.label_imagen.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_imagen)

        self.setLayout(layout)
        self.dibujar_imagen(self.imagen_actual)

    def dibujar_imagen(self, matriz):
        self.imagen_actual = matriz
        h, w, ch = matriz.shape
        bytes_por_linea = ch * w
        imagen_qt = QImage(matriz.data, w, h, bytes_por_linea, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(imagen_qt)

        self.label_imagen.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        if self.imagen_actual is not None:
            self.dibujar_imagen(self.imagen_actual)
        super().resizeEvent(event)