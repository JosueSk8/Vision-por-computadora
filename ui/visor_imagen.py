from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


class VisorImagen(QWidget):
    def __init__(self, matriz_imagen):
        super().__init__()
        # Memoria individual de esta ventanita
        self.imagen_original = matriz_imagen
        self.imagen_actual = matriz_imagen

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

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

        # SmoothTransformation evita que la imagen se pixelee al hacer pequeña la ventana
        self.label_imagen.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        # Cuando el usuario estira la ventana, repintamos la imagen para que se adapte
        if self.imagen_actual is not None:
            self.dibujar_imagen(self.imagen_actual)
        super().resizeEvent(event)