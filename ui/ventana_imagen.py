from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


class VentanaImagen(QWidget):
    def __init__(self, matriz_imagen, titulo="Visor de Imagen"):
        super().__init__()
        # Aparecerá a la derecha del panel de control
        self.setGeometry(550, 100, 700, 600)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")

        layout = QVBoxLayout()
        self.label_imagen = QLabel()
        self.label_imagen.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_imagen)
        self.setLayout(layout)

        self.actualizar_imagen(matriz_imagen, titulo)

    def actualizar_imagen(self, matriz, titulo="Visor de Imagen"):
        # Cambia el título de la ventana y repinta la imagen
        self.setWindowTitle(titulo)

        h, w, ch = matriz.shape
        bytes_por_linea = ch * w
        imagen_qt = QImage(matriz.data, w, h, bytes_por_linea, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(imagen_qt)

        self.label_imagen.setPixmap(pixmap.scaled(680, 580, Qt.KeepAspectRatio))