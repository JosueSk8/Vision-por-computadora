from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QColorDialog
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt


class DialogoMapaPersonalizado(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Diseñador de Mapa de Color")
        self.setFixedSize(300, 350)
        self.setStyleSheet("background-color: #333; color: white;")

        self.colores = [QColor(255, 0, 0), QColor(255, 255, 0), QColor(0, 255, 0),
                        QColor(0, 255, 255), QColor(0, 0, 255)]

        layout = QVBoxLayout(self)
        label_instruccion = QLabel("Selecciona 5 colores para tu mapa:")
        label_instruccion.setAlignment(Qt.AlignCenter)
        layout.addWidget(label_instruccion)

        self.botones_color = []
        for i in range(5):
            btn = QPushButton(f"Elegir Nivel de Color {i + 1}")
            btn.setStyleSheet(
                f"background-color: {self.colores[i].name()}; color: black; font-weight: bold; padding: 10px; border-radius: 5px;")
            btn.clicked.connect(lambda checked, idx=i: self.elegir_color(idx))
            self.botones_color.append(btn)
            layout.addWidget(btn)

        self.btn_aplicar = QPushButton(" Aplicar a la Imagen")
        self.btn_aplicar.setStyleSheet(
            "background-color: #2e7d32; font-weight: bold; padding: 12px; border-radius: 5px; margin-top: 15px;")
        self.btn_aplicar.clicked.connect(self.accept)
        layout.addWidget(self.btn_aplicar)

    def elegir_color(self, idx):
        color = QColorDialog.getColor(self.colores[idx], self, f"Seleccionar Color {idx + 1}")
        if color.isValid():
            self.colores[idx] = color
            self.botones_color[idx].setStyleSheet(
                f"background-color: {color.name()}; color: black; font-weight: bold; padding: 10px; border-radius: 5px;")

    def obtener_colores_normalizados(self):
        return [(c.red() / 255.0, c.green() / 255.0, c.blue() / 255.0) for c in self.colores]