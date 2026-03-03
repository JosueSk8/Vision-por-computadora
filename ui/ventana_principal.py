import cv2
from PyQt5.QtWidgets import (QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QWidget, QFileDialog, QMessageBox,
                             QFrame, QMdiArea, QMdiSubWindow)

# Importamos las matemáticas
from core.procesamiento import cargar_imagen
from core.filtros import filtro_grises, aplicar_mapa_cv2, mapa_pastel, mapa_tierra, \
    mapa_neon_termico

#  Importamos el nuevo módulo visual
from ui.visor_imagen import VisorImagen


class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Procesador de Filtros MDI - Práctica 1")
        self.setGeometry(100, 100, 1100, 700)
        self.setStyleSheet("background-color: #262626; color: white;")

        layout_principal = QHBoxLayout()
        layout_principal.setContentsMargins(15, 15, 15, 15)

        # --- 1. PANEL IZQUIERDO (Menú Fijo) ---
        marco_controles = QFrame()
        marco_controles.setFixedWidth(380)
        marco_controles.setStyleSheet("""
            QFrame {
                background-color: #3a3a3a;
                border: 2px solid #555555;
                border-radius: 10px;
            }
        """)

        layout_marco = QVBoxLayout()
        layout_marco.setContentsMargins(20, 20, 20, 20)

        estilo_btn = """
            QPushButton { background-color: #505050; border: 1px solid #707070; padding: 10px; font-size: 13px; border-radius: 5px; }
            QPushButton:hover { background-color: #606060; border: 1px solid #909090; }
        """

        self.btn_cargar = QPushButton("Cargar Imagen Nueva")
        self.btn_cargar.setStyleSheet(
            "background-color: #2a5a8a; font-weight: bold; padding: 12px; border-radius: 5px;")
        self.btn_cargar.clicked.connect(self.abrir_explorador)
        layout_marco.addWidget(self.btn_cargar)

        grid_filtros = QGridLayout()
        grid_filtros.setSpacing(10)

        botones = [
            ("Original", lambda: self.aplicar_filtro("Original")),
            ("Grises", lambda: self.aplicar_filtro("Grises")),
            ("Mapa JET", lambda: self.aplicar_filtro("Mapa JET")),
            ("Mapa HOT", lambda: self.aplicar_filtro("Mapa HOT")),
            ("Mapa OCEAN", lambda: self.aplicar_filtro("Mapa OCEAN")),
            ("Mapa BONE", lambda: self.aplicar_filtro("Mapa BONE")),
            ("Mapa PINK", lambda: self.aplicar_filtro("Mapa PINK")),
            ("Mapa Pastel", lambda: self.aplicar_filtro("Mapa Pastel")),
            ("Mapa Tierra", lambda: self.aplicar_filtro("Mapa Tierra")),
            ("Mapa Facial", lambda: self.aplicar_filtro("Mapa Facial"))
        ]

        fila, col = 0, 0
        for texto, funcion in botones:
            btn = QPushButton(texto)
            btn.setStyleSheet(estilo_btn)
            btn.clicked.connect(funcion)
            grid_filtros.addWidget(btn, fila, col)
            col += 1
            if col > 1:
                col = 0
                fila += 1

        layout_marco.addLayout(grid_filtros)
        layout_marco.addStretch()
        marco_controles.setLayout(layout_marco)

        # --- 2. PANEL DERECHO (El Escritorio Virtual MDI) ---
        self.mdi_area = QMdiArea()
        self.mdi_area.setStyleSheet("""
            QMdiArea {
                background-color: #1e1e1e;
                border: 2px dashed #555555;
                border-radius: 10px;
            }
        """)

        layout_principal.addWidget(marco_controles)
        layout_principal.addWidget(self.mdi_area, 1)

        widget_central = QWidget()
        widget_central.setLayout(layout_principal)
        self.setCentralWidget(widget_central)

    def abrir_explorador(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Seleccionar Imagen", "", "Imágenes (*.png *.jpg *.jpeg)")
        if ruta:
            matriz = cargar_imagen(ruta)
            if matriz is not None:
                # Usamos la clase importada de tu nuevo módulo
                visor = VisorImagen(matriz)

                sub_ventana = QMdiSubWindow()
                sub_ventana.setWidget(visor)
                sub_ventana.setWindowTitle("Imagen Original")
                sub_ventana.resize(400, 450)

                self.mdi_area.addSubWindow(sub_ventana)
                sub_ventana.show()

    def aplicar_filtro(self, tipo_filtro):
        ventana_activa = self.mdi_area.activeSubWindow()

        if not ventana_activa:
            QMessageBox.warning(self, "Cuidado", "Haz clic en una imagen primero para seleccionarla.")
            return

        visor = ventana_activa.widget()
        matriz_base = visor.imagen_original

        if tipo_filtro == "Original":
            resultado = matriz_base
        elif tipo_filtro == "Grises":
            resultado = filtro_grises(matriz_base)
        elif tipo_filtro == "Mapa JET":
            resultado = aplicar_mapa_cv2(matriz_base, cv2.COLORMAP_JET)
        elif tipo_filtro == "Mapa HOT":
            resultado = aplicar_mapa_cv2(matriz_base, cv2.COLORMAP_HOT)
        elif tipo_filtro == "Mapa OCEAN":
            resultado = aplicar_mapa_cv2(matriz_base, cv2.COLORMAP_OCEAN)
        elif tipo_filtro == "Mapa BONE":
            resultado = aplicar_mapa_cv2(matriz_base, cv2.COLORMAP_BONE)
        elif tipo_filtro == "Mapa PINK":
            resultado = aplicar_mapa_cv2(matriz_base, cv2.COLORMAP_PINK)
        elif tipo_filtro == "Mapa Pastel":
            resultado = mapa_pastel(matriz_base)
        elif tipo_filtro == "Mapa Tierra":
            resultado = mapa_tierra(matriz_base)
        elif tipo_filtro == "Mapa Facial":
            resultado = mapa_neon_termico(matriz_base)

        visor.dibujar_imagen(resultado)
        ventana_activa.setWindowTitle(f"Filtro: {tipo_filtro}")