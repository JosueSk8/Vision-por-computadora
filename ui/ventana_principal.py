import cv2
from PyQt5.QtWidgets import (QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QWidget, QFileDialog, QMessageBox,
                             QFrame, QMdiArea, QMdiSubWindow)

from core.procesamiento import cargar_imagen
from core.filtros import (filtro_grises, aplicar_mapa_cv2, mapa_pastel,
                          mapa_tierra, mapa_neon_termico, extraer_rostro)
from ui.visor_imagen import VisorImagen


class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Procesador de Filtros MDI - Práctica 1")
        self.setGeometry(100, 100, 1100, 750)
        self.setStyleSheet("background-color: #262626; color: white;")

        layout_principal = QHBoxLayout()
        layout_principal.setContentsMargins(15, 15, 15, 15)

        # --- PANEL IZQUIERDO ---
        marco_controles = QFrame()
        marco_controles.setFixedWidth(380)
        marco_controles.setStyleSheet("""
            QFrame { background-color: #3a3a3a; border: 2px solid #555555; border-radius: 10px; }
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
            ("Mapa Calor Facial", lambda: self.aplicar_filtro("Mapa Calor Facial"))
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

        # --- BOTÓN SWITCH: RECORTE FACIAL ---
        self.btn_recorte = QPushButton("IA: Recorte Facial [INACTIVO]")
        self.btn_recorte.setCheckable(True)
        self.btn_recorte.setStyleSheet(
            "background-color: #555555; border: 1px solid #707070; padding: 12px; font-weight: bold; border-radius: 5px; margin-top: 15px;")
        self.btn_recorte.clicked.connect(self.toggle_recorte)
        layout_marco.addWidget(self.btn_recorte)

        layout_marco.addStretch()

        # --- BOTÓN DE GUARDADO ---
        self.btn_guardar = QPushButton("Guardar Imagen Actual")
        self.btn_guardar.setStyleSheet(
            "background-color: #2e7d32; font-weight: bold; padding: 12px; border-radius: 5px; margin-top: 10px;")
        self.btn_guardar.clicked.connect(self.guardar_imagen)
        layout_marco.addWidget(self.btn_guardar)

        marco_controles.setLayout(layout_marco)

        # --- PANEL DERECHO (Área MDI) ---
        self.mdi_area = QMdiArea()
        self.mdi_area.setStyleSheet(
            "QMdiArea { background-color: #1e1e1e; border: 2px dashed #555555; border-radius: 10px; }")

        self.mdi_area.subWindowActivated.connect(self.al_cambiar_ventana)

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
                visor = VisorImagen(matriz)
                sub_ventana = QMdiSubWindow()
                sub_ventana.setWidget(visor)
                sub_ventana.setWindowTitle("Imagen Original")

                h, w = matriz.shape[:2]
                alto_base = 500
                ancho_base = int((w / h) * alto_base)
                sub_ventana.resize(ancho_base, alto_base)

                self.mdi_area.addSubWindow(sub_ventana)
                sub_ventana.show()

    def al_cambiar_ventana(self, sub_ventana):
        if sub_ventana:
            visor = sub_ventana.widget()
            self.btn_recorte.setChecked(visor.recorte_activo)
            self.actualizar_color_boton_recorte(visor.recorte_activo)
        else:
            self.btn_recorte.setChecked(False)
            self.actualizar_color_boton_recorte(False)

    def actualizar_color_boton_recorte(self, activo):
        if activo:
            self.btn_recorte.setText("IA: Recorte Facial [ACTIVADO]")
            self.btn_recorte.setStyleSheet(
                "background-color: #c62828; border: 1px solid #e53935; padding: 12px; font-weight: bold; border-radius: 5px; margin-top: 15px;")
        else:
            self.btn_recorte.setText("IA: Recorte Facial [INACTIVO]")
            self.btn_recorte.setStyleSheet(
                "background-color: #555555; border: 1px solid #707070; padding: 12px; font-weight: bold; border-radius: 5px; margin-top: 15px;")

    def toggle_recorte(self, checked):
        ventana_activa = self.mdi_area.activeSubWindow()
        if not ventana_activa:
            self.btn_recorte.setChecked(False)
            QMessageBox.warning(self, "Cuidado", "Selecciona una imagen primero.")
            return

        visor = ventana_activa.widget()
        visor.recorte_activo = checked
        self.actualizar_color_boton_recorte(checked)
        self.procesar_visor(visor, ventana_activa)

    def aplicar_filtro(self, tipo_filtro):
        ventana_activa = self.mdi_area.activeSubWindow()
        if not ventana_activa:
            QMessageBox.warning(self, "Cuidado", "Haz clic en una imagen primero para seleccionarla.")
            return

        visor = ventana_activa.widget()
        visor.filtro_actual = tipo_filtro
        self.procesar_visor(visor, ventana_activa)

    def procesar_visor(self, visor, ventana_activa):
        matriz_base = visor.imagen_original

        if visor.recorte_activo:
            recorte = extraer_rostro(matriz_base)
            if recorte is not None:
                matriz_base = recorte
            else:
                QMessageBox.warning(self, "Sin Rostros", "La IA no detectó ningún rostro frontal.")
                visor.recorte_activo = False
                self.btn_recorte.setChecked(False)
                self.actualizar_color_boton_recorte(False)

        tipo_filtro = visor.filtro_actual

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
        elif tipo_filtro == "Mapa Calor Facial":
            resultado = mapa_neon_termico(matriz_base)

        visor.dibujar_imagen(resultado)
        ventana_activa.setWindowTitle(f"Filtro: {tipo_filtro}")

    def guardar_imagen(self):
        ventana_activa = self.mdi_area.activeSubWindow()
        if not ventana_activa:
            QMessageBox.warning(self, "Error", "Selecciona primero la imagen que quieres guardar.")
            return

        visor = ventana_activa.widget()
        ruta_guardado, _ = QFileDialog.getSaveFileName(self, "Guardar Imagen", "", "PNG (*.png);;JPEG (*.jpg *.jpeg)")

        if ruta_guardado:
            imagen_bgr = cv2.cvtColor(visor.imagen_actual, cv2.COLOR_RGB2BGR)
            cv2.imwrite(ruta_guardado, imagen_bgr)
            QMessageBox.information(self, "Éxito", "Imagen guardada correctamente.")