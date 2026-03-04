import cv2
from PyQt5.QtWidgets import (QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QWidget, QFileDialog, QMessageBox,
                             QFrame, QMdiArea, QMdiSubWindow, QTabWidget, QSlider, QLabel)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

from core.procesamiento import cargar_imagen
from core.filtros import (filtro_grises, aplicar_mapa_cv2, mapa_pastel,
                          mapa_tierra, mapa_neon_termico, extraer_rostro,
                          canal_rojo, canal_verde, canal_azul, binarizar_imagen,
                          modelo_hsv, generar_histograma)
from ui.visor_imagen import VisorImagen


class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Procesador de Imágenes Estadístico")
        self.setGeometry(50, 50, 1200, 850)  # Hacemos la ventana un poco más alta
        self.setStyleSheet("background-color: #262626; color: white;")

        layout_principal = QHBoxLayout()
        layout_principal.setContentsMargins(15, 15, 15, 15)

        # =========================================================
        # PANEL IZQUIERDO (Controles + HISTOGRAMA FIJO)
        # =========================================================
        marco_controles_v = QVBoxLayout()  # Diseño vertical para el panel izquierdo

        # --- SUB-PANEL: Botones y Pestañas ---
        marco_botones = QFrame()
        marco_botones.setFixedWidth(390)
        marco_botones.setStyleSheet(
            "QFrame { background-color: #3a3a3a; border: 2px solid #555555; border-radius: 10px; }")

        layout_marco = QVBoxLayout()
        layout_marco.setContentsMargins(20, 20, 20, 20)

        self.estilo_btn = """
            QPushButton { background-color: #505050; border: 1px solid #707070; padding: 10px; font-size: 13px; border-radius: 5px; }
            QPushButton:hover { background-color: #606060; border: 1px solid #909090; }
        """

        # --- BOTÓN CARGAR ---
        self.btn_cargar = QPushButton("Cargar Imagen Nueva")
        self.btn_cargar.setStyleSheet(
            "background-color: #2a5a8a; font-weight: bold; padding: 12px; border-radius: 5px; margin-bottom: 10px;")
        self.btn_cargar.clicked.connect(self.abrir_explorador)
        layout_marco.addWidget(self.btn_cargar)

        # =========================================================
        # PESTAÑAS (TABS)
        # =========================================================
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #555; border-radius: 5px; background-color: #2b2b2b; }
            QTabBar::tab { background: #444; color: white; padding: 10px; border: 1px solid #555; border-top-left-radius: 5px; border-top-right-radius: 5px; margin-right: 2px;}
            QTabBar::tab:selected { background: #2a5a8a; font-weight: bold; }
        """)

        # --- PESTAÑA 1: Básicos y Rangos ---
        tab_basicos = QWidget()
        layout_basicos = QVBoxLayout(tab_basicos)
        grid_basicos = QGridLayout()

        self.crear_boton("Original", grid_basicos, 0, 0, lambda: self.aplicar_filtro("Original"))
        self.crear_boton("Grises", grid_basicos, 0, 1, lambda: self.aplicar_filtro("Grises"))
        self.crear_boton("Modelo HSV", grid_basicos, 1, 0, lambda: self.aplicar_filtro("Modelo HSV"))
        self.crear_boton("Canal Rojo", grid_basicos, 2, 0, lambda: self.aplicar_filtro("Canal Rojo"))
        self.crear_boton("Canal Verde", grid_basicos, 2, 1, lambda: self.aplicar_filtro("Canal Verde"))
        self.crear_boton("Canal Azul", grid_basicos, 3, 0, lambda: self.aplicar_filtro("Canal Azul"))

        layout_basicos.addLayout(grid_basicos)

        # SLIDER DE BINARIZACIÓN
        marco_slider = QFrame()
        marco_slider.setStyleSheet(
            "background-color: #333; border: 1px solid #555; border-radius: 5px; margin-top: 10px;")
        layout_slider = QVBoxLayout(marco_slider)

        self.label_umbral = QLabel("Umbral de Binarización: 128")
        self.label_umbral.setAlignment(Qt.AlignCenter)
        self.label_umbral.setStyleSheet("border: none; font-weight: bold;")

        self.slider_umbral = QSlider(Qt.Horizontal)
        self.slider_umbral.setRange(0, 255)
        self.slider_umbral.setValue(128)
        self.slider_umbral.valueChanged.connect(self.cambiar_umbral)

        btn_binarizar = QPushButton("Aplicar Binarización")
        btn_binarizar.setStyleSheet(self.estilo_btn)
        btn_binarizar.clicked.connect(lambda: self.aplicar_filtro("Binarizar"))

        layout_slider.addWidget(self.label_umbral)
        layout_slider.addWidget(self.slider_umbral)
        layout_slider.addWidget(btn_binarizar)

        layout_basicos.addWidget(marco_slider)
        layout_basicos.addStretch()
        self.tabs.addTab(tab_basicos, "Básicos")

        # --- PESTAÑA 2: Pseudocolor e IA ---
        tab_color = QWidget()
        layout_color = QVBoxLayout(tab_color)
        grid_color = QGridLayout()
        grid_color.setSpacing(10)

        self.crear_boton("Mapa JET", grid_color, 0, 0, lambda: self.aplicar_filtro("Mapa JET"))
        self.crear_boton("Mapa HOT", grid_color, 0, 1, lambda: self.aplicar_filtro("Mapa HOT"))
        self.crear_boton("Mapa BONE", grid_color, 1, 0, lambda: self.aplicar_filtro("Mapa BONE"))
        self.crear_boton("Mapa Pastel", grid_color, 1, 1, lambda: self.aplicar_filtro("Mapa Pastel"))
        self.crear_boton("Mapa Tierra", grid_color, 2, 0, lambda: self.aplicar_filtro("Mapa Tierra"))
        self.crear_boton("Calor Facial", grid_color, 2, 1, lambda: self.aplicar_filtro("Mapa Calor Facial"))

        layout_color.addLayout(grid_color)

        # Movemos el botón de IA aquí para que la pestaña de Análisis quede limpia para el histograma
        self.btn_recorte = QPushButton("IA: Recorte Facial [INACTIVO]")
        self.btn_recorte.setCheckable(True)
        self.btn_recorte.setStyleSheet(
            "background-color: #555555; border: 1px solid #707070; padding: 12px; font-weight: bold; border-radius: 5px; margin-top: 15px;")
        self.btn_recorte.clicked.connect(self.toggle_recorte)
        layout_color.addWidget(self.btn_recorte)
        layout_color.addStretch()

        self.tabs.addTab(tab_color, "Pseudocolor e IA")

        layout_marco.addWidget(self.tabs)
        layout_marco.addStretch()

        # --- BOTÓN GUARDAR ---
        self.btn_guardar = QPushButton(" Guardar Imagen Actual")
        self.btn_guardar.setStyleSheet(
            "background-color: #2e7d32; font-weight: bold; padding: 12px; border-radius: 5px;")
        self.btn_guardar.clicked.connect(self.guardar_imagen)
        layout_marco.addWidget(self.btn_guardar)

        marco_botones.setLayout(layout_marco)

        # =========================================================
        # --- SUB-PANEL: HISTOGRAMA FIJO ---
        # =========================================================
        self.marco_hist = QFrame()
        self.marco_hist.setFixedWidth(390)
        self.marco_hist.setFixedHeight(300)  # Altura fija para la gráfica
        self.marco_hist.setStyleSheet("""
            QFrame { background-color: #1e1e1e; border: 2px solid #555555; border-radius: 10px; margin-top: 10px; }
        """)

        layout_hist = QVBoxLayout(self.marco_hist)
        layout_hist.setContentsMargins(10, 10, 10, 10)

        label_titulo_hist = QLabel("Análisis Estadístico (Histograma)")
        label_titulo_hist.setAlignment(Qt.AlignCenter)
        label_titulo_hist.setStyleSheet("border: none; font-weight: bold; font-size: 14px; color: #aaa;")

        # Este es el lienzo donde dibujaremos el histograma
        self.label_hist_display = QLabel("Selecciona una imagen para ver su histograma")
        self.label_hist_display.setAlignment(Qt.AlignCenter)
        self.label_hist_display.setStyleSheet("border: none; color: #666; font-size: 12px;")

        layout_hist.addWidget(label_titulo_hist)
        layout_hist.addWidget(self.label_hist_display, 1)  # El '1' le da todo el espacio sobrante

        # Ensamblamos el panel izquierdo verticalmente
        marco_controles_v.addWidget(marco_botones)
        marco_controles_v.addWidget(self.marco_hist)

        # =========================================================
        # PANEL DERECHO (Área MDI)
        # =========================================================
        self.mdi_area = QMdiArea()
        self.mdi_area.setStyleSheet(
            "QMdiArea { background-color: #1e1e1e; border: 2px dashed #555555; border-radius: 10px; }")

        # Sensor para actualizar el histograma al cambiar de ventana
        self.mdi_area.subWindowActivated.connect(self.al_cambiar_ventana)

        layout_principal.addLayout(marco_controles_v)  # Agregamos todo el panel izquierdo
        layout_principal.addWidget(self.mdi_area, 1)

        widget_central = QWidget()
        widget_central.setLayout(layout_principal)
        self.setCentralWidget(widget_central)

    def crear_boton(self, texto, layout, fila, col, funcion):
        btn = QPushButton(texto)
        btn.setStyleSheet(self.estilo_btn)
        btn.clicked.connect(funcion)
        layout.addWidget(btn, fila, col)

    def abrir_explorador(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Seleccionar Imagen", "", "Imágenes (*.png *.jpg *.jpeg)")
        if ruta:
            matriz = cargar_imagen(ruta)
            if matriz is not None:
                self.crear_subventana(matriz, "Imagen Original")

    def crear_subventana(self, matriz, titulo):
        visor = VisorImagen(matriz)
        sub_ventana = QMdiSubWindow()
        sub_ventana.setWidget(visor)
        sub_ventana.setWindowTitle(titulo)

        h, w = matriz.shape[:2]
        alto_base = 450
        ancho_base = int((w / h) * alto_base)
        sub_ventana.resize(ancho_base, alto_base)

        self.mdi_area.addSubWindow(sub_ventana)
        sub_ventana.show()

    # --- LÓGICA DE ACTUALIZACIÓN (Automaticay MDI) ---
    def al_cambiar_ventana(self, sub_ventana):
        """Se activa al cambiar de imagen"""
        if sub_ventana:
            visor = sub_ventana.widget()
            self.btn_recorte.setChecked(visor.recorte_activo)
            self.actualizar_color_boton_recorte(visor.recorte_activo)

            # MAGIA: Actualizamos el histograma fijo automáticamente
            self.actualizar_histograma_fijo(visor)
        else:
            self.btn_recorte.setChecked(False)
            self.actualizar_color_boton_recorte(False)
            self.label_hist_display.setText("Selecciona una imagen para ver su histograma")
            self.label_hist_display.setPixmap(QPixmap())  # Borramos la gráfica

    def actualizar_histograma_fijo(self, visor):
        """Calcula el histograma de la foto seleccionada y lo dibuja en el panel fijo"""
        # Obtenemos la matriz base (original) de la ventanita
        matriz_base = visor.imagen_original

        # Si tiene el recorte activado, el histograma debe ser SOLO del rostro
        if visor.recorte_activo:
            recorte = extraer_rostro(matriz_base)
            if recorte is not None: matriz_base = recorte

        # Generamos la gráfica estadísticay la convertimos en matriz
        img_hist = generar_histograma(matriz_base)

        # La convertimos para PyQt5
        h, w, ch = img_hist.shape
        bytes_por_linea = ch * w
        imagen_qt = QImage(img_hist.data, w, h, bytes_por_linea, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(imagen_qt)

        # La pintamos en el lienzo fijo escalándola para que quepa perfecto
        self.label_hist_display.setPixmap(
            pixmap.scaled(self.label_hist_display.width(), self.label_hist_display.height(), Qt.KeepAspectRatio,
                          Qt.SmoothTransformation))

    # --- LÓGICA DE BOTONES Y SLIDER (No toques esto, ya funcionaba) ---
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

        # También actualizamos el histograma para que refleje el recorte
        self.actualizar_histograma_fijo(visor)

    def cambiar_umbral(self, valor):
        self.label_umbral.setText(f"Umbral de Binarización: {valor}")
        ventana_activa = self.mdi_area.activeSubWindow()
        if ventana_activa:
            visor = ventana_activa.widget()
            visor.filtro_actual = "Binarizar"
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
        elif tipo_filtro == "Binarizar":
            resultado = binarizar_imagen(matriz_base, self.slider_umbral.value())
        elif tipo_filtro == "Modelo HSV":
            resultado = modelo_hsv(matriz_base)
        elif tipo_filtro == "Canal Rojo":
            resultado = canal_rojo(matriz_base)
        elif tipo_filtro == "Canal Verde":
            resultado = canal_verde(matriz_base)
        elif tipo_filtro == "Canal Azul":
            resultado = canal_azul(matriz_base)
        elif tipo_filtro == "Mapa JET":
            resultado = aplicar_mapa_cv2(matriz_base, cv2.COLORMAP_JET)
        elif tipo_filtro == "Mapa HOT":
            resultado = aplicar_mapa_cv2(matriz_base, cv2.COLORMAP_HOT)
        elif tipo_filtro == "Mapa BONE":
            resultado = aplicar_mapa_cv2(matriz_base, cv2.COLORMAP_BONE)
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