import cv2
from PyQt5.QtWidgets import (QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QWidget, QFileDialog, QMessageBox,
                             QFrame, QMdiArea, QMdiSubWindow, QTabWidget, QSlider, QLabel,
                             QDialog, QColorDialog)
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import Qt

from core.procesamiento import cargar_imagen
from core.filtros import (filtro_grises, aplicar_mapa_cv2, mapa_pastel,
                          mapa_tierra, mapa_neon_termico, extraer_rostro,
                          canal_rojo, canal_verde, canal_azul, binarizar_imagen,
                          modelo_hsv, modelo_cmy, modelo_yiq, modelo_hsi,  # <-- Agregamos los nuevos modelos aquí
                          generar_histograma, aplicar_mapa_personalizado, calcular_estadisticas)
from ui.visor_imagen import VisorImagen


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
        self.btn_aplicar = QPushButton("🚀 Aplicar a la Imagen")
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


class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Procesador de Imágenes Estadístico - Prácticas 1 y 2")
        self.setGeometry(50, 50, 1200, 850)
        self.setStyleSheet("background-color: #262626; color: white;")

        layout_principal = QHBoxLayout()
        layout_principal.setContentsMargins(15, 15, 15, 15)

        marco_controles_v = QVBoxLayout()
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

        self.btn_cargar = QPushButton("Cargar Imagen Nueva")
        self.btn_cargar.setStyleSheet(
            "background-color: #2a5a8a; font-weight: bold; padding: 12px; border-radius: 5px; margin-bottom: 10px;")
        self.btn_cargar.clicked.connect(self.abrir_explorador)
        layout_marco.addWidget(self.btn_cargar)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #555; border-radius: 5px; background-color: #2b2b2b; }
            QTabBar::tab { background: #444; color: white; padding: 10px; border: 1px solid #555; border-top-left-radius: 5px; border-top-right-radius: 5px; margin-right: 2px;}
            QTabBar::tab:selected { background: #2a5a8a; font-weight: bold; }
        """)

        # PESTAÑA 1: Básicos y Modelos de Color
        tab_basicos = QWidget()
        layout_basicos = QVBoxLayout(tab_basicos)
        grid_basicos = QGridLayout()

        self.crear_boton("Modelo RGB", grid_basicos, 0, 0, lambda: self.aplicar_filtro("Original"))
        self.crear_boton("Grises", grid_basicos, 0, 1, lambda: self.aplicar_filtro("Grises"))
        self.crear_boton("Modelo CMY", grid_basicos, 1, 0, lambda: self.aplicar_filtro("Modelo CMY"))
        self.crear_boton("Modelo YIQ", grid_basicos, 1, 1, lambda: self.aplicar_filtro("Modelo YIQ"))
        self.crear_boton("Modelo HSI", grid_basicos, 2, 0, lambda: self.aplicar_filtro("Modelo HSI"))
        self.crear_boton("Modelo HSV", grid_basicos, 2, 1, lambda: self.aplicar_filtro("Modelo HSV"))
        self.crear_boton("Canal Rojo", grid_basicos, 3, 0, lambda: self.aplicar_filtro("Canal Rojo"))
        self.crear_boton("Canal Verde", grid_basicos, 3, 1, lambda: self.aplicar_filtro("Canal Verde"))
        self.crear_boton("Canal Azul", grid_basicos, 4, 0, lambda: self.aplicar_filtro("Canal Azul"))

        layout_basicos.addLayout(grid_basicos)

        marco_slider = QFrame()
        marco_slider.setStyleSheet(
            "background-color: #333; border: 1px solid #555; border-radius: 5px; margin-top: 10px;")
        layout_slider = QVBoxLayout(marco_slider)
        self.label_umbral = QLabel("Umbral de Binarización: 128")
        self.label_umbral.setAlignment(Qt.AlignCenter)
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
        self.tabs.addTab(tab_basicos, "Modelos y Canales")

        # PESTAÑA 2: Pseudocolor e IA
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

        self.btn_crear_mapa = QPushButton("🎨 Crear Paleta de Color Propia")
        self.btn_crear_mapa.setStyleSheet(
            "background-color: #d84315; font-weight: bold; padding: 12px; border-radius: 5px; margin-top: 5px;")
        self.btn_crear_mapa.clicked.connect(self.abrir_dialogo_personalizado)
        layout_color.addWidget(self.btn_crear_mapa)

        self.btn_recorte = QPushButton("IA: Recorte Facial [INACTIVO]")
        self.btn_recorte.setCheckable(True)
        self.btn_recorte.setStyleSheet(
            "background-color: #555555; border: 1px solid #707070; padding: 12px; font-weight: bold; border-radius: 5px; margin-top: 10px;")
        self.btn_recorte.clicked.connect(self.toggle_recorte)
        layout_color.addWidget(self.btn_recorte)
        layout_color.addStretch()

        self.tabs.addTab(tab_color, "Pseudocolor e IA")
        layout_marco.addWidget(self.tabs)
        layout_marco.addStretch()

        self.btn_guardar = QPushButton("💾 Guardar Imagen Actual")
        self.btn_guardar.setStyleSheet(
            "background-color: #2e7d32; font-weight: bold; padding: 12px; border-radius: 5px;")
        self.btn_guardar.clicked.connect(self.guardar_imagen)
        layout_marco.addWidget(self.btn_guardar)

        marco_botones.setLayout(layout_marco)

        # HISTOGRAMA Y ESTADÍSTICAS
        self.marco_hist = QFrame()
        self.marco_hist.setFixedWidth(390)
        self.marco_hist.setFixedHeight(340)
        self.marco_hist.setStyleSheet(
            "QFrame { background-color: #1e1e1e; border: 2px solid #555555; border-radius: 10px; margin-top: 10px; }")
        layout_hist = QVBoxLayout(self.marco_hist)

        label_titulo_hist = QLabel("Análisis Estadístico")
        label_titulo_hist.setAlignment(Qt.AlignCenter)
        label_titulo_hist.setStyleSheet("border: none; font-weight: bold; font-size: 14px; color: #aaa;")

        self.label_hist_display = QLabel("Selecciona una imagen para ver su histograma")
        self.label_hist_display.setAlignment(Qt.AlignCenter)
        self.label_hist_display.setStyleSheet("border: none; color: #666; font-size: 12px;")

        self.label_estadisticas = QLabel("Media: - | Varianza: - | Entropía: -\nAsimetría: - | Energía: -")
        self.label_estadisticas.setAlignment(Qt.AlignCenter)
        self.label_estadisticas.setStyleSheet(
            "border: none; color: #90caf9; font-weight: bold; font-size: 12px; margin-top: 5px;")

        layout_hist.addWidget(label_titulo_hist)
        layout_hist.addWidget(self.label_hist_display, 1)
        layout_hist.addWidget(self.label_estadisticas)

        marco_controles_v.addWidget(marco_botones)
        marco_controles_v.addWidget(self.marco_hist)

        # PANEL DERECHO
        self.mdi_area = QMdiArea()
        self.mdi_area.setStyleSheet(
            "QMdiArea { background-color: #1e1e1e; border: 2px dashed #555555; border-radius: 10px; }")
        self.mdi_area.subWindowActivated.connect(self.al_cambiar_ventana)

        layout_principal.addLayout(marco_controles_v)
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

    def al_cambiar_ventana(self, sub_ventana):
        if sub_ventana:
            visor = sub_ventana.widget()
            self.btn_recorte.setChecked(visor.recorte_activo)
            self.actualizar_color_boton_recorte(visor.recorte_activo)
            self.actualizar_histograma_fijo(visor)
        else:
            self.btn_recorte.setChecked(False)
            self.actualizar_color_boton_recorte(False)
            self.label_hist_display.setText("Selecciona una imagen para ver su histograma")
            self.label_hist_display.setPixmap(QPixmap())
            self.label_estadisticas.setText("Media: - | Varianza: - | Entropía: -\nAsimetría: - | Energía: -")

    def actualizar_histograma_fijo(self, visor):
        matriz_base = visor.imagen_original
        if visor.recorte_activo:
            recorte = extraer_rostro(matriz_base)
            if recorte is not None: matriz_base = recorte

        img_hist = generar_histograma(matriz_base)
        h, w, ch = img_hist.shape
        bytes_por_linea = ch * w
        imagen_qt = QImage(img_hist.data, w, h, bytes_por_linea, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(imagen_qt)
        self.label_hist_display.setPixmap(
            pixmap.scaled(self.label_hist_display.width(), self.label_hist_display.height(), Qt.KeepAspectRatio,
                          Qt.SmoothTransformation))

        media, varianza, entropia, asimetria, energia = calcular_estadisticas(matriz_base)
        texto_stats = f"Media: {media:.2f}  |  Varianza: {varianza:.2f}  |  Entropía: {entropia:.2f}\nAsimetría: {asimetria:.2f}  |  Energía: {energia:.4f}"
        self.label_estadisticas.setText(texto_stats)

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
        self.actualizar_histograma_fijo(visor)

    def cambiar_umbral(self, valor):
        self.label_umbral.setText(f"Umbral de Binarización: {valor}")
        ventana_activa = self.mdi_area.activeSubWindow()
        if ventana_activa:
            visor = ventana_activa.widget()
            visor.filtro_actual = "Binarizar"
            self.procesar_visor(visor, ventana_activa)

    def abrir_dialogo_personalizado(self):
        ventana_activa = self.mdi_area.activeSubWindow()
        if not ventana_activa:
            QMessageBox.warning(self, "Cuidado", "Abre una imagen primero para aplicarle tu mapa de color.")
            return

        dialogo = DialogoMapaPersonalizado(self)
        if dialogo.exec_():
            colores_nuevos = dialogo.obtener_colores_normalizados()
            visor = ventana_activa.widget()
            visor.filtro_actual = "Mapa Creado"
            visor.colores_custom = colores_nuevos
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
        elif tipo_filtro == "Modelo CMY":
            resultado = modelo_cmy(matriz_base)
        elif tipo_filtro == "Modelo YIQ":
            resultado = modelo_yiq(matriz_base)
        elif tipo_filtro == "Modelo HSI":
            resultado = modelo_hsi(matriz_base)
        elif tipo_filtro == "Modelo HSV":
            resultado = modelo_hsv(matriz_base)
        elif tipo_filtro == "Canal Rojo":
            resultado = canal_rojo(matriz_base)
        elif tipo_filtro == "Canal Verde":
            resultado = canal_verde(matriz_base)
        elif tipo_filtro == "Canal Azul":
            resultado = canal_azul(matriz_base)
        elif tipo_filtro == "Binarizar":
            resultado = binarizar_imagen(matriz_base, self.slider_umbral.value())
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
        elif tipo_filtro == "Mapa Creado":
            resultado = aplicar_mapa_personalizado(matriz_base, visor.colores_custom)

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