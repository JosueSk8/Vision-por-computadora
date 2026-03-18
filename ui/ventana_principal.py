import cv2
from PyQt5.QtWidgets import (QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QWidget, QFileDialog, QMessageBox,
                             QFrame, QMdiArea, QMdiSubWindow, QTabWidget, QSlider, QLabel,
                             QToolBox)  # <-- NUEVA IMPORTACIÓN
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

from core.procesamiento import cargar_imagen
from core.filtros import (filtro_grises, aplicar_mapa_cv2, mapa_pastel,
                          mapa_tierra, mapa_neon_termico, extraer_rostro,
                          canal_rojo, canal_verde, canal_azul, binarizar_imagen,
                          aplicar_mapa_personalizado)
from core.modelos_color import (modelo_hsv, modelo_cmy, modelo_yiq, modelo_hsi)
from core.estadisticas import generar_histograma, calcular_estadisticas

from core.ruidos import agregar_ruido_sal_pimienta, agregar_ruido_gaussiano
from core.operaciones import (suavizar_ruido, operacion_not, operacion_and,
                              operacion_or, operacion_xor, contar_objetos,
                              operacion_suma, operacion_resta, operacion_multiplicacion,
                              operacion_mayor, operacion_menor, operacion_igual)

from ui.visor_imagen import VisorImagen
from ui.dialogo_personalizado import DialogoMapaPersonalizado


class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Procesador de Imágenes - Prácticas 1, 2 y 3")
        self.setGeometry(50, 50, 1200, 850)
        self.setStyleSheet("background-color: #262626; color: white;")

        layout_principal = QHBoxLayout()
        layout_principal.setContentsMargins(15, 15, 15, 15)

        # PANEL IZQUIERDO
        marco_controles_v = QVBoxLayout()
        marco_botones = QFrame()
        marco_botones.setFixedWidth(430)
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

        # --- PESTAÑA 1: Básicos ---
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
        self.label_umbral = QLabel("Umbral General (Bin/Relacional): 128")
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
        self.tabs.addTab(tab_basicos, "Modelos")

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

        self.btn_crear_mapa = QPushButton("🎨 Crear Paleta Propia")
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

        self.tabs.addTab(tab_color, "Filtros e IA")

        # --- PESTAÑA 3: MINIPROYECTO (ACORDEÓN / QTOOLBOX) ---
        tab_p3 = QWidget()
        layout_p3 = QVBoxLayout(tab_p3)

        # Creamos el Acordeón
        self.acordeon_p3 = QToolBox()
        self.acordeon_p3.setStyleSheet("""
            QToolBox::tab { background-color: #4a4a4a; color: white; font-weight: bold; border-radius: 4px; padding: 5px; }
            QToolBox::tab:selected { background-color: #2a5a8a; font-style: italic; }
        """)

        # CAJA 1: Ruido
        widget_ruido = QWidget()
        grid_ruido = QGridLayout(widget_ruido)
        self.crear_boton("Polvo: Sal/Pimienta", grid_ruido, 0, 0, lambda: self.aplicar_filtro("Ruido SP"))
        self.crear_boton("Estática: Gaussiano", grid_ruido, 0, 1, lambda: self.aplicar_filtro("Ruido Gaussiano"))
        self.crear_boton("Suavizador (Gauss)", grid_ruido, 1, 0, lambda: self.aplicar_filtro("Filtro Suavizador"))
        self.acordeon_p3.addItem(widget_ruido, "▶ 1. Ruido y Filtros (3-a)")

        # CAJA 2: Aritméticas
        widget_arit = QWidget()
        grid_arit = QGridLayout(widget_arit)
        self.crear_boton("Suma (+ Brillo)", grid_arit, 0, 0, lambda: self.aplicar_filtro("Op Suma"))
        self.crear_boton("Resta (- Brillo)", grid_arit, 0, 1, lambda: self.aplicar_filtro("Op Resta"))
        self.crear_boton("Multiplicar (Contraste)", grid_arit, 1, 0, lambda: self.aplicar_filtro("Op Mult"))
        self.acordeon_p3.addItem(widget_arit, "▶ 2. Operaciones Aritméticas (3-b)")

        # CAJA 3: Relacionales
        widget_relac = QWidget()
        grid_relac = QGridLayout(widget_relac)
        self.crear_boton("Mayor que (>)", grid_relac, 0, 0, lambda: self.aplicar_filtro("Op Mayor"))
        self.crear_boton("Menor que (<)", grid_relac, 0, 1, lambda: self.aplicar_filtro("Op Menor"))
        self.crear_boton("Igual a (==)", grid_relac, 1, 0, lambda: self.aplicar_filtro("Op Igual"))
        self.acordeon_p3.addItem(widget_relac, "▶ 3. Op. Relacionales (Usan Slider) (3-b)")

        # CAJA 4: Lógicas
        widget_logic = QWidget()
        grid_logic = QGridLayout(widget_logic)
        self.crear_boton("Op. AND", grid_logic, 0, 0, lambda: self.aplicar_filtro("Operación AND"))
        self.crear_boton("Op. OR", grid_logic, 0, 1, lambda: self.aplicar_filtro("Operación OR"))
        self.crear_boton("Op. XOR", grid_logic, 1, 0, lambda: self.aplicar_filtro("Operación XOR"))
        self.crear_boton("Invertir (NOT)", grid_logic, 1, 1, lambda: self.aplicar_filtro("Operación NOT"))
        self.acordeon_p3.addItem(widget_logic, "▶ 4. Compuertas Lógicas (Piden 2da foto)")

        # CAJA 5: Conteo
        widget_conteo = QWidget()
        grid_conteo = QGridLayout(widget_conteo)
        self.crear_boton("Contar (Vecindad-4)", grid_conteo, 0, 0, lambda: self.aplicar_filtro("Contar V-4"))
        self.crear_boton("Contar (Vecindad-8)", grid_conteo, 0, 1, lambda: self.aplicar_filtro("Contar V-8"))
        self.acordeon_p3.addItem(widget_conteo, "▶ 5. Conteo de Objetos (3-c)")

        # Agregamos el acordeón a la pestaña
        layout_p3.addWidget(self.acordeon_p3)

        self.tabs.addTab(tab_p3, "Práctica 3")

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
        self.marco_hist.setFixedWidth(430)
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
        self.label_umbral.setText(f"Umbral General (Bin/Relacional): {valor}")
        ventana_activa = self.mdi_area.activeSubWindow()
        if ventana_activa:
            visor = ventana_activa.widget()
            if visor.filtro_actual in ["Binarizar", "Op Mayor", "Op Menor", "Op Igual", "Contar V-4", "Contar V-8"]:
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

        if tipo_filtro in ["Operación AND", "Operación OR", "Operación XOR"]:
            ruta_img2, _ = QFileDialog.getOpenFileName(self, f"Selecciona la 2DA IMAGEN para la {tipo_filtro}", "",
                                                       "Imágenes (*.png *.jpg *.jpeg)")
            if not ruta_img2:
                return
            visor.img2_cache = cargar_imagen(ruta_img2)

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

        # --- PRÁCTICA 3 ---
        elif tipo_filtro == "Ruido SP":
            gris = cv2.cvtColor(matriz_base, cv2.COLOR_RGB2GRAY)
            ruido = agregar_ruido_sal_pimienta(gris, cantidad=0.05)
            resultado = cv2.cvtColor(ruido, cv2.COLOR_GRAY2RGB)

        elif tipo_filtro == "Ruido Gaussiano":
            gris = cv2.cvtColor(matriz_base, cv2.COLOR_RGB2GRAY)
            ruido = agregar_ruido_gaussiano(gris, media=0, sigma=30)
            resultado = cv2.cvtColor(ruido, cv2.COLOR_GRAY2RGB)

        elif tipo_filtro == "Filtro Suavizador":
            resultado = suavizar_ruido(matriz_base)

        elif tipo_filtro == "Op Suma":
            resultado = operacion_suma(matriz_base, 50)

        elif tipo_filtro == "Op Resta":
            resultado = operacion_resta(matriz_base, 50)

        elif tipo_filtro == "Op Mult":
            resultado = operacion_multiplicacion(matriz_base, 1.5)

        elif tipo_filtro == "Op Mayor":
            resultado = operacion_mayor(matriz_base, self.slider_umbral.value())

        elif tipo_filtro == "Op Menor":
            resultado = operacion_menor(matriz_base, self.slider_umbral.value())

        elif tipo_filtro == "Op Igual":
            resultado = operacion_igual(matriz_base, self.slider_umbral.value())

        elif tipo_filtro == "Operación NOT":
            resultado = operacion_not(matriz_base)

        elif tipo_filtro == "Operación AND":
            resultado = operacion_and(matriz_base, visor.img2_cache)

        elif tipo_filtro == "Operación OR":
            resultado = operacion_or(matriz_base, visor.img2_cache)

        elif tipo_filtro == "Operación XOR":
            resultado = operacion_xor(matriz_base, visor.img2_cache)

        elif tipo_filtro == "Contar V-4":
            resultado, conteo = contar_objetos(matriz_base, vecindad=4, valor_umbral=self.slider_umbral.value())
            QMessageBox.information(self, "Resultado", f"¡Se detectaron {conteo} objetos usando Vecindad-4!")

        elif tipo_filtro == "Contar V-8":
            resultado, conteo = contar_objetos(matriz_base, vecindad=8, valor_umbral=self.slider_umbral.value())
            QMessageBox.information(self, "Resultado", f"¡Se detectaron {conteo} objetos usando Vecindad-8!")

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