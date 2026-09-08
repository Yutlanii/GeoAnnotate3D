"""
ui/manual_dialog.py — Manual de uso in-app (Ayuda → Manual de uso).

Antes, entender el flujo completo, cada herramienta de etiquetado, o
qué es y para qué sirve `.ga3d_bin` requería leer el código fuente o
encontrar `docs/` en GitHub — nada de esto era descubrible desde
dentro de la propia app. Este diálogo resume el mismo contenido que
`docs/user-guide.md` (la fuente de verdad más completa, con más
detalle) para que quede accesible sin salir del programa.

Contenido derivado directamente del código (no inventado):
  - Pasos del flujo: `ui/main_window.py::STEP_NAMES/STEP_DESCRIPTIONS`.
  - Herramientas y teclas: `annotation/tools.py` / `annotation/region_growing.py`.
  - Formato GA3D-Bin: `core/heavy_cloud.py` (header, umbral de 200M pts, mmap).
"""
from __future__ import annotations
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QWidget,
)
from PyQt5.QtCore import Qt

from ui.icons import icon as qicon, pixmap as qpixmap
from ui.theme import (
    SURFACE, SURFACE_2, BORDER, BORDER_SOFT,
    TEXT, TEXT_DIM, TEXT_MUTE, ACCENT_STRONG, ACCENT_SOFT,
)

# (icono, título, párrafos de cuerpo — lista de str, cada uno un párrafo)
MANUAL_SECTIONS = [
    ("cloud-arrow-up", "Paso 1 — Nube", [
        "Carga un archivo LAS, LAZ, E57, PLY, XYZ, TXT, CSV, NPY o el "
        "formato propio .ga3d_bin (Ctrl+N, o arrastra el archivo a la "
        "ventana). Si tiene color RGB se abre en modo RGB; si no, "
        "coloreado por elevación.",
        "Para nubes muy densas, configura una grilla de tiles (tamaño "
        "en metros) — cada tile se puede abrir por separado a densidad "
        "completa, sin cargar toda la nube de golpe. Arrastra la grilla "
        "o usa Offset X/Y/Rotación para alinearla con la nube.",
    ]),
    ("magic", "Paso 2 — Pre-clasificar", [
        "Dos métodos automáticos, independientes entre sí, para reducir "
        "el etiquetado manual:",
        "AGL (altura sobre el suelo): clasifica por rangos de altura "
        "relativa al punto más bajo de la nube. \"Configurar rangos…\" "
        "abre una tabla donde cada clase tiene un rango Desde/Hasta — "
        "mover un límite desliza automáticamente el límite vecino para "
        "que los rangos no queden con huecos ni se solapen.",
        "CSF (Cloth Simulation Filter): el mismo algoritmo de detección "
        "de suelo que usa CloudCompare — simula una tela cayendo sobre "
        "la nube invertida.",
    ]),
    ("tag", "Paso 3 — Etiquetar", [
        "Clases (panel izquierdo): cada clase es un círculo numerado y "
        "coloreado — teclas 1-9 o clic para activarla. Doble clic para "
        "renombrar/cambiar color.",
        "Herramientas (panel derecho), cada una con su tecla: Pincel "
        "(B, esférico, Ctrl+arrastra), Disco (D), Region Growing (G, "
        "Ctrl+clic — crece por geometría conectada y similar), Polígono "
        "(L), Caja (X), Esfera (H, Ctrl+clic instantáneo), Corte Z (C), "
        "Pick (I, muestrea la clase de un punto), Medir (M).",
        "E alterna modo borrar (quita la clase, el punto sigue en la "
        "nube). Supr alterna modo eliminar puntos (saca el punto de la "
        "nube — para limpiar ruido del sensor; funciona con cualquier "
        "herramienta de selección de arriba). Ctrl+Z/Ctrl+Shift+Z "
        "deshacen/rehacen ambos.",
    ]),
    ("box-arrow-up", "Paso 4 — Exportar", [
        "Elige una o más arquitecturas destino (RandLA-Net, PointNet++, "
        "KPConv) — el formato de exportación se elige automáticamente "
        "según cuál. El split train/val/test es espacial (bloques "
        "geográficos separados, sin fuga de datos entre splits).",
        "Marca \"Usar códigos ASPRS estándar\" al exportar el .las "
        "clasificado para que abra correctamente en CloudCompare/QGIS/"
        "ArcGIS con la clasificación estándar en vez de tus IDs internos.",
    ]),
    ("cpu", "Paso 5 — Entrenar", [
        "Tres arquitecturas ya implementadas dentro de la app (RandLA-"
        "Net, PointNet++, KPConv), sin depender de un framework externo "
        "de entrenamiento. Apunta \"Dataset\" a la carpeta exportada y "
        "pulsa Iniciar entrenamiento.",
        "Puedes reanudar un entrenamiento detenido desde cualquier "
        "checkpoint guardado (.pth) — restaura también el optimizador y "
        "el learning-rate scheduler, no solo los pesos.",
        "Al terminar, además del modelo verás un reporte por clase "
        "(precision/recall/IoU de cada clase por separado, no solo el "
        "mIoU agregado) — útil para saber qué clase necesita más datos.",
    ]),
    ("bullseye", "Paso 6 — Inferir", [
        "Carga un modelo .pth entrenado y ejecuta inferencia sobre la "
        "nube cargada — el resultado se aplica directo al lienzo como "
        "anotaciones, revisables y corregibles con las herramientas de "
        "etiquetado.",
        "Inferencia por lote: selecciona una carpeta y procesa todos "
        "los archivos compatibles con el mismo modelo, sin supervisión.",
    ]),
    ("cloud-check", "¿Qué es .ga3d_bin?", [
        "Es el formato binario propio de GeoAnnotate3D — no un formato "
        "de intercambio como LAS/LAZ, sino una CACHÉ rápida que la app "
        "genera automáticamente junto al archivo original cuando supera "
        "200 millones de puntos.",
        "LAS/LAZ necesitan parsear (y, en LAZ, descomprimir) su formato "
        "de registros en cada lectura. GA3D-Bin es un layout binario "
        "plano — un header de 512 bytes seguido de XYZ en float32 "
        "crudo, sin compresión ni cuantización — que se puede abrir con "
        "mapeo de memoria (mmap) en vez de cargarlo todo a RAM: permite "
        "trabajar con nubes más grandes que tu memoria disponible, y "
        "reabrirlo es casi instantáneo porque no hay que volver a "
        "descomprimir ni parsear nada.",
        "No hay que gestionarlo a mano: la primera vez que abres un "
        ".las/.laz/.e57 muy pesado, la app ofrece convertirlo una vez; "
        "la siguiente vez que abras ESE MISMO archivo, detecta el "
        ".ga3d_bin existente y pregunta si usarlo. Es seguro borrarlo — "
        "se puede regenerar del original cuando haga falta.",
    ]),
]


class ManualDialog(QDialog):
    """Manual de uso in-app — mismo contenido resumido que docs/user-guide.md."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual de uso")
        self.setMinimumWidth(520)
        self.setMinimumHeight(480)
        self.setStyleSheet(f"QDialog{{background:{SURFACE};}}")
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(4)

        hdr = QHBoxLayout(); hdr.setSpacing(8)
        ic = QLabel(); ic.setPixmap(qpixmap("info-circle", ACCENT_STRONG, 16))
        hdr.addWidget(ic)
        title = QLabel("Manual de uso")
        title.setStyleSheet(f"color:{TEXT};font-size:15px;font-weight:700;background:transparent;")
        hdr.addWidget(title); hdr.addStretch()
        root.addLayout(hdr)

        subtitle = QLabel(
            "Resumen de cada paso del flujo y de las herramientas de etiquetado. "
            "Para más detalle, ver el manual completo en el repositorio de GitHub "
            "(docs/user-guide.md).")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;background:transparent;")
        root.addWidget(subtitle)
        root.addSpacing(10)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background:transparent;")
        content = QWidget(); content.setStyleSheet(f"background:{SURFACE};")
        cl = QVBoxLayout(content)
        cl.setContentsMargins(0, 0, 4, 0); cl.setSpacing(10)

        for icon_name, section_title, paragraphs in MANUAL_SECTIONS:
            card = QFrame(); card.setObjectName("manualCard")
            card.setStyleSheet(
                f"QFrame#manualCard{{background:{SURFACE_2};border-radius:5px;}}")
            card_l = QVBoxLayout(card)
            card_l.setContentsMargins(12, 10, 12, 11); card_l.setSpacing(6)

            hrow = QHBoxLayout(); hrow.setSpacing(7)
            hic = QLabel(); hic.setPixmap(qpixmap(icon_name, ACCENT_STRONG, 14))
            hrow.addWidget(hic)
            hlbl = QLabel(section_title)
            hlbl.setStyleSheet(f"color:{ACCENT_STRONG};font-size:11.5px;font-weight:700;background:transparent;")
            hrow.addWidget(hlbl, 1)
            card_l.addLayout(hrow)

            for para in paragraphs:
                p = QLabel(para)
                p.setWordWrap(True)
                p.setStyleSheet(f"color:{TEXT_DIM};font-size:10.5px;background:transparent;")
                card_l.addWidget(p)

            cl.addWidget(card)

        cl.addStretch()
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        close_btn = QPushButton("Cerrar")
        close_btn.setStyleSheet(
            f"QPushButton{{background:{SURFACE_2};border:1px solid {BORDER};border-radius:4px;"
            f"color:{TEXT_DIM};padding:8px 18px;font-size:10.5px;font-weight:600;}}"
            f"QPushButton:hover{{border-color:{ACCENT_STRONG};}}")
        close_btn.clicked.connect(self.accept)
        btn_row = QHBoxLayout(); btn_row.addStretch(); btn_row.addWidget(close_btn)
        root.addSpacing(10)
        root.addLayout(btn_row)
