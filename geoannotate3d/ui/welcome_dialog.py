"""
ui/welcome_dialog.py — Diálogo de bienvenida / onboarding

Aparece la primera vez que el usuario abre el software (y opcionalmente
siempre si no marca "No mostrar al inicio").

Explica el flujo de 6 pasos de forma visual y concisa para que
el usuario entienda el propósito de cada paso antes de comenzar.
"""
from __future__ import annotations
import os
from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QFrame, QScrollArea, QWidget, QSizePolicy,
)
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QFont, QPixmap, QColor, QPainter, QBrush

from ui.icons import icon as qicon, pixmap as qpixmap
from ui.theme import (
    SURFACE, SURFACE_2, SURFACE_3, BORDER, BORDER_SOFT,
    TEXT, TEXT_DIM, TEXT_MUTE,
    ACCENT, ACCENT_STRONG, ACCENT_SOFT, ACCENT_BORDER,
    OK, OK_SOFT, TEXT_FAINT,
)


# ── Datos de cada paso ────────────────────────────────────────────────────────

STEPS = [
    {
        "num":   "1",
        "icon":  "cloud-arrow-up",
        "title": "Nube de puntos",
        "color": ACCENT,
        "desc":  (
            "Carga cualquier archivo LAS, LAZ, E57 o GA3D-Bin. "
            "El software soporta nubes de cientos de millones de puntos "
            "gracias al sistema de tiles y al motor de LOD progresivo. "
            "Configura la cuadrícula espacial para dividir la nube en "
            "sectores manejables."
        ),
        "tips":  [
            "Nubes > 500M pts: se recomienda convertir a GA3D-Bin para mejor rendimiento",
            "Si la nube ya tiene clasificación LAS, se puede importar como anotaciones",
        ],
    },
    {
        "num":   "2",
        "icon":  "layers-half",
        "title": "Pre-clasificar",
        "color": OK,
        "desc":  (
            "Usa algoritmos geoespaciales para pre-clasificar la nube "
            "automáticamente antes de la anotación manual. "
            "CSF (Cloth Simulation Filter) detecta el suelo. "
            "AGL (Height Above Ground) clasifica por altura relativa "
            "para separar vegetación baja, media y alta, edificios, etc."
        ),
        "tips":  [
            "CSF funciona mejor en terrenos con pendiente moderada",
            "AGL requiere que el suelo esté clasificado previamente",
            "La pre-clasificación automática puede ahorrar horas de trabajo manual",
        ],
    },
    {
        "num":   "3",
        "icon":  "pencil-square",
        "title": "Etiquetar",
        "color": ACCENT,
        "desc":  (
            "Anota manualmente la nube con las herramientas de etiquetado: "
            "pincel esférico, polígono 2D/3D, caja 3D y región creciente. "
            "Puedes superponer ortomosaicos (.tif) y vectoriales (.shp) "
            "con coordenadas reales para guiar el etiquetado. "
            "Doble clic en una clase para cambiar su color y nombre."
        ),
        "tips":  [
            "Tecla Ctrl+Z deshace la última anotación",
            "Rueda del ratón cambia el tamaño del pincel en tiempo real",
            "Carga un ortomosaico en el panel derecho para ver la imagen aérea",
        ],
    },
    {
        "num":   "4",
        "icon":  "box-arrow-up",
        "title": "Exportar dataset",
        "color": ACCENT,
        "desc":  (
            "Exporta las anotaciones como dataset listo para entrenamiento "
            "de redes neuronales. El software genera automáticamente "
            "los archivos de puntos (.npy), etiquetas y el módulo "
            "custom_dataset.py compatible con PyTorch. "
            "La división train/val/test es automática."
        ),
        "tips":  [
            "Asegúrate de tener al menos 5,000 puntos por clase antes de exportar",
            "El dataset se puede reutilizar para entrenar múltiples arquitecturas",
        ],
    },
    {
        "num":   "5",
        "icon":  "cpu-fill",
        "title": "Entrenar",
        "color": TEXT_MUTE,
        "desc":  (
            "Entrena una red neuronal directamente dentro del software. "
            "Soporta tres arquitecturas: RandLA-Net (rápida, ideal para nubes grandes), "
            "PointNet++ (mejor captura multi-escala) y KPConv (detalle fino en bordes). "
            "Monitorea loss y mIoU en tiempo real con gráficas. "
            "Requiere PyTorch instalado (con o sin GPU/CUDA)."
        ),
        "tips":  [
            "GPU CUDA acelera el entrenamiento 10-50x respecto a CPU",
            "RandLA-Net es la arquitectura más rápida para empezar",
            "El mejor modelo se guarda automáticamente según el mIoU de validación",
        ],
    },
    {
        "num":   "6",
        "icon":  "magic",
        "title": "Inferir",
        "color": TEXT_MUTE,
        "desc":  (
            "Aplica el modelo entrenado a una nube completa para clasificarla "
            "automáticamente. La inferencia por parches solapados cubre toda "
            "la nube sin importar su tamaño. El resultado se aplica directamente "
            "al canvas como anotaciones que puedes revisar y corregir en el Paso 3, "
            "cerrando el ciclo de mejora continua."
        ),
        "tips":  [
            "El overlap del 50% suaviza las transiciones entre parches",
            "Después de inferir, corrige los errores y re-exporta para mejorar el modelo",
            "El ciclo anotar→entrenar→inferir→corregir mejora el mIoU con cada iteración",
        ],
    },
]


# ── Widget de un paso ─────────────────────────────────────────────────────────

class _StepCard(QFrame):
    def __init__(self, step: dict, parent=None):
        super().__init__(parent)
        # ID-scoped selector: la versión anterior usaba el selector de
        # CLASE "QFrame{...}", que cascadea a TODO QFrame descendiente
        # que no lo sobrescriba explícitamente — incluyendo el propio
        # separador VLine de abajo (mismo bug de fondo documentado en
        # NOTES_CLAUDE.md, aquí por selector de clase en vez de bare
        # rule). Con setObjectName + `QFrame#stepCard` el borde de
        # acento solo aplica a esta tarjeta.
        self.setObjectName("stepCard")
        self.setStyleSheet(
            f"QFrame#stepCard{{background:{SURFACE};border:1px solid {BORDER};"
            f"border-left:3px solid {step['color']};border-radius:3px;"
            f"padding:0px;}}")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(14)

        # Número + icono
        badge = QVBoxLayout(); badge.setSpacing(2); badge.setAlignment(Qt.AlignCenter)
        ic = QLabel(); ic.setAlignment(Qt.AlignCenter)
        ic.setPixmap(qpixmap(step['icon'], step['color'], 22))
        ic.setStyleSheet("background:transparent;border:none;")
        num = QLabel(step['num']); num.setAlignment(Qt.AlignCenter)
        num.setStyleSheet(
            f"color:{step['color']};font-size:13px;font-weight:700;"
            f"background:transparent;border:none;")
        badge.addWidget(ic); badge.addWidget(num)
        badge_w = QWidget(); badge_w.setLayout(badge); badge_w.setFixedWidth(44)
        badge_w.setStyleSheet("background:transparent;")
        lay.addWidget(badge_w)

        # Separador
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"background:{step['color']};max-width:1px;")
        lay.addWidget(sep)

        # Texto
        txt = QVBoxLayout()
        txt.setSpacing(4)

        title = QLabel(f"Paso {step['num']} — {step['title']}")
        title.setStyleSheet(
            f"color:{step['color']};font-size:11px;font-weight:700;"
            f"background:transparent;border:none;")
        txt.addWidget(title)

        desc = QLabel(step['desc'])
        desc.setWordWrap(True)
        desc.setStyleSheet(
            f"color:{TEXT_MUTE};font-size:10.5px;background:transparent;border:none;")
        txt.addWidget(desc)

        if step.get('tips'):
            tips_row = QHBoxLayout(); tips_row.setSpacing(5)
            tips_ic = QLabel(); tips_ic.setPixmap(qpixmap("info-circle", TEXT_MUTE, 10))
            tips_ic.setStyleSheet("background:transparent;border:none;")
            tips_ic.setAlignment(Qt.AlignTop)
            tips_lbl = QLabel("   •   ".join(step['tips']))
            tips_lbl.setWordWrap(True)
            tips_lbl.setStyleSheet(
                f"color:{TEXT_DIM};font-size:10.5px;background:transparent;border:none;"
                f"font-style:italic;")
            tips_row.addWidget(tips_ic); tips_row.addWidget(tips_lbl, 1)
            txt.addLayout(tips_row)

        lay.addLayout(txt, 1)


# ── Diálogo principal ─────────────────────────────────────────────────────────

class WelcomeDialog(QDialog):
    """
    Diálogo de bienvenida con flujo de 6 pasos.
    Se muestra al inicio si el usuario no lo ha desactivado.
    """
    SETTINGS_KEY = "show_welcome_on_start"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bienvenido a GeoAnnotate3D")
        self.setMinimumWidth(760)
        self.setMinimumHeight(600)
        self.resize(800, 700)
        self.setModal(True)
        self.setStyleSheet(
            f"QDialog{{background:{SURFACE_2};}}"
            f"QScrollArea{{background:{SURFACE_2};border:none;}}"
            f"QScrollBar:vertical{{background:{SURFACE_2};width:8px;border:none;}}"
            f"QScrollBar::handle:vertical{{background:{BORDER};border-radius:4px;min-height:20px;}}"
            f"QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{{height:0;}}")
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = QWidget()
        hdr.setObjectName("welcomeHdr")
        # Antes tenía un degradado teñido de acento (ACCENT_SOFT→SURFACE_2)
        # detrás del nombre "GeoAnnotate3D" — el usuario pidió que el fondo
        # sea el mismo color neutro del resto de la UI, no un color propio.
        hdr.setStyleSheet(
            f"QWidget#welcomeHdr{{background:{SURFACE};border-bottom:1px solid {BORDER};}}")
        hdr.setFixedHeight(90)
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(24, 0, 24, 0)

        title_block = QVBoxLayout()
        title_block.setSpacing(2)
        app_name = QLabel("GeoAnnotate3D")
        app_name.setStyleSheet(
            f"color:{ACCENT_STRONG};font-size:22px;font-weight:700;"
            f"letter-spacing:1px;background:transparent;border:none;")
        subtitle = QLabel(
            "Plataforma integral para anotación LiDAR y entrenamiento de redes neuronales geoespaciales")
        subtitle.setStyleSheet(
            f"color:{TEXT_DIM};font-size:10.5px;background:transparent;border:none;")
        subtitle.setWordWrap(True)
        title_block.addWidget(app_name)
        title_block.addWidget(subtitle)
        hl.addLayout(title_block, 1)

        # Version badge
        ver = QLabel("v1.0")
        ver.setObjectName("verBadge")
        ver.setStyleSheet(
            f"QLabel#verBadge{{color:{ACCENT_STRONG};background:{ACCENT_SOFT};border:1px solid {ACCENT_BORDER};"
            f"border-radius:4px;padding:4px 10px;font-size:10.5px;font-weight:700;}}")
        hl.addWidget(ver)
        root.addWidget(hdr)

        # ── Intro text ────────────────────────────────────────────────────────
        intro_w = QWidget()
        intro_w.setStyleSheet(f"background:{SURFACE_2};border-bottom:1px solid {BORDER};")
        il = QHBoxLayout(intro_w)
        il.setContentsMargins(24, 12, 24, 12)
        intro = QLabel(
            "GeoAnnotate3D es el único software que integra el ciclo completo de GeoAI en un solo entorno: "
            "desde la visualización de nubes de puntos masivas hasta el entrenamiento e inferencia de redes "
            "neuronales, con herramientas geoespaciales reales (AGL, CSF, ortomosaicos, shapefiles). "
            "El flujo de trabajo se divide en 6 pasos que se recorren de izquierda a derecha:"
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color:{TEXT_DIM};font-size:10.5px;background:transparent;border:none;")
        il.addWidget(intro)
        root.addWidget(intro_w)

        # ── Steps scroll area ─────────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        content.setStyleSheet(f"background:{SURFACE_2};")
        cl = QVBoxLayout(content)
        cl.setContentsMargins(20, 16, 20, 16)
        cl.setSpacing(8)

        for step in STEPS:
            cl.addWidget(_StepCard(step))

        # Cycle tip
        cycle_w = QFrame(); cycle_w.setObjectName("cycleTip")
        cycle_w.setStyleSheet(
            f"QFrame#cycleTip{{background:{OK_SOFT};border:1px solid {OK_SOFT};"
            f"border-radius:3px;}}")
        cycle_l = QHBoxLayout(cycle_w); cycle_l.setContentsMargins(10,10,10,10); cycle_l.setSpacing(6)
        cycle_ic = QLabel(); cycle_ic.setPixmap(qpixmap("arrow-counterclockwise", OK, 12))
        cycle_ic.setStyleSheet("background:transparent;")
        cycle = QLabel(
            "El ciclo de mejora continua:  "
            "Etiquetar → Exportar → Entrenar → Inferir → Corregir → Mejorar"
        )
        cycle.setStyleSheet(f"color:{TEXT_DIM};background:transparent;font-size:10.5px;")
        cycle.setWordWrap(True)
        cycle_l.addWidget(cycle_ic); cycle_l.addWidget(cycle, 1)
        cl.addWidget(cycle_w)

        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        # ── Footer ────────────────────────────────────────────────────────────
        footer = QWidget()
        footer.setStyleSheet(
            f"background:{SURFACE_2};border-top:1px solid {BORDER};")
        footer.setFixedHeight(54)
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(20, 0, 20, 0)

        self._chk = QCheckBox("No mostrar al iniciar")
        self._chk.setStyleSheet(
            f"QCheckBox{{color:{TEXT_DIM};font-size:10.5px;background:transparent;}}"
            f"QCheckBox::indicator{{width:14px;height:14px;"
            f"background:{SURFACE};border:1px solid {BORDER};border-radius:3px;}}"
            f"QCheckBox::indicator:checked{{background:{ACCENT};border-color:{ACCENT};}}")
        self._chk.setChecked(not self._should_show())
        fl.addWidget(self._chk)
        fl.addStretch()

        help_lbl = QLabel("Puedes volver a ver este tutorial desde  Ayuda → Bienvenida")
        help_lbl.setStyleSheet(
            f"color:{TEXT_MUTE};font-size:10.5px;background:transparent;border:none;")
        fl.addWidget(help_lbl)
        fl.addSpacing(16)

        btn_start = QPushButton("  Comenzar")
        btn_start.setIcon(qicon("play-fill", "#ffffff"))
        btn_start.setFixedSize(130, 34)
        btn_start.setStyleSheet(
            f"QPushButton{{background:{ACCENT};color:#ffffff;border:none;"
            f"border-radius:3px;font-size:10.5px;font-weight:700;}}"
            f"QPushButton:hover{{background:{ACCENT_STRONG};}}")
        btn_start.clicked.connect(self._on_start)
        fl.addWidget(btn_start)

        root.addWidget(footer)

    # ── Logic ────────────────────────────────────────────────────────────────

    def _on_start(self):
        # Save preference
        settings = QSettings("GeoAnnotate3D", "GeoAnnotate3D")
        settings.setValue(self.SETTINGS_KEY, not self._chk.isChecked())
        self.accept()

    @staticmethod
    def _should_show() -> bool:
        settings = QSettings("GeoAnnotate3D", "GeoAnnotate3D")
        return settings.value(WelcomeDialog.SETTINGS_KEY, True, type=bool)

    @classmethod
    def show_if_needed(cls, parent=None) -> None:
        """Muestra el diálogo solo si el usuario no lo ha desactivado."""
        if cls._should_show():
            dlg = cls(parent)
            dlg.exec_()

    @classmethod
    def show_always(cls, parent=None) -> None:
        """Muestra el diálogo siempre (desde menú Ayuda)."""
        dlg = cls(parent)
        dlg.exec_()
