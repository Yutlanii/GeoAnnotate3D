"""
ui/welcome_dialog.py — Diálogo de bienvenida / onboarding

Aparece la primera vez que el usuario abre el software (y opcionalmente
siempre si no marca "No mostrar al inicio").

REDISEÑO 2026-09-06 (v2 — tour guiado en vez de pared de texto):
  - v1 apilaba las 6 tarjetas de paso en una sola lista vertical con
    scroll, cada una con un párrafo denso + tips en cursiva unidos con
    "•" — se leía como una hoja de especificaciones, no como una
    bienvenida. Pedido explícito del usuario: rediseñar para que sea
    "más amigable y entendible".
  - v2: un tour guiado de un paso a la vez (como Slack/Notion/VS Code
    en su primer arranque) — un riel a la izquierda con los 6 pasos
    como círculos numerados coloreados (mismo lenguaje visual que
    class_panel.py: número + color fusionados en un badge, en vez de
    un icono y un número por separado), y a la derecha SOLO el paso
    activo en grande, con una descripción corta (1-2 frases) y UN tip
    destacado — no un párrafo + una lista de tips en cursiva.
  - Navegación: click en cualquier paso del riel, o Anterior/Siguiente.
"""
from __future__ import annotations
import os
from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QFrame, QWidget, QSizePolicy, QStackedWidget,
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


def _readable_on(hex_color: str) -> str:
    """Blanco o gris oscuro, el que dé mejor contraste — mismo helper
    que ya existe en class_panel.py, duplicado aquí a propósito para no
    crear un acoplamiento entre dos módulos de UI por una función de
    3 líneas."""
    try:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        luma = 0.299*r + 0.587*g + 0.114*b
        return "#2b2d30" if luma > 150 else "#ffffff"
    except Exception:
        return "#ffffff"


# ── Datos de cada paso (descripciones acortadas a 1-2 frases + UN tip) ────────

STEPS = [
    {
        "num": "1", "icon": "cloud-arrow-up", "title": "Nube de puntos", "color": ACCENT,
        "desc": "Carga un archivo LAS, LAZ, E57 o GA3D-Bin. Soporta cientos de "
                "millones de puntos gracias al sistema de tiles y LOD progresivo.",
        "tip": "Si la nube ya tiene clasificación LAS, se importa automáticamente como anotaciones.",
    },
    {
        "num": "2", "icon": "layers-half", "title": "Pre-clasificar", "color": OK,
        "desc": "Clasifica automáticamente antes de anotar a mano: CSF detecta el "
                "suelo, AGL clasifica por altura (vegetación, edificios, etc.).",
        "tip": "La pre-clasificación automática puede ahorrarte horas de trabajo manual.",
    },
    {
        "num": "3", "icon": "pencil-square", "title": "Etiquetar", "color": ACCENT,
        "desc": "Anota a mano con pincel, disco, polígono, caja o crecimiento de "
                "región. Puedes superponer ortomosaicos y capas vectoriales reales.",
        "tip": "Ctrl+Z deshace; la rueda del ratón cambia el tamaño del pincel al vuelo.",
    },
    {
        "num": "4", "icon": "box-arrow-up", "title": "Exportar dataset", "color": ACCENT,
        "desc": "Genera un dataset listo para PyTorch (RandLA-Net, PointNet++ o "
                "KPConv), con split train/val/test automático.",
        "tip": "Asegúrate de tener al menos ~5,000 puntos por clase antes de exportar.",
    },
    {
        "num": "5", "icon": "cpu-fill", "title": "Entrenar", "color": TEXT_MUTE,
        "desc": "Entrena directamente aquí — sin salir del programa ni escribir "
                "código. Monitorea loss y mIoU en vivo, con o sin GPU.",
        "tip": "Puedes pausar y luego reanudar el entrenamiento desde el último checkpoint.",
    },
    {
        "num": "6", "icon": "magic", "title": "Inferir", "color": TEXT_MUTE,
        "desc": "Aplica un modelo entrenado para clasificar una nube completa "
                "automáticamente — sola o en lote sobre una carpeta entera.",
        "tip": "Corrige los errores del modelo y re-exporta: el ciclo mejora el mIoU cada vez.",
    },
]


# ── Ítem del riel de pasos (mismo lenguaje visual que class_panel._ClassRow) ──

class _StepRailItem(QFrame):
    clicked_step = None   # placeholder de tipo, la señal real es un signal Qt

    def __init__(self, step: dict, parent=None):
        super().__init__(parent)
        from PyQt5.QtCore import pyqtSignal
        self._step = step
        self._active = False
        self.setObjectName("welcomeStepItem")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(46)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 10, 0)
        lay.setSpacing(10)

        color = step["color"]
        fg = _readable_on(color)
        self._badge = QLabel(step["num"])
        self._badge.setFixedSize(28, 28)
        self._badge.setAlignment(Qt.AlignCenter)
        self._badge.setStyleSheet(
            f"background:{color};border-radius:14px;color:{fg};"
            f"font-size:12px;font-weight:700;")
        lay.addWidget(self._badge)

        self._title = QLabel(step["title"])
        self._title.setWordWrap(True)
        lay.addWidget(self._title, 1)

        self._apply_style()

    def _apply_style(self):
        if self._active:
            self.setStyleSheet(
                f"QFrame#welcomeStepItem{{background:{ACCENT_SOFT};border-radius:6px;}}")
            self._title.setStyleSheet(
                f"color:{TEXT};font-size:11.5px;font-weight:700;background:transparent;")
        else:
            self.setStyleSheet(
                f"QFrame#welcomeStepItem{{background:transparent;border-radius:6px;}}"
                f"QFrame#welcomeStepItem:hover{{background:{SURFACE_2};}}")
            self._title.setStyleSheet(
                f"color:{TEXT_DIM};font-size:11.5px;font-weight:600;background:transparent;")

    def set_active(self, active: bool):
        self._active = active
        self._apply_style()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and callable(self._on_click):
            self._on_click()

    def set_on_click(self, fn):
        self._on_click = fn


# ── Panel de un paso (grande, a la derecha) ───────────────────────────────────

class _StepDetail(QWidget):
    def __init__(self, step: dict, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)

        color = step["color"]
        hdr = QHBoxLayout(); hdr.setSpacing(12)
        ic_bg = QFrame(); ic_bg.setObjectName("stepIconBg")
        ic_bg.setFixedSize(52, 52)
        ic_bg.setStyleSheet(f"QFrame#stepIconBg{{background:{color};border-radius:12px;}}")
        ic_l = QVBoxLayout(ic_bg); ic_l.setContentsMargins(0,0,0,0)
        ic = QLabel(); ic.setAlignment(Qt.AlignCenter)
        ic.setPixmap(qpixmap(step["icon"], _readable_on(color), 26))
        ic.setStyleSheet("background:transparent;")
        ic_l.addWidget(ic)
        hdr.addWidget(ic_bg)

        title_col = QVBoxLayout(); title_col.setSpacing(2)
        step_lbl = QLabel(f"PASO {step['num']} DE {len(STEPS)}")
        step_lbl.setStyleSheet(f"color:{TEXT_MUTE};font-size:9.5px;font-weight:700;letter-spacing:0.5px;background:transparent;")
        title = QLabel(step["title"])
        title.setStyleSheet(f"color:{TEXT};font-size:19px;font-weight:700;background:transparent;")
        title_col.addWidget(step_lbl); title_col.addWidget(title)
        hdr.addLayout(title_col, 1)
        lay.addLayout(hdr)

        desc = QLabel(step["desc"])
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color:{TEXT_DIM};font-size:12.5px;background:transparent;line-height:150%;")
        lay.addWidget(desc)

        tip = QFrame(); tip.setObjectName("stepTip")
        tip.setStyleSheet(f"QFrame#stepTip{{background:{SURFACE_2};border-radius:6px;}}")
        tip_l = QHBoxLayout(tip); tip_l.setContentsMargins(12, 10, 12, 10); tip_l.setSpacing(8)
        tip_ic = QLabel(); tip_ic.setPixmap(qpixmap("info-circle", ACCENT_STRONG, 13))
        tip_ic.setStyleSheet("background:transparent;")
        tip_ic.setAlignment(Qt.AlignTop)
        tip_txt = QLabel(step["tip"])
        tip_txt.setWordWrap(True)
        tip_txt.setStyleSheet(f"color:{TEXT_DIM};font-size:10.5px;background:transparent;")
        tip_l.addWidget(tip_ic); tip_l.addWidget(tip_txt, 1)
        lay.addWidget(tip)
        lay.addStretch()


# ── Diálogo principal ─────────────────────────────────────────────────────────

class WelcomeDialog(QDialog):
    """
    Diálogo de bienvenida — tour guiado de 6 pasos, uno a la vez.
    Se muestra al inicio si el usuario no lo ha desactivado.
    """
    SETTINGS_KEY = "show_welcome_on_start"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bienvenido a GeoAnnotate3D")
        self.setMinimumWidth(720)
        self.setMinimumHeight(560)
        self.resize(760, 600)
        self.setModal(True)
        self.setStyleSheet(f"QDialog{{background:{SURFACE_2};}}")
        self._current = 0
        self._rail_items = []
        self._build_ui()
        self._select_step(0)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = QWidget()
        hdr.setObjectName("welcomeHdr")
        hdr.setStyleSheet(
            f"QWidget#welcomeHdr{{background:{SURFACE};border-bottom:1px solid {BORDER};}}")
        hdr.setFixedHeight(64)
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(24, 0, 24, 0)
        logo_ic = QLabel(); logo_ic.setPixmap(qpixmap("cloud-check", ACCENT, 20))
        logo_ic.setStyleSheet("background:transparent;")
        hl.addWidget(logo_ic)
        app_name = QLabel("GeoAnnotate3D")
        app_name.setStyleSheet(
            f"color:{ACCENT_STRONG};font-size:17px;font-weight:700;background:transparent;")
        hl.addSpacing(4); hl.addWidget(app_name)
        subtitle = QLabel("· anotación LiDAR y entrenamiento de redes neuronales, todo en un solo lugar")
        subtitle.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;background:transparent;")
        hl.addWidget(subtitle)
        hl.addStretch()
        ver = QLabel("v1.0")
        ver.setObjectName("verBadge")
        ver.setFixedHeight(22)
        ver.setAlignment(Qt.AlignCenter)
        ver.setStyleSheet(
            f"QLabel#verBadge{{color:{ACCENT_STRONG};background:{ACCENT_SOFT};border:1px solid {ACCENT_BORDER};"
            f"border-radius:11px;padding:0 10px;font-size:10px;font-weight:700;}}")
        hl.addWidget(ver)
        root.addWidget(hdr)

        # ── Cuerpo: riel de pasos (izq) + detalle del paso activo (der) ────────
        body = QWidget(); body.setStyleSheet(f"background:{SURFACE_2};")
        bl = QHBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)

        rail = QWidget(); rail.setFixedWidth(200)
        rail.setStyleSheet(f"background:{SURFACE};border-right:1px solid {BORDER};")
        rail_l = QVBoxLayout(rail)
        rail_l.setContentsMargins(10, 14, 10, 14); rail_l.setSpacing(2)
        for i, step in enumerate(STEPS):
            item = _StepRailItem(step)
            item.set_on_click(lambda idx=i: self._select_step(idx))
            self._rail_items.append(item)
            rail_l.addWidget(item)
        rail_l.addStretch()

        cycle = QFrame(); cycle.setObjectName("cycleTip")
        cycle.setStyleSheet(f"QFrame#cycleTip{{background:{OK_SOFT};border-radius:5px;}}")
        cycle_l = QVBoxLayout(cycle); cycle_l.setContentsMargins(9, 8, 9, 8); cycle_l.setSpacing(3)
        cycle_ic_row = QHBoxLayout(); cycle_ic_row.setSpacing(5)
        cycle_ic = QLabel(); cycle_ic.setPixmap(qpixmap("arrow-counterclockwise", OK, 11))
        cycle_ic.setStyleSheet("background:transparent;")
        cycle_lbl = QLabel("El ciclo de mejora")
        cycle_lbl.setStyleSheet(f"color:{OK};font-size:9.5px;font-weight:700;background:transparent;")
        cycle_ic_row.addWidget(cycle_ic); cycle_ic_row.addWidget(cycle_lbl); cycle_ic_row.addStretch()
        cycle_l.addLayout(cycle_ic_row)
        cycle_txt = QLabel("Etiquetar → Exportar → Entrenar → Inferir → Corregir")
        cycle_txt.setWordWrap(True)
        cycle_txt.setStyleSheet(f"color:{TEXT_DIM};font-size:9.5px;background:transparent;")
        cycle_l.addWidget(cycle_txt)
        rail_l.addWidget(cycle)

        bl.addWidget(rail)

        detail_wrap = QWidget(); detail_wrap.setStyleSheet(f"background:{SURFACE_2};")
        dw_l = QVBoxLayout(detail_wrap)
        dw_l.setContentsMargins(28, 26, 28, 20); dw_l.setSpacing(0)
        self._stack = QStackedWidget()
        for step in STEPS:
            self._stack.addWidget(_StepDetail(step))
        dw_l.addWidget(self._stack, 1)

        nav_row = QHBoxLayout(); nav_row.setSpacing(8)
        self._prev_btn = QPushButton("←  Anterior")
        self._prev_btn.setStyleSheet(
            f"QPushButton{{background:{SURFACE};border:1px solid {BORDER};border-radius:4px;"
            f"color:{TEXT_DIM};padding:7px 14px;font-size:10.5px;font-weight:600;}}"
            f"QPushButton:hover{{border-color:{ACCENT_BORDER};}}"
            f"QPushButton:disabled{{color:{TEXT_FAINT};}}")
        self._prev_btn.clicked.connect(lambda: self._select_step(self._current - 1))
        self._next_btn = QPushButton("Siguiente  →")
        self._next_btn.setStyleSheet(
            f"QPushButton{{background:{ACCENT_SOFT};border:1px solid {ACCENT_BORDER};border-radius:4px;"
            f"color:{ACCENT_STRONG};padding:7px 14px;font-size:10.5px;font-weight:700;}}"
            f"QPushButton:hover{{border-color:{ACCENT};}}")
        self._next_btn.clicked.connect(lambda: self._select_step(self._current + 1))
        nav_row.addWidget(self._prev_btn); nav_row.addWidget(self._next_btn); nav_row.addStretch()
        dw_l.addSpacing(14)
        dw_l.addLayout(nav_row)

        bl.addWidget(detail_wrap, 1)
        root.addWidget(body, 1)

        # ── Footer ────────────────────────────────────────────────────────────
        footer = QWidget()
        footer.setStyleSheet(f"background:{SURFACE};border-top:1px solid {BORDER};")
        footer.setFixedHeight(54)
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(20, 0, 20, 0)

        self._chk = QCheckBox("No mostrar al iniciar")
        self._chk.setStyleSheet(
            f"QCheckBox{{color:{TEXT_DIM};font-size:10.5px;background:transparent;}}"
            f"QCheckBox::indicator{{width:14px;height:14px;"
            f"background:{SURFACE_2};border:1px solid {BORDER};border-radius:3px;}}"
            f"QCheckBox::indicator:checked{{background:{ACCENT};border-color:{ACCENT};}}")
        self._chk.setChecked(not self._should_show())
        fl.addWidget(self._chk)
        fl.addStretch()

        help_lbl = QLabel("Puedes volver a ver este tour desde  Ayuda → Bienvenida / Tutorial")
        help_lbl.setStyleSheet(f"color:{TEXT_MUTE};font-size:10px;background:transparent;")
        fl.addWidget(help_lbl)
        fl.addSpacing(16)

        btn_start = QPushButton("  Comenzar")
        btn_start.setIcon(qicon("play-fill", "#ffffff"))
        btn_start.setFixedSize(130, 34)
        btn_start.setStyleSheet(
            f"QPushButton{{background:{ACCENT};color:#ffffff;border:none;"
            f"border-radius:4px;font-size:10.5px;font-weight:700;}}"
            f"QPushButton:hover{{background:{ACCENT_STRONG};}}")
        btn_start.clicked.connect(self._on_start)
        fl.addWidget(btn_start)

        root.addWidget(footer)

    # ── Navegación entre pasos ────────────────────────────────────────────────

    def _select_step(self, idx: int) -> None:
        idx = max(0, min(len(STEPS) - 1, idx))
        self._current = idx
        self._stack.setCurrentIndex(idx)
        for i, item in enumerate(self._rail_items):
            item.set_active(i == idx)
        self._prev_btn.setEnabled(idx > 0)
        self._next_btn.setText("Siguiente  →" if idx < len(STEPS) - 1 else "Ir al final  →")
        if idx == len(STEPS) - 1:
            self._next_btn.setEnabled(False)
        else:
            self._next_btn.setEnabled(True)

    # ── Logic ────────────────────────────────────────────────────────────────

    def _on_start(self):
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
