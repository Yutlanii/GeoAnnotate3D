"""
ui/layer_properties.py — Ventana flotante de propiedades de capa

Vectorial: color, grosor de línea, estilo (sólido/guiones/puntos), opacidad
Raster:    opacidad, ajuste Z, brillo, contraste

REDISEÑO 2026-09-05: colores movidos a ui/theme.py (ya eran los valores
correctos, solo sueltos inline) + icono real en el título en vez de
glifos ⬟/⬛.
"""
from __future__ import annotations
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton,
    QComboBox, QColorDialog, QFrame, QWidget, QDoubleSpinBox,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor

from ui.icons import icon as qicon, pixmap as qpixmap
from ui.theme import (
    SURFACE, SURFACE_2, SURFACE_3, BORDER, BORDER_SOFT,
    TEXT, TEXT_DIM, TEXT_MUTE,
    ACCENT, ACCENT_STRONG, ACCENT_BORDER,
)

S_LABEL  = f"color:{TEXT_MUTE};font-size:10.5px;"
S_VAL    = f"color:{ACCENT_STRONG};font-size:10.5px;min-width:36px;text-align:right;"
S_SLIDER = (
    f"QSlider::groove:horizontal{{background:{SURFACE_3};height:4px;border-radius:2px;}}"
    f"QSlider::handle:horizontal{{background:{ACCENT};width:10px;height:10px;"
    f"margin:-3px 0;border-radius:3px;}}"
    f"QSlider::sub-page:horizontal{{background:{ACCENT_BORDER};border-radius:2px;}}")
S_COMBO = (
    f"QComboBox{{background:{SURFACE_2};border:1px solid {BORDER};border-radius:3px;"
    f"color:{TEXT_DIM};padding:3px 6px;font-size:10.5px;}}"
    f"QComboBox:hover{{border-color:{ACCENT};}}"
    f"QComboBox QAbstractItemView{{background:{SURFACE};color:{TEXT_DIM};"
    f"selection-background-color:{SURFACE_2};selection-color:{ACCENT_STRONG};}}")
S_BTN = (
    f"QPushButton{{background:{SURFACE_2};border:1px solid {BORDER};border-radius:3px;"
    f"color:{TEXT_DIM};padding:5px 10px;font-size:10.5px;}}"
    f"QPushButton:hover{{border-color:{ACCENT};color:{ACCENT_STRONG};}}")
S_DLG = (
    f"QDialog{{background:{SURFACE_2};}}"
    f"QLabel{{color:{TEXT_DIM};}}"
    f"QFrame[frameShape='4']{{background:{BORDER_SOFT};max-height:1px;}}")


class LayerPropertiesDialog(QDialog):
    """
    Ventana flotante de propiedades de capa.
    No bloquea la ventana principal (NonModal).
    Emite señales al cambiar propiedades para que el canvas actualice en tiempo real.
    """
    # Vectorial
    color_changed      = pyqtSignal(object, tuple)   # (layer, (r,g,b) 0-1)
    line_width_changed = pyqtSignal(object, float)
    line_style_changed = pyqtSignal(object, str)     # 'solid','dash','dot','dashdot'
    opacity_changed    = pyqtSignal(object, float)
    # Raster
    raster_opacity_changed  = pyqtSignal(object, float)
    raster_z_changed        = pyqtSignal(object, float)
    raster_brightness_changed = pyqtSignal(object, float)  # -1..1
    raster_contrast_changed = pyqtSignal(object, float)    # 0.5..2.0

    def __init__(self, layer, parent=None):
        super().__init__(parent)
        self._layer = layer
        self.setWindowTitle(f"Propiedades — {layer.name}")
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setModal(False)
        self.setMinimumWidth(300)
        self.setStyleSheet(S_DLG)
        self._build(layer.kind)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build(self, kind: str):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(10)

        # Title
        title_row = QHBoxLayout(); title_row.setSpacing(6)
        ic = QLabel()
        ic.setPixmap(qpixmap("diagram-3" if kind == "vector" else "layers", ACCENT_STRONG, 14))
        title_row.addWidget(ic)
        title = QLabel(f"{'Vectorial' if kind=='vector' else 'Raster'}  ·  {self._layer.name}")
        title.setStyleSheet(f"color:{ACCENT_STRONG};font-size:11px;font-weight:700;")
        title.setWordWrap(True)
        title_row.addWidget(title, 1)
        lay.addLayout(title_row)
        lay.addWidget(self._sep())

        if kind == "vector":
            self._build_vector(lay)
        else:
            self._build_raster(lay)

        lay.addWidget(self._sep())

        # Close button
        close_btn = QPushButton("  Cerrar")
        close_btn.setIcon(qicon("x-lg", TEXT_MUTE))
        close_btn.setStyleSheet(S_BTN)
        close_btn.clicked.connect(self.close)
        lay.addWidget(close_btn)

    def _build_vector(self, lay):
        # ── Color ──────────────────────────────────────────────────────────────
        self._vec_color = (1.0, 0.85, 0.1)  # default yellow
        color_row = QHBoxLayout()
        color_lbl = QLabel("Color de línea:"); color_lbl.setStyleSheet(S_LABEL)
        color_row.addWidget(color_lbl)
        self._color_preview = QPushButton()
        self._color_preview.setFixedSize(40, 22)
        self._update_color_preview()
        self._color_preview.clicked.connect(self._pick_color)
        color_row.addWidget(self._color_preview)
        color_row.addStretch()
        lay.addLayout(color_row)

        # ── Grosor de línea ────────────────────────────────────────────────────
        self._width_sl, width_val = self._slider_row(
            lay, "Grosor de línea", 1, 80, 25, fmt=lambda v: f"{v/10:.1f}px")
        self._width_sl.valueChanged.connect(
            lambda v: self.line_width_changed.emit(self._layer, v / 10.0))

        # ── Estilo de línea ────────────────────────────────────────────────────
        style_row = QHBoxLayout()
        style_lbl = QLabel("Estilo:")
        style_lbl.setStyleSheet(S_LABEL)
        style_row.addWidget(style_lbl)
        self._style_combo = QComboBox()
        self._style_combo.setStyleSheet(S_COMBO)
        for name, val in [("Sólido", "solid"), ("Guiones", "dash"),
                          ("Puntos", "dot"), ("Guion-Punto", "dashdot")]:
            self._style_combo.addItem(name, val)
        self._style_combo.currentIndexChanged.connect(
            lambda i: self.line_style_changed.emit(
                self._layer, self._style_combo.currentData()))
        style_row.addWidget(self._style_combo, 1)
        lay.addLayout(style_row)

        # ── Opacidad ───────────────────────────────────────────────────────────
        self._op_sl, _ = self._slider_row(
            lay, "Opacidad", 0, 100, 85, fmt=lambda v: f"{v}%")
        self._op_sl.valueChanged.connect(
            lambda v: self.opacity_changed.emit(self._layer, v / 100.0))

    def _build_raster(self, lay):
        # ── Opacidad ───────────────────────────────────────────────────────────
        self._r_op_sl, _ = self._slider_row(
            lay, "Opacidad", 0, 100, 85, fmt=lambda v: f"{v}%")
        self._r_op_sl.valueChanged.connect(
            lambda v: self.raster_opacity_changed.emit(self._layer, v / 100.0))

        # ── Altura Z ────────────────────────────────────────────────────────────
        z_row = QHBoxLayout()
        z_lbl = QLabel("Altura Z (offset):")
        z_lbl.setStyleSheet(S_LABEL); z_row.addWidget(z_lbl)
        self._z_spin = QDoubleSpinBox()
        self._z_spin.setRange(-50.0, 200.0); self._z_spin.setValue(0.0)
        self._z_spin.setSuffix(" m"); self._z_spin.setSingleStep(0.5)
        self._z_spin.setDecimals(1); self._z_spin.setFixedWidth(90)
        self._z_spin.setStyleSheet(
            f"QDoubleSpinBox{{background:{SURFACE_2};border:1px solid {BORDER};"
            f"border-radius:3px;color:{TEXT_DIM};padding:3px 5px;font-size:10.5px;}}"
            f"QDoubleSpinBox:hover{{border-color:{ACCENT};}}")
        self._z_spin.valueChanged.connect(
            lambda v: self.raster_z_changed.emit(self._layer, v))
        z_row.addWidget(self._z_spin); z_row.addStretch()
        lay.addLayout(z_row)

        # ── Brillo ──────────────────────────────────────────────────────────────
        self._bri_sl, _ = self._slider_row(
            lay, "Brillo", -50, 50, 0, fmt=lambda v: f"{v:+d}%")
        self._bri_sl.valueChanged.connect(
            lambda v: self.raster_brightness_changed.emit(self._layer, v / 50.0))

        # ── Contraste ────────────────────────────────────────────────────────────
        self._con_sl, _ = self._slider_row(
            lay, "Contraste", 50, 200, 100, fmt=lambda v: f"{v}%")
        self._con_sl.valueChanged.connect(
            lambda v: self.raster_contrast_changed.emit(self._layer, v / 100.0))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _slider_row(self, parent_lay, label, mn, mx, default, fmt=None):
        row = QHBoxLayout(); row.setSpacing(6)
        lbl = QLabel(label); lbl.setStyleSheet(S_LABEL); lbl.setMinimumWidth(78)
        row.addWidget(lbl)
        val_lbl = QLabel(fmt(default) if fmt else str(default))
        val_lbl.setStyleSheet(S_VAL); val_lbl.setAlignment(Qt.AlignRight)
        sl = QSlider(Qt.Horizontal)
        sl.setRange(mn, mx); sl.setValue(default)
        sl.setStyleSheet(S_SLIDER)
        if fmt:
            sl.valueChanged.connect(lambda v, l=val_lbl, f=fmt: l.setText(f(v)))
        row.addWidget(sl, 1); row.addWidget(val_lbl)
        parent_lay.addLayout(row)
        return sl, val_lbl

    def _pick_color(self):
        r, g, b = self._vec_color
        initial = QColor(int(r*255), int(g*255), int(b*255))
        col = QColorDialog.getColor(initial, self, "Seleccionar color")
        if col.isValid():
            self._vec_color = (col.redF(), col.greenF(), col.blueF())
            self._update_color_preview()
            self.color_changed.emit(self._layer, self._vec_color)

    def _update_color_preview(self):
        r, g, b = self._vec_color
        hex_col = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        self._color_preview.setStyleSheet(
            f"QPushButton{{background:{hex_col};border:1px solid {BORDER};"
            f"border-radius:3px;}}"
            f"QPushButton:hover{{border-color:{ACCENT};}}")

    def _sep(self):
        f = QFrame(); f.setFrameShape(QFrame.HLine)
        f.setStyleSheet(f"background:{BORDER_SOFT};max-height:1px;margin:2px 0;")
        return f
