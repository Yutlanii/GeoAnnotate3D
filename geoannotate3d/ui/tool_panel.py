"""
ui/tool_panel.py — Panel de herramientas v3.0 (rediseño claro 2026-09-04)

Cambios vs v2.0:
  - Iconos REALES (Bootstrap Icons vía ui/icons.py) en vez de dibujos a
    mano con QPainter — de paso corrige 2 herramientas (Disco, Region
    Growing) que antes se quedaban sin icono porque _draw_tool_icon no
    tenía un caso para ellas.
  - Paleta clara (ui/theme.py) — nada de fondos oscuros.
  - Se ELIMINÓ la sección "Capas superpuestas" que duplicaba a
    OverlayPanel (ahora accesible desde "Capas de referencia" en el
    riel de main_window.py). Un solo lugar para gestionar capas.

Layout:
  ── Selector de herramienta (grid 4×3, 9 herramientas) ───
  ── Modo erase / solo-vacíos (toggles globales) ──────────
  ── Contexto dinámico por herramienta ─────────────────────
  ── Visualización ─────────────────────────────────────────
"""
from __future__ import annotations
from typing import Dict, Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QPushButton, QFrame, QComboBox,
    QSizePolicy, QGridLayout, QDoubleSpinBox,
    QSpinBox, QCheckBox,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor

from annotation.tools import ALL_TOOLS
from ui.icons import pixmap as qpixmap, icon as qicon
from ui.theme import (
    SURFACE, SURFACE_2, SURFACE_3, BORDER, BORDER_SOFT,
    TEXT, TEXT_DIM, TEXT_MUTE,
    ACCENT, ACCENT_STRONG, ACCENT_SOFT, ACCENT_BORDER,
    OK, WARN, WARN_SOFT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-widgets
# ─────────────────────────────────────────────────────────────────────────────

class _Toggle(QWidget):
    """Switch tipo pill — dibujado a mano (es un control genérico, no un
    icono; se queda como estaba, solo con los colores nuevos)."""
    toggled = pyqtSignal(bool)

    def __init__(self, checked: bool = True, parent=None):
        super().__init__(parent)
        self._on = checked
        self.setFixedSize(34, 18)
        self.setCursor(Qt.PointingHandCursor)

    def is_checked(self): return self._on

    def set_checked(self, v: bool):
        if self._on != v:
            self._on = v; self.update()

    def mousePressEvent(self, e):
        self._on = not self._on; self.update(); self.toggled.emit(self._on)

    def paintEvent(self, e):
        from PyQt5.QtGui import QPainter, QBrush
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.setBrush(QBrush(QColor(ACCENT if self._on else SURFACE_3)))
        p.setPen(Qt.NoPen); p.drawRoundedRect(0, 0, w, h, h//2, h//2)
        tx = w - h + 2 if self._on else 2
        p.setBrush(QBrush(QColor("#f4f5f6")))
        p.drawEllipse(tx, 2, h - 4, h - 4)
        p.end()


class _SliderRow(QWidget):
    value_changed = pyqtSignal(float)

    def __init__(self, label: str, unit: str, min_v: int, max_v: int,
                 init_v: int, scale: float = 1.0, parent=None):
        super().__init__(parent)
        self._scale = scale
        self._unit  = unit
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 4); lay.setSpacing(2)
        h = QHBoxLayout(); h.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label); lbl.setStyleSheet(f"color:{TEXT_DIM}; font-size:10.5px;")
        self._val = QLabel(self._fmt(init_v))
        self._val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._val.setStyleSheet(f"color:{ACCENT_STRONG}; font-size:10.5px; font-weight:600; min-width:46px; font-family:Consolas;")
        h.addWidget(lbl); h.addWidget(self._val); lay.addLayout(h)
        self._s = QSlider(Qt.Horizontal)
        self._s.setRange(min_v, max_v)
        self._s.setValue(init_v)
        self._s.setStyleSheet(
            f"QSlider::groove:horizontal{{height:4px;background:{SURFACE_3};border-radius:2px;}}"
            f"QSlider::handle:horizontal{{width:13px;height:13px;margin:-5px 0;"
            f"border-radius:4px;background:{ACCENT};border:2px solid #f4f5f6;}}"
            f"QSlider::sub-page:horizontal{{background:{ACCENT};border-radius:2px;}}")
        self._s.valueChanged.connect(self._on)
        lay.addWidget(self._s)

    def _fmt(self, v: int) -> str:
        r = v * self._scale
        if self._scale < 1.0:
            return f"{r:.2f} {self._unit}"
        elif self._scale == 1.0:
            return f"{v} {self._unit}"
        else:
            return f"{r:.1f} {self._unit}"

    def _on(self, v: int):
        self._val.setText(self._fmt(v)); self.value_changed.emit(v * self._scale)

    def value(self) -> float:
        return self._s.value() * self._scale

    def set_value(self, real_v: float):
        v = int(round(real_v / self._scale)) if self._scale != 1.0 else int(real_v)
        self._s.blockSignals(True); self._s.setValue(v); self._s.blockSignals(False)
        self._val.setText(self._fmt(v))


# ─────────────────────────────────────────────────────────────────────────────
# Botón de herramienta — iconos reales (Bootstrap Icons)
# ─────────────────────────────────────────────────────────────────────────────

# Nombre de la herramienta (BaseTool.name) → icono real en ui/icons/*.svg
_TOOL_ICON = {
    "Pincel":         "brush",
    "Disco":          "record-circle",
    "Region Growing": "share",
    "Polígono":       "pentagon",
    "Caja":           "bounding-box",
    "Esfera":         "globe2",
    "Corte Z":        "layers-half",
    "Relleno":        "paint-bucket",
    "Pick":           "eyedropper",
    "Medir":          "rulers",
}


class _ToolButton(QWidget):
    """
    Botón de herramienta estilo TOOLBAR (QGIS / Cyclone 3DR): icono solo,
    tamaño fijo compacto, el nombre y el atajo van en el tooltip — no como
    texto permanente. Antes cada botón era una tarjeta grande de 56px con
    icono+nombre+atajo apilados, ocupando mucho espacio vertical para una
    sola fila de 9 herramientas; ahora son botones de 34×34 en una barra
    que se acomoda en varias filas según el ancho del panel, igual que
    cualquier barra de herramientas de un SIG.
    """
    clicked = pyqtSignal(str)
    SIZE = 34

    def __init__(self, tool_cls: type, parent=None):
        super().__init__(parent)
        self._name    = tool_cls.name
        self._key     = tool_cls.key
        self._icon    = _TOOL_ICON.get(self._name, "grid-3x3")
        self._tooltip = getattr(tool_cls, "tooltip", "")
        self._active  = False
        self.setCursor(Qt.PointingHandCursor)
        key_hint = f"  [{self._key}]" if self._key else ""
        self.setToolTip(f"{tool_cls.name}{key_hint}\n{self._tooltip}")
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setAttribute(Qt.WA_StyledBackground, True)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._icon_lbl = QLabel()
        self._icon_lbl.setAlignment(Qt.AlignCenter)
        # Se neutraliza border/radio explícitamente: un stylesheet "pelado"
        # (sin selector) en el padre se filtra a los hijos en Qt — sin esto
        # el icono heredaría el borde redondeado del botón por debajo.
        self._icon_lbl.setStyleSheet("background:transparent;border:none;")
        lay.addWidget(self._icon_lbl)

        self._update_style()

    def set_active(self, a: bool):
        self._active = a; self._update_style()

    def _update_style(self):
        color = ACCENT_STRONG if self._active else TEXT_DIM
        self._icon_lbl.setPixmap(qpixmap(self._icon, color, 18))
        if self._active:
            self.setStyleSheet(
                f"background:{ACCENT_SOFT};border:1px solid {ACCENT_BORDER};border-radius:3px;")
        else:
            self.setStyleSheet(
                f"background:{SURFACE};border:1px solid {BORDER};border-radius:3px;")

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self._name)

    def enterEvent(self, e):
        if not self._active:
            self.setStyleSheet(
                f"background:{SURFACE_2};border:1px solid {ACCENT_BORDER};border-radius:3px;")

    def leaveEvent(self, e):
        self._update_style()


# ─────────────────────────────────────────────────────────────────────────────
# Panel de contexto dinámico por herramienta
# ─────────────────────────────────────────────────────────────────────────────

_BTN_SS = (
    f"QPushButton{{background:{SURFACE};border:1px solid {BORDER};border-radius:3px;"
    f"color:{TEXT_DIM};padding:6px;font-size:10.5px;font-weight:600;}}"
    f"QPushButton:hover{{border-color:{ACCENT_BORDER};color:{ACCENT_STRONG};background:{ACCENT_SOFT};}}"
)

class _ContextSection(QWidget):
    # Señales de parámetros
    brush_radius_changed     = pyqtSignal(float)
    brush_overlap_changed    = pyqtSignal(float)
    brush_thickness_changed  = pyqtSignal(float)
    sphere_radius_changed    = pyqtSignal(float)
    flood_step_changed       = pyqtSignal(float)
    flood_max_pts_changed    = pyqtSignal(int)
    flood_z_tol_changed      = pyqtSignal(float)
    slice_mode_changed       = pyqtSignal(str)
    lasso_close_requested    = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current = ""

    def update_for_tool(self, name: str) -> None:
        if name == self._current: return
        self._current = name
        lay = self.layout()
        if lay:
            while lay.count():
                item = lay.takeAt(0)
                if item.widget(): item.widget().deleteLater()
            try:
                import sip; sip.delete(lay)
            except Exception:
                pass

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        hdr = QLabel(f"OPCIONES — {name.upper()}")
        hdr.setWordWrap(True)   # nombres largos ("REGION GROWING") no se recortan
        hdr.setStyleSheet(f"color:{TEXT_MUTE}; font-size:10.5px; font-weight:700; letter-spacing:0.5px;")
        root.addWidget(hdr)

        # ── Pincel ──────────────────────────────────────────────────────────
        if name == "Pincel":
            self._br = _SliderRow("Radio", "m", 1, 500, 18)
            self._br.value_changed.connect(self.brush_radius_changed)
            self._bov = _SliderRow("Densidad trazo", "%", 0, 100, 80)
            self._bov.value_changed.connect(self.brush_overlap_changed)
            self._bthk = _SliderRow("Grosor disco", "%", 5, 100, 40)
            self._bthk.value_changed.connect(self.brush_thickness_changed)
            root.addWidget(self._br)
            root.addWidget(self._bov)
            root.addWidget(self._bthk)
            root.addWidget(self._hint("Ctrl+arrastrar · sigue la superficie", OK))
            root.addWidget(self._hint("Grosor 100%=esfera · 5%=disco fino"))

        # ── Region Growing ─────────────────────────────────────────────────────
        elif name == "Region Growing":
            self._rg_dist    = _SliderRow("Radio maximo",    "m",     1, 200, 30)
            self._rg_tol_z   = _SliderRow("Tolerancia Z",   "dm",    1, 100, 10)
            self._rg_tol_rgb = _SliderRow("Tolerancia color","0-255", 0, 255, 30)
            # Las señales están en ToolPanel (padre), no en _ContextSection (self)
            tp = self.parent()
            while tp is not None and not isinstance(tp, ToolPanel):
                tp = getattr(tp, 'parent', lambda: None)()
            if tp is not None:
                self._rg_dist.value_changed.connect(
                    lambda v, _t=tp: _t.rg_dist_changed.emit(float(v)))
                self._rg_tol_z.value_changed.connect(
                    lambda v, _t=tp: _t.rg_tol_z_changed.emit(v * 0.1))
                self._rg_tol_rgb.value_changed.connect(
                    lambda v, _t=tp: _t.rg_tol_rgb_changed.emit(float(v)))
            root.addWidget(self._rg_dist)
            root.addWidget(self._rg_tol_z)
            root.addWidget(self._rg_tol_rgb)
            root.addWidget(self._hint("Ctrl+clic → crece conectado desde el punto", OK))
            root.addWidget(self._hint("Paso: distancia local · Tolerancias: vs la semilla"))

        # ── Polígono ─────────────────────────────────────────────────────────
        elif name == "Polígono":
            root.addWidget(self._hint("Ctrl+clic → añade vértice"))
            root.addWidget(self._hint("Clic inicio / Enter → aplica"))
            root.addWidget(self._hint("Z → invertir selección"))
            close_btn = QPushButton("Cerrar polígono  (Enter)")
            close_btn.setStyleSheet(_BTN_SS)
            close_btn.clicked.connect(self.lasso_close_requested)
            root.addWidget(close_btn)
            root.addWidget(self._hint("Selecciona TODO a cualquier profundidad", OK))

        # ── Caja (BoxSelect) ─────────────────────────────────────────────────
        elif name == "Caja":
            root.addWidget(self._hint("Ctrl+arrastrar → rectángulo"))
            root.addWidget(self._hint("Selecciona columna 3D completa", OK))
            root.addWidget(self._hint("Escape → cancelar"))

        # ── Esfera ────────────────────────────────────────────────────────────
        elif name == "Esfera":
            self._sr = _SliderRow("Radio", "m", 1, 500, 5)
            self._sr.value_changed.connect(self.sphere_radius_changed)
            root.addWidget(self._sr)
            root.addWidget(self._hint("Ctrl+clic para seleccionar"))

        # ── Corte Z ───────────────────────────────────────────────────────────
        elif name == "Corte Z":
            root.addWidget(self._hint("Ctrl+clic → base Z"))
            root.addWidget(self._hint("Arrastrar → define altura"))
            root.addWidget(self._hint("T → cambiar modo"))

            self._slice_mode_lbl = QLabel("Modo: Entre dos planos")
            self._slice_mode_lbl.setStyleSheet(f"color:{ACCENT_STRONG}; font-size:10.5px; font-weight:600; margin-top:4px;")
            root.addWidget(self._slice_mode_lbl)

            mode_btn = QPushButton("T  →  Cambiar modo")
            mode_btn.setStyleSheet(_BTN_SS)
            mode_btn.clicked.connect(self._cycle_slice_mode)
            root.addWidget(mode_btn)
            self._slice_mode_idx = 0

        # ── Relleno ───────────────────────────────────────────────────────────
        elif name == "Relleno":
            self._fs = _SliderRow("Paso", "m", 1, 200, 5, 0.1)
            self._fs.value_changed.connect(self.flood_step_changed)
            root.addWidget(self._fs)

            self._fz = _SliderRow("Tol. Z", "m", 0, 200, 0, 0.1)
            self._fz.value_changed.connect(self.flood_z_tol_changed)
            root.addWidget(self._fz)

            maxp_row = QHBoxLayout()
            maxp_row.setContentsMargins(0,0,0,0)
            maxp_lbl = QLabel("Máx. puntos"); maxp_lbl.setStyleSheet(f"color:{TEXT_DIM}; font-size:10.5px;")
            self._maxp = QSpinBox()
            self._maxp.setRange(1000, 5_000_000)
            self._maxp.setValue(500_000)
            self._maxp.setSingleStep(50_000)
            self._maxp.setStyleSheet(
                f"QSpinBox{{background:{SURFACE};border:1px solid {BORDER};border-radius:3px;"
                f"color:{ACCENT_STRONG};font-size:10.5px;font-weight:600;padding:2px;}}")
            self._maxp.valueChanged.connect(self.flood_max_pts_changed)
            maxp_row.addWidget(maxp_lbl); maxp_row.addWidget(self._maxp)
            root.addLayout(maxp_row)
            root.addWidget(self._hint("Ctrl+clic → crece desde semilla"))
            root.addWidget(self._hint("Tol.Z=0 → sin límite de altura"))

        # ── Disco ─────────────────────────────────────────────────────────────
        elif name == "Disco":
            root.addWidget(self._hint("Ctrl+clic → coloca el disco"))
            root.addWidget(self._hint("Se orienta según la superficie local", OK))

        # ── Pick ──────────────────────────────────────────────────────────────
        elif name == "Pick":
            self._pick_cls = self._info_row("Clase", "—")
            self._pick_e   = self._info_row("E",     "—")
            self._pick_n   = self._info_row("N",     "—")
            self._pick_z   = self._info_row("Z",     "—")
            for w in [self._pick_cls, self._pick_e, self._pick_n, self._pick_z]:
                root.addWidget(w)
            root.addWidget(self._hint("Ctrl+clic para inspeccionar"))

        # ── Medir ─────────────────────────────────────────────────────────────
        elif name == "Medir":
            self._m3d = self._info_row("Dist. 3D", "—")
            self._m2d = self._info_row("Dist. 2D", "—")
            self._mz  = self._info_row("dZ",       "—")
            for w in [self._m3d, self._m2d, self._mz]:
                root.addWidget(w)
            root.addWidget(self._hint("Clic A → clic B"))
            root.addWidget(self._hint("Escape → reiniciar"))

        root.addStretch()

    # ── Helpers de layout ────────────────────────────────────────────────────

    def _hint(self, text: str, color: str = None) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color:{color or TEXT_MUTE}; font-size:10.5px;"
                          f"{'font-weight:600;' if color else ''}")
        lbl.setWordWrap(True)
        return lbl

    def _info_row(self, key: str, val: str) -> QWidget:
        w   = QWidget(); w.setStyleSheet("background:transparent;")
        lay = QHBoxLayout(w); lay.setContentsMargins(0, 0, 0, 2)
        kl  = QLabel(key); kl.setStyleSheet(f"color:{TEXT_MUTE}; font-size:10.5px;")
        vl  = QLabel(val); vl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        vl.setStyleSheet(f"color:{ACCENT_STRONG}; font-size:10.5px; font-weight:600;")
        vl.setObjectName(f"val_{key.replace(' ', '_')}")
        lay.addWidget(kl); lay.addWidget(vl)
        return w

    def _set_val(self, widget: QWidget, text: str) -> None:
        for lbl in widget.findChildren(QLabel):
            if lbl.objectName().startswith("val_"):
                lbl.setText(text); return

    def _cycle_slice_mode(self):
        modes = [("between", "Entre dos planos"), ("above", "Encima del plano"),
                 ("below", "Debajo del plano")]
        self._slice_mode_idx = (self._slice_mode_idx + 1) % 3
        m_id, m_label = modes[self._slice_mode_idx]
        try:
            self._slice_mode_lbl.setText(f"Modo: {m_label}")
        except Exception:
            pass
        self.slice_mode_changed.emit(m_id)

    # ── API ───────────────────────────────────────────────────────────────────

    def show_pick_result(self, cls_name: str, e: str, n: str, z: str) -> None:
        try:
            self._set_val(self._pick_cls, cls_name)
            self._set_val(self._pick_e, e)
            self._set_val(self._pick_n, n)
            self._set_val(self._pick_z, z)
        except Exception: pass

    def show_measure(self, d3d: float, dh: float, dz: float) -> None:
        try:
            self._set_val(self._m3d, f"{d3d:.3f} m")
            self._set_val(self._m2d, f"{dh:.3f} m")
            self._set_val(self._mz,  f"{dz:.3f} m")
        except Exception: pass


# ─────────────────────────────────────────────────────────────────────────────
# ToolPanel principal
# ─────────────────────────────────────────────────────────────────────────────

class ToolPanel(QWidget):
    """
    Panel derecho con selector de herramientas, opciones globales,
    contexto dinámico y visualización.

    La gestión de capas de referencia YA NO vive aquí (duplicaba a
    OverlayPanel) — ver "Capas de referencia" en el riel de main_window.py.
    """
    # Señales de herramienta
    tool_changed            = pyqtSignal(str)
    # Señales de parámetros (reenviadas desde _ContextSection)
    brush_radius_changed    = pyqtSignal(float)
    brush_overlap_changed   = pyqtSignal(float)
    brush_thickness_changed = pyqtSignal(float)
    rg_dist_changed      = pyqtSignal(float)
    rg_tol_z_changed     = pyqtSignal(float)
    rg_tol_rgb_changed   = pyqtSignal(float)
    rg_use_rgb_changed   = pyqtSignal(bool)
    rg_use_z_changed     = pyqtSignal(bool)
    radius_changed          = pyqtSignal(float)    # sphere radius
    # Señales de opciones globales
    erase_mode_changed      = pyqtSignal(bool)
    delete_mode_changed     = pyqtSignal(bool)
    only_unlabeled_changed  = pyqtSignal(bool)
    # Señales de herramientas específicas
    flood_step_changed      = pyqtSignal(float)
    flood_max_pts_changed   = pyqtSignal(int)
    flood_z_tol_changed     = pyqtSignal(float)
    slice_mode_changed      = pyqtSignal(str)
    lasso_close_requested   = pyqtSignal()
    # Señales de visualización
    color_mode_changed      = pyqtSignal(str)
    point_size_changed      = pyqtSignal(float)
    show_unlabeled_toggled  = pyqtSignal(bool)
    grid_toggled            = pyqtSignal(bool)
    edl_toggled              = pyqtSignal(bool)
    # Exportar
    export_requested        = pyqtSignal()

    # Señales legacy para compatibilidad con main_window actual
    box_z_changed           = pyqtSignal(float)
    box_mode_changed        = pyqtSignal(str)
    box_confirm_requested   = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(180)
        self.setMaximumWidth(400)
        self._btns:       Dict[str, _ToolButton] = {}
        self._active_name = ALL_TOOLS[0].name
        self._build_ui()

    # ── Construcción UI ───────────────────────────────────────────────────────

    def _card(self, title: str, icon_name: str = None):
        """
        Tarjeta agrupada estilo panel de propiedades de QGIS: borde fino,
        esquinas redondeadas, encabezado con icono. Usa un selector con
        objectName (no un stylesheet "pelado") para que el fondo/borde NO
        se filtre a los widgets hijos — es un gotcha real de Qt: un
        `setStyleSheet("background:...;border:...")` sin selector aplica
        esas propiedades a todos los descendientes que no las overridee.

        Retorna (frame, content_layout) — añadir directamente al layout.
        """
        frame = QFrame()
        frame.setObjectName("card")
        frame.setStyleSheet(
            f"QFrame#card{{background:{SURFACE};border:1px solid {BORDER};border-radius:4px;}}")
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(12, 10, 12, 12)
        outer.setSpacing(8)

        if title:
            hdr_row = QHBoxLayout(); hdr_row.setSpacing(6)
            if icon_name:
                ic = QLabel(); ic.setStyleSheet("background:transparent;border:none;")
                ic.setPixmap(qpixmap(icon_name, TEXT_MUTE, 13))
                hdr_row.addWidget(ic)
            hdr_row.addWidget(self._section_lbl(title))
            hdr_row.addStretch()
            outer.addLayout(hdr_row)

        content = QVBoxLayout()
        content.setSpacing(6)
        outer.addLayout(content)
        return frame, content

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ── Barra de herramientas (estilo QGIS/Cyclone 3DR) ───────────────────
        # Tira de iconos compacta, varias filas si hace falta — el nombre y
        # el atajo van en el tooltip, no como texto fijo bajo cada icono.
        toolbar_frame = QFrame()
        toolbar_frame.setObjectName("toolbar")
        toolbar_frame.setStyleSheet(
            f"QFrame#toolbar{{background:{SURFACE_2};border:1px solid {BORDER};border-radius:4px;}}")
        tb_outer = QVBoxLayout(toolbar_frame)
        tb_outer.setContentsMargins(8, 6, 8, 8)
        tb_outer.setSpacing(4)
        # "Ctrl+clic" ya no va en el título — cada botón lo explica en su
        # tooltip; en paneles angostos el título completo forzaba un salto
        # de línea a mitad de palabra ("Ctrl+" / "clic").
        tb_outer.addWidget(self._section_lbl("HERRAMIENTAS"))

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        ncols = 7
        for i, tc in enumerate(ALL_TOOLS):
            btn = _ToolButton(tc)
            btn.clicked.connect(self._on_tool_clicked)
            self._btns[tc.name] = btn
            grid.addWidget(btn, i // ncols, i % ncols, Qt.AlignLeft)
        grid_host = QHBoxLayout(); grid_host.addLayout(grid); grid_host.addStretch()
        tb_outer.addLayout(grid_host)
        root.addWidget(toolbar_frame)

        # ── Modo de anotación ─────────────────────────────────────────────────
        opt_card, opt_l = self._card("ANOTACIÓN", "sliders2")

        # Borrar
        self._erase_btn = QPushButton("  Modo borrar   [E]")
        self._erase_btn.setIcon(qicon("eraser", WARN))
        self._erase_btn.setCheckable(True)
        self._erase_btn.setStyleSheet(
            f"QPushButton{{background:{SURFACE};border:1px solid {BORDER};"
            f"border-radius:3px;color:{TEXT_DIM};font-size:10.5px;font-weight:600;padding:7px;text-align:left;}}"
            f"QPushButton:hover{{border-color:{WARN};color:{WARN};background:{WARN_SOFT};}}"
            f"QPushButton:checked{{background:{WARN_SOFT};border-color:{WARN};"
            f"color:{WARN};font-weight:700;}}")
        self._erase_btn.toggled.connect(self.erase_mode_changed)
        self._erase_btn.toggled.connect(
            lambda v: self._delete_btn.setChecked(False) if v else None)
        opt_l.addWidget(self._erase_btn)

        # Eliminar puntos — distinto de "Modo borrar": borrar solo quita la
        # clase (el punto sigue en la nube, sin etiquetar); esto saca el
        # punto de la nube por completo (para limpiar ruido). Se puede
        # deshacer con Ctrl+Z igual que cualquier anotación — se avisa en
        # el tooltip para que no dé miedo de más usarlo.
        self._delete_btn = QPushButton("  Eliminar puntos   [Supr]")
        self._delete_btn.setIcon(qicon("trash", WARN))
        self._delete_btn.setCheckable(True)
        self._delete_btn.setToolTip(
            "Con esta opción activa, cualquier herramienta de selección\n"
            "ELIMINA los puntos seleccionados de la nube (no solo su clase)\n"
            "— útil para limpiar ruido. Se puede deshacer con Ctrl+Z.")
        self._delete_btn.setStyleSheet(
            f"QPushButton{{background:{SURFACE};border:1px solid {BORDER};"
            f"border-radius:3px;color:{TEXT_DIM};font-size:10.5px;font-weight:600;padding:7px;text-align:left;}}"
            f"QPushButton:hover{{border-color:{WARN};color:{WARN};background:{WARN_SOFT};}}"
            f"QPushButton:checked{{background:{WARN};border-color:{WARN};"
            f"color:#ffffff;font-weight:700;}}")
        self._delete_btn.toggled.connect(self.delete_mode_changed)
        self._delete_btn.toggled.connect(
            lambda v: self._erase_btn.setChecked(False) if v else None)
        opt_l.addWidget(self._delete_btn)

        # Solo sin etiquetar
        unl_row = QWidget(); unl_row.setStyleSheet("background:transparent;")
        unl_l   = QHBoxLayout(unl_row); unl_l.setContentsMargins(0,0,0,0)
        unl_lbl = QLabel("Solo sin etiquetar")
        unl_lbl.setStyleSheet(f"color:{TEXT_DIM}; font-size:10.5px; background:transparent;")
        self._only_unl = _Toggle(False)
        self._only_unl.toggled.connect(self.only_unlabeled_changed)
        unl_l.addWidget(unl_lbl); unl_l.addStretch(); unl_l.addWidget(self._only_unl)
        opt_l.addWidget(unl_row)
        root.addWidget(opt_card)

        # ── Contexto dinámico ─────────────────────────────────────────────────
        ctx_card, ctx_l = self._card(None)
        self._ctx = _ContextSection()
        self._ctx.brush_radius_changed.connect(self.brush_radius_changed)
        self._ctx.brush_overlap_changed.connect(self.brush_overlap_changed)
        self._ctx.brush_thickness_changed.connect(self.brush_thickness_changed)
        self._ctx.sphere_radius_changed.connect(self.radius_changed)
        self._ctx.flood_step_changed.connect(self.flood_step_changed)
        self._ctx.flood_max_pts_changed.connect(self.flood_max_pts_changed)
        self._ctx.flood_z_tol_changed.connect(self.flood_z_tol_changed)
        self._ctx.slice_mode_changed.connect(self.slice_mode_changed)
        self._ctx.lasso_close_requested.connect(self.lasso_close_requested)
        ctx_l.addWidget(self._ctx)
        root.addWidget(ctx_card, 1)

        # ── Visualización ─────────────────────────────────────────────────────
        vis_card, vis_l = self._card("VISUALIZACIÓN", "palette")

        self._pt_size = _SliderRow("Tamaño punto", "px", 5, 80, 14, 0.1)
        self._pt_size.value_changed.connect(self.point_size_changed)
        vis_l.addWidget(self._pt_size)

        for lbl_text, attr, default, sig in [
            ("Ver sin etiquetar", "_tog_unl",  True,  self.show_unlabeled_toggled),
            ("Grilla de fondo",   "_tog_grid", False, self.grid_toggled),
            ("Eye-Dome Lighting", "_tog_edl",  False, self.edl_toggled),
        ]:
            row = QWidget(); row.setStyleSheet("background:transparent;")
            rl  = QHBoxLayout(row); rl.setContentsMargins(0, 2, 0, 2)
            ll = QLabel(lbl_text); ll.setStyleSheet(f"color:{TEXT_DIM}; font-size:10.5px;")
            tog = _Toggle(default); tog.toggled.connect(sig)
            if attr == "_tog_edl":
                # EDL es un término técnico (sombreado por profundidad) —
                # explicar qué hace, igual que CloudCompare/Potree lo
                # describen al usuario.
                tip = ("Sombreado por profundidad: resalta el relieve/los "
                       "bordes de la nube sin necesitar normales — más "
                       "fácil de leer la geometría en modo Elevación o "
                       "Color único. Tiene un costo de rendimiento leve.")
                ll.setToolTip(tip); tog.setToolTip(tip)
            setattr(self, attr, tog)
            rl.addWidget(ll); rl.addStretch(); rl.addWidget(tog)
            vis_l.addWidget(row)

        cc_row = QWidget(); cc_row.setStyleSheet("background:transparent;")
        cc_l   = QHBoxLayout(cc_row); cc_l.setContentsMargins(0, 6, 0, 0)
        cc_lbl = QLabel("Color"); cc_lbl.setStyleSheet(f"color:{TEXT_DIM}; font-size:10.5px;")
        self._color_combo = QComboBox()
        self._color_combo.addItems([
            "Anotación", "Elevación", "RGB",
            "Intensidad", "Clasificación", "Confianza", "Color único",
        ])
        self._color_combo.setStyleSheet(
            f"QComboBox{{background:{SURFACE};border:1px solid {BORDER};border-radius:3px;"
            f"color:{TEXT_DIM};font-size:10.5px;padding:3px 6px;}}"
            f"QComboBox::drop-down{{width:18px;}}"
            f"QComboBox QAbstractItemView{{background:{SURFACE};color:{TEXT};"
            f"selection-background-color:{ACCENT_SOFT};selection-color:{ACCENT_STRONG};}}")
        self._color_combo.currentTextChanged.connect(self.color_mode_changed)
        cc_l.addWidget(cc_lbl); cc_l.addWidget(self._color_combo, 1)
        vis_l.addWidget(cc_row)
        root.addWidget(vis_card)

        # Activar primera herramienta
        first = ALL_TOOLS[0].name
        if first in self._btns:
            self._btns[first].set_active(True)
        self._ctx.update_for_tool(first)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _add_sep(self, layout) -> None:
        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"background:{BORDER_SOFT}; max-height:1px;")
        layout.addWidget(sep)

    def _section_lbl(self, text: str) -> QLabel:
        lbl = QLabel(text)
        # word-wrap como red de seguridad: antes este texto se recortaba
        # silenciosamente ("Ctrl+clic" quedaba en "Ctrl+cli") en paneles
        # angostos (~180-210px) porque QLabel no envuelve por defecto.
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color:{TEXT_MUTE}; font-size:10.5px; font-weight:700; letter-spacing:0.5px;")
        return lbl

    # ── Eventos ───────────────────────────────────────────────────────────────

    def _on_tool_clicked(self, name: str) -> None:
        if name == self._active_name: return
        if self._active_name in self._btns:
            self._btns[self._active_name].set_active(False)
        self._active_name = name
        if name in self._btns:
            self._btns[name].set_active(True)
        self._ctx.update_for_tool(name)
        self.tool_changed.emit(name)

    # ── API pública ───────────────────────────────────────────────────────────

    def set_active_tool(self, name: str) -> None:
        self._on_tool_clicked(name)

    def set_erase_mode(self, v: bool) -> None:
        self._erase_btn.setChecked(v)

    def set_delete_mode(self, v: bool) -> None:
        self._delete_btn.setChecked(v)

    def set_cloud(self, pc) -> None:
        pass

    def set_color_mode(self, mode: str) -> None:
        """
        Sincroniza el combo de modo de color con el modo REALMENTE activo en
        el canvas, sin re-emitir color_mode_changed (evita eco/recursión).
        """
        idx = self._color_combo.findText(mode)
        if idx < 0 or idx == self._color_combo.currentIndex():
            return
        self._color_combo.blockSignals(True)
        self._color_combo.setCurrentIndex(idx)
        self._color_combo.blockSignals(False)

    def update_export_note(self, n: int) -> None:
        if hasattr(self,'_export_note'): self._export_note.setText(f"{n:,} pts etiquetados")

    def show_pick_result(self, cls_name: str, cls_color: str,
                         e: str, n: str, z: str) -> None:
        self._ctx.show_pick_result(cls_name, e, n, z)

    def show_measure_result(self, d3d: float, dh: float, dz: float) -> None:
        self._ctx.show_measure(d3d, dh, dz)

    def show_measure_waiting(self, has_a: bool) -> None:
        pass

    def get_box_z_extend(self) -> float:
        return 5.0
