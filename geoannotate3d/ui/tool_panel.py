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
    QSpinBox, QCheckBox, QScrollArea,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QFontMetrics

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
    "Ajustar plano":  "bricks",
    "Polígono":       "pentagon",
    "Caja":           "bounding-box",
    "Esfera":         "globe2",
    "Corte Z":        "layers-half",
    "Relleno":        "paint-bucket",
    "Pick":           "eyedropper",
    "Medir":          "rulers",
    "Etiqueta 3D":    "tag",
    "Polilínea":      "diagram-3",
    "Perfil":         "graph-up",
}


class _ToolButton(QWidget):
    """
    Botón de herramienta — icono + NOMBRE visible siempre debajo.

    REDISEÑO (2026-09-09): la versión anterior (icono solo, 34×34px, el
    nombre solo en el tooltip) recibió la queja explícita de que "no se
    sabe cuál herramienta es cuál" sin pasar el mouse sobre cada una y
    esperar. Con 11 herramientas ya no alcanza con memorizar posiciones
    en una barra — cada botón ahora es una tarjeta más grande con el
    nombre siempre a la vista (envuelto a 2 líneas si hace falta) y un
    icono más grande, así se puede escanear la grilla de un vistazo en
    vez de tener que recordar qué icono es cuál herramienta.
    """
    clicked = pyqtSignal(str)
    # Tamaño de referencia (un poco más chico que la versión anterior,
    # 74×80, a pedido explícito del usuario) — el tamaño REAL en pantalla
    # lo decide ToolPanel._relayout_tool_grid() dentro de [MIN_W, MAX_W],
    # igual de responsivo que TileGridWidget con el tamaño de celda: un
    # panel angosto encoge las tarjetas (nunca las saca de vista), uno
    # ancho las agranda un poco en vez de dejar hueco vacío.
    W, H = 64, 70
    # MIN_W=58 (no 52): a un ancho menor, ni el tamaño de fuente más
    # chico permitido (6px) alcanza a acomodar los nombres más largos
    # ("Etiqueta 3D", "Polígono") sin desbordar un par de píxeles —
    # confirmado con test. 58 deja margen de sobra en cualquier caso real.
    MIN_W, MAX_W = 58, 84
    _ASPECT = H / W

    def __init__(self, tool_cls: type, parent=None):
        super().__init__(parent)
        self._name    = tool_cls.name
        self._key     = tool_cls.key
        self._icon    = _TOOL_ICON.get(self._name, "grid-3x3")
        self._tooltip = getattr(tool_cls, "tooltip", "")
        self._active  = False
        self._font_px = 8
        self._icon_px = 22
        self.setCursor(Qt.PointingHandCursor)
        key_hint = f"  [{self._key}]" if self._key else ""
        self.setToolTip(f"{tool_cls.name}{key_hint}\n{self._tooltip}")
        self.setAttribute(Qt.WA_StyledBackground, True)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 5, 2, 5)
        lay.setSpacing(2)
        self._icon_lbl = QLabel()
        self._icon_lbl.setAlignment(Qt.AlignCenter)
        # Se neutraliza border/radio explícitamente: un stylesheet "pelado"
        # (sin selector) en el padre se filtra a los hijos en Qt — sin esto
        # el icono heredaría el borde redondeado del botón por debajo.
        self._icon_lbl.setStyleSheet("background:transparent;border:none;")
        lay.addWidget(self._icon_lbl)

        self._name_lbl = QLabel(self._name)
        self._name_lbl.setAlignment(Qt.AlignCenter)
        self._name_lbl.setWordWrap(True)
        lay.addWidget(self._name_lbl)

        self.set_size(self.W, self.H)

    def set_size(self, w: int, h: int) -> None:
        """
        Redimensiona la tarjeta completa — icono y nombre se reescalan
        con ella (no solo el marco exterior). Llamado por
        ToolPanel._relayout_tool_grid() en cada resize del panel.
        """
        w = max(self.MIN_W, min(self.MAX_W, int(w)))
        h = max(1, int(h))
        if (w, h) == (self.width(), self.height()) and self.width() > 0:
            return
        self.setFixedSize(w, h)
        self._icon_px = max(15, min(26, int(w * 0.33)))
        self._icon_lbl.setFixedHeight(self._icon_px + 4)
        self._name_lbl.setFixedHeight(max(20, int(h * 0.42)))
        # Tamaño de fuente elegido para que la palabra más larga del
        # nombre ("Growing", "Etiqueta"...) quepa de verdad en el ancho
        # ACTUAL — mismo principio que el fix del grid de tiles: sin
        # esto, nombres largos como "Region Growing" se recortaban a
        # mitad de palabra ("Growin▐") en vez de ajustarse, y además
        # tarjetas más chicas necesitan una fuente más chica todavía.
        self._font_px = self._fit_font_size(self._name, w - 6)
        self._update_style()

    @staticmethod
    def _fit_font_size(text: str, max_w: int) -> int:
        """
        Tamaño de fuente ENTERO (px) más grande, entre 9 y 5, tal que la
        palabra más ancha de `text` quepa en max_w — evita que nombres
        largos ("Growing", "Etiqueta") se recorten a mitad de palabra.

        BUG REAL ENCONTRADO (confirmado con captura): esto antes probaba
        tamaños con medio píxel (8.5, 7.5...) validándolos con
        `QFont.setPixelSize(int(size))` (TRUNCA — 7.5→7px para medir),
        pero el CSS aplicado de verdad era `font-size:7.5px` — que Qt
        REDONDEA (7.5→8px) al renderizar. Medía a 7px, pintaba a 8px: un
        texto que "cabía" en la medición se desbordaba en pantalla
        ("Etiqueta" quedaba como "▐tiquet▐"). Fix: solo tamaños ENTEROS,
        para que medir y aplicar sean exactamente lo mismo.

        Rango extendido a 5px (antes 6 era el mínimo) al agregar la
        herramienta "Polilínea": es una sola palabra de 9 caracteres (no
        se puede partir en 2 líneas como "Region Growing") que a 6px
        (54px) no entra en el ancho mínimo de tarjeta (MIN_W=58 → max_w=52)
        por apenas 2px — confirmado con el arnés de tests, no a ojo.
        """
        words = text.split(" ") or [text]
        for size in (9, 8, 7, 6, 5):
            f = QFont(); f.setPixelSize(size); f.setBold(True)
            fm = QFontMetrics(f)
            if all(fm.horizontalAdvance(w) <= max_w for w in words):
                return size
        return 5

    def set_active(self, a: bool):
        self._active = a; self._update_style()

    def _update_style(self):
        color = ACCENT_STRONG if self._active else TEXT_DIM
        self._icon_lbl.setPixmap(qpixmap(self._icon, color, self._icon_px))
        self._name_lbl.setStyleSheet(
            f"background:transparent;border:none;"
            f"font-size:{self._font_px}px;font-weight:600;color:{color};")
        if self._active:
            self.setStyleSheet(
                f"background:{ACCENT_SOFT};border:1.5px solid {ACCENT};border-radius:4px;")
        else:
            self.setStyleSheet(
                f"background:{SURFACE};border:1px solid {BORDER};border-radius:4px;")

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self._name)

    def enterEvent(self, e):
        if not self._active:
            self.setStyleSheet(
                f"background:{SURFACE_2};border:1px solid {ACCENT_BORDER};border-radius:4px;")

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
    plane_radius_changed     = pyqtSignal(float)
    plane_threshold_changed  = pyqtSignal(float)
    label_font_size_changed  = pyqtSignal(float)
    label_color_changed      = pyqtSignal(tuple)
    profile_buffer_changed   = pyqtSignal(float)
    polyline_width_changed   = pyqtSignal(float)
    polyline_color_changed   = pyqtSignal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current = ""

    def _hide_and_delete_layout_contents(self, lay) -> None:
        """
        Vacía `lay` recursivamente y esconde/marca-para-borrar cada widget
        que encuentra — en TODOS los niveles, no solo los widgets
        agregados directo con `root.addWidget(...)`.

        BUG REAL reportado por el usuario: al cambiar de "Etiqueta 3D" o
        "Polilínea" a otra herramienta, la fila de color (el QLabel
        "Color de la próxima..." + el botón de color) se quedaba
        pintada encima del panel de la herramienta nueva. Causa: esta
        fila se agrega con `root.addLayout(color_row)`, no
        `root.addWidget(...)` — el bucle de limpieza de antes solo
        llamaba `item.widget()` (que da None para un item que es un
        LAYOUT anidado, no un widget) así que nunca les tocaba el turno
        de `hide()`/`deleteLater()`; quedaban huérfanos de layout pero
        seguían siendo hijos visibles de _ContextSection para siempre.
        Ahora se revisa `item.layout()` también, y si hay un layout
        anidado se recorre con la misma lógica antes de seguir.
        """
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None:
                # hide() inmediato — deleteLater() es diferido (espera al
                # próximo ciclo del event loop), así que sin esto el
                # widget viejo puede seguir PINTÁNDOSE encima del nuevo
                # contenido durante esa ventana, produciendo texto
                # superpuesto/fantasma (confirmado con captura real al
                # cambiar de "Pincel" a "Ajustar plano").
                w.hide()
                w.deleteLater()
                continue
            sub = item.layout()
            if sub is not None:
                self._hide_and_delete_layout_contents(sub)

    def update_for_tool(self, name: str) -> None:
        if name == self._current: return
        self._current = name
        lay = self.layout()
        if lay:
            self._hide_and_delete_layout_contents(lay)
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

        # ── Ajustar plano (RANSAC) ────────────────────────────────────────────
        elif name == "Ajustar plano":
            root.addWidget(self._hint(
                "Ctrl+clic → ajusta el plano dentro del radio y pinta "
                "solo lo plano (círculo naranja = radio)", OK))
            self._pr = _SliderRow("Radio", "m", 1, 100, int(5))
            self._pr.value_changed.connect(self.plane_radius_changed)
            root.addWidget(self._pr)
            self._pth = _SliderRow("Tolerancia", "cm", 1, 100, 8)
            self._pth.value_changed.connect(lambda v: self.plane_threshold_changed.emit(v / 100.0))
            root.addWidget(self._pth)
            root.addWidget(self._hint("Tolerancia alta = acepta superficies más rugosas"))

        # ── Etiqueta 3D ───────────────────────────────────────────────────────
        elif name == "Etiqueta 3D":
            root.addWidget(self._hint("Ctrl+clic → coloca una etiqueta"))
            root.addWidget(self._hint(
                "Ctrl+clic sobre una ya puesta → editarla o eliminarla", OK))
            self._lfs = _SliderRow("Tamaño de letra", "pt", 8, 48, 14)
            self._lfs.value_changed.connect(self.label_font_size_changed)
            root.addWidget(self._lfs)

            color_row = QHBoxLayout()
            color_lbl = QLabel("Color de la próxima etiqueta")
            color_lbl.setStyleSheet(f"color:{TEXT_DIM}; font-size:10.5px;")
            color_row.addWidget(color_lbl)
            self._label_color = (1.0, 0.85, 0.2)
            self._label_color_btn = QPushButton()
            self._label_color_btn.setFixedSize(46, 20)
            self._update_label_color_btn()
            self._label_color_btn.clicked.connect(self._pick_label_color)
            color_row.addWidget(self._label_color_btn)
            color_row.addStretch()
            root.addLayout(color_row)

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
            root.addWidget(self._hint("Clic A → clic B mide"))
            root.addWidget(self._hint("Ctrl+clic sobre una medida → editarla o eliminarla", OK))
            root.addWidget(self._hint("Escape → cancelar el punto A"))

        # ── Polilínea ─────────────────────────────────────────────────────────
        elif name == "Polilínea":
            root.addWidget(self._hint("Ctrl+clic → agrega un vértice · Enter la guarda", OK))
            root.addWidget(self._hint("Backspace quita el último vértice · Escape cancela"))
            root.addWidget(self._hint("Ctrl+clic sobre una ya puesta → editarla o eliminarla", OK))

            self._plw = _SliderRow("Grosor de línea", "px", 1, 12, 3, 0.5)
            self._plw.value_changed.connect(self.polyline_width_changed)
            root.addWidget(self._plw)

            color_row = QHBoxLayout()
            color_lbl = QLabel("Color de la próxima polilínea")
            color_lbl.setStyleSheet(f"color:{TEXT_DIM}; font-size:10.5px;")
            color_row.addWidget(color_lbl)
            self._polyline_color = (0.2, 0.85, 1.0)
            self._polyline_color_btn = QPushButton()
            self._polyline_color_btn.setFixedSize(46, 20)
            self._update_polyline_color_btn()
            self._polyline_color_btn.clicked.connect(self._pick_polyline_color)
            color_row.addWidget(self._polyline_color_btn)
            color_row.addStretch()
            root.addLayout(color_row)

        # ── Perfil (corte vertical) ──────────────────────────────────────────
        elif name == "Perfil":
            root.addWidget(self._hint("Ctrl+clic A · clic B → corte vertical de la franja", OK))
            self._pfb = _SliderRow("Ancho de franja", "m", 1, 100, int(2))
            self._pfb.value_changed.connect(self.profile_buffer_changed)
            root.addWidget(self._pfb)
            root.addWidget(self._hint("Escape → cancelar el punto A"))

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

    def _update_label_color_btn(self) -> None:
        r, g, b = (int(max(0, min(1, c)) * 255) for c in self._label_color)
        self._label_color_btn.setStyleSheet(
            f"background: rgb({r},{g},{b}); border: 1px solid {BORDER}; border-radius: 3px;")

    def _pick_label_color(self) -> None:
        from PyQt5.QtWidgets import QColorDialog
        r, g, b = (int(max(0, min(1, c)) * 255) for c in self._label_color)
        picked = QColorDialog.getColor(QColor(r, g, b), self, "Color de la etiqueta")
        if picked.isValid():
            self._label_color = (picked.redF(), picked.greenF(), picked.blueF())
            self._update_label_color_btn()
            self.label_color_changed.emit(self._label_color)

    def _update_polyline_color_btn(self) -> None:
        r, g, b = (int(max(0, min(1, c)) * 255) for c in self._polyline_color)
        self._polyline_color_btn.setStyleSheet(
            f"background: rgb({r},{g},{b}); border: 1px solid {BORDER}; border-radius: 3px;")

    def _pick_polyline_color(self) -> None:
        from PyQt5.QtWidgets import QColorDialog
        r, g, b = (int(max(0, min(1, c)) * 255) for c in self._polyline_color)
        picked = QColorDialog.getColor(QColor(r, g, b), self, "Color de la polilínea")
        if picked.isValid():
            self._polyline_color = (picked.redF(), picked.greenF(), picked.blueF())
            self._update_polyline_color_btn()
            self.polyline_color_changed.emit(self._polyline_color)

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
    plane_radius_changed    = pyqtSignal(float)
    plane_threshold_changed = pyqtSignal(float)
    label_font_size_changed = pyqtSignal(float)
    label_color_changed     = pyqtSignal(tuple)
    profile_buffer_changed  = pyqtSignal(float)
    polyline_width_changed  = pyqtSignal(float)
    polyline_color_changed  = pyqtSignal(tuple)
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
    clip_box_toggled         = pyqtSignal(bool)
    # Exportar
    export_requested        = pyqtSignal()

    # Señales legacy para compatibilidad con main_window actual
    box_z_changed           = pyqtSignal(float)
    box_mode_changed        = pyqtSignal(str)
    box_confirm_requested   = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(180)
        # Antes 400 — el usuario pidió poder ensancharlo más de lo que
        # dejaba. La grilla de herramientas ahora es responsiva en tamaño
        # Y columnas (ver _relayout_tool_grid), así que un panel más ancho
        # se aprovecha bien en vez de dejar espacio vacío.
        self.setMaximumWidth(650)
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
        # BUG REAL REPORTADO (2026-09-09): sin scroll, el panel entero
        # dependía de que TODAS las tarjetas (Herramientas + Anotación +
        # Contexto dinámico + Visualización) cupieran en la altura fija
        # del dock. Con una herramienta SIN parámetros (ej. Pick) cabía
        # bien: pero con una que sí tiene varios controles (Pincel: 3
        # sliders + 2 hints; Ajustar plano: 2 sliders + 3 hints) el panel
        # se quedaba sin alto y Qt comprimía/amontonaba todo — el layout
        # se veía "roto" literalmente porque no tenía a dónde crecer. Fix:
        # todo el contenido vive en un widget interno dentro de un
        # QScrollArea — si una herramienta necesita más espacio que el
        # que hay disponible, el panel simplemente da scroll en vez de
        # aplastar el contenido.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet(
            "QScrollArea{background:transparent;border:none;}"
            f"QScrollBar:vertical{{background:{SURFACE_2};width:8px;border:none;margin:0;}}"
            f"QScrollBar::handle:vertical{{background:{BORDER};border-radius:4px;min-height:24px;}}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}"
            "QScrollBar::add-page:vertical,QScrollBar::sub-page:vertical{background:none;}")

        content = QWidget()
        content.setStyleSheet("background:transparent;")
        root = QVBoxLayout(content)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ── Barra de herramientas (estilo QGIS/Cyclone 3DR) ───────────────────
        # Tira de iconos compacta, varias filas si hace falta — el nombre y
        # el atajo van en el tooltip, no como texto fijo bajo cada icono.
        #
        # REDISEÑO (2026-09-09) — grilla RESPONSIVA en vez de un número fijo
        # de columnas: con `ncols` fijo, un panel angosto podía dejar
        # herramientas recortadas/fuera de vista por la derecha, y uno
        # ancho dejaba una franja de espacio vacío en vez de aprovecharla
        # para acomodar más columnas (y así menos filas — más bajo de
        # alto). Ahora `_relayout_tool_grid()` recalcula cuántas columnas
        # caben de verdad según el ancho disponible cada vez que el panel
        # cambia de tamaño (ver resizeEvent), y reacomoda los mismos
        # botones — nunca se crean ni se destruyen widgets, solo se
        # mueven de celda.
        self._toolbar_frame = QFrame()
        self._toolbar_frame.setObjectName("toolbar")
        self._toolbar_frame.setStyleSheet(
            f"QFrame#toolbar{{background:{SURFACE_2};border:1px solid {BORDER};border-radius:4px;}}")
        tb_outer = QVBoxLayout(self._toolbar_frame)
        tb_outer.setContentsMargins(8, 6, 8, 8)
        tb_outer.setSpacing(4)
        # "Ctrl+clic" ya no va en el título — cada botón lo explica en su
        # tooltip; en paneles angostos el título completo forzaba un salto
        # de línea a mitad de palabra ("Ctrl+" / "clic").
        tb_outer.addWidget(self._section_lbl("HERRAMIENTAS"))

        self._tool_grid = QGridLayout()
        self._tool_grid.setContentsMargins(0, 0, 0, 0)
        self._tool_grid.setSpacing(4)
        self._tool_btn_order: list = []
        for tc in ALL_TOOLS:
            btn = _ToolButton(tc)
            btn.clicked.connect(self._on_tool_clicked)
            self._btns[tc.name] = btn
            self._tool_btn_order.append(btn)
        grid_host = QHBoxLayout(); grid_host.addLayout(self._tool_grid); grid_host.addStretch()
        tb_outer.addLayout(grid_host)
        root.addWidget(self._toolbar_frame)
        self._relayout_tool_grid(force=True)

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
        self._ctx.plane_radius_changed.connect(self.plane_radius_changed)
        self._ctx.plane_threshold_changed.connect(self.plane_threshold_changed)
        self._ctx.label_font_size_changed.connect(self.label_font_size_changed)
        self._ctx.label_color_changed.connect(self.label_color_changed)
        self._ctx.profile_buffer_changed.connect(self.profile_buffer_changed)
        self._ctx.polyline_width_changed.connect(self.polyline_width_changed)
        self._ctx.polyline_color_changed.connect(self.polyline_color_changed)
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
            ("Caja de recorte",   "_tog_clipbox", False, self.clip_box_toggled),
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
            elif attr == "_tog_clipbox":
                tip = ("Caja 3D interactiva para aislar un volumen — "
                       "arrastra sus caras (naranja) para recortar la "
                       "vista. No borra nada, solo oculta temporalmente "
                       "lo que queda fuera (igual que en CloudCompare).")
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

        scroll.setWidget(content)
        outer.addWidget(scroll)

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

    # ── Grilla de herramientas responsiva ────────────────────────────────────

    def _relayout_tool_grid(self, force: bool = False) -> None:
        """
        Recalcula CUÁNTAS COLUMNAS caben Y el TAMAÑO de cada tarjeta en el
        ancho actual del panel — mismo espíritu que TileGridWidget
        (core del grid de tiles): ahí la cantidad de filas/columnas la
        pone la nube, pero el tamaño de celda se adapta al ancho
        disponible; aquí es al revés (el número de herramientas es fijo),
        así que se adapta tanto cuántas entran por fila como qué tan
        grande se ve cada una.

        - Panel angosto: primero se buscan más columnas achicando las
          tarjetas hasta MIN_W — nunca se recortan/desaparecen fuera de
          vista por la derecha, se ven más chicas pero completas.
        - Panel ancho: las tarjetas crecen hasta MAX_W para llenar el
          espacio en vez de dejarlo vacío; más allá de eso, se suman
          columnas.
        """
        if not hasattr(self, "_tool_grid") or not self._tool_btn_order:
            return
        # Ancho disponible real: el del propio panel menos márgenes/
        # padding acumulados de root + tarjeta (10+10 del root, 8+8 del
        # frame) y un margen para la barra de scroll vertical.
        avail = max(_ToolButton.MIN_W, self.width() - 44)
        spacing = self._tool_grid.spacing() or 4
        n_tools = len(self._tool_btn_order)

        # 1) Punto de partida: cuántas columnas caben a tamaño de
        # referencia (W=64) — así un resize pequeño solo cambia el
        # TAMAÑO de las tarjetas (suave, como el cell_sz del grid de
        # tiles), y el número de columnas solo salta cuando de verdad
        # hace falta (paso 2).
        ncols = max(1, min(n_tools, round((avail + spacing) / (_ToolButton.W + spacing))))
        # 2) Corrección: si con ese ncols las tarjetas quedarían más
        # chicas que MIN_W, sacar una columna (repartir el mismo ancho
        # entre menos tarjetas → cada una más grande); si quedarían más
        # grandes que MAX_W, agregar una columna (space de sobra para
        # una más, en vez de dejarlo vacío).
        while ncols > 1 and (avail - spacing * (ncols - 1)) / ncols < _ToolButton.MIN_W:
            ncols -= 1
        while ncols < n_tools and (avail - spacing * (ncols - 1)) / ncols > _ToolButton.MAX_W:
            ncols += 1

        btn_w = (avail - spacing * (ncols - 1)) / ncols
        btn_w = int(max(_ToolButton.MIN_W, min(_ToolButton.MAX_W, btn_w)))
        btn_h = int(round(btn_w * _ToolButton._ASPECT))

        state = (ncols, btn_w)
        if not force and state == getattr(self, "_tool_grid_state", None):
            return
        self._tool_grid_state = state
        self._tool_grid_ncols = ncols   # expuesto para tests/depuración
        for i, btn in enumerate(self._tool_btn_order):
            btn.set_size(btn_w, btn_h)
            self._tool_grid.addWidget(btn, i // ncols, i % ncols, Qt.AlignLeft)

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self._relayout_tool_grid()

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
