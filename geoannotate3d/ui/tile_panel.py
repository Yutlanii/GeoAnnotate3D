"""
ui/tile_panel.py — Panel de navegación por tiles v1.1

Cambios vs v1.0:
  - Tile-to-tile: se puede hacer click en cualquier tile sin salir al overview.
    Si se está en tile mode, el click emite tile_selected directamente.
    El botón "← Vista global" sigue siendo la ruta para volver al overview.
  - Layout rediseñado: controles compactos en una sola fila, grid ocupa
    todo el espacio disponible, leyenda integrada en el header.
  - TileGridWidget: celdas con color de fondo visible en ambos modos (dark).
"""
from __future__ import annotations
from typing import Optional

import numpy as np
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QComboBox, QSizePolicy, QFrame, QStackedWidget,
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QRect, QPoint
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QFontMetrics

from ui.icons import icon as qicon, pixmap as qpixmap
from ui.theme import (
    SURFACE, SURFACE_2, SURFACE_3, BORDER, BORDER_SOFT,
    TEXT, TEXT_DIM, TEXT_MUTE,
    ACCENT, ACCENT_STRONG, ACCENT_SOFT, ACCENT_BORDER, OK, OK_SOFT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de color — REDISEÑO 2026-09-05: antes usaban QColor(r,g,b) con
# enteros hardcodeados del tema oscuro viejo. La sustitución automática de
# colores de la sesión anterior solo buscaba strings "#hex", así que estos
# NUNCA se recolorearon — por eso el grid de tiles seguía viéndose oscuro
# aunque el resto del panel ya era claro.
# ─────────────────────────────────────────────────────────────────────────────

def _pct_to_fill(pct: float, active: bool) -> QColor:
    if active:
        return QColor(ACCENT_BORDER)      # acento activo (tinte)
    if pct < 1:
        return QColor(SURFACE)            # sin anotar — gris neutro liso
    if pct >= 99.5:
        return QColor(OK_SOFT)            # completo — verde claro
    # Gradiente acento → verde según % etiquetado
    t = pct / 100.0
    c0 = QColor(ACCENT_SOFT); c1 = QColor(OK_SOFT)
    return QColor(
        int(c0.red()   * (1-t) + c1.red()   * t),
        int(c0.green() * (1-t) + c1.green() * t),
        int(c0.blue()  * (1-t) + c1.blue()  * t),
    )

def _pct_to_border(pct: float, active: bool, hover: bool) -> QColor:
    if active:  return QColor(ACCENT_STRONG)
    if hover:   return QColor(ACCENT)
    if pct >= 99.5: return QColor(OK)
    if pct > 0:     return QColor(ACCENT_BORDER)
    return QColor(BORDER)


# ─────────────────────────────────────────────────────────────────────────────
# Grid widget
# ─────────────────────────────────────────────────────────────────────────────

CELL_MIN = 12
CELL_MAX = 56
CELL_GAP = 2
CELL_PAD = 6

class TileGridWidget(QWidget):
    tile_clicked       = pyqtSignal(object)   # TileInfo
    tile_hover_changed = pyqtSignal(object)   # TileInfo | None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tm       = None
        self._active   = None
        self._hover    = None
        self._cell_sz  = 32
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumSize(60, 60)

    # ── API ──────────────────────────────────────────────────────────────────

    def set_tile_manager(self, tm) -> None:
        self._tm     = tm
        self._active = None
        self._hover  = None
        self._recompute_cell()
        self.update()

    def set_active_tile(self, tile) -> None:
        self._active = tile
        self.update()

    def refresh(self) -> None:
        self.update()

    # ── Layout ───────────────────────────────────────────────────────────────

    def _recompute_cell(self) -> None:
        if self._tm is None:
            self._cell_sz = CELL_MAX; return
        avail = max(60, self.width() - CELL_PAD * 2)
        n     = self._tm.n_cols
        fitted = (avail - CELL_GAP * max(n-1, 0)) // max(n, 1)
        self._cell_sz = max(CELL_MIN, min(CELL_MAX, fitted))

    def _grid_h(self) -> int:
        if self._tm is None: return 60
        c = self._cell_sz
        return CELL_PAD * 2 + self._tm.n_rows * (c + CELL_GAP) - CELL_GAP

    def sizeHint(self) -> QSize:
        return QSize(200, max(60, self._grid_h()))

    def _cell_rect(self, row: int, col: int) -> QRect:
        c = self._cell_sz
        return QRect(
            CELL_PAD + col * (c + CELL_GAP),
            CELL_PAD + row * (c + CELL_GAP),
            c, c)

    def _tile_at(self, pos: QPoint):
        if self._tm is None: return None
        c   = self._cell_sz
        col = (pos.x() - CELL_PAD) // (c + CELL_GAP)
        row = (pos.y() - CELL_PAD) // (c + CELL_GAP)
        if 0 <= row < self._tm.n_rows and 0 <= col < self._tm.n_cols:
            tid = row * self._tm.n_cols + col
            if 0 <= tid < len(self._tm.tiles):
                return self._tm.tiles[tid]
        return None

    # ── Eventos ──────────────────────────────────────────────────────────────

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._recompute_cell()
        self.update()

    def mouseMoveEvent(self, ev):
        t = self._tile_at(ev.pos())
        if t != self._hover:
            self._hover = t; self.update()
            self.tile_hover_changed.emit(t)
            # A celdas pequeñas (< 22px) el texto interno (coords/%) se
            # oculta por espacio — sin nada más, la grilla se ve como
            # puros cuadrados sin significado (reportado por el usuario:
            # "a veces solo se ven cuadrados y el usuario puede
            # confundirse"). Un tooltip nativo con la misma info funciona
            # a CUALQUIER tamaño de celda, sin depender del espacio
            # disponible para dibujar texto.
            if t is not None:
                status = ("Completo" if t.labeled_pct >= 99.5 else
                          "Parcial" if t.labeled_pct > 0 else "Sin anotar")
                extra = "  ·  activo" if t is self._active else ""
                self.setToolTip(f"Tile ({t.col}, {t.row})\n"
                                f"{t.labeled_pct:.0f}% etiquetado — {status}{extra}")
            else:
                self.setToolTip("")

    def leaveEvent(self, ev):
        if self._hover is not None:
            self._hover = None; self.update()
            self.tile_hover_changed.emit(None)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            t = self._tile_at(ev.pos())
            if t is not None and t != self._active:
                self.tile_clicked.emit(t)

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        p.fillRect(self.rect(), QColor(SURFACE_2))

        if self._tm is None:
            p.setPen(QColor(TEXT_MUTE))
            p.drawText(self.rect(), Qt.AlignCenter, "Sin nube cargada")
            return

        c        = self._cell_sz
        show_txt = c >= 22
        show_pct = c >= 36

        # BUG REAL ENCONTRADO (2026-09-09, confirmado con captura real —
        # ver skill de rediseño de este panel): el tamaño de fuente escalaba
        # como `c - 18` sin límite, así que en celdas grandes (cerca de
        # CELL_MAX=56px) el texto salía a ~38px de alto — mucho más ANCHO
        # que la celda para strings de 3-4 caracteres ("100%", "2,0") — el
        # texto se desbordaba sobre las celdas vecinas y quedaba ilegible
        # (números superpuestos). Fix: elegir el tamaño de fuente más
        # grande que quepa de verdad dentro de la celda (con margen),
        # medido con QFontMetrics contra el string MÁS ANCHO posible
        # ("100%"), no una fórmula lineal ciega al ancho real del texto.
        if show_txt:
            f = QFont()
            widest = "100%"
            max_w  = c - 6   # margen de ~3px por lado
            size   = min(16, c - 8)   # techo razonable, nunca gigante
            while size > 6:
                f.setPixelSize(size)
                fm = QFontMetrics(f)
                if fm.horizontalAdvance(widest) <= max_w and fm.height() <= c - 2:
                    break
                size -= 1
            f.setPixelSize(max(6, size))
            p.setFont(f)

        for tile in self._tm.tiles:
            rect   = self._cell_rect(tile.row, tile.col)
            is_act = tile is self._active
            is_hov = tile is self._hover

            fill   = _pct_to_fill(tile.labeled_pct, is_act)
            border = _pct_to_border(tile.labeled_pct, is_act, is_hov)

            p.fillRect(rect, fill)
            p.setPen(QPen(border, 1.0 if not is_act else 1.8))
            p.drawRect(rect)

            if show_txt:
                if show_pct and tile.labeled_pct > 0:
                    txt = f"{tile.labeled_pct:.0f}%"
                else:
                    txt = f"{tile.col},{tile.row}"
                tc = QColor(ACCENT_STRONG) if is_act else (
                     QColor(TEXT_DIM) if tile.labeled_pct > 0 else
                     QColor(TEXT_MUTE))
                p.setPen(tc)
                # Clip explícito al rect de la celda: aunque el tamaño ya
                # se eligió para caber, esto es una segunda barrera de
                # seguridad — nunca más debería poder pintarse texto de
                # una celda encima de la vecina, pase lo que pase.
                p.save()
                p.setClipRect(rect)
                p.drawText(rect, Qt.AlignCenter, txt)
                p.restore()


# ─────────────────────────────────────────────────────────────────────────────
# Panel completo
# ─────────────────────────────────────────────────────────────────────────────

class TilePanel(QWidget):
    """
    Dock panel de tiles.

    Navegación:
      - Click en tile (overview o tile mode) → tile_selected
      - Botón "← Overview" → exit_requested (solo salida al overview)
      - Escape (en main_window) → también sale
    """
    tile_selected          = pyqtSignal(object)
    exit_requested         = pyqtSignal()
    tile_size_changed      = pyqtSignal(float)
    view_mode_requested    = pyqtSignal(str)
    tile_clicked           = pyqtSignal(object)
    transform_changed      = pyqtSignal(float, float, float)
    tile_hover_2d          = pyqtSignal(object)
    grid_edit_mode_changed = pyqtSignal(str)
    load_cloud_requested   = pyqtSignal()   # "Cargar nube" del estado vacío

    TILE_SIZES = [10, 25, 50, 100, 200, 500]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tm     = None
        self._active = None
        self._build_ui()

    # ── Construcción ─────────────────────────────────────────────────────────

    def _icon_btn(self, icon_name: str, tooltip: str, checkable: bool = False) -> QPushButton:
        """Botón de icono compacto estilo barra de herramientas de SIG."""
        b = QPushButton()
        b.setIcon(qicon(icon_name, TEXT_DIM))
        b.setIconSize(QSize(15, 15))
        b.setFixedSize(28, 26)
        b.setCheckable(checkable)
        b.setToolTip(tooltip)
        b.setStyleSheet(
            f"QPushButton{{background:{SURFACE};border:1px solid {BORDER};border-radius:3px;}}"
            f"QPushButton:hover{{border-color:{ACCENT_BORDER};background:{ACCENT_SOFT};}}"
            f"QPushButton:checked{{background:{ACCENT_SOFT};border-color:{ACCENT};}}")
        return b

    def _icon_text_btn(self, icon_name: str, text: str, tooltip: str,
                       checkable: bool = False) -> QPushButton:
        """
        Botón icono + texto corto — REDISEÑO (2026-09-09): antes "modo de
        vista", "cámara" y "modo de edición del grid" eran filas de
        botones SOLO-ICONO (28×26px, sin texto) — había que pasar el
        mouse y esperar el tooltip para saber qué hacía cada uno
        (reportado como "no intuitivo", "no se sabe cómo volver a la
        vista general/poner densidad máxima/vista cenital"). Este botón
        SIEMPRE muestra icono + una palabra corta, el tooltip queda solo
        como refuerzo (atajo de teclado, explicación más larga).
        """
        b = QPushButton(f"  {text}")
        b.setIcon(qicon(icon_name, TEXT_DIM))
        b.setIconSize(QSize(14, 14))
        b.setFixedHeight(28)
        b.setCheckable(checkable)
        b.setToolTip(tooltip)
        b.setStyleSheet(
            f"QPushButton{{background:{SURFACE};border:1px solid {BORDER};border-radius:4px;"
            f"color:{TEXT_DIM};font-size:10px;font-weight:600;padding:0 6px;text-align:left;}}"
            f"QPushButton:hover{{border-color:{ACCENT_BORDER};background:{ACCENT_SOFT};color:{ACCENT_STRONG};}}"
            f"QPushButton:checked{{background:{ACCENT_SOFT};border-color:{ACCENT};color:{ACCENT_STRONG};}}")
        return b

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color:{TEXT_MUTE};font-size:9.5px;font-weight:700;"
            f"letter-spacing:0.4px;background:transparent;margin-top:2px;")
        return lbl

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Antes, este panel mostraba SIEMPRE la barra de "Tamaño de tile" +
        # una grilla vacía con solo el texto "Sin nube cargada" — controles
        # para algo que todavía no existe, sin ningún llamado a la acción.
        # Pedido explícito del usuario: la pantalla donde está "el botón de
        # cargar nube" debe ser más amigable. Ahora hay un estado vacío
        # de verdad (icono + mensaje + botón grande) antes de cargar un
        # proyecto, y los controles reales (que si tienen sentido una vez
        # hay una nube) viven en una segunda página que se activa sola en
        # cuanto set_tile_manager() recibe un tile manager real.
        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_empty_state())
        self._stack.addWidget(self._build_loaded_page())
        root.addWidget(self._stack)

    def _build_empty_state(self) -> QWidget:
        w = QWidget(); w.setStyleSheet(f"background:{SURFACE_2};")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(24, 40, 24, 24)
        lay.setSpacing(14)
        lay.addStretch()

        ic_bg = QFrame(); ic_bg.setObjectName("emptyIconBg")
        ic_bg.setFixedSize(72, 72)
        ic_bg.setStyleSheet(f"QFrame#emptyIconBg{{background:{ACCENT_SOFT};border-radius:36px;}}")
        ic_l = QVBoxLayout(ic_bg); ic_l.setContentsMargins(0, 0, 0, 0)
        ic = QLabel(); ic.setAlignment(Qt.AlignCenter)
        ic.setPixmap(qpixmap("cloud-arrow-up", ACCENT_STRONG, 34))
        ic.setStyleSheet("background:transparent;")
        ic_l.addWidget(ic)
        ic_row = QHBoxLayout(); ic_row.addStretch(); ic_row.addWidget(ic_bg); ic_row.addStretch()
        lay.addLayout(ic_row)

        title = QLabel("Ninguna nube cargada")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color:{TEXT};font-size:14px;font-weight:700;background:transparent;")
        lay.addWidget(title)

        desc = QLabel(
            "Carga un archivo LAS, LAZ, E57 o GA3D-Bin para empezar. "
            "También puedes arrastrar el archivo directamente sobre esta ventana.")
        desc.setAlignment(Qt.AlignCenter)
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;background:transparent;")
        lay.addWidget(desc)

        btn = QPushButton("  Cargar nube")
        btn.setIcon(qicon("cloud-arrow-up", "#ffffff"))
        btn.setFixedHeight(38)
        btn.setStyleSheet(
            f"QPushButton{{background:{ACCENT};color:#ffffff;border:none;"
            f"border-radius:5px;font-size:11px;font-weight:700;padding:0 18px;}}"
            f"QPushButton:hover{{background:{ACCENT_STRONG};}}")
        btn.clicked.connect(self.load_cloud_requested)
        btn_row = QHBoxLayout(); btn_row.addStretch(); btn_row.addWidget(btn); btn_row.addStretch()
        lay.addLayout(btn_row)

        lay.addStretch()
        return w

    def _build_loaded_page(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── Fila superior: modo + botón salida ──────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(4)

        self._mode_lbl = QLabel("Overview")
        self._mode_lbl.setStyleSheet(
            f"color:{TEXT_MUTE}; font-size:12px; font-weight:700;")
        top.addWidget(self._mode_lbl, 1)

        # REDISEÑO: antes un botón "Vista global" con borde sutil, mismo
        # peso visual que cualquier otro botón secundario — la acción más
        # importante para "salir de un tile" pasaba desapercibida
        # (reportado: "no se sabe bien cómo regresar a la vista completa").
        # Ahora, mientras HAY un tile activo, se pinta con fondo de acento
        # sólido (mismo tratamiento que "Cargar nube" en el estado vacío)
        # para que se note de inmediato que es LA acción para volver.
        self._exit_btn = QPushButton("  ← Volver a vista general")
        self._exit_btn.setIcon(qicon("box-arrow-left", "#ffffff"))
        self._exit_btn.setFixedHeight(26)
        self._exit_btn.setStyleSheet(
            f"QPushButton{{background:{ACCENT};border:1px solid {ACCENT};"
            f"border-radius:4px;color:#ffffff;font-size:10.5px;font-weight:700;padding:2px 10px;}}"
            f"QPushButton:hover{{background:{ACCENT_STRONG};border-color:{ACCENT_STRONG};}}"
            f"QPushButton:disabled{{background:{SURFACE};color:{TEXT_MUTE};border-color:{BORDER_SOFT};}}")
        self._exit_btn.setEnabled(False)
        self._exit_btn.clicked.connect(self._on_exit_clicked)
        top.addWidget(self._exit_btn)
        root.addLayout(top)

        # ── Tarjeta "barra de herramientas": tamaño + vista + cámara ─────────
        tb = QFrame(); tb.setObjectName("tilesToolbar")
        tb.setStyleSheet(
            f"QFrame#tilesToolbar{{background:{SURFACE_2};border:1px solid {BORDER};border-radius:4px;}}")
        tb_l = QVBoxLayout(tb); tb_l.setContentsMargins(8, 8, 8, 8); tb_l.setSpacing(7)

        # Fila 1: tamaño de tile — etiqueta arriba, controles abajo (antes
        # iban todos en una sola fila y quedaban muy apretados a la derecha).
        lbl = QLabel("Tamaño de tile")
        lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:10.5px;font-weight:600;background:transparent;")
        tb_l.addWidget(lbl)

        size_row = QHBoxLayout(); size_row.setSpacing(5)
        self._size_combo = QComboBox()
        self._size_combo.setFixedHeight(24)
        self._size_combo.setStyleSheet(
            f"QComboBox{{background:{SURFACE};border:1px solid {BORDER};"
            f"border-radius:4px;color:{TEXT_DIM};font-size:10.5px;padding:1px 4px;}}"
            f"QComboBox::drop-down{{width:16px;}}"
            f"QComboBox QAbstractItemView{{background:{SURFACE};color:{TEXT_DIM};"
            f"selection-background-color:{ACCENT_SOFT};border:1px solid {BORDER};}}")
        for s in self.TILE_SIZES:
            self._size_combo.addItem(f"{s} m", s)
        self._size_combo.setCurrentIndex(2)
        self._size_combo.currentIndexChanged.connect(self._on_size_changed)
        size_row.addWidget(self._size_combo, 1)

        # Entrada manual de tamaño personalizado
        self._custom_size = QDoubleSpinBox()
        self._custom_size.setRange(1.0, 5000.0)
        self._custom_size.setDecimals(1)
        self._custom_size.setSingleStep(5.0)
        self._custom_size.setValue(50.0)
        self._custom_size.setSuffix(" m")
        self._custom_size.setFixedHeight(24)
        self._custom_size.setToolTip("Tamaño personalizado\nEscribe el valor y presiona Enter o el botón →")
        self._custom_size.setStyleSheet(
            f"QDoubleSpinBox{{background:{SURFACE};border:1px solid {BORDER};"
            f"border-radius:4px;color:{TEXT_DIM};font-size:10.5px;padding:1px 4px;}}"
            f"QDoubleSpinBox:hover{{border-color:{ACCENT};}}"
            f"QDoubleSpinBox::up-button,QDoubleSpinBox::down-button"
            f"{{width:14px;background:{SURFACE};}}")
        self._custom_size.editingFinished.connect(self._on_custom_size_apply)
        size_row.addWidget(self._custom_size, 1)

        self._apply_size_btn = QPushButton("→")
        self._apply_size_btn.setFixedSize(24, 24)
        self._apply_size_btn.setToolTip("Aplicar tamaño personalizado")
        self._apply_size_btn.setStyleSheet(
            f"QPushButton{{background:{SURFACE};border:1px solid {BORDER};"
            f"border-radius:4px;color:{ACCENT_STRONG};font-size:11px;font-weight:700;}}"
            f"QPushButton:hover{{background:{ACCENT_SOFT};border-color:{ACCENT};}}")
        self._apply_size_btn.clicked.connect(self._on_custom_size_apply)
        size_row.addWidget(self._apply_size_btn)
        tb_l.addLayout(size_row)

        divider = QFrame(); divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet(f"background:{BORDER_SOFT};max-height:1px;")
        tb_l.addWidget(divider)

        # Fila 2: modo de vista — REDISEÑO: antes 3 iconos sueltos sin texto
        # ("Sparse"/"Full"/"Volar" solo se sabían por tooltip). Ahora cada
        # botón dice qué hace ("Disperso"/"Densidad máx."/"Volar"), y el
        # nombre interno ya no dice "Full" (ambiguo) sino que el texto
        # visible es explícito sobre qué es "máxima densidad".
        tb_l.addWidget(self._section_label("MODO DE VISTA"))
        mode_row = QHBoxLayout(); mode_row.setSpacing(4)
        self._mode_btns = {}
        for mode, icon_name, label, tip in [
            ("Sparse", "grid-3x3",   "Disperso",     "Vista dispersa — rápida, ideal para nubes enormes.\nMenos puntos en pantalla, no crece automáticamente."),
            ("Full",   "layers",    "Densidad máx.", "Vista completa — carga progresivamente hasta la\nmáxima densidad disponible del tile/nube."),
            ("Volar",  "controller","Volar",         "Modo vuelo: cámara libre estilo videojuego (WASD +\nmouse para mirar).  Atajo: F"),
        ]:
            b = self._icon_text_btn(icon_name, label, tip, checkable=True)
            b.clicked.connect(lambda checked, m=mode: self._on_mode_btn(m))
            mode_row.addWidget(b, 1)
            self._mode_btns[mode] = b
        tb_l.addLayout(mode_row)

        self._stats_lbl = QLabel("–")
        self._stats_lbl.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;background:transparent;")
        self._stats_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        tb_l.addWidget(self._stats_lbl)

        # Fila 3: cámara — mismo rediseño: texto explícito en vez de solo icono.
        tb_l.addWidget(self._section_label("CÁMARA"))
        cam_row = QHBoxLayout(); cam_row.setSpacing(4)
        self._btn_top   = self._icon_text_btn("arrow-bar-up",      "Cenital",   "Vista desde arriba (mirando hacia abajo en Z).  Atajo: V")
        self._btn_side  = self._icon_text_btn("arrow-bar-right",   "Lateral",   "Vista lateral (desde el eje Y).  Atajo: Y")
        self._btn_reset = self._icon_text_btn("arrows-fullscreen", "Encuadrar", "Encuadra/centra la cámara para ver la nube completa.  Atajo: R")
        self._btn_top.clicked.connect(lambda: self._emit_view_cmd("top"))
        self._btn_side.clicked.connect(lambda: self._emit_view_cmd("side"))
        self._btn_reset.clicked.connect(lambda: self._emit_view_cmd("reset"))
        cam_row.addWidget(self._btn_top, 1)
        cam_row.addWidget(self._btn_side, 1)
        cam_row.addWidget(self._btn_reset, 1)
        tb_l.addLayout(cam_row)

        root.addWidget(tb)

        # ── Grid ────────────────────────────────────────────────────────────
        self._grid = TileGridWidget()
        self._grid.tile_clicked.connect(self._on_tile_clicked)
        self._grid.tile_hover_changed.connect(self._on_grid_hover)

        scroll = QScrollArea()
        scroll.setWidget(self._grid)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet(
            "QScrollArea{background:#e8e9eb;border:none;}"
            "QScrollBar:vertical{background:#e8e9eb;width:6px;border:none;}"
            "QScrollBar::handle:vertical{background:#d6d8da;border-radius:3px;min-height:20px;}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}"
            "QScrollBar:horizontal{background:#e8e9eb;height:6px;border:none;}"
            "QScrollBar::handle:horizontal{background:#d6d8da;border-radius:3px;min-width:20px;}"
            "QScrollBar::add-line:horizontal,QScrollBar::sub-line:horizontal{width:0;}")
        root.addWidget(scroll, 1)

        # ── Leyenda compacta ─────────────────────────────────────────────────
        leg = QHBoxLayout()
        leg.setSpacing(6)
        leg.setContentsMargins(0, 2, 0, 0)
        # Borde más marcado que BORDER normal — los rellenos son tintes muy
        # pálidos (SURFACE/ACCENT_SOFT/OK_SOFT) que se confunden con el
        # fondo del panel si el borde es demasiado sutil.
        for hex_col, txt in [(SURFACE,"Sin anotar"),(ACCENT_SOFT,"Parcial"),(OK_SOFT,"Completo")]:
            dot = QLabel()
            dot.setFixedSize(10, 10)
            dot.setStyleSheet(
                f"background:{hex_col};border:1.5px solid {TEXT_MUTE};"
                f"border-radius:2px;")
            lbl = QLabel(txt)
            lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:10.5px;background:transparent;")
            leg.addWidget(dot)
            leg.addWidget(lbl)
        # "Activo" — antes la leyenda solo explicaba el color de RELLENO
        # (progreso), no el borde grueso de acento que marca el tile
        # seleccionado — un usuario podía ver ese borde resaltado sin
        # saber qué significaba.
        dot_active = QLabel()
        dot_active.setFixedSize(10, 10)
        dot_active.setStyleSheet(
            f"background:{SURFACE};border:2px solid {ACCENT_STRONG};border-radius:2px;")
        lbl_active = QLabel("Activo")
        lbl_active.setStyleSheet(f"color:{TEXT_DIM};font-size:10.5px;background:transparent;")
        leg.addWidget(dot_active); leg.addWidget(lbl_active)
        leg.addStretch()
        root.addLayout(leg)

        # ── Transform del grid ────────────────────────────────────────────────
        self._build_transform_section(root)
        return page

    # ── API pública ──────────────────────────────────────────────────────────

    def set_tile_manager(self, tm) -> None:
        self._tm     = tm
        self._active = None
        self._grid.set_tile_manager(tm)
        self._update_stats()
        # Estado vacío (página 0) mientras no haya tile manager real —
        # ver docstring de _build_ui.
        if hasattr(self, "_stack"):
            self._stack.setCurrentIndex(1 if tm is not None else 0)

    def set_active_tile(self, tile) -> None:
        self._active = tile
        self._grid.set_active_tile(tile)
        self._update_header()
        self._update_stats()

    def refresh_grid(self) -> None:
        self._grid.refresh()
        self._update_stats()

    def get_selected_tile_size(self) -> float:
        return float(self._size_combo.currentData())

    # ── Privado ───────────────────────────────────────────────────────────────

    def _update_header(self):
        if self._active is not None:
            t = self._active
            pts = f"{t.n_points/1e6:.1f}M" if t.n_points > 0 else "?"
            self._mode_lbl.setText(f"Tile ({t.col},{t.row}) · {pts} pts")
            self._mode_lbl.setStyleSheet(
                f"color:{ACCENT_STRONG};font-size:12px;font-weight:700;")
            self._exit_btn.setEnabled(True)
        else:
            self._mode_lbl.setText("Overview")
            self._mode_lbl.setStyleSheet(
                f"color:{TEXT_MUTE};font-size:12px;font-weight:700;")
            self._exit_btn.setEnabled(False)

    def _update_stats(self):
        if self._tm is None:
            self._stats_lbl.setText("–"); return
        tm  = self._tm
        pct = tm.overall_labeled_pct
        self._stats_lbl.setText(
            f"{tm.n_tiles_complete}/{tm.n_tiles}  {pct:.0f}%")

    def _on_grid_hover(self, tile) -> None:
        """Panel 2D → canvas 3D: resaltar el tile al pasar el mouse en el mapa."""
        mw = self.window()
        if hasattr(mw, '_canvas') and hasattr(mw._canvas, 'set_panel_hover_tile'):
            mw._canvas.set_panel_hover_tile(tile)
        self.tile_hover_2d.emit(tile)

    def _on_tile_clicked(self, tile) -> None:
        # FIX v1.1: permite tile-to-tile directamente sin pasar por overview.
        # El botón "← Overview" es la única forma de volver al overview.
        if tile is self._active:
            return   # ya estamos en este tile
        self.tile_selected.emit(tile)

    def set_view_mode(self, mode: str) -> None:
        """Actualiza el botón activo según el modo de vista del canvas."""
        if not hasattr(self, '_mode_btns'): return
        for m, btn in self._mode_btns.items():
            btn.setChecked(m == mode)


    def set_hover_tile(self, tile) -> None:
        """Resalta visualmente el tile al pasar el mouse desde el canvas 3D."""
        if hasattr(self, '_grid'):
            self._grid._hover = tile
            self._grid.update()

    def _build_transform_section(self, root):
        """Controles de posición y rotación del grid."""
        from PyQt5.QtWidgets import QDoubleSpinBox

        # ── Transform del grid — tarjeta ────────────────────────────────────
        tr_frame = QFrame(); tr_frame.setObjectName("transformCard")
        tr_frame.setStyleSheet(
            f"QFrame#transformCard{{background:{SURFACE};border:1px solid {BORDER};border-radius:4px;}}")
        tr_l = QVBoxLayout(tr_frame); tr_l.setContentsMargins(10,9,10,10); tr_l.setSpacing(6)

        hdr = QHBoxLayout()
        _ic = QLabel(); _ic.setStyleSheet("background:transparent;border:none;")
        _ic.setPixmap(qicon("arrows-move", TEXT_MUTE).pixmap(QSize(12, 12)))
        _sec = QLabel("TRANSFORM")
        _sec.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;font-weight:700;letter-spacing:0.4px;background:transparent;")
        hdr.addWidget(_ic); hdr.addWidget(_sec)
        hdr.addStretch()
        reset_btn = QPushButton("Reset")
        # FIX: con setFixedSize(50, 22) el texto salía cortado — mismo tipo
        # de desajuste de redondeo de fuente que ya afectó a otros botones
        # (ver CHANGELOG, "Bug de fuentes Qt"): a 10.5px/600 weight, "Reset"
        # más el padding del QPushButton no entra en 50px de ancho. Se
        # ensancha un poco y se deja padding horizontal explícito en vez de
        # depender de un ancho fijo calculado a ojo.
        reset_btn.setFixedHeight(22)
        reset_btn.setMinimumWidth(64)
        reset_btn.setStyleSheet(
            f"QPushButton{{background:{SURFACE_2};border:1px solid {BORDER};"
            f"border-radius:4px;color:{TEXT_DIM};font-size:10.5px;font-weight:600;"
            f"padding:0 8px;}}"
            f"QPushButton:hover{{border-color:{ACCENT};color:{ACCENT_STRONG};}}")
        reset_btn.clicked.connect(self._reset_transform)
        hdr.addWidget(reset_btn)
        tr_l.addLayout(hdr)

        # Botones de modo — REDISEÑO: antes 3 iconos sueltos ("compass",
        # "arrows-move", "arrow-repeat") sin texto — y "compass" colisionaba
        # visualmente con el botón de "Volar" de más arriba (mismo icono,
        # significado distinto), reforzando la confusión. Ahora cada botón
        # dice qué hace, y "Volar" ya usa un icono distinto (controller).
        tr_l.addWidget(self._section_label("QUÉ HACE EL MOUSE SOBRE LA NUBE"))
        mode_row = QHBoxLayout(); mode_row.setSpacing(4)
        self._btn_navigate = self._icon_text_btn(
            "compass", "Navegar", "Modo normal: el mouse mueve la CÁMARA.\n"
                       "Rueda = zoom · Clic+arrastrar = girar · Ctrl+arrastrar = panear",
            checkable=True)
        self._btn_move   = self._icon_text_btn(
            "arrows-move", "Mover grid",
            "El mouse MUEVE el grid de tiles (en vez de la cámara) —\n"
            "arrastra para reposicionarlo sobre la nube.", checkable=True)
        self._btn_rotate = self._icon_text_btn(
            "arrow-repeat", "Rotar grid",
            "El mouse ROTA el grid de tiles (en vez de la cámara) —\n"
            "arrastra para girarlo.", checkable=True)
        self._btn_navigate.setChecked(True)   # activo por defecto

        for btn, mode in [
            (self._btn_navigate, None),
            (self._btn_move,    "move"),
            (self._btn_rotate,  "rotate")
        ]:
            btn.clicked.connect(lambda c, m=mode: self._on_edit_mode(m))
            mode_row.addWidget(btn, 1)
        tr_l.addLayout(mode_row)

        # Spinboxes
        SP = (f"QDoubleSpinBox{{background:{SURFACE_2};border:1px solid {BORDER};"
              f"border-radius:4px;color:{TEXT_DIM};padding:2px 4px;font-size:10.5px;}}"
              f"QDoubleSpinBox:hover{{border-color:{ACCENT};}}")
        for label, attr in [("Offset X", "_sp_ox"), ("Offset Y", "_sp_oy")]:
            row = QHBoxLayout()
            lbl = QLabel(label); lbl.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;min-width:58px;background:transparent;")
            sp = QDoubleSpinBox()
            sp.setRange(-10000, 10000); sp.setValue(0.0)
            sp.setSuffix(" m"); sp.setSingleStep(1.0); sp.setDecimals(1)
            sp.setStyleSheet(SP)
            sp.valueChanged.connect(self._on_transform_spin)
            setattr(self, attr, sp)
            row.addWidget(lbl); row.addWidget(sp, 1)
            tr_l.addLayout(row)

        rot_row = QHBoxLayout()
        lbl_r = QLabel("Rotación"); lbl_r.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;min-width:58px;background:transparent;")
        self._sp_rot = QDoubleSpinBox()
        self._sp_rot.setRange(-180, 180); self._sp_rot.setValue(0.0)
        self._sp_rot.setSuffix("°"); self._sp_rot.setSingleStep(0.5); self._sp_rot.setDecimals(1)
        self._sp_rot.setStyleSheet(SP)
        self._sp_rot.valueChanged.connect(self._on_transform_spin)
        rot_row.addWidget(lbl_r); rot_row.addWidget(self._sp_rot, 1)
        tr_l.addLayout(rot_row)

        root.addWidget(tr_frame)

    def _on_edit_mode(self, mode) -> None:
        """Cambia modo de edición. mode=None → navegación normal."""
        nav = (mode is None)
        self._btn_navigate.setChecked(nav)
        self._btn_move.setChecked(mode == "move")
        self._btn_rotate.setChecked(mode == "rotate")
        # Emitir señal — None significa volver a navegación
        self.grid_edit_mode_changed.emit(mode or "")

    def _on_transform_spin(self) -> None:
        ox = self._sp_ox.value() if hasattr(self, '_sp_ox') else 0.0
        oy = self._sp_oy.value() if hasattr(self, '_sp_oy') else 0.0
        rot = self._sp_rot.value() if hasattr(self, '_sp_rot') else 0.0
        self.transform_changed.emit(ox, oy, rot)

    def _reset_transform(self) -> None:
        for attr in ('_sp_ox', '_sp_oy', '_sp_rot'):
            if hasattr(self, attr):
                getattr(self, attr).blockSignals(True)
                getattr(self, attr).setValue(0.0)
                getattr(self, attr).blockSignals(False)
        self.transform_changed.emit(0.0, 0.0, 0.0)

    def update_transform_spinboxes(self, ox: float, oy: float, rot: float) -> None:
        """Actualiza los spinboxes sin emitir señales (para sincronización desde proyecto)."""
        for attr, val in [('_sp_ox', ox), ('_sp_oy', oy), ('_sp_rot', rot)]:
            if hasattr(self, attr):
                getattr(self, attr).blockSignals(True)
                getattr(self, attr).setValue(val)
                getattr(self, attr).blockSignals(False)

    def _emit_view_cmd(self, cmd: str) -> None:
        """Emite solicitud de vista al canvas via main_window."""
        mw = self.window()
        if cmd == "top" and hasattr(mw, '_set_top_view'):
            mw._set_top_view()
        elif cmd == "side" and hasattr(mw, '_set_side_view'):
            mw._set_side_view()
        elif cmd == "reset" and hasattr(mw, '_reset_camera_view'):
            mw._reset_camera_view()

    def _on_mode_btn(self, mode: str) -> None:
        self.view_mode_requested.emit(mode)
        if hasattr(self, '_mode_btns'):
            for m, btn in self._mode_btns.items():
                btn.setChecked(m == mode)

    def get_tile_size(self) -> float:
        """Returns the current tile size in meters (combo or custom input)."""
        if hasattr(self, '_custom_size') and self._custom_size.hasFocus():
            return self._custom_size.value()
        if hasattr(self, '_size_combo'):
            try:
                return float(self._size_combo.currentText().split()[0])
            except Exception:
                pass
        return 50.0

    def get_transform(self) -> tuple:
        """Returns (offset_x, offset_y, rotation_deg)."""
        ox  = self._sp_ox.value()  if hasattr(self, '_sp_ox')  else 0.0
        oy  = self._sp_oy.value()  if hasattr(self, '_sp_oy')  else 0.0
        rot = self._sp_rot.value() if hasattr(self, '_sp_rot') else 0.0
        return (ox, oy, rot)

    def set_schema(self, schema) -> None:
        """Pasa el schema al grid para colorear tiles por clase."""
        if hasattr(self, '_grid') and hasattr(self._grid, 'set_schema'):
            self._grid.set_schema(schema)

    def _on_exit_clicked(self) -> None:
        self.exit_requested.emit()

    def _on_custom_size_apply(self) -> None:
        """Aplica el tamaño personalizado introducido manualmente."""
        val = self._custom_size.value()
        if val <= 0:
            return
        # Seleccionar en el combo el valor más cercano si existe, sino dejarlo custom
        for i in range(self._size_combo.count()):
            if abs(self._size_combo.itemData(i) - val) < 0.01:
                self._size_combo.setCurrentIndex(i)
                return   # _on_size_changed se encargará
        # Valor personalizado no está en el combo — emitir directamente
        if self._tm is not None:
            self.tile_size_changed.emit(val)

    def _on_size_changed(self, _idx: int) -> None:
        if self._tm is None: return
        self.tile_size_changed.emit(float(self._size_combo.currentData()))
