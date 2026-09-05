"""
ui/class_panel.py
=================
Panel izquierdo — lista de clases semánticas y clase activa.

REDISEÑO 2026-09-05 (v3 — badges numerados coloreados, más amigable):
  - v2 (Photoshop/Illustrator layers-panel) era funcional pero frío:
    swatch minúsculo + badge de tecla separados, acento genérico
    (siempre el mismo teal) para marcar "activa" sin importar la
    clase. Difícil de asociar rápidamente número↔color↔clase.
  - v3 fusiona número de tecla + color en un solo badge circular
    grande coloreado con el color REAL de la clase (metáfora tipo
    "paleta de pintura" — CVAT/Labelbox/Procreate usan variantes de
    esto). El estado "activa" ahora se resalta con EL COLOR DE ESA
    CLASE (fondo tintado + borde), no con un acento genérico — así
    aprender qué color pertenece a qué clase es inmediato, y la fila
    activa siempre destaca sin importar cuál sea.
  - Filas más altas (36px) y con más aire → objetivo más grande y
    amigable para hacer clic, en vez de una lista densa tipo tabla.
  - El chip de "clase activa" ahora se presenta como "PINTANDO CON"
    con icono de pincel — metáfora de "color de primer plano" de
    herramientas de dibujo (Photoshop/Procreate), más intuitiva que
    la etiqueta técnica "clase activa".

Layout:
  ┌─ PINTANDO CON ──────────────────┐
  │ 🖌  ⬤ Edificio         11,230   │  ← "color de pincel" actual
  ├─ CLASES              [+][✎] ───┤
  │ (1) Suelo             38,420    │  ← badge circular = tecla + color
  │ (2) Veg. baja         12,180    │
  │  ...                            │
  ├─ COBERTURA TOTAL ───────────────┤
  │  ████░░░░░░░  6.2%  74,618 pts  │
  └─────────────────────────────────┘
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QScrollArea, QFrame, QSizePolicy, QPushButton,
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QColor, QPainter, QBrush, QPen

from core.project import Project, SemanticClass
from ui.icons import icon as qicon, pixmap as qpixmap
from ui.theme import (
    SURFACE, SURFACE_2, SURFACE_3, BORDER, BORDER_SOFT,
    TEXT, TEXT_DIM, TEXT_MUTE,
    ACCENT, ACCENT_STRONG, ACCENT_SOFT, ACCENT_BORDER,
)


def _readable_on(hex_color: str) -> str:
    """Blanco o gris oscuro, el que dé mejor contraste sobre hex_color.
    Necesario porque el badge circular usa el color REAL de cada clase
    (puede ser oscuro o claro) como fondo con el número encima."""
    try:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        luma = 0.299*r + 0.587*g + 0.114*b
        return "#2b2d30" if luma > 150 else "#ffffff"
    except Exception:
        return "#ffffff"


def _tint(hex_color: str, amount: float = 0.85) -> str:
    """Mezcla hex_color con blanco (amount=1 → blanco puro) — para el
    fondo suave de la fila/chip activa, coloreado según la clase."""
    try:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        r = int(r + (255-r)*amount); g = int(g + (255-g)*amount); b = int(b + (255-b)*amount)
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return SURFACE_2


# ── Widget de barra de progreso personalizada ────────────────────────────────

class _ProgressBar(QWidget):
    """Barra de progreso con el acento del tema."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(6)
        self._value = 0.0   # 0.0–1.0

    def set_value(self, v: float) -> None:
        self._value = max(0.0, min(1.0, v))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(SURFACE_3)))
        p.drawRect(0, 0, w, h)
        fill_w = int(w * self._value)
        if fill_w > 0:
            p.setBrush(QBrush(QColor(ACCENT)))
            p.drawRect(0, 0, fill_w, h)
        p.end()


# ── Fila de clase ─────────────────────────────────────────────────────────────

class _ClassRow(QWidget):
    """
    Una fila en la lista de clases: [badge circular: tecla + color][nombre][count]

    El badge circular fusiona la tecla de atajo Y el color de la clase
    en un solo elemento — más fácil de asociar de un vistazo que los
    dos elementos separados y diminutos de la v2 (key-pill + swatch).
    La fila activa se resalta con EL COLOR PROPIO de esa clase (fondo
    tintado + borde), no con un acento genérico — el objetivo es que
    el usuario aprenda "número → color → clase" de un vistazo.
    """

    clicked = pyqtSignal(int)  # class_id
    color_changed = pyqtSignal(int, str)   # class_id, new_color_hex
    name_changed  = pyqtSignal(int, str)   # class_id, new_name

    def __init__(self, sc: SemanticClass, row_index: int = 0, parent=None):
        super().__init__(parent)
        self.sc = sc
        self._active = False
        self._alt    = (row_index % 2 == 1)
        self.setFixedHeight(36)
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setToolTip("Clic: pintar con esta clase\nDoble clic: editar color y nombre")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 0, 10, 0)
        lay.setSpacing(9)

        # Badge circular: tecla de atajo sobre el color real de la clase
        self._badge = QLabel(sc.key if hasattr(sc, "key") else "")
        self._badge.setFixedSize(24, 24)
        self._badge.setAlignment(Qt.AlignCenter)

        # Nombre
        self._name_lbl = QLabel(sc.name)
        self._name_lbl.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred)

        # Conteo
        self._count_lbl = QLabel("—")
        self._count_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._count_lbl.setStyleSheet(
            f"color: {TEXT_MUTE}; font-size:10.5px; min-width: 56px; font-family: Consolas; background:transparent;")

        lay.addWidget(self._badge)
        lay.addWidget(self._name_lbl)
        lay.addWidget(self._count_lbl)

        self._apply_style()

    def _apply_style(self) -> None:
        base_bg = SURFACE_2 if self._alt else SURFACE
        color = self.sc.color
        badge_fg = _readable_on(color)
        self._badge.setStyleSheet(
            f"background:{color};border-radius:12px;color:{badge_fg};"
            f"font-size:10.5px;font-weight:700;")
        if self._active:
            # Antes llevaba también un border-left de acento — el usuario
            # pidió quitar esa línea vertical (quedaban dos: esta y la del
            # chip "PINTANDO CON" de arriba, redundantes). El fondo teñido
            # con el color de la clase ya deja clara la fila activa.
            row_bg = _tint(color, 0.82)
            self.setStyleSheet(f"background: {row_bg}; border: none;")
            self._name_lbl.setStyleSheet(
                f"color: {TEXT}; font-size: 11.5px; font-weight: 700;"
                f" background: transparent; border: none; padding: 0;")
        else:
            self.setStyleSheet(f"background: {base_bg}; border: none;")
            self._name_lbl.setStyleSheet(
                f"color: {TEXT_DIM}; font-size: 11.5px;"
                f" background: transparent; border: none; padding: 0;")

    def set_active(self, active: bool) -> None:
        self._active = active
        self._apply_style()

    def set_count(self, n: int) -> None:
        self._count_lbl.setText(f"{n:,}" if n > 0 else "—")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.sc.id)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._open_editor()

    def _open_editor(self):
        """Abre un diálogo flotante para editar color y nombre de la clase."""
        from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                     QPushButton, QLineEdit, QColorDialog)
        from PyQt5.QtGui import QColor
        from PyQt5.QtCore import Qt

        dlg = QDialog(self.window())
        dlg.setWindowTitle(f"Editar clase: {self.sc.name}")
        dlg.setFixedWidth(300)
        dlg.setModal(False)
        dlg.setStyleSheet(f"QDialog{{background:{SURFACE};}} QLabel{{color:{TEXT_DIM};}}")

        lay = QVBoxLayout(dlg); lay.setContentsMargins(14,14,14,14); lay.setSpacing(10)

        # Color picker button
        col_row = QHBoxLayout()
        col_lbl = QLabel("Color:"); col_lbl.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;")
        self._col_preview = QPushButton()
        self._col_preview.setFixedSize(48, 24)
        self._current_color = self.sc.color
        self._update_color_btn()
        self._col_preview.clicked.connect(lambda: self._pick_color(dlg))
        col_row.addWidget(col_lbl); col_row.addWidget(self._col_preview); col_row.addStretch()
        lay.addLayout(col_row)

        # Name editor
        name_row = QHBoxLayout()
        name_lbl = QLabel("Nombre:"); name_lbl.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;")
        self._name_edit = QLineEdit(self.sc.name)
        self._name_edit.setStyleSheet(
            f"QLineEdit{{background:{SURFACE};border:1px solid {BORDER};border-radius:3px;"
            f"color:{TEXT_DIM};padding:4px;font-size:11px;}}"
            f"QLineEdit:focus{{border-color:{ACCENT};}}")
        name_row.addWidget(name_lbl); name_row.addWidget(self._name_edit, 1)
        lay.addLayout(name_row)

        # Buttons
        btn_row = QHBoxLayout()
        btn_cancel = QPushButton("Cancelar")
        btn_ok     = QPushButton("Aplicar")
        for btn, sty in [
            (btn_cancel, f"background:{SURFACE};border:1px solid {BORDER};border-radius:3px;"
                         f"color:{TEXT_MUTE};padding:5px 12px;font-size:10.5px;"),
            (btn_ok,     f"background:{ACCENT};color:#ffffff;border:none;border-radius:3px;"
                         f"padding:5px 12px;font-size:10.5px;font-weight:600;")
        ]:
            btn.setStyleSheet(f"QPushButton{{{sty}}} QPushButton:hover{{opacity:0.85;}}")
        btn_cancel.clicked.connect(dlg.reject)
        btn_ok.clicked.connect(lambda: self._apply_edit(dlg))
        btn_row.addWidget(btn_cancel); btn_row.addStretch(); btn_row.addWidget(btn_ok)
        lay.addLayout(btn_row)

        dlg.show()

    def _update_color_btn(self):
        c = self._current_color
        self._col_preview.setStyleSheet(
            f"QPushButton{{background:{c};border:1px solid {BORDER};border-radius:3px;}}"
            f"QPushButton:hover{{border-color:{ACCENT};}}")

    def _pick_color(self, parent):
        from PyQt5.QtWidgets import QColorDialog
        from PyQt5.QtGui import QColor
        col = QColorDialog.getColor(QColor(self._current_color), parent, "Seleccionar color")
        if col.isValid():
            self._current_color = col.name()
            self._update_color_btn()

    def _apply_edit(self, dlg):
        new_color = self._current_color
        new_name  = self._name_edit.text().strip() or self.sc.name
        self.sc.color = new_color
        self.sc.name  = new_name
        self._name_lbl.setText(new_name)
        self._apply_style()   # refresca el badge circular con el nuevo color
        self.color_changed.emit(self.sc.id, new_color)
        self.name_changed.emit(self.sc.id, new_name)
        dlg.accept()

    def set_key(self, key: str) -> None:
        self._badge.setText(key)


# ── Chip de clase activa ──────────────────────────────────────────────────────

class _ActiveChip(QWidget):
    """
    Indicador de "con qué clase estás pintando ahora" — metáfora del
    color de primer plano en herramientas de dibujo (Photoshop,
    Procreate): un círculo grande con el color actual + un icono de
    pincel, en vez de la etiqueta técnica "clase activa" + un swatch
    diminuto. Fondo tintado con el color de la clase (no un acento
    genérico) para que se sienta como "estás pintando de este color".
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(50)
        self.setAttribute(Qt.WA_StyledBackground, True)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 0, 12, 0)
        lay.setSpacing(10)

        self._brush_ic = QLabel()
        self._brush_ic.setFixedSize(14, 14)
        self._brush_ic.setStyleSheet("background:transparent;")

        self._swatch = QLabel()
        self._swatch.setFixedSize(20, 20)

        self._name = QLabel("—")
        self._name.setStyleSheet(f"font-size: 13.5px; font-weight: 700; color: {ACCENT_STRONG}; background:transparent;")
        self._name.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self._count = QLabel("")
        self._count.setStyleSheet(f"color: {TEXT_MUTE}; font-size:10.5px; font-family:Consolas; background:transparent;")

        lay.addWidget(self._brush_ic)
        lay.addWidget(self._swatch)
        lay.addWidget(self._name)
        lay.addWidget(self._count)

    def update_class(self, sc: SemanticClass, count: int) -> None:
        color = sc.color
        fg = _readable_on(color)
        self._brush_ic.setPixmap(qpixmap("brush", color, 14))
        self._swatch.setStyleSheet(f"border-radius: 10px; background: {color};")
        self._name.setText(sc.name)
        self._name.setStyleSheet(
            f"font-size: 13.5px; font-weight: 700; color: {TEXT}; background:transparent;")
        # Sin border-left: el usuario pidió quitar las líneas verticales de
        # acento (esta y la de la fila activa en la lista, redundantes
        # entre sí) — el fondo teñido con el color de la clase + el
        # círculo de color ya comunican "con qué color estás pintando".
        self.setStyleSheet(f"background: {_tint(color, 0.88)}; border: none;")
        self._count.setText(f"{count:,}" if count > 0 else "")


# ── Panel principal ───────────────────────────────────────────────────────────

class ClassPanel(QWidget):
    """
    Panel izquierdo de clases semánticas.

    Señales:
        active_class_changed(int)  — class_id de la nueva clase activa
    """

    active_class_changed = pyqtSignal(int)
    schema_changed       = pyqtSignal(list)   # nuevo schema completo

    def __init__(self, parent=None):
        super().__init__(parent)
        # setFixedWidth() impedía que el panel respondiera al redimensionar
        # el dock/splitter que lo contiene (el usuario podía arrastrar el
        # borde pero el contenido nunca cambiaba de ancho) — con min/max
        # en su lugar, todo el layout interno (QHBoxLayout con stretch en
        # nombre/badge) se adapta cuando el panel se hace más ancho o más
        # angosto, igual que el resto de paneles (ToolPanel, TilePanel).
        self.setMinimumWidth(190)
        self.setMaximumWidth(480)
        self.setStyleSheet(f"background:{SURFACE};")

        self._project: Optional[Project]        = None
        self._rows: Dict[int, _ClassRow]        = {}
        self._active_id: int                    = 1
        self._visible_ids: Optional[Set[int]]   = None  # None = mostrar todas

        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sección: clase activa ──────────────────────────────────────
        sec_ac_lbl = QLabel("PINTANDO CON")
        sec_ac_lbl.setProperty("role", "section")
        sec_ac_lbl.setWordWrap(True)
        sec_ac_lbl.setContentsMargins(12, 10, 12, 4)

        self._chip = _ActiveChip()

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        sep1.setStyleSheet(f"background: {BORDER}; max-height:1px;")

        root.addWidget(sec_ac_lbl)
        root.addWidget(self._chip)
        root.addWidget(sep1)

        # ── Barra de herramientas del panel: título + iconos ────────────
        # Estilo panel de capas (Photoshop/Illustrator): las acciones del
        # panel viven como iconos pequeños junto al título, no como un
        # botón de texto grande y suelto al final.
        tb = QWidget(); tb.setStyleSheet(f"background:{SURFACE_2};")
        tb_l = QHBoxLayout(tb); tb_l.setContentsMargins(12, 6, 8, 6); tb_l.setSpacing(4)
        # "TECLAS 1-9" se quitó del título: las filas ya muestran el badge
        # de atajo directamente, era redundante y forzaba un salto de línea
        # incómodo ("CLASES ·" / "TECLAS 1-9") en el ancho normal del panel.
        sec_list_lbl = QLabel("CLASES")
        sec_list_lbl.setProperty("role", "section")
        sec_list_lbl.setWordWrap(True)
        tb_l.addWidget(sec_list_lbl)
        tb_l.addStretch()

        mgr_btn = QPushButton()
        mgr_btn.setIcon(qicon("pencil-square", TEXT_DIM))
        mgr_btn.setIconSize(QSize(13, 13))
        mgr_btn.setFixedSize(24, 22)
        mgr_btn.setToolTip("Gestionar clases (añadir, quitar, reordenar)")
        mgr_btn.setStyleSheet(
            f"QPushButton{{background:transparent;border:1px solid transparent;border-radius:3px;}}"
            f"QPushButton:hover{{background:{SURFACE};border-color:{ACCENT_BORDER};}}")
        mgr_btn.clicked.connect(self._open_class_manager)
        tb_l.addWidget(mgr_btn)
        root.addWidget(tb)

        sep_tb = QFrame(); sep_tb.setFrameShape(QFrame.HLine)
        sep_tb.setStyleSheet(f"background:{BORDER_SOFT};max-height:1px;")
        root.addWidget(sep_tb)

        # Área scrollable para las filas
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent;")

        self._list_widget = QWidget()
        self._list_widget.setStyleSheet("background: transparent;")
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(0)
        self._list_layout.addStretch()

        scroll.setWidget(self._list_widget)
        root.addWidget(scroll, 1)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet(f"background: {BORDER}; max-height:1px;")
        root.addWidget(sep2)

        # ── Cobertura total ───────────────────────────────────────────
        prog_container = QWidget()
        prog_container.setStyleSheet(f"background: {SURFACE_2};")
        prog_lay = QVBoxLayout(prog_container)
        prog_lay.setContentsMargins(12, 9, 12, 11)
        prog_lay.setSpacing(6)

        sec_prog_lbl = QLabel("COBERTURA TOTAL")
        sec_prog_lbl.setProperty("role", "section")
        sec_prog_lbl.setWordWrap(True)
        prog_lay.addWidget(sec_prog_lbl)

        self._prog_bar = _ProgressBar()
        prog_lay.addWidget(self._prog_bar)

        prog_nums = QHBoxLayout()
        prog_nums.setContentsMargins(0, 0, 0, 0)
        self._prog_labeled = QLabel("0 etiquetados")
        self._prog_labeled.setStyleSheet(f"color: {TEXT_MUTE}; font-size:10.5px; background:transparent;")
        self._prog_pct = QLabel("0.0%")
        self._prog_pct.setAlignment(Qt.AlignRight)
        self._prog_pct.setStyleSheet(f"color: {ACCENT_STRONG}; font-size:10.5px; font-weight:700; font-family:Consolas; background:transparent;")
        prog_nums.addWidget(self._prog_labeled)
        prog_nums.addWidget(self._prog_pct)
        prog_lay.addLayout(prog_nums)

        root.addWidget(prog_container)

    # ── API pública ───────────────────────────────────────────────────

    def _on_class_color_changed(self, class_id: int, color: str) -> None:
        """Reasigna el color de la clase y propaga al canvas via schema_changed."""
        self.schema_changed.emit(self._project.schema if self._project else [])

    def _on_class_name_changed(self, class_id: int, name: str) -> None:
        """Propaga cambio de nombre."""
        self.schema_changed.emit(self._project.schema if self._project else [])

    def set_project(self, project: Project) -> None:
        """Carga el schema del proyecto y construye las filas."""
        self._project = project
        self._rebuild_rows()

    def set_active_class(self, class_id: int) -> None:
        """Cambia la clase activa programáticamente (desde teclado o click)."""
        if class_id == self._active_id:
            return
        if class_id not in self._rows and self._project:
            ids = [sc.id for sc in self._project.schema if sc.id > 0]
            if class_id not in ids:
                return

        # Desactivar anterior
        if self._active_id in self._rows:
            self._rows[self._active_id].set_active(False)

        self._active_id = class_id

        # Activar nueva
        if class_id in self._rows:
            self._rows[class_id].set_active(True)

        # Actualizar chip
        if self._project:
            sc = next((s for s in self._project.schema if s.id == class_id), None)
            if sc:
                count = self._rows[class_id]._count_lbl.text()
                n = int(count.replace(",", "")) if count != "—" else 0
                self._chip.update_class(sc, n)

        self.active_class_changed.emit(class_id)

    def update_counts(self, counts: Dict[int, int]) -> None:
        """Actualiza los conteos de puntos por clase."""
        for class_id, row in self._rows.items():
            row.set_count(counts.get(class_id, 0))

        # Actualizar chip con el conteo de la clase activa
        if self._project and self._active_id in self._rows:
            sc = next((s for s in self._project.schema
                       if s.id == self._active_id), None)
            if sc:
                self._chip.update_class(sc, counts.get(self._active_id, 0))

    def update_progress(self, n_labeled: int, n_total: int) -> None:
        """Actualiza la barra de progreso."""
        pct = n_labeled / n_total if n_total > 0 else 0.0
        self._prog_bar.set_value(pct)
        self._prog_labeled.setText(f"{n_labeled:,} etiquetados")
        self._prog_pct.setText(f"{pct * 100:.1f}%")

    # ── Construcción interna ──────────────────────────────────────────

    def _rebuild_rows(self) -> None:
        """Reconstruye la lista de filas desde el schema del proyecto."""
        if self._project is None:
            return

        while self._list_layout.count() > 1:
            item = self._list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._rows.clear()

        keys = [str(i) for i in range(1, 10)]
        key_idx = 0
        row_idx = 0

        for sc in self._project.schema:
            if sc.id == 0:
                continue
            # Filtro de visibilidad
            if self._visible_ids is not None and sc.id not in self._visible_ids:
                continue

            row = _ClassRow(sc, row_index=row_idx)
            row_idx += 1
            if key_idx < len(keys):
                row.set_key(keys[key_idx])
                key_idx += 1

            row.clicked.connect(self.set_active_class)
            # Connect color/name editor signals → propagate schema change to canvas
            row.color_changed.connect(self._on_class_color_changed)
            row.name_changed.connect(self._on_class_name_changed)
            self._rows[sc.id] = row
            self._list_layout.insertWidget(self._list_layout.count() - 1, row)

        first_id = next(
            (sc.id for sc in self._project.schema
             if sc.id > 0 and (self._visible_ids is None or sc.id in (self._visible_ids or {sc.id}))),
            1)
        self._active_id = first_id
        if first_id in self._rows:
            self._rows[first_id].set_active(True)
            sc = next(s for s in self._project.schema if s.id == first_id)
            self._chip.update_class(sc, 0)

    def _open_class_manager(self) -> None:
        if self._project is None:
            return
        from ui.class_manager import ClassManagerDialog
        counts = {}
        if self._project.labels is not None:
            import numpy as np
            lbl = self._project.labels
            unique, cnts = np.unique(lbl[lbl > 0], return_counts=True)
            counts = {int(u): int(c) for u, c in zip(unique, cnts)}

        dlg = ClassManagerDialog(self._project.schema, counts, self)
        dlg.schema_updated.connect(self._apply_schema_update)
        dlg.exec_()

    def _apply_schema_update(self, new_schema: list, visible_ids: set) -> None:
        """Aplica el nuevo schema y filtro de visibilidad al proyecto y panel."""
        if self._project is None:
            return
        self._project.schema = new_schema
        self._visible_ids = visible_ids if visible_ids else None
        self._rebuild_rows()
        self.schema_changed.emit(new_schema)
