"""
ui/class_manager.py — Diálogo para gestionar el schema de clases.

Permite:
  - Ver todas las clases con su color y conteo
  - Añadir clase nueva
  - Eliminar clases (con confirmación)
  - Cambiar nombre y color de cada clase
  - Filtrar qué clases son visibles en el panel lateral

REDISEÑO 2026-09-05: colores movidos a ui/theme.py (ya usaban los
valores correctos de la paleta gris/cian, pero sueltos inline) +
iconos reales en vez de "×"/"✕" de texto. Bug de contraste real
corregido: el `QLineEdit` de nombre editable usaba `color:#a8acb0`
(TEXT_FAINT — pensado para texto secundario apenas visible) en el
texto PRINCIPAL editable de cada clase, dificultando la lectura;
ahora usa TEXT (texto normal).
"""
from __future__ import annotations
from typing import List, Optional

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QWidget, QScrollArea, QFrame, QLineEdit, QColorDialog,
    QCheckBox, QMessageBox, QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QBrush

from core.project import SemanticClass
from ui.icons import icon as qicon, pixmap as qpixmap
from ui.theme import (
    SURFACE, SURFACE_2, SURFACE_3, BORDER, BORDER_SOFT,
    TEXT, TEXT_DIM, TEXT_MUTE, TEXT_FAINT,
    ACCENT, ACCENT_STRONG, ACCENT_SOFT, ACCENT_BORDER,
    OK, OK_SOFT, WARN, WARN_SOFT,
)


class _ColorSwatch(QWidget):
    """Swatch clicable que abre el color picker."""
    color_changed = pyqtSignal(str)   # hex string

    def __init__(self, color: str, parent=None):
        super().__init__(parent)
        self._color = color
        self.setFixedSize(22, 22)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Cambiar color")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            c = QColorDialog.getColor(QColor(self._color), self,
                                      "Seleccionar color de clase")
            if c.isValid():
                self._color = c.name()
                self.update()
                self.color_changed.emit(self._color)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QBrush(QColor(self._color)))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(2, 2, 18, 18, 4, 4)
        p.end()

    def set_color(self, c: str):
        self._color = c; self.update()

    @property
    def color(self): return self._color


class _ClassRow(QFrame):
    """Fila de una clase en el editor."""
    deleted = pyqtSignal(int)   # class_id

    def __init__(self, sc: SemanticClass, count: int, parent=None):
        super().__init__(parent)
        self._sc = sc
        self.setObjectName("classRow")
        self.setFixedHeight(42)
        self.setStyleSheet(f"QFrame#classRow{{background:{SURFACE};border-radius:3px;}}")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 0, 8, 0)
        lay.setSpacing(8)

        # Visible en panel (toggle de icono ojo)
        self._vis_btn = QPushButton()
        self._vis_btn.setCheckable(True); self._vis_btn.setChecked(True)
        self._vis_btn.setFixedSize(22, 22)
        self._vis_btn.setToolTip("Visible en panel de clases")
        self._vis_btn.setIcon(qicon("eye", ACCENT_STRONG))
        self._vis_btn.setStyleSheet(
            f"QPushButton{{background:transparent;border:none;border-radius:3px;}}"
            f"QPushButton:hover{{background:{SURFACE_2};}}")
        self._vis_btn.toggled.connect(self._on_vis_toggled)
        lay.addWidget(self._vis_btn)

        # ID badge
        id_lbl = QLabel(str(sc.id))
        id_lbl.setFixedWidth(20)
        id_lbl.setAlignment(Qt.AlignCenter)
        id_lbl.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;")
        lay.addWidget(id_lbl)

        # Color swatch
        self._swatch = _ColorSwatch(sc.color)
        self._swatch.color_changed.connect(lambda c: setattr(self._sc, 'color', c))
        lay.addWidget(self._swatch)

        # Nombre editable
        self._name_edit = QLineEdit(sc.name)
        self._name_edit.setStyleSheet(
            f"QLineEdit{{background:{SURFACE_2};border:1px solid {BORDER};"
            f"border-radius:3px;color:{TEXT};padding:4px 7px;font-size:11px;}}"
            f"QLineEdit:focus{{border-color:{ACCENT};}}")
        self._name_edit.textChanged.connect(lambda t: setattr(self._sc, 'name', t))
        lay.addWidget(self._name_edit, 1)

        # Conteo
        cnt_lbl = QLabel(f"{count:,}" if count > 0 else "—")
        cnt_lbl.setFixedWidth(64)
        cnt_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        cnt_lbl.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;font-family:'Consolas';")
        lay.addWidget(cnt_lbl)

        # Botón eliminar (no para clase 0)
        if sc.id != 0:
            del_btn = QPushButton()
            del_btn.setIcon(qicon("trash", TEXT_MUTE))
            del_btn.setFixedSize(22, 22)
            del_btn.setStyleSheet(
                f"QPushButton{{background:transparent;border:none;border-radius:3px;}}"
                f"QPushButton:hover{{background:{WARN_SOFT};}}")
            del_btn.setToolTip("Eliminar clase")
            del_btn.clicked.connect(lambda: self.deleted.emit(self._sc.id))
            lay.addWidget(del_btn)
        else:
            lay.addSpacing(22)

    def _on_vis_toggled(self, checked: bool):
        self._vis_btn.setIcon(qicon("eye" if checked else "eye-slash",
                                    ACCENT_STRONG if checked else TEXT_MUTE))

    @property
    def visible_in_panel(self) -> bool:
        return self._vis_btn.isChecked()

    @property
    def sc(self) -> SemanticClass:
        return self._sc


class ClassManagerDialog(QDialog):
    """
    Diálogo completo de gestión de schema de clases.

    Retorna via `result()`:
        - schema actualizado
        - set de IDs visibles en el panel lateral
    """

    schema_updated = pyqtSignal(list, set)  # schema, visible_ids

    def __init__(self, schema: List[SemanticClass],
                 counts: dict, parent=None):
        super().__init__(parent)
        self._schema  = [SemanticClass(sc.id, sc.name, sc.color)
                         for sc in schema]  # copia mutable
        self._counts  = counts
        self._rows: List[_ClassRow] = []
        self._next_id = max((sc.id for sc in schema), default=0) + 1

        self.setWindowTitle("Gestión de clases")
        self.setMinimumWidth(500)
        self.setMinimumHeight(420)
        self.setModal(True)
        self.setStyleSheet(f"QDialog{{background:{SURFACE_2};}}")

        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        hdr = QWidget()
        hdr.setStyleSheet(f"background:{SURFACE}; border-bottom:1px solid {BORDER};")
        hdr_lay = QHBoxLayout(hdr)
        hdr_lay.setContentsMargins(16, 12, 16, 12); hdr_lay.setSpacing(8)
        hic = QLabel(); hic.setPixmap(qpixmap("pencil-square", ACCENT_STRONG, 15))
        hdr_lay.addWidget(hic)
        title = QLabel("Gestión de clases")
        title.setStyleSheet(f"color:{ACCENT_STRONG}; font-size:13px; font-weight:700;")
        hdr_lay.addWidget(title)
        hdr_lay.addStretch()
        close_btn = QPushButton()
        close_btn.setIcon(qicon("x-lg", TEXT_MUTE))
        close_btn.setFixedSize(24, 24)
        close_btn.setStyleSheet(
            f"QPushButton{{background:transparent;border:none;border-radius:4px;}}"
            f"QPushButton:hover{{background:{SURFACE_2};}}")
        close_btn.clicked.connect(self.reject)
        hdr_lay.addWidget(close_btn)
        root.addWidget(hdr)

        # Leyenda de columnas
        legend = QWidget()
        legend.setStyleSheet(f"background:{SURFACE_2}; border-bottom:1px solid {BORDER_SOFT};")
        leg_lay = QHBoxLayout(legend)
        leg_lay.setContentsMargins(8, 5, 8, 5); leg_lay.setSpacing(8)
        for text, width in [("VIS", 22), ("ID", 20), ("", 22), ("NOMBRE", 0), ("PUNTOS", 64), ("", 22)]:
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color:{TEXT_MUTE}; font-size:10.5px; font-weight:600; letter-spacing:0.5px;")
            if width: lbl.setFixedWidth(width); lbl.setAlignment(Qt.AlignCenter)
            else: lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            leg_lay.addWidget(lbl)
        root.addWidget(legend)

        # Lista scrollable
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"background:{SURFACE_2};")

        self._list_w = QWidget()
        self._list_w.setStyleSheet(f"background:{SURFACE_2};")
        self._list_lay = QVBoxLayout(self._list_w)
        self._list_lay.setContentsMargins(6, 6, 6, 6)
        self._list_lay.setSpacing(3)
        self._list_lay.addStretch()

        scroll.setWidget(self._list_w)
        root.addWidget(scroll, 1)

        # Toolbar: añadir clase
        add_bar = QWidget()
        add_bar.setStyleSheet(f"background:{SURFACE}; border-top:1px solid {BORDER};")
        add_lay = QHBoxLayout(add_bar)
        add_lay.setContentsMargins(12, 9, 12, 9); add_lay.setSpacing(8)

        self._new_name = QLineEdit()
        self._new_name.setPlaceholderText("nombre de nueva clase…")
        self._new_name.setStyleSheet(
            f"QLineEdit{{background:{SURFACE_2};border:1px solid {BORDER};"
            f"border-radius:4px;color:{TEXT};padding:6px 8px;}}"
            f"QLineEdit:focus{{border-color:{ACCENT};}}")
        self._new_name.returnPressed.connect(self._add_class)

        self._new_color = _ColorSwatch(TEXT_MUTE)

        add_btn = QPushButton("  Añadir clase")
        add_btn.setIcon(qicon("plus-lg", ACCENT_STRONG))
        add_btn.setStyleSheet(
            f"QPushButton{{background:{ACCENT_SOFT};border:1px solid {ACCENT_BORDER};"
            f"border-radius:4px;color:{ACCENT_STRONG};padding:7px 14px;font-size:11px;font-weight:600;}}"
            f"QPushButton:hover{{border-color:{ACCENT};}}")
        add_btn.clicked.connect(self._add_class)

        add_lay.addWidget(self._new_color)
        add_lay.addWidget(self._new_name, 1)
        add_lay.addWidget(add_btn)
        root.addWidget(add_bar)

        # Footer
        ftr = QWidget()
        ftr.setStyleSheet(f"background:{SURFACE}; border-top:1px solid {BORDER};")
        ftr_lay = QHBoxLayout(ftr)
        ftr_lay.setContentsMargins(16, 10, 16, 10)
        ftr_lay.setSpacing(8)
        ftr_lay.addStretch()

        cancel_btn = QPushButton("Cancelar")
        cancel_btn.setStyleSheet(
            f"QPushButton{{background:transparent;border:1px solid {BORDER};border-radius:4px;"
            f"color:{TEXT_DIM};padding:7px 16px;font-size:11px;}}"
            f"QPushButton:hover{{border-color:{TEXT_MUTE};}}")
        cancel_btn.clicked.connect(self.reject)

        apply_btn = QPushButton("Aplicar")
        apply_btn.setIcon(qicon("check2", "#ffffff"))
        apply_btn.setStyleSheet(
            f"QPushButton{{background:{ACCENT};color:#ffffff;border:none;"
            f"border-radius:4px;font-size:11px;font-weight:700;padding:7px 20px;}}"
            f"QPushButton:hover{{background:{ACCENT_STRONG};}}")
        apply_btn.clicked.connect(self._apply)

        ftr_lay.addWidget(cancel_btn)
        ftr_lay.addWidget(apply_btn)
        root.addWidget(ftr)

        # Populate
        self._rebuild_rows()

    def _rebuild_rows(self) -> None:
        while self._list_lay.count() > 1:
            item = self._list_lay.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self._rows.clear()

        for sc in sorted(self._schema, key=lambda s: s.id):
            row = _ClassRow(sc, self._counts.get(sc.id, 0))
            row.deleted.connect(self._delete_class)
            self._rows.append(row)
            self._list_lay.insertWidget(self._list_lay.count() - 1, row)

    def _add_class(self) -> None:
        name = self._new_name.text().strip()
        if not name:
            self._new_name.setFocus()
            return
        color = self._new_color.color
        sc = SemanticClass(self._next_id, name, color)
        self._next_id += 1
        self._schema.append(sc)
        self._new_name.clear()
        self._rebuild_rows()

    def _delete_class(self, class_id: int) -> None:
        cnt = self._counts.get(class_id, 0)
        msg = f"¿Eliminar clase ID {class_id}?"
        if cnt > 0:
            msg += f"\n\n{cnt:,} puntos etiquetados serán marcados como 'sin etiquetar'."
        reply = QMessageBox.question(self, "Eliminar clase", msg,
                                     QMessageBox.Yes | QMessageBox.Cancel)
        if reply == QMessageBox.Yes:
            self._schema = [s for s in self._schema if s.id != class_id]
            self._rebuild_rows()

    def _apply(self) -> None:
        # Recoger schema actualizado desde las filas
        updated: List[SemanticClass] = []
        visible_ids: set = set()
        for row in self._rows:
            updated.append(row.sc)
            if row.visible_in_panel:
                visible_ids.add(row.sc.id)
        self.schema_updated.emit(updated, visible_ids)
        self.accept()
