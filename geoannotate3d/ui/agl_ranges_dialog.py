"""
ui/agl_ranges_dialog.py — Diálogo "Configurar rangos" (AGL por altura).

REDISEÑO 2026-09-05 (v4): el editor de rangos AGL vivía embebido como
tarjetas apiladas dentro del panel angosto de Pre-clasificar (~200-260px
de ancho) — el usuario pidió un diseño de TABLA (Clase | Desde (m) |
Hasta (m), fila por clase, separadores finos) que necesita más ancho
del que ese panel lateral puede dar. Solución: mover la edición real a
un diálogo modal ancho (como `ClassManagerDialog`/`ExportDialog`), y
dejar en el panel lateral solo un resumen compacto de solo lectura +
un botón "Configurar rangos…" que abre este diálogo.

También implementa el pedido de encadenar rangos: el "Hasta" de una
clase se liga automáticamente al "Desde" de la siguiente (y viceversa)
para que los rangos queden siempre continuos sin huecos al ajustar un
límite — igual que el ejemplo de referencia que pasó el usuario.
"""
from __future__ import annotations
from typing import List, Optional

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QWidget, QScrollArea, QFrame, QComboBox, QDoubleSpinBox,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal

from ui.icons import icon as qicon, pixmap as qpixmap
from ui.theme import (
    SURFACE, SURFACE_2, SURFACE_3, BORDER, BORDER_SOFT,
    TEXT, TEXT_DIM, TEXT_MUTE,
    ACCENT, ACCENT_STRONG, ACCENT_SOFT, ACCENT_BORDER,
    OK, OK_SOFT, WARN, WARN_SOFT,
)


class AGLRangesDialog(QDialog):
    """
    Editor de rangos AGL en formato tabla ancha.

    `layers`: lista de dicts {"class_id": int, "lo": float, "hi": float}
    en el orden en que deben aplicarse. Se trabaja sobre una COPIA —
    "Cancelar" descarta los cambios, igual que ClassManagerDialog.
    """

    def __init__(self, schema: list, layers: List[dict], parent=None):
        super().__init__(parent)
        self._schema = schema
        self._layers = [dict(l) for l in layers]   # copia mutable
        self._rows: List[dict] = []   # [{combo, sp_lo, sp_hi, dot, row_widget}]

        self.setWindowTitle("Configurar rangos")
        self.setMinimumWidth(680)
        self.setMinimumHeight(420)
        self.setModal(True)
        self.setStyleSheet(f"QDialog{{background:{SURFACE};}}")

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 18)
        root.setSpacing(4)

        title = QLabel("Configurar rangos")
        title.setStyleSheet(f"color:{TEXT};font-size:16px;font-weight:700;background:transparent;")
        root.addWidget(title)

        subtitle = QLabel(
            "Define los intervalos de altura para cada clase. Los rangos "
            "deben ser continuos y no superponerse.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;background:transparent;")
        root.addSpacing(2)
        root.addWidget(subtitle)
        root.addSpacing(14)

        # ── Encabezado de columnas ──────────────────────────────────────
        hdr = QHBoxLayout(); hdr.setSpacing(14)
        h_clase = QLabel("CLASE")
        h_clase.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;font-weight:700;letter-spacing:0.5px;background:transparent;")
        h_clase.setMinimumWidth(180)
        h_desde = QLabel("DESDE (m)")
        h_desde.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;font-weight:700;letter-spacing:0.5px;background:transparent;")
        h_hasta = QLabel("HASTA (m)")
        h_hasta.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;font-weight:700;letter-spacing:0.5px;background:transparent;")
        hdr.addWidget(h_clase, 2)
        hdr.addWidget(h_desde, 1)
        hdr.addWidget(h_hasta, 1)
        hdr.addSpacing(28)   # espacio del botón eliminar
        root.addLayout(hdr)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"background:{BORDER};max-height:1px;")
        root.addWidget(sep)

        # ── Tabla scrollable ──────────────────────────────────────────────
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background:transparent;")
        self._table_w = QWidget(); self._table_w.setStyleSheet(f"background:{SURFACE};")
        self._table_l = QVBoxLayout(self._table_w)
        self._table_l.setContentsMargins(0, 0, 0, 0); self._table_l.setSpacing(0)
        scroll.setWidget(self._table_w)
        root.addWidget(scroll, 1)

        # Botón agregar
        add_row = QHBoxLayout()
        add_btn = QPushButton("+  Agregar clase")
        add_btn.setIcon(qicon("plus-lg", ACCENT_STRONG))
        add_btn.setStyleSheet(
            f"QPushButton{{background:transparent;border:1px dashed {BORDER};border-radius:4px;"
            f"color:{ACCENT_STRONG};padding:8px;font-size:10.5px;font-weight:600;}}"
            f"QPushButton:hover{{background:{ACCENT_SOFT};border-color:{ACCENT};}}")
        add_btn.clicked.connect(self._add_row)
        add_row.addWidget(add_btn)
        root.addSpacing(8)
        root.addLayout(add_row)

        # ── Aviso de solapamiento ─────────────────────────────────────────
        self._overlap_lbl = QLabel("")
        self._overlap_lbl.setWordWrap(True)
        self._overlap_lbl.setStyleSheet(f"color:{WARN};font-size:10.5px;font-weight:600;background:transparent;")
        root.addWidget(self._overlap_lbl)

        # Poblar la tabla ahora que _overlap_lbl ya existe (_check_overlaps,
        # llamado desde _rebuild_rows, lo necesita).
        self._rebuild_rows()

        # ── Nota informativa ──────────────────────────────────────────────
        note = QFrame(); note.setObjectName("aglNote")
        note.setStyleSheet(f"QFrame#aglNote{{background:{SURFACE_2};border-radius:5px;}}")
        note_l = QHBoxLayout(note); note_l.setContentsMargins(12, 9, 12, 9); note_l.setSpacing(8)
        note_ic = QLabel(); note_ic.setPixmap(qpixmap("info-circle", TEXT_MUTE, 13))
        note_ic.setStyleSheet("background:transparent;")
        note_txt = QLabel("Los rangos se aplicarán en el orden mostrado.")
        note_txt.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;background:transparent;")
        note_l.addWidget(note_ic); note_l.addWidget(note_txt, 1)
        root.addSpacing(10)
        root.addWidget(note)

        # ── Footer ────────────────────────────────────────────────────────
        ftr = QHBoxLayout(); ftr.setSpacing(8)
        ftr.addStretch()
        cancel_btn = QPushButton("Cancelar")
        cancel_btn.setStyleSheet(
            f"QPushButton{{background:transparent;border:1px solid {BORDER};border-radius:4px;"
            f"color:{TEXT_DIM};padding:8px 18px;font-size:10.5px;}}"
            f"QPushButton:hover{{border-color:{TEXT_MUTE};}}")
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton("  Guardar clasificador")
        save_btn.setIcon(qicon("check2", "#ffffff"))
        save_btn.setStyleSheet(
            f"QPushButton{{background:{TEXT};color:#ffffff;border:none;border-radius:4px;"
            f"padding:8px 18px;font-size:10.5px;font-weight:700;}}"
            f"QPushButton:hover{{background:{ACCENT_STRONG};}}")
        save_btn.clicked.connect(self._on_save)
        ftr.addWidget(cancel_btn); ftr.addWidget(save_btn)
        root.addSpacing(14)
        root.addLayout(ftr)

    # ── Filas de la tabla ───────────────────────────────────────────────────

    def _class_color(self, class_id: int) -> str:
        for sc in self._schema:
            if sc.id == class_id: return sc.color
        return TEXT_MUTE

    def _rebuild_rows(self) -> None:
        for i in reversed(range(self._table_l.count())):
            item = self._table_l.itemAt(i)
            if item.widget(): item.widget().deleteLater()
        self._rows = []

        for idx, layer in enumerate(self._layers):
            self._add_row_widget(layer, idx, last=(idx == len(self._layers) - 1))

        self._check_overlaps()

    def _add_row_widget(self, layer: dict, idx: int, last: bool) -> None:
        row = QFrame(); row.setObjectName("aglTableRow")
        row.setStyleSheet(f"QFrame#aglTableRow{{background:transparent;border:none;}}")
        rl = QHBoxLayout(row); rl.setContentsMargins(0, 10, 0, 10); rl.setSpacing(14)

        # Clase: punto de color + combo
        cls_box = QHBoxLayout(); cls_box.setSpacing(8)
        dot = QLabel(); dot.setFixedSize(10, 10)
        dot.setStyleSheet(f"background:{self._class_color(layer['class_id'])};border-radius:5px;")
        combo = QComboBox()
        combo.addItem("(No clasificar)", 0)
        for sc in self._schema:
            if sc.id > 0: combo.addItem(sc.name, sc.id)
        for i in range(combo.count()):
            if combo.itemData(i) == layer["class_id"]:
                combo.setCurrentIndex(i); break
        combo.setStyleSheet(
            f"QComboBox{{background:transparent;border:none;color:{TEXT};"
            f"font-size:11.5px;padding:2px 4px;}}"
            f"QComboBox:hover{{background:{SURFACE_2};border-radius:3px;}}"
            f"QComboBox QAbstractItemView{{background:{SURFACE};color:{TEXT_DIM};"
            f"selection-background-color:{ACCENT_SOFT};selection-color:{ACCENT_STRONG};}}")
        combo.currentIndexChanged.connect(
            lambda _, d=dot, c=combo: d.setStyleSheet(
                f"background:{self._class_color(c.currentData() or 0)};border-radius:5px;"))
        cls_box.addWidget(dot); cls_box.addWidget(combo, 1)
        cls_w = QWidget(); cls_w.setLayout(cls_box); cls_w.setStyleSheet("background:transparent;")
        rl.addWidget(cls_w, 2)

        # Desde
        sp_lo = QDoubleSpinBox()
        sp_lo.setRange(-50, 9990); sp_lo.setDecimals(2); sp_lo.setSingleStep(0.1)
        sp_lo.setValue(layer["lo"])
        sp_lo.setStyleSheet(
            f"QDoubleSpinBox{{background:{SURFACE};border:1px solid {BORDER};border-radius:4px;"
            f"color:{TEXT_DIM};padding:6px 8px;font-size:11px;}}"
            f"QDoubleSpinBox:hover{{border-color:{ACCENT};}}"
            f"QDoubleSpinBox:focus{{border-color:{ACCENT};}}")
        rl.addWidget(sp_lo, 1)

        # Hasta
        sp_hi = QDoubleSpinBox()
        sp_hi.setRange(-50, 9999); sp_hi.setDecimals(2); sp_hi.setSingleStep(0.1)
        if layer["hi"] >= 9998:
            sp_hi.setValue(9999); sp_hi.setSpecialValueText("sin lím.")
        else:
            sp_hi.setValue(layer["hi"])
        sp_hi.setStyleSheet(sp_lo.styleSheet())
        rl.addWidget(sp_hi, 1)

        # Eliminar
        del_btn = QPushButton()
        del_btn.setIcon(qicon("trash", TEXT_MUTE))
        del_btn.setFixedSize(24, 24)
        del_btn.setStyleSheet(
            f"QPushButton{{background:transparent;border:none;border-radius:4px;}}"
            f"QPushButton:hover{{background:{WARN_SOFT};}}")
        del_btn.setToolTip("Eliminar este rango")
        rl.addWidget(del_btn)

        entry = {"row": row, "combo": combo, "sp_lo": sp_lo, "sp_hi": sp_hi, "dot": dot}
        self._rows.append(entry)

        # Encadenar: cambiar "Hasta" empuja el "Desde" de la fila SIGUIENTE;
        # cambiar "Desde" empuja el "Hasta" de la fila ANTERIOR — así los
        # rangos quedan continuos automáticamente (pedido explícito del
        # usuario, como en la imagen de referencia).
        def on_hi_changed(v, i=idx):
            if i + 1 < len(self._rows):
                nxt = self._rows[i + 1]["sp_lo"]
                if abs(nxt.value() - v) > 1e-6:
                    nxt.blockSignals(True); nxt.setValue(v); nxt.blockSignals(False)
            self._check_overlaps()

        def on_lo_changed(v, i=idx):
            if i - 1 >= 0:
                prev = self._rows[i - 1]["sp_hi"]
                if abs(prev.value() - v) > 1e-6:
                    prev.blockSignals(True); prev.setValue(v); prev.blockSignals(False)
            self._check_overlaps()

        sp_hi.valueChanged.connect(on_hi_changed)
        sp_lo.valueChanged.connect(on_lo_changed)
        del_btn.clicked.connect(lambda: self._remove_row(idx))

        self._table_l.addWidget(row)
        if not last:
            rsep = QFrame(); rsep.setFrameShape(QFrame.HLine)
            rsep.setStyleSheet(f"background:{BORDER_SOFT};max-height:1px;")
            self._table_l.addWidget(rsep)

    def _add_row(self) -> None:
        # Nueva clase: Desde = Hasta de la última fila (continuidad)
        lo = self._rows[-1]["sp_hi"].value() if self._rows else 0.0
        self._layers = self._collect() + [{"class_id": 0, "lo": lo, "hi": lo + 5.0}]
        self._rebuild_rows()

    def _remove_row(self, idx: int) -> None:
        layers = self._collect()
        if 0 <= idx < len(layers):
            layers.pop(idx)
        # Re-encadenar para cerrar el hueco que deja la clase eliminada.
        # Antes, borrar una clase intermedia (p.ej. 1-2, 2-3, 3-4 → borrar
        # 2-3) dejaba las restantes con un hueco (1-2, 3-4) — pedido
        # explícito del usuario: deben reacomodarse para encajar entre sí
        # (1-2, 2-3), evitando huecos y solapamientos. Se preserva el
        # ANCHO (hi-lo) original de cada rango restante, solo se desliza
        # su "Desde" para que empiece donde termina el rango anterior;
        # la primera fila queda fija como ancla.
        for i in range(1, len(layers)):
            width = layers[i]["hi"] - layers[i]["lo"]
            layers[i]["lo"] = layers[i - 1]["hi"]
            layers[i]["hi"] = layers[i]["lo"] + width
        self._layers = layers
        self._rebuild_rows()

    def _collect(self) -> List[dict]:
        out = []
        for r in self._rows:
            hi = r["sp_hi"].value()
            if hi >= 9998: hi = 9999.0
            out.append({
                "class_id": r["combo"].currentData() or 0,
                "lo": r["sp_lo"].value(),
                "hi": hi,
            })
        return out

    def _check_overlaps(self) -> None:
        layers = self._collect()
        overlaps = []
        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                if layers[i]["lo"] < layers[j]["hi"] and layers[j]["lo"] < layers[i]["hi"]:
                    overlaps.append((i + 1, j + 1))
        if overlaps:
            pairs = ", ".join(f"{a} y {b}" for a, b in overlaps[:3])
            self._overlap_lbl.setText(f"⚠ Rangos {pairs} se solapan")
        else:
            self._overlap_lbl.setText("")

    def _on_save(self) -> None:
        self._layers = self._collect()
        self.accept()

    def get_layers(self) -> List[dict]:
        """Lista final [{class_id, lo, hi}] tras aceptar el diálogo."""
        return self._layers
