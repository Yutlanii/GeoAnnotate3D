"""
ui/overlay_panel.py — Panel de gestión de capas superpuestas (vector y raster)

Muestra la lista de capas cargadas con:
  - Nombre y tipo (vector / raster)
  - Estado (OK, advertencia, error)
  - Slider de opacidad
  - Toggle de visibilidad
  - Botón de eliminar

REDISEÑO 2026-09-05: cada fila era UNA sola línea horizontal con hasta
7 controles apretados (icono + nombre + estado + checkbox + slider +
slider-Z + borrar) — en el ancho real del panel (~200-230px) eso se
recortaba o quedaba ilegible. Reescrito como tarjeta de 2-3 líneas
(nombre+estado+borrar / visibilidad+opacidad / Z-offset si es raster),
mismo patrón `_card()` que el resto de paneles, iconos reales en vez
de emoji/glifos (⬟ ⬛ ⚠ ℹ ⬆).
"""
from __future__ import annotations
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QScrollArea, QFrame, QFileDialog, QMessageBox,
    QCheckBox,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon

from ui.icons import icon as qicon, pixmap as qpixmap
from ui.theme import (
    SURFACE, SURFACE_2, SURFACE_3, BORDER, BORDER_SOFT,
    TEXT, TEXT_DIM, TEXT_MUTE,
    ACCENT, ACCENT_STRONG, ACCENT_SOFT, ACCENT_BORDER,
    OK, OK_SOFT, WARN, WARN_SOFT, INFO_TEAL,
)

VECTOR_EXTS = (".shp", ".geojson", ".gpkg", ".kml", ".dxf")
RASTER_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


class _LayerRow(QFrame):
    removed    = pyqtSignal(object)   # emite la OverlayLayer
    toggled    = pyqtSignal(object, bool)
    opacity_c  = pyqtSignal(object, float)
    z_offset_c = pyqtSignal(object, float)

    def __init__(self, layer, parent=None):
        super().__init__(parent)
        self._layer = layer
        self.setObjectName("layerRow")
        self.setStyleSheet(
            f"QFrame#layerRow{{background:{SURFACE};border:1px solid {BORDER_SOFT};border-radius:4px;}}")
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 7); outer.setSpacing(5)

        # ── Línea 1: icono + nombre + estado + borrar ────────────────────────
        top = QHBoxLayout(); top.setSpacing(6)
        icon_name = "diagram-3" if self._layer.kind == "vector" else "layers"
        icon_color = ACCENT_STRONG if self._layer.kind == "vector" else TEXT_MUTE
        ic = QLabel(); ic.setPixmap(qpixmap(icon_name, icon_color, 13))
        ic.setStyleSheet("background:transparent;")
        top.addWidget(ic)

        name_lbl = QLabel(self._layer.name)
        name_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:10.5px;font-weight:600;background:transparent;")
        name_lbl.setWordWrap(True)
        top.addWidget(name_lbl, 1)

        if not self._layer.ok:
            status = QLabel(); status.setPixmap(qpixmap("exclamation-triangle", WARN, 12))
            status.setStyleSheet("background:transparent;")
            status.setToolTip("\n".join(self._layer.warnings))
            top.addWidget(status)
        elif self._layer.warnings:
            status = QLabel(); status.setPixmap(qpixmap("info-circle", ACCENT, 12))
            status.setStyleSheet("background:transparent;")
            status.setToolTip("\n".join(self._layer.warnings))
            top.addWidget(status)

        del_btn = QPushButton()
        del_btn.setIcon(qicon("trash", WARN)); del_btn.setFixedSize(22, 20)
        del_btn.setStyleSheet(
            f"QPushButton{{background:{WARN_SOFT};border:1px solid {WARN_SOFT};border-radius:3px;}}"
            f"QPushButton:hover{{border-color:{WARN};}}")
        del_btn.setToolTip("Eliminar capa")
        del_btn.clicked.connect(lambda: self.removed.emit(self._layer))
        top.addWidget(del_btn)
        outer.addLayout(top)

        # ── Línea 2: visibilidad + opacidad ───────────────────────────────────
        mid = QHBoxLayout(); mid.setSpacing(6)
        self._vis_btn = QPushButton()
        self._vis_btn.setIcon(qicon("eye", ACCENT_STRONG)); self._vis_btn.setFixedSize(22, 20)
        self._vis_btn.setCheckable(True); self._vis_btn.setChecked(True)
        self._vis_btn.setStyleSheet(
            f"QPushButton{{background:{SURFACE_2};border:1px solid {BORDER};border-radius:3px;}}"
            f"QPushButton:hover{{border-color:{ACCENT};}}"
            f"QPushButton:!checked{{background:{SURFACE_3};}}")
        self._vis_btn.setToolTip("Mostrar/ocultar capa")
        self._vis_btn.toggled.connect(self._on_vis_toggled)
        mid.addWidget(self._vis_btn)

        op_lbl = QLabel("Opacidad")
        op_lbl.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;background:transparent;")
        mid.addWidget(op_lbl)

        sl = QSlider(Qt.Horizontal)
        sl.setRange(10, 100); sl.setValue(85)
        sl.setStyleSheet(
            f"QSlider::groove:horizontal{{background:{SURFACE_3};height:3px;border-radius:1px;}}"
            f"QSlider::handle:horizontal{{background:{ACCENT};width:9px;height:9px;"
            f"margin:-3px 0;border-radius:4px;}}"
            f"QSlider::sub-page:horizontal{{background:{ACCENT_BORDER};border-radius:1px;}}")
        sl.valueChanged.connect(lambda v: self.opacity_c.emit(self._layer, v / 100.0))
        mid.addWidget(sl, 1)
        outer.addLayout(mid)

        # ── Línea 3: Z-offset (solo raster) ───────────────────────────────────
        if self._layer.kind == "raster":
            zrow = QHBoxLayout(); zrow.setSpacing(6)
            z_lbl = QLabel("Altura Z")
            z_lbl.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;background:transparent;min-width:28px;")
            zrow.addWidget(z_lbl)
            z_sl = QSlider(Qt.Horizontal)
            z_sl.setRange(-200, 500)   # -20m a +50m en décimas
            z_sl.setValue(0)
            z_sl.setToolTip("Ajustar altura Z del raster\n(subir para verlo junto a la nube)")
            z_sl.setStyleSheet(
                f"QSlider::groove:horizontal{{background:{SURFACE_3};height:3px;border-radius:1px;}}"
                f"QSlider::handle:horizontal{{background:{ACCENT_STRONG};width:9px;height:9px;"
                f"margin:-3px 0;border-radius:4px;}}")
            z_sl.valueChanged.connect(lambda v: self.z_offset_c.emit(self._layer, v * 0.1))
            zrow.addWidget(z_sl, 1)
            outer.addLayout(zrow)

    def _on_vis_toggled(self, checked: bool):
        self._vis_btn.setIcon(qicon("eye" if checked else "eye-slash", ACCENT_STRONG if checked else TEXT_MUTE))
        self.toggled.emit(self._layer, checked)


class OverlayPanel(QWidget):
    """Panel lateral de gestión de capas superpuestas."""

    layer_added      = pyqtSignal(object)
    layer_removed    = pyqtSignal(object)
    layer_toggled    = pyqtSignal(object, bool)
    layer_opacity    = pyqtSignal(object, float)
    layer_z_offset   = pyqtSignal(object, float)
    top_view_clicked = pyqtSignal()
    bg_opacity_changed = pyqtSignal(float)   # transparencia fondo VTK
    render_needed    = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list = []
        self._build_ui()

    def _card(self, title: str, icon_name: str = None):
        frame = QFrame(); frame.setObjectName("ovCard")
        frame.setStyleSheet(
            f"QFrame#ovCard{{background:{SURFACE};border:1px solid {BORDER};border-radius:4px;}}")
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(11, 10, 11, 11); outer.setSpacing(7)
        hdr_row = QHBoxLayout(); hdr_row.setSpacing(6)
        if icon_name:
            ic = QLabel(); ic.setStyleSheet("background:transparent;")
            ic.setPixmap(qpixmap(icon_name, ACCENT_STRONG, 13))
            hdr_row.addWidget(ic)
        title_lbl = QLabel(title); title_lbl.setWordWrap(True)
        title_lbl.setStyleSheet(f"color:{ACCENT_STRONG};font-size:10.5px;font-weight:700;background:transparent;")
        hdr_row.addWidget(title_lbl, 1)
        outer.addLayout(hdr_row)
        content = QVBoxLayout(); content.setSpacing(6)
        outer.addLayout(content)
        return frame, content

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(9)
        self.setStyleSheet(f"background:{SURFACE_2};")

        # Cabecera
        hdr = QHBoxLayout(); hdr.setSpacing(6)
        hic = QLabel(); hic.setPixmap(qpixmap("layers", ACCENT_STRONG, 14))
        hdr.addWidget(hic)
        title = QLabel("Capas superpuestas")
        title.setStyleSheet(f"color:{ACCENT_STRONG};font-size:11px;font-weight:700;")
        hdr.addWidget(title); hdr.addStretch()
        root.addLayout(hdr)

        # Botones de carga
        btn_row = QHBoxLayout(); btn_row.setSpacing(6)

        self._btn_vec = QPushButton("  Vector")
        self._btn_vec.setIcon(qicon("diagram-3", ACCENT_STRONG))
        self._btn_vec.setToolTip(
            "Cargar capa vectorial\n.shp, .geojson, .gpkg, .kml, .dxf")
        self._btn_vec.setStyleSheet(
            f"QPushButton{{background:{ACCENT_SOFT};border:1px solid {ACCENT_BORDER};"
            f"border-radius:4px;color:{ACCENT_STRONG};padding:6px;font-size:10.5px;font-weight:600;}}"
            f"QPushButton:hover{{border-color:{ACCENT};}}")
        self._btn_vec.clicked.connect(self._on_add_vector)
        btn_row.addWidget(self._btn_vec)

        self._btn_ras = QPushButton("  Raster")
        self._btn_ras.setIcon(qicon("layers", OK))
        self._btn_ras.setToolTip(
            "Cargar capa raster / ortomosaico\n.tif, .tiff, .png, .jpg")
        self._btn_ras.setStyleSheet(
            f"QPushButton{{background:{OK_SOFT};border:1px solid {OK_SOFT};"
            f"border-radius:4px;color:{OK};padding:6px;font-size:10.5px;font-weight:600;}}"
            f"QPushButton:hover{{border-color:{OK};}}")
        self._btn_ras.clicked.connect(self._on_add_raster)
        btn_row.addWidget(self._btn_ras)
        root.addLayout(btn_row)

        # Hint de dependencias
        self._deps_hint = QLabel(
            "Requiere: pip install geopandas rasterio\n"
            "Para transformación CRS: pip install pyproj")
        self._deps_hint.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;")
        self._deps_hint.setWordWrap(True)
        root.addWidget(self._deps_hint)

        # Lista de capas
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background:transparent;")
        self._list_widget = QWidget()
        self._list_widget.setStyleSheet("background:transparent;")
        self._list_vlay = QVBoxLayout(self._list_widget)
        self._list_vlay.setContentsMargins(0, 0, 0, 0)
        self._list_vlay.setSpacing(6)
        self._list_vlay.addStretch()
        scroll.setWidget(self._list_widget)
        root.addWidget(scroll, 1)

        self._empty_lbl = QLabel("Sin capas cargadas")
        self._empty_lbl.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;padding:10px;")
        self._empty_lbl.setAlignment(Qt.AlignCenter)
        self._list_vlay.insertWidget(0, self._empty_lbl)

        # Herramientas de vista
        gb_view, vl = self._card("Vista y aspecto", "compass")

        # Botón vista cenital
        top_btn = QPushButton("  Vista cenital (top-down)")
        top_btn.setIcon(qicon("arrow-bar-up", ACCENT_STRONG))
        top_btn.setToolTip(
            "Posiciona la cámara mirando desde arriba\n"
            "para ver la nube y el raster alineados")
        top_btn.setStyleSheet(
            f"QPushButton{{background:{ACCENT_SOFT};border:1px solid {ACCENT_BORDER};"
            f"border-radius:4px;color:{ACCENT_STRONG};padding:7px;font-size:10.5px;"
            f"font-weight:600;text-align:left;}}"
            f"QPushButton:hover{{border-color:{ACCENT};}}")
        top_btn.clicked.connect(self.top_view_clicked)
        vl.addWidget(top_btn)

        # Opacidad de la nube de puntos (para ver el raster debajo)
        cloud_op_lbl = QLabel("Opacidad nube de puntos")
        cloud_op_lbl.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;background:transparent;")
        vl.addWidget(cloud_op_lbl)

        cloud_op_row = QHBoxLayout(); cloud_op_row.setSpacing(6)
        self._cloud_op_sl = QSlider(Qt.Horizontal)
        self._cloud_op_sl.setRange(0, 100); self._cloud_op_sl.setValue(100)
        self._cloud_op_sl.setStyleSheet(
            f"QSlider::groove:horizontal{{background:{SURFACE_3};height:4px;border-radius:2px;}}"
            f"QSlider::handle:horizontal{{background:{ACCENT};width:11px;height:11px;"
            f"margin:-4px 0;border-radius:3px;}}"
            f"QSlider::sub-page:horizontal{{background:{ACCENT_BORDER};border-radius:2px;}}")
        self._cloud_op_val = QLabel("100%")
        self._cloud_op_val.setStyleSheet(f"color:{ACCENT_STRONG};font-size:10.5px;min-width:34px;font-weight:600;")
        self._cloud_op_sl.valueChanged.connect(
            lambda v: (self._cloud_op_val.setText(f"{v}%"),
                       self.bg_opacity_changed.emit(v / 100.0)))
        cloud_op_row.addWidget(self._cloud_op_sl, 1); cloud_op_row.addWidget(self._cloud_op_val)
        vl.addLayout(cloud_op_row)

        hint = QLabel(
            "Baja la opacidad de la nube para ver mejor el "
            "raster/ortomosaico debajo.")
        hint.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;background:transparent;")
        hint.setWordWrap(True)
        vl.addWidget(hint)

        root.addWidget(gb_view)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_add_vector(self):
        exts = " ".join(f"*{e}" for e in VECTOR_EXTS)
        path, _ = QFileDialog.getOpenFileName(
            self, "Cargar capa vectorial", "",
            f"Archivos vectoriales ({exts});;Todos los archivos (*)")
        if path:
            self.layer_added.emit(("vector", path))

    def _on_add_raster(self):
        exts = " ".join(f"*{e}" for e in RASTER_EXTS)
        path, _ = QFileDialog.getOpenFileName(
            self, "Cargar ortomosaico / raster", "",
            f"Archivos raster ({exts});;Todos los archivos (*)")
        if path:
            self.layer_added.emit(("raster", path))

    # ── API pública ────────────────────────────────────────────────────────────

    def add_layer_ui(self, layer) -> None:
        """Añade una fila en la lista para la capa cargada."""
        self._empty_lbl.setVisible(False)
        row = _LayerRow(layer)
        row.removed.connect(self._on_row_removed)
        row.toggled.connect(self.layer_toggled)
        row.opacity_c.connect(self.layer_opacity)
        row.z_offset_c.connect(self.layer_z_offset)
        # Insert before the stretch
        idx = max(0, self._list_vlay.count() - 1)
        self._list_vlay.insertWidget(idx, row)
        self._rows.append((layer, row))

        if layer.warnings:
            kind = "Error" if not layer.ok else "Aviso"
            msg = "\n".join(layer.warnings)
            if not layer.ok:
                QMessageBox.warning(self, f"{kind} al cargar {layer.name}", msg)
            else:
                QMessageBox.information(self, f"{kind} — {layer.name}", msg)

    def _on_row_removed(self, layer):
        self.layer_removed.emit(layer)
        for i, (l, row) in enumerate(self._rows):
            if l is layer:
                row.setParent(None)
                self._rows.pop(i)
                break
        if not self._rows:
            self._empty_lbl.setVisible(True)
