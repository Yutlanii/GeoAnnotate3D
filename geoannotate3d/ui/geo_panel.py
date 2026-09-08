"""
ui/geo_panel.py — Panel geoespacial: detección de terreno y clasificación AGL.

REDISEÑO 2026-09-05 (v2 — paleta gris/cian + corrección de recortes):
  - Las secciones eran QGroupBox con título de texto plano — igual que los
    QLabel, un QGroupBox NO envuelve su título por defecto: en el ancho
    normal del panel ("PASO 1  Detectar el terreno", "Clasificar Suelo —
    CSF (Cloth Simulation Filter)") se recortaba en seco. Reemplazadas por
    tarjetas propias (mismo patrón que ToolPanel/TilePanel) con encabezado
    en QLabel con word-wrap.
  - La tabla de capas AGL tenía 3 columnas fijas (85px + 85px + combo) en
    una sola fila — no cabía en el ancho del panel y todo se recortaba
    ("Desde (m AGL) Hasta (m AGL) Clas..."). Ahora cada capa ocupa DOS
    líneas: rango arriba, clase + eliminar abajo.
  - Textos largos acortados donde clip; los que quedaron largos llevan
    word-wrap.

REDISEÑO 2026-09-05 (v4 — editor de rangos movido a diálogo ancho):
  - El usuario pidió un formato de tabla (Clase | Desde (m) | Hasta (m))
    con encadenado de límites (el "Hasta" de una clase liga con el
    "Desde" de la siguiente) siguiendo una imagen de referencia — ese
    formato de tabla necesita más ancho del que este panel lateral
    puede dar de forma realista. La edición real vive ahora en
    `ui/agl_ranges_dialog.py::AGLRangesDialog` (diálogo modal ancho,
    como ClassManagerDialog); aquí solo queda un resumen compacto de
    solo lectura + el botón "Configurar rangos…".
"""
from __future__ import annotations
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QFrame, QScrollArea, QComboBox,
    QSizePolicy, QDoubleSpinBox, QSpinBox,
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QPainter, QLinearGradient, QColor

from ui.icons import icon as qicon, pixmap as qpixmap
from ui.theme import (
    SURFACE, SURFACE_2, SURFACE_3, BORDER, BORDER_SOFT,
    TEXT, TEXT_DIM, TEXT_MUTE,
    ACCENT, ACCENT_STRONG, ACCENT_SOFT, ACCENT_BORDER,
    OK, OK_SOFT, WARN, WARN_SOFT,
)
from ui.agl_ranges_dialog import AGLRangesDialog


class _AGLBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(14)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._min_v = 0.; self._max_v = 30.
        self._lo = 0.; self._hi = 5.

    def set_range(self, mn, mx): self._min_v=mn; self._max_v=mx; self.update()
    def set_selection(self, lo, hi): self._lo=lo; self._hi=hi; self.update()

    def paintEvent(self, e):
        p = QPainter(self); w, h = self.width(), self.height()
        # Gradiente terreno→dosel — dato, no tema (ver NOTES_CLAUDE.md #23/24)
        gr = QLinearGradient(0,0,w,0)
        gr.setColorAt(0.0, QColor("#5a3518")); gr.setColorAt(0.1, QColor("#8c6018"))
        gr.setColorAt(0.2, QColor("#78a018")); gr.setColorAt(0.5, QColor("#1a6c0e"))
        gr.setColorAt(0.8, QColor("#88b020")); gr.setColorAt(1.0, QColor("#e8e880"))
        p.fillRect(0,0,w,h,gr)
        span = max(self._max_v - self._min_v, 0.001)
        x1 = max(0, min(w, int(w*(self._lo-self._min_v)/span)))
        x2 = max(0, min(w, int(w*(self._hi-self._min_v)/span)))
        if x2 > x1:
            p.fillRect(x1, 0, x2-x1, h, QColor(255,200,50,100))
            p.setPen(QColor(255,180,30,220)); p.drawRect(x1,0,x2-x1,h-1)
        p.end()


class GeoPanel(QWidget):
    agl_select_requested        = pyqtSignal(float, float)
    rules_apply_all_requested   = pyqtSignal()
    agl_auto_classify_requested = pyqtSignal()
    csf_classify_requested      = pyqtSignal()
    sor_detect_requested        = pyqtSignal(int, float)   # (k, std_ratio)

    _DEFAULT_LAYERS = [
        ("suelo",    ["suelo","ground","terreno","tierra","floor"],      -0.5,  0.3),
        ("veg_baja", ["veg. baja","veg baja","low veg","hierba","pasto"], 0.3,  2.0),
        ("veg_med",  ["veg. media","arbusto","shrub","mid veg"],          2.0,  5.0),
        ("arbol",    ["arbol","árbol","tree","forest","bosque","veg. alta"], 5.0, 999.),
        ("edificio", ["edificio","building","construcc"],                  0.5, 999.),
        ("vehiculo", ["vehiculo","vehículo","car","vehicle","coche"],      0.0,  3.5),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._schema: list = []
        self._rules:  list = []
        self._agl_data: list = []   # [{class_id, lo, hi}] — ver AGLRangesDialog
        self._build_ui()

    # ── construcción de la UI ─────────────────────────────────────────────────

    def _card(self, title: str, icon_name: str = None):
        """Tarjeta con encabezado que SÍ envuelve (ver docstring del módulo)."""
        frame = QFrame(); frame.setObjectName("geoCard")
        frame.setStyleSheet(
            f"QFrame#geoCard{{background:{SURFACE};border:1px solid {BORDER};border-radius:4px;}}")
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(11, 10, 11, 11)
        outer.setSpacing(7)

        hdr_row = QHBoxLayout(); hdr_row.setSpacing(6)
        if icon_name:
            ic = QLabel(); ic.setStyleSheet("background:transparent;border:none;")
            ic.setPixmap(qpixmap(icon_name, ACCENT_STRONG, 14))
            ic.setFixedWidth(14)
            hdr_row.addWidget(ic)
        title_lbl = QLabel(title)
        title_lbl.setWordWrap(True)
        title_lbl.setStyleSheet(f"color:{ACCENT_STRONG};font-size:11px;font-weight:700;background:transparent;")
        hdr_row.addWidget(title_lbl, 1)
        outer.addLayout(hdr_row)

        content = QVBoxLayout(); content.setSpacing(7)
        outer.addLayout(content)
        return frame, content

    def _desc(self, text: str) -> QLabel:
        d = QLabel(text)
        d.setWordWrap(True)
        d.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;background:transparent;")
        return d

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background:transparent;")
        content = QWidget(); content.setStyleSheet(f"background:{SURFACE_2};")
        lay = QVBoxLayout(content)
        lay.setContentsMargins(10,12,10,12); lay.setSpacing(10)

        # ── PASO 1: clasificar por capas AGL ──────────────────────────────────
        # (El antiguo "PASO 1 — Detectar el terreno" se quitó: no hacía
        # nada realmente — la clasificación AGL calcula la altura relativa
        # al punto más bajo de la nube directamente, sin usar un MDT. Ver
        # NOTES_CLAUDE.md.)
        gb2, l2 = self._card("PASO 1 — Clasificar por altura (AGL)", "layers-half")

        l2.addWidget(self._desc(
            "Asigna una clase a cada rango de altura sobre el suelo. "
            "Pulsa \"Configurar rangos…\" para editarlos en una tabla."))

        # Resumen compacto de solo lectura — la edición real vive en
        # AGLRangesDialog (ver docstring del módulo).
        self._layers_widget = QWidget()
        self._layers_widget.setStyleSheet("background:transparent;")
        self._layers_vlay = QVBoxLayout(self._layers_widget)
        self._layers_vlay.setContentsMargins(0,2,0,0); self._layers_vlay.setSpacing(3)
        l2.addWidget(self._layers_widget)

        configure_btn = QPushButton("  Configurar rangos…")
        configure_btn.setIcon(qicon("sliders2", ACCENT_STRONG))
        configure_btn.setStyleSheet(
            f"QPushButton{{background:{ACCENT_SOFT};border:1px solid {ACCENT_BORDER};"
            f"border-radius:4px;color:{ACCENT_STRONG};padding:8px;font-size:10.5px;"
            f"font-weight:600;text-align:left;}}"
            f"QPushButton:hover{{border-color:{ACCENT};}}")
        configure_btn.clicked.connect(self._open_ranges_dialog)
        l2.addWidget(configure_btn)
        self._refresh_layers_summary()   # placeholder inicial (sin schema aún)

        self._agl_auto_btn = QPushButton("  Clasificar por capas AGL")
        self._agl_auto_btn.setIcon(qicon("cloud-check", "#ffffff"))
        self._agl_auto_btn.setStyleSheet(
            f"QPushButton{{background:{ACCENT};color:#ffffff;border:none;"
            f"border-radius:4px;padding:9px;font-size:10.5px;font-weight:600;text-align:left;}}"
            f"QPushButton:hover{{background:{ACCENT_STRONG};}}"
            f"QPushButton:disabled{{background:{ACCENT_SOFT};color:{TEXT_MUTE};border:none;}}")
        self._agl_auto_btn.setEnabled(False)
        self._agl_auto_btn.clicked.connect(self.agl_auto_classify_requested)
        l2.addWidget(self._agl_auto_btn)
        lay.addWidget(gb2)

        # ── PASO 2: clasificar suelo por simulación de tela (CSF) ─────────────
        gb3, l3 = self._card("PASO 2 — Suelo por simulación de tela (CSF)", "grid-3x3")

        l3.addWidget(self._desc(
            "Cloth Simulation Filter: simula una tela cayendo sobre la nube "
            "invertida para detectar el suelo. Mismo algoritmo que CloudCompare."))

        # Clase objetivo
        cr = QHBoxLayout(); cr.setSpacing(6)
        cr_lbl = QLabel("Asignar a clase:")
        cr_lbl.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;background:transparent;")
        cr.addWidget(cr_lbl)
        self._csf_combo = QComboBox()
        self._csf_combo.setStyleSheet(
            f"QComboBox{{background:{SURFACE_2};border:1px solid {BORDER};border-radius:3px;"
            f"color:{TEXT_DIM};padding:3px 6px;font-size:10.5px;}}"
            f"QComboBox:hover{{border-color:{ACCENT};}}"
            f"QComboBox QAbstractItemView{{background:{SURFACE};color:{TEXT_DIM};"
            f"selection-background-color:{ACCENT_SOFT};selection-color:{ACCENT_STRONG};}}")
        cr.addWidget(self._csf_combo, 1); l3.addLayout(cr)

        # Resolución de la tela
        self._csf_resolution = QDoubleSpinBox()
        self._csf_resolution.setRange(0.1, 10.0); self._csf_resolution.setValue(0.5)
        self._csf_resolution.setSuffix(" m"); self._csf_resolution.setSingleStep(0.1)
        self._csf_resolution.setDecimals(1)
        self._csf_resolution.setStyleSheet(
            f"QDoubleSpinBox{{background:{SURFACE_2};border:1px solid {BORDER};"
            f"border-radius:3px;color:{TEXT_DIM};padding:2px 4px;font-size:10.5px;}}"
            f"QDoubleSpinBox:hover{{border-color:{ACCENT};}}")
        res_row = QHBoxLayout(); res_row.setSpacing(6)
        res_lbl = QLabel("Resolución tela:")
        res_lbl.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;background:transparent;")
        res_row.addWidget(res_lbl); res_row.addWidget(self._csf_resolution, 1)
        l3.addLayout(res_row)

        csf_hint = QLabel("Menor valor = más detalle (más lento)")
        csf_hint.setWordWrap(True)
        csf_hint.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;background:transparent;")
        l3.addWidget(csf_hint)

        self._csf_btn = QPushButton("  Detectar suelo (CSF)")
        self._csf_btn.setIcon(qicon("grid-3x3", OK))
        self._csf_btn.setEnabled(False)
        self._csf_btn.setStyleSheet(
            f"QPushButton{{background:{OK_SOFT};border:1px solid {OK_SOFT};"
            f"border-radius:4px;color:{OK};padding:8px;font-size:10.5px;font-weight:600;}}"
            f"QPushButton:hover{{border-color:{OK};}}"
            f"QPushButton:disabled{{background:{SURFACE_2};color:{TEXT_MUTE};border-color:{SURFACE_2};}}")
        self._csf_btn.clicked.connect(self.csf_classify_requested)
        l3.addWidget(self._csf_btn)
        lay.addWidget(gb3)

        # ── PASO 3: limpiar ruido (Statistical Outlier Removal) ───────────────
        # Mismo algoritmo que CloudCompare/PCL — pensado para combinar con la
        # herramienta de "eliminar puntos" ya existente: detecta candidatos a
        # ruido del sensor (puntos anormalmente aislados) y ofrece borrarlos.
        gb4, l4 = self._card("PASO 3 — Limpiar ruido (SOR)", "eraser")

        l4.addWidget(self._desc(
            "Statistical Outlier Removal: detecta puntos anormalmente "
            "aislados de sus vecinos (ruido del sensor) y los ofrece "
            "para eliminar — mismo algoritmo que CloudCompare."))

        k_row = QHBoxLayout(); k_row.setSpacing(6)
        k_lbl = QLabel("Vecinos (k):")
        k_lbl.setStyleSheet(f"color:{TEXT_MUTE};font-size:10px;background:transparent;")
        k_row.addWidget(k_lbl)
        self._sor_k = QSpinBox()
        self._sor_k.setRange(3, 50); self._sor_k.setValue(8)
        self._sor_k.setStyleSheet(
            f"QSpinBox{{background:{SURFACE_2};border:1px solid {BORDER};border-radius:3px;"
            f"color:{TEXT_DIM};padding:3px 6px;font-size:10px;}}"
            f"QSpinBox:hover{{border-color:{ACCENT};}}")
        k_row.addWidget(self._sor_k, 1)
        l4.addLayout(k_row)

        ratio_row = QHBoxLayout(); ratio_row.setSpacing(6)
        ratio_lbl = QLabel("Sensibilidad:")
        ratio_lbl.setStyleSheet(f"color:{TEXT_MUTE};font-size:10px;background:transparent;")
        ratio_row.addWidget(ratio_lbl)
        self._sor_ratio = QDoubleSpinBox()
        self._sor_ratio.setRange(0.5, 5.0); self._sor_ratio.setValue(2.0)
        self._sor_ratio.setSingleStep(0.1); self._sor_ratio.setDecimals(1)
        self._sor_ratio.setStyleSheet(
            f"QDoubleSpinBox{{background:{SURFACE_2};border:1px solid {BORDER};border-radius:3px;"
            f"color:{TEXT_DIM};padding:3px 6px;font-size:10px;}}"
            f"QDoubleSpinBox:hover{{border-color:{ACCENT};}}")
        ratio_row.addWidget(self._sor_ratio, 1)
        l4.addLayout(ratio_row)

        sor_hint = QLabel("Menor sensibilidad = detecta más ruido (más agresivo)")
        sor_hint.setWordWrap(True)
        sor_hint.setStyleSheet(f"color:{TEXT_MUTE};font-size:9.5px;background:transparent;")
        l4.addWidget(sor_hint)

        self._sor_btn = QPushButton("  Detectar ruido (SOR)")
        self._sor_btn.setIcon(qicon("eraser", WARN))
        self._sor_btn.setStyleSheet(
            f"QPushButton{{background:{WARN_SOFT};border:1px solid {WARN_SOFT};"
            f"border-radius:4px;color:{WARN};padding:8px;font-size:10.5px;font-weight:600;}}"
            f"QPushButton:hover{{border-color:{WARN};}}"
            f"QPushButton:disabled{{background:{SURFACE_2};color:{TEXT_MUTE};border-color:{SURFACE_2};}}")
        self._sor_btn.clicked.connect(
            lambda: self.sor_detect_requested.emit(self._sor_k.value(), self._sor_ratio.value()))
        l4.addWidget(self._sor_btn)
        lay.addWidget(gb4)

        lay.addStretch()

        scroll.setWidget(content)
        root.addWidget(scroll)

    # ── API pública ───────────────────────────────────────────────────────────

    def set_schema(self, schema: list) -> None:
        self._schema = schema
        if hasattr(self, '_csf_combo'):
            self._csf_combo.clear()
            for sc in schema:
                if sc.id > 0: self._csf_combo.addItem(f"  {sc.name}", sc.id)
            # Pre-select class named "suelo" or similar
            for i in range(self._csf_combo.count()):
                name = self._csf_combo.itemText(i).lower()
                if any(n in name for n in ["suelo","ground","terreno","tierra"]):
                    self._csf_combo.setCurrentIndex(i); break
            self._csf_btn.setEnabled(bool(schema))
        self._init_default_agl_data()
        self._refresh_layers_summary()

    def _init_default_agl_data(self) -> None:
        """Construye self._agl_data ([{class_id, lo, hi}]) con las capas
        por defecto, mapeadas por nombre al schema real del proyecto."""
        self._agl_data = []
        if not self._schema:
            return
        for key, names, lo, hi in self._DEFAULT_LAYERS:
            best_id = 0
            for sc in self._schema:
                for nm in names:
                    if nm in sc.name.lower() or sc.name.lower() in nm:
                        best_id = sc.id; break
                if best_id: break
            self._agl_data.append({"class_id": best_id, "lo": lo, "hi": hi})
        self._agl_auto_btn.setEnabled(True)

    def _open_ranges_dialog(self) -> None:
        if not self._schema:
            return
        dlg = AGLRangesDialog(self._schema, self._agl_data, self)
        if dlg.exec_() == dlg.Accepted:
            self._agl_data = dlg.get_layers()
            self._refresh_layers_summary()
            self._agl_auto_btn.setEnabled(True)

    def _class_color(self, class_id: int) -> str:
        for sc in self._schema:
            if sc.id == class_id: return sc.color
        return TEXT_MUTE

    def _class_name(self, class_id: int) -> str:
        for sc in self._schema:
            if sc.id == class_id: return sc.name
        return "(No clasificar)"

    def _refresh_layers_summary(self) -> None:
        """Redibuja el resumen de solo lectura desde self._agl_data —
        la edición real pasa por AGLRangesDialog (ver docstring)."""
        for i in reversed(range(self._layers_vlay.count())):
            w = self._layers_vlay.itemAt(i).widget()
            if w: w.setParent(None)

        if not self._schema:
            lbl = QLabel("(Carga una nube con clases para configurar los rangos)")
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;padding:4px;background:transparent;")
            self._layers_vlay.addWidget(lbl)
            return

        if not self._agl_data:
            lbl = QLabel("Sin rangos configurados.")
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;padding:4px;background:transparent;")
            self._layers_vlay.addWidget(lbl)
            return

        for layer in self._agl_data:
            row = QWidget(); row.setStyleSheet("background:transparent;")
            rl = QHBoxLayout(row); rl.setContentsMargins(2, 2, 2, 2); rl.setSpacing(7)
            dot = QLabel(); dot.setFixedSize(8, 8)
            dot.setStyleSheet(f"background:{self._class_color(layer['class_id'])};border-radius:4px;")
            name = QLabel(self._class_name(layer["class_id"]))
            name.setStyleSheet(f"color:{TEXT_DIM};font-size:10.5px;background:transparent;")
            hi_txt = "∞" if layer["hi"] >= 9998 else f"{layer['hi']:.1f}"
            rng = QLabel(f"{layer['lo']:.1f} – {hi_txt} m")
            rng.setAlignment(Qt.AlignRight)
            rng.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;font-family:'Consolas';background:transparent;")
            rl.addWidget(dot); rl.addWidget(name, 1); rl.addWidget(rng)
            self._layers_vlay.addWidget(row)

        # Aviso de solapamiento (informativo, no bloqueante)
        overlaps = []
        for i in range(len(self._agl_data)):
            lo1, hi1 = self._agl_data[i]["lo"], self._agl_data[i]["hi"]
            for j in range(i+1, len(self._agl_data)):
                lo2, hi2 = self._agl_data[j]["lo"], self._agl_data[j]["hi"]
                if lo1 < hi2 and lo2 < hi1:
                    overlaps.append((i+1, j+1))
        if overlaps:
            pairs = ", ".join(f"{a} y {b}" for a, b in overlaps[:3])
            warn = QLabel(f"⚠ Rangos {pairs} se solapan")
            warn.setWordWrap(True)
            warn.setStyleSheet(f"color:{WARN};font-size:10.5px;font-weight:600;padding:2px;background:transparent;")
            self._layers_vlay.addWidget(warn)

    def get_agl_layers(self) -> list:
        """Retorna [(class_id, agl_min, agl_max)] desde self._agl_data."""
        result = []
        for layer in self._agl_data:
            cid, lo, hi = layer["class_id"], layer["lo"], layer["hi"]
            if hi >= 9998: hi = 9999.0
            if cid and cid > 0 and hi > lo:
                result.append((int(cid), float(lo), float(hi)))
        return result

    def get_active_rules(self):
        return [r for r in self._rules if r.enabled]

    def is_flat_mode(self) -> bool:
        """Siempre True — modo plano es el único disponible tras quitar el AVANZADO."""
        return True

    def get_csf_params(self) -> dict:
        """Returns CSF parameters: class_id and cloth_resolution."""
        cid = self._csf_combo.currentData() if hasattr(self, '_csf_combo') else 1
        res = self._csf_resolution.value() if hasattr(self, '_csf_resolution') else 0.5
        return {"class_id": int(cid or 1), "resolution": float(res)}

    def get_agl_class_id(self) -> int:
        return 1   # sin selector manual — usar capas AGL del panel

    def _on_select_click(self):
        self.agl_select_requested.emit(0.0, 5.0)
