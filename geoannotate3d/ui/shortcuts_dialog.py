"""
ui/shortcuts_dialog.py — Referencia de atajos de teclado (Ayuda → Atajos).

Añadido para la primera versión pública: ningún atajo era descubrible
sin leer el código fuente (Ctrl+Z, 1-9 para clases, letras de
herramienta, etc.). Esta lista se construyó leyendo directamente
`ui/main_window.py::keyPressEvent` y `annotation/tools.py` — no es una
lista adivinada, por lo que si algún atajo cambia ahí, hay que
actualizar esta también.

De paso, construir esta lista encontró 2 bugs reales: `DiscTool` (tecla
"V") y `SphereSelectTool` (tecla "R") colisionaban con los atajos de
cámara (V=vista cenital, R=reset cámara), que se comprueban ANTES que
el mapa de herramientas en `keyPressEvent` — esas 2 herramientas nunca
respondían a su tecla. Reasignadas a "D" y "H" (ver NOTES_CLAUDE.md).
"""
from __future__ import annotations
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QWidget,
)
from PyQt5.QtCore import Qt

from ui.icons import icon as qicon, pixmap as qpixmap
from ui.theme import (
    SURFACE, SURFACE_2, BORDER, BORDER_SOFT,
    TEXT, TEXT_DIM, TEXT_MUTE, ACCENT_STRONG,
)

# (categoría, [(atajo, descripción), ...])
SHORTCUT_GROUPS = [
    ("General", [
        ("Ctrl+Z",        "Deshacer última anotación"),
        ("Ctrl+Shift+Z",  "Rehacer"),
        ("Ctrl+S",        "Guardar proyecto"),
        ("Ctrl+N",        "Nuevo proyecto / cargar nube"),
        ("Ctrl+T",        "Ir al panel de Tiles"),
        ("Espacio",       "Reiniciar vista de cámara al encuadre completo"),
        ("Escape",        "Salir del modo tile (volver a Overview)"),
    ]),
    ("Clases", [
        ("1 – 9",         "Cambiar la clase activa (\"pintando con\")"),
    ]),
    ("Cámara", [
        ("V",             "Vista cenital (top-down)"),
        ("Y",             "Vista lateral"),
        ("R",             "Restablecer cámara"),
        ("F",             "Alternar modo de vuelo (fly mode)"),
    ]),
    ("Herramientas de anotación", [
        ("B",             "Pincel (brush) esférico"),
        ("D",             "Disco 3D orientado a la superficie"),
        ("G",             "Region Growing (crecimiento de región)"),
        ("L",             "Polígono 2D/3D"),
        ("X",             "Caja de selección"),
        ("H",             "Esfera (Ctrl+clic — selección instantánea)"),
        ("C",             "Corte Z / slice"),
        ("I",             "Pick (muestrear clase de un punto)"),
        ("M",             "Medir distancias"),
        ("E",             "Alternar modo borrar en la herramienta activa"),
        ("Ctrl+arrastrar","Pintar/seleccionar (según la herramienta)"),
        ("Rueda del ratón","Cambiar el tamaño del pincel/esfera activa"),
    ]),
]


class ShortcutsDialog(QDialog):
    """Diálogo de solo lectura con la lista de atajos de teclado."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Atajos de teclado")
        self.setMinimumWidth(460)
        self.setMinimumHeight(420)
        self.setStyleSheet(f"QDialog{{background:{SURFACE};}}")
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(4)

        hdr = QHBoxLayout(); hdr.setSpacing(8)
        ic = QLabel(); ic.setPixmap(qpixmap("sliders2", ACCENT_STRONG, 16))
        hdr.addWidget(ic)
        title = QLabel("Atajos de teclado")
        title.setStyleSheet(f"color:{TEXT};font-size:15px;font-weight:700;background:transparent;")
        hdr.addWidget(title); hdr.addStretch()
        root.addLayout(hdr)

        subtitle = QLabel("Referencia rápida de todos los atajos disponibles en GeoAnnotate3D.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;background:transparent;")
        root.addWidget(subtitle)
        root.addSpacing(10)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background:transparent;")
        content = QWidget(); content.setStyleSheet(f"background:{SURFACE};")
        cl = QVBoxLayout(content)
        cl.setContentsMargins(0, 0, 4, 0); cl.setSpacing(14)

        for group_name, items in SHORTCUT_GROUPS:
            grp = QLabel(group_name.upper())
            grp.setStyleSheet(
                f"color:{ACCENT_STRONG};font-size:10.5px;font-weight:700;"
                f"letter-spacing:0.5px;background:transparent;")
            cl.addWidget(grp)

            for key, desc in items:
                row = QFrame(); row.setObjectName("shortcutRow")
                row.setStyleSheet(
                    f"QFrame#shortcutRow{{background:{SURFACE_2};border-radius:4px;}}")
                rl = QHBoxLayout(row); rl.setContentsMargins(10, 7, 10, 7); rl.setSpacing(10)
                key_lbl = QLabel(key)
                key_lbl.setStyleSheet(
                    f"color:{TEXT};font-size:10.5px;font-weight:700;font-family:'Consolas';"
                    f"background:{SURFACE};border:1px solid {BORDER};border-radius:3px;"
                    f"padding:2px 8px;")
                key_lbl.setMinimumWidth(96)
                key_lbl.setAlignment(Qt.AlignCenter)
                desc_lbl = QLabel(desc)
                desc_lbl.setWordWrap(True)
                desc_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:10.5px;background:transparent;")
                rl.addWidget(key_lbl); rl.addWidget(desc_lbl, 1)
                cl.addWidget(row)

        cl.addStretch()
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        close_btn = QPushButton("Cerrar")
        close_btn.setStyleSheet(
            f"QPushButton{{background:{SURFACE_2};border:1px solid {BORDER};border-radius:4px;"
            f"color:{TEXT_DIM};padding:8px 18px;font-size:10.5px;font-weight:600;}}"
            f"QPushButton:hover{{border-color:{ACCENT_STRONG};}}")
        close_btn.clicked.connect(self.accept)
        btn_row = QHBoxLayout(); btn_row.addStretch(); btn_row.addWidget(close_btn)
        root.addSpacing(10)
        root.addLayout(btn_row)
