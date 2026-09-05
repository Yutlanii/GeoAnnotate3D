"""
ui/export_dialog.py — Modal de exportación simplificado.

Diseño:
  Selecciona arquitectura → define el formato de salida implícitamente.
  Checkboxes extra: "compatible con todas las redes", "exportar .las clasificado".

REDISEÑO 2026-09-05: colores movidos a ui/theme.py + iconos reales.
Bug de contraste real corregido: `_ArchButton.set_selected` pintaba el
subtítulo con `#c9cbce`/`#d6d8da` — esos son tonos de BORDE (BORDER/
BORDER_SOFT en theme.py), casi invisibles como texto sobre los fondos
claros del botón (`#dceef0` seleccionado / `#e8e9eb` normal). Ahora usa
TEXT_MUTE/TEXT_DIM, que sí están pensados para texto secundario.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QWidget, QFrame, QCheckBox, QProgressBar,
    QMessageBox, QSizePolicy,
)
from PyQt5.QtCore import Qt

from annotation.exporter import ExportConfig, ExportWorker
from ui.icons import icon as qicon, pixmap as qpixmap
from ui.theme import (
    SURFACE, SURFACE_2, SURFACE_3, BORDER, BORDER_SOFT,
    TEXT, TEXT_DIM, TEXT_MUTE,
    ACCENT, ACCENT_STRONG, ACCENT_SOFT, ACCENT_BORDER,
    OK, OK_SOFT, WARN, WARN_SOFT,
)


# ── Botón de arquitectura ─────────────────────────────────────────────────────

class _ArchButton(QWidget):
    """Botón seleccionable con nombre, subtítulo y formato implícito."""

    def __init__(self, name: str, subtitle: str, fmt: str,
                 on_select: Callable, parent=None):
        super().__init__(parent)
        self._name      = name
        self._fmt       = fmt        # formato implícito de esta arquitectura
        self._selected  = False
        self._on_select = on_select
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_StyledBackground, True)
        # ID-scoped selector: un setStyleSheet() sin selector en este
        # QWidget cascadearía su `border` a los QLabel hijos que no lo
        # sobrescriben (mismo bug documentado en NOTES_CLAUDE.md — ver
        # tool_panel.py/_card()), produciendo una caja dentro de otra
        # alrededor de cada línea de texto envuelta.
        self.setObjectName("archBtn")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 10)
        lay.setSpacing(4)
        lay.setAlignment(Qt.AlignCenter)

        self._name_lbl = QLabel(name)
        self._name_lbl.setAlignment(Qt.AlignCenter)
        self._name_lbl.setStyleSheet(
            f"color:{TEXT_DIM}; font-size:12px; font-weight:700; background:transparent;")

        self._sub_lbl = QLabel(subtitle)
        self._sub_lbl.setAlignment(Qt.AlignCenter)
        self._sub_lbl.setStyleSheet(f"color:{TEXT_MUTE}; font-size:10.5px; background:transparent;")
        self._sub_lbl.setWordWrap(True)

        self._fmt_lbl = QLabel(f"→ {fmt}")
        self._fmt_lbl.setAlignment(Qt.AlignCenter)
        self._fmt_lbl.setStyleSheet(f"color:{TEXT_MUTE}; font-size:10.5px; font-style:italic; background:transparent;")

        lay.addWidget(self._name_lbl)
        lay.addWidget(self._sub_lbl)
        lay.addWidget(self._fmt_lbl)
        self._set_style(False)

    def set_selected(self, v: bool) -> None:
        self._selected = v
        self._set_style(v)
        self._name_lbl.setStyleSheet(
            f"color:{ACCENT_STRONG if v else TEXT_DIM}; font-size:12px; font-weight:700; background:transparent;")
        self._sub_lbl.setStyleSheet(
            f"color:{TEXT_DIM if v else TEXT_MUTE}; font-size:10.5px; background:transparent;")
        self._fmt_lbl.setStyleSheet(
            f"color:{ACCENT_STRONG if v else TEXT_MUTE}; font-size:10.5px; font-style:italic; background:transparent;")

    def _set_style(self, sel: bool) -> None:
        self.setStyleSheet(
            f"QWidget#archBtn{{background:{ACCENT_SOFT}; border:1.5px solid {ACCENT}; border-radius:4px;}}"
            if sel else
            f"QWidget#archBtn{{background:{SURFACE}; border:1px solid {BORDER}; border-radius:4px;}}")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._on_select(self)

    def enterEvent(self, event):
        if not self._selected:
            self.setStyleSheet(
                f"QWidget#archBtn{{background:{SURFACE_2}; border:1px solid {ACCENT_BORDER}; border-radius:4px;}}")

    def leaveEvent(self, event):
        if not self._selected:
            self._set_style(False)

    @property
    def arch_name(self) -> str: return self._name

    @property
    def fmt_value(self) -> str: return self._fmt


# ── Split info ────────────────────────────────────────────────────────────────

class _SplitInfo(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("splitInfo")
        self.setStyleSheet(
            f"QWidget#splitInfo{{background:{OK_SOFT}; border-radius:4px; border-left:3px solid {OK};}}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(11, 8, 11, 8)
        lay.setSpacing(2)
        l1 = QLabel("Split espacial automático  ·  bloques 50m × 50m")
        l1.setStyleSheet(f"color:{OK}; font-size:10.5px; font-weight:600; background:transparent;")
        l1.setWordWrap(True)
        l2 = QLabel("train 70%  ·  val 20%  ·  test 10%  —  sin data leakage geográfico")
        l2.setStyleSheet(f"color:{TEXT_DIM}; font-size:10.5px; background:transparent;")
        l2.setWordWrap(True)
        lay.addWidget(l1); lay.addWidget(l2)


# ── Dialog principal ──────────────────────────────────────────────────────────

class ExportDialog(QDialog):
    """
    Modal de exportación simplificado.

    El usuario elige la arquitectura objetivo → el formato se selecciona
    automáticamente. Opciones extra como checkboxes.
    """

    def __init__(self, pc, project, parent=None):
        super().__init__(parent)
        self._pc      = pc
        self._project = project
        self._worker: Optional[ExportWorker] = None

        self.setWindowTitle("Exportar dataset")
        self.setMinimumWidth(560)
        self.setModal(True)
        self.setStyleSheet(f"QDialog{{background:{SURFACE_2};}}")

        self._arch_btns: List[_ArchButton] = []
        self._active_arch: Optional[_ArchButton] = None

        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ─────────────────────────────────────────────────────
        hdr = QWidget()
        hdr.setStyleSheet(f"background:{SURFACE}; border-bottom:1px solid {BORDER};")
        hdr_lay = QHBoxLayout(hdr)
        hdr_lay.setContentsMargins(20, 14, 20, 14); hdr_lay.setSpacing(8)
        hic = QLabel(); hic.setPixmap(qpixmap("box-arrow-up", ACCENT_STRONG, 16))
        hdr_lay.addWidget(hic)
        title = QLabel("Exportar dataset")
        title.setStyleSheet(f"color:{ACCENT_STRONG}; font-size:14px; font-weight:700;")
        close_btn = QPushButton()
        close_btn.setIcon(qicon("x-lg", TEXT_MUTE))
        close_btn.setFixedSize(26, 26)
        close_btn.setStyleSheet(
            f"QPushButton{{background:transparent;border:none;border-radius:4px;}}"
            f"QPushButton:hover{{background:{SURFACE_2};}}")
        close_btn.clicked.connect(self.reject)
        hdr_lay.addWidget(title); hdr_lay.addStretch(); hdr_lay.addWidget(close_btn)
        root.addWidget(hdr)

        # ── Body ───────────────────────────────────────────────────────
        body = QWidget(); body.setStyleSheet("background:transparent;")
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(20, 18, 20, 14)
        body_lay.setSpacing(14)

        # Arquitectura + formato implícito
        body_lay.addWidget(self._sect("Arquitectura objetivo — define el formato de salida"))
        arch_row = QHBoxLayout(); arch_row.setSpacing(8)
        for name, sub, fmt in [
            ("RandLA-Net", "LiDAR urbano · 1M+ pts",      ".pkl"),
            ("KPConv",     "Alta precisión · escenas med.", ".npy + json"),
            ("PointNet++", "Baseline · objetos 3D",        ".h5 (HDF5)"),
        ]:
            btn = _ArchButton(name, sub, fmt, self._select_arch)
            self._arch_btns.append(btn)
            arch_row.addWidget(btn)
        body_lay.addLayout(arch_row)
        self._select_arch(self._arch_btns[0])

        # Split info
        body_lay.addWidget(_SplitInfo())

        # Checkboxes
        self._chk_all_arch = self._checkbox(
            "También exportar formato universal (.npy + json)",
            "compatible directamente con todas las redes sin conversión", True)
        self._chk_classified = self._checkbox(
            "Exportar nube clasificada (.las con campo classification)",
            "útil para visualizar en CloudCompare / QGIS", True)
        self._chk_only_labeled = self._checkbox(
            "Solo puntos etiquetados (excluir label=0)",
            "descarta puntos sin etiquetar del dataset final", False)
        self._chk_asprs = self._checkbox(
            "Usar códigos de clasificación ASPRS estándar en el .las",
            "en vez de los IDs internos del schema — mejor compatibilidad "
            "con CloudCompare/QGIS/ArcGIS; lo no reconocido usa 64+ID", False)
        for w in [self._chk_all_arch, self._chk_classified,
                  self._chk_only_labeled, self._chk_asprs]:
            body_lay.addWidget(w)

        root.addWidget(body)

        # ── Footer ─────────────────────────────────────────────────────
        ftr = QWidget()
        ftr.setStyleSheet(f"background:{SURFACE}; border-top:1px solid {BORDER};")
        ftr_lay = QHBoxLayout(ftr)
        ftr_lay.setContentsMargins(20, 12, 20, 12)
        ftr_lay.setSpacing(10)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        self._progress.setFixedHeight(5)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet(
            f"QProgressBar{{background:{SURFACE_3};border:none;border-radius:2px;}}"
            f"QProgressBar::chunk{{background:{ACCENT};border-radius:2px;}}")
        self._prog_msg = QLabel("")
        self._prog_msg.setStyleSheet(f"color:{TEXT_DIM}; font-size:10.5px;")
        self._prog_msg.setVisible(False)
        ftr_lay.addWidget(self._progress, 1)
        ftr_lay.addWidget(self._prog_msg)

        cancel_btn = QPushButton("Cancelar")
        cancel_btn.setStyleSheet(
            f"QPushButton{{background:transparent;border:1px solid {BORDER};border-radius:4px;"
            f"color:{TEXT_DIM};padding:8px 18px;font-size:11px;}}"
            f"QPushButton:hover{{border-color:{TEXT_MUTE};}}")
        cancel_btn.clicked.connect(self.reject)

        self._export_btn = QPushButton("  Exportar dataset")
        self._export_btn.setIcon(qicon("box-arrow-up", "#ffffff"))
        self._export_btn.setStyleSheet(
            f"QPushButton{{background:{ACCENT};color:#ffffff;border:none;border-radius:4px;"
            f"font-size:11px;font-weight:700;padding:9px 22px;letter-spacing:0.3px;}}"
            f"QPushButton:hover{{background:{ACCENT_STRONG};}}"
            f"QPushButton:disabled{{background:{ACCENT_SOFT};color:{TEXT_MUTE};}}")
        self._export_btn.clicked.connect(self._on_export)
        ftr_lay.addWidget(cancel_btn)
        ftr_lay.addWidget(self._export_btn)
        root.addWidget(ftr)

    # ── Helpers ───────────────────────────────────────────────────────

    def _sect(self, text: str) -> QLabel:
        lbl = QLabel(text.upper())
        lbl.setStyleSheet(f"color:{TEXT_MUTE}; font-size:10.5px; font-weight:600; letter-spacing:0.6px;")
        lbl.setWordWrap(True)
        return lbl

    def _checkbox(self, label: str, subtitle: str, checked: bool) -> QWidget:
        w = QWidget(); w.setStyleSheet("background:transparent;")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 2, 0, 2); lay.setSpacing(10)
        chk = QCheckBox(); chk.setChecked(checked)
        chk.setStyleSheet(f"""
            QCheckBox::indicator{{width:15px;height:15px;border:1px solid {BORDER};
                border-radius:3px;background:{SURFACE};}}
            QCheckBox::indicator:checked{{background:{ACCENT};border-color:{ACCENT};}}
        """)
        txt = QVBoxLayout(); txt.setSpacing(1)
        l1 = QLabel(label); l1.setStyleSheet(f"color:{TEXT_DIM}; font-size:11px; background:transparent;")
        l1.setWordWrap(True)
        l2 = QLabel(subtitle); l2.setStyleSheet(f"color:{TEXT_MUTE}; font-size:10.5px; background:transparent;")
        l2.setWordWrap(True)
        txt.addWidget(l1); txt.addWidget(l2)
        lay.addWidget(chk); lay.addLayout(txt); lay.addStretch()
        w._checkbox = chk
        return w

    def _select_arch(self, btn: _ArchButton) -> None:
        for b in self._arch_btns: b.set_selected(False)
        btn.set_selected(True)
        self._active_arch = btn

    # ── Exportación ───────────────────────────────────────────────────

    def _on_export(self) -> None:
        out_dir = QFileDialog.getExistingDirectory(
            self, "Directorio de exportación", str(Path.home()))
        if not out_dir:
            return

        arch_map = {"RandLA-Net": "randlanet", "KPConv": "kpconv",
                    "PointNet++": "pointnetpp"}
        arch     = self._active_arch
        arch_key = arch_map.get(arch.arch_name if arch else "", "randlanet")
        export_las = self._chk_classified._checkbox.isChecked()
        only_lbl   = self._chk_only_labeled._checkbox.isChecked()

        # ¿Exportar todas las arquitecturas?
        if getattr(self, '_chk_all_arch', None) and self._chk_all_arch._checkbox.isChecked():
            archs = ["randlanet", "pointnetpp", "kpconv"]
        else:
            archs = [arch_key]

        config = ExportConfig(
            output_dir    = out_dir,
            architectures = archs,
            export_las    = export_las,
            only_labeled  = only_lbl,
            asprs_codes   = self._chk_asprs._checkbox.isChecked(),
        )

        self._export_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._prog_msg.setVisible(True)
        self._run_export(config)

    def _run_export(self, config: ExportConfig, then=None) -> None:
        self._worker = ExportWorker(
            self._pc, self._project,
            label_store=None,
            config=config,
            tile_manager=getattr(self, '_tile_manager', None),
            parent=self)
        self._worker.progress.connect(self._on_progress)
        if then:
            self._worker.finished.connect(lambda _: then())
        else:
            self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, pct: int, msg: str) -> None:
        self._progress.setValue(pct)
        self._prog_msg.setText(msg)

    def _on_finished(self, out_dir: str) -> None:
        self._export_btn.setEnabled(True)
        QMessageBox.information(
            self, "Exportación completa",
            f"Dataset exportado en:\n{out_dir}\n\n"
            "Para abrir un .npy en Python:\n"
            "  import numpy as np\n"
            "  data = np.load('cloud_train.npy')\n"
            "  print(data.shape)  # (N, 7)")
        self.accept()

    def _on_error(self, msg: str) -> None:
        self._export_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._prog_msg.setVisible(False)
        QMessageBox.critical(self, "Error de exportación", msg[:500])
