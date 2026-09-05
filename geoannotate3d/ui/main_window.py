"""
ui/main_window.py — Ventana principal de GeoAnnotate3D v4.0

Cambios vs v3.0:
  - Feature 1: tile_hover_2d → canvas.set_panel_hover_tile (resalta tile en 3D desde panel 2D).
  - Feature 2: grid_edit_mode_changed → canvas.set_grid_edit_mode (drag Mover/Rotar).
  - grid_transform_preview → actualizar spinboxes en tiempo real durante drag.
  - grid_transform_set → rebuild completo del TileManager al soltar el mouse.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QDockWidget, QHBoxLayout, QVBoxLayout,
    QTabWidget, QStackedWidget, QFrame,
    QFileDialog, QMessageBox, QLabel, QPushButton, QMenu, QAction,
    QSizePolicy, QProgressBar,
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSettings
from PyQt5.QtGui import QFont

from ui.styles import QSS
from ui.theme import (BG, SURFACE, SURFACE_2, SURFACE_3, BORDER, BORDER_SOFT,
                       TEXT, TEXT_DIM, TEXT_MUTE, TEXT_FAINT,
                       ACCENT, ACCENT_STRONG, ACCENT_SOFT, ACCENT_BORDER, OK, WARN)
from ui.icons import icon as qicon, pixmap as qpixmap
from ui.class_panel   import ClassPanel
from ui.overlay_panel import OverlayPanel
from ui.geo_panel     import GeoPanel
from ui.tool_panel import ToolPanel
from ui.status_bar import AnnotationStatusBar
from ui.export_dialog import ExportDialog
from ui.tile_panel import TilePanel

from core.pointcloud import PointCloud, CloudPipeline
from core.project import Project
from core.terrain import TerrainModel
from core.rules_engine import RulesEngine
from core.tile_manager import TileManager, TileExtractionWorker
from annotation.label_store import LabelStore
from annotation.tools import (ALL_TOOLS, TOOL_BY_NAME, TOOL_BY_KEY,
                               BrushTool, BaseTool)
from render.canvas import AnnotationCanvas


# ── Header bar ────────────────────────────────────────────────────────────────

class _HeaderBar(QWidget):
    """Barra superior: logo | archivo | CRS | modo | Proyecto▾ | timer | guardado"""

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self._win = window
        self.setFixedHeight(44)
        # BG (el fondo neutro base de toda la app), no SURFACE (un tono
        # más claro pensado para tarjetas/superficies elevadas) — el
        # usuario pidió que la cabecera no tenga un fondo "blanco"
        # propio, solo el color de fondo original de la UI.
        self.setStyleSheet(f"background:{BG};border-bottom:1px solid {BORDER};")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 0, 16, 0)
        lay.setSpacing(10)

        # Logo (icono de "nube de puntos" + nombre)
        logo_row = QHBoxLayout(); logo_row.setSpacing(8)
        logo_ic = QLabel(); logo_ic.setPixmap(qpixmap("cloud-check", ACCENT, 18))
        logo = QLabel("GeoAnnotate3D")
        logo.setStyleSheet(f"color:{TEXT};font-size:14px;font-weight:700;")
        logo_row.addWidget(logo_ic); logo_row.addWidget(logo)
        lay.addLayout(logo_row)

        lay.addWidget(self._sep(), 0, Qt.AlignVCenter)

        self._file_lbl = QLabel("sin proyecto")
        self._file_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:10.5px;font-family:Consolas;background:transparent;")
        self._file_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._file_lbl.setMaximumWidth(400)
        lay.addWidget(self._file_lbl, 0, Qt.AlignVCenter)

        # Estado (CRS / modo): antes eran "chips" con fondo y borde propios
        # — el usuario reportó que ese fondo seguía sin encajar bien con el
        # fondo general de la barra ("contrasta mal"). Quitado del todo:
        # ahora son texto plano (con un icono como pista visual) sobre el
        # mismo fondo de la barra — sin caja que pueda desentonar.
        self._crs_ic = QLabel(); self._crs_ic.setPixmap(qpixmap("globe2", TEXT_MUTE, 12))
        self._crs_ic.setStyleSheet("background:transparent;")
        self._crs_ic.setVisible(False)
        lay.addWidget(self._crs_ic, 0, Qt.AlignVCenter)
        self._crs_lbl = QLabel("")
        self._crs_lbl.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;font-family:Consolas;background:transparent;")
        self._crs_lbl.setVisible(False)
        lay.addWidget(self._crs_lbl, 0, Qt.AlignVCenter)

        self._mode_ic = QLabel(); self._mode_ic.setPixmap(qpixmap("arrows-fullscreen", TEXT_MUTE, 11))
        self._mode_ic.setStyleSheet("background:transparent;")
        lay.addWidget(self._mode_ic, 0, Qt.AlignVCenter)
        self._mode_badge = QLabel("Overview")
        self._mode_badge.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;font-weight:600;background:transparent;")
        lay.addWidget(self._mode_badge, 0, Qt.AlignVCenter)

        # Proyecto menu — botón plano: sin borde/caja visible en reposo,
        # el fondo/borde solo aparece al pasar el mouse (affordance de
        # clic sin sumar otra caja permanente a la barra).
        self._proj_btn = QPushButton("Proyecto")
        self._proj_btn.setIcon(qicon("folder2", TEXT_DIM))
        self._proj_btn.setFixedHeight(28)
        self._proj_btn.setStyleSheet(
            f"QPushButton{{background:transparent;border:1px solid transparent;border-radius:5px;"
            f"color:{TEXT_DIM};font-size:10.5px;font-weight:600;padding:0 12px;}}"
            f"QPushButton:hover{{border-color:{ACCENT_BORDER};color:{ACCENT_STRONG};background:{ACCENT_SOFT};}}"
            f"QPushButton::menu-indicator{{width:0;}}")

        proj_menu = QMenu(self._proj_btn)
        proj_menu.setStyleSheet(
            f"QMenu{{background:{SURFACE};border:1px solid {BORDER};border-radius:4px;"
            f"padding:4px 0;color:{TEXT_DIM};}}"
            f"QMenu::item{{padding:7px 32px 7px 14px;font-size:10.5px;}}"
            f"QMenu::item:selected{{background:{ACCENT_SOFT};color:{ACCENT_STRONG};}}"
            f"QMenu::separator{{height:1px;background:{BORDER_SOFT};margin:3px 0;}}")

        def _act(label, shortcut, fn):
            a = QAction(label, self)
            if shortcut: a.setShortcut(shortcut)
            a.triggered.connect(fn)
            return a

        proj_menu.addAction(_act("Nuevo proyecto",            "Ctrl+N", window.new_project))
        proj_menu.addAction(_act("Abrir nube (.las/.laz…)",   "Ctrl+O", window.new_project))
        proj_menu.addAction(_act("Reemplazar nube…",          "Ctrl+R", window.replace_cloud))
        proj_menu.addSeparator()
        proj_menu.addAction(_act("Guardar proyecto (.geoa3d)", "Ctrl+S", window.save_project))
        proj_menu.addAction(_act("Cargar proyecto (.geoa3d)",  "Ctrl+L", window.load_project))
        proj_menu.addSeparator()
        self._recent_menu = QMenu("Abrir reciente  ▶", proj_menu)
        self._recent_menu.setStyleSheet(proj_menu.styleSheet())
        proj_menu.addMenu(self._recent_menu)
        self._proj_btn.setMenu(proj_menu)
        lay.addWidget(self._proj_btn, 0, Qt.AlignVCenter)

        # Ayuda — antes no existía ningún menú de ayuda pese a que
        # WelcomeDialog ya mencionaba "Ayuda → Bienvenida" en su pie de
        # página (footer text sin nada real detrás). Con esto ese texto
        # por fin es cierto, y además da acceso a la referencia de
        # atajos de teclado (ningún atajo era descubrible sin leer el
        # código fuente).
        self._help_btn = QPushButton("Ayuda")
        self._help_btn.setIcon(qicon("info-circle", TEXT_DIM))
        self._help_btn.setFixedHeight(28)
        self._help_btn.setStyleSheet(
            f"QPushButton{{background:transparent;border:1px solid transparent;border-radius:5px;"
            f"color:{TEXT_DIM};font-size:10.5px;font-weight:600;padding:0 12px;}}"
            f"QPushButton:hover{{border-color:{ACCENT_BORDER};color:{ACCENT_STRONG};background:{ACCENT_SOFT};}}"
            f"QPushButton::menu-indicator{{width:0;}}")
        help_menu = QMenu(self._help_btn)
        help_menu.setStyleSheet(proj_menu.styleSheet())
        help_menu.addAction(_act("Atajos de teclado",      "F1", window.show_shortcuts))
        help_menu.addAction(_act("Bienvenida / Tutorial",   None, window.show_welcome_dialog))
        help_menu.addSeparator()
        help_menu.addAction(_act("Abrir carpeta de registros (logs)", None, window.open_logs_folder))
        self._help_btn.setMenu(help_menu)
        lay.addWidget(self._help_btn, 0, Qt.AlignVCenter)

        lay.addStretch(1)

        self._timer_lbl = QLabel("00:00:00")
        self._timer_lbl.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;min-width:58px;font-family:Consolas;background:transparent;")
        self._timer_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.addWidget(self._timer_lbl, 0, Qt.AlignVCenter)

        self._save_lbl = QLabel("")
        self._save_lbl.setStyleSheet(f"color:{TEXT_MUTE};font-size:10.5px;font-weight:600;background:transparent;")
        self._save_lbl.setVisible(False)
        lay.addWidget(self._save_lbl, 0, Qt.AlignVCenter)

    def _sep(self):
        s = QLabel("·"); s.setStyleSheet(f"color:{TEXT_FAINT};background:transparent;"); return s

    def set_file(self, filename, n_pts):
        self._file_lbl.setText(
            f"{filename}  —  {n_pts/1e6:.2f}M pts" if n_pts > 0 else filename)

    def set_crs(self, crs):
        if crs:
            self._crs_lbl.setText(crs[:30])
            self._crs_lbl.setVisible(True)
            self._crs_ic.setVisible(True)
        else:
            self._crs_lbl.setVisible(False)
            self._crs_ic.setVisible(False)

    def set_mode(self, tile_mode, tile=None):
        if tile_mode and tile is not None:
            self._mode_badge.setText(f"Tile ({tile.col},{tile.row})")
            self._mode_badge.setStyleSheet(
                f"color:{ACCENT_STRONG};font-size:10.5px;font-weight:700;background:transparent;")
            self._mode_ic.setPixmap(qpixmap("arrows-fullscreen", ACCENT_STRONG, 11))
        else:
            self._mode_badge.setText("Overview")
            self._mode_badge.setStyleSheet(
                f"color:{TEXT_MUTE};font-size:10.5px;font-weight:600;background:transparent;")
            self._mode_ic.setPixmap(qpixmap("arrows-fullscreen", TEXT_MUTE, 11))

    def set_session_time(self, h, m, s):
        self._timer_lbl.setText(f"{h:02d}:{m:02d}:{s:02d}")

    def set_saved(self, saved):
        if saved:
            self._save_lbl.setText("●  Guardado")
            self._save_lbl.setStyleSheet(f"color:{OK};font-size:10.5px;font-weight:600;background:transparent;")
        else:
            self._save_lbl.setText("●  Sin guardar")
            self._save_lbl.setStyleSheet(f"color:{WARN};font-size:10.5px;font-weight:600;background:transparent;")
        self._save_lbl.setVisible(True)


# ── Franja de paso (breadcrumb) ────────────────────────────────────────────────

class _StepBreadcrumb(QWidget):
    """Franja delgada bajo la cabecera: 'Paso 3 de 6 — Etiquetar — descripción'.

    Sustituye la orientación que antes daba la barra de pasos horizontal
    grande; con la navegación movida al riel lateral (_StepRail), esta
    franja es la que deja clarísimo en qué paso está el usuario en todo
    momento, sin tener que mirar al riel.
    """
    N_STEPS = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)
        # Antes tenía su propio fondo teñido (ACCENT_SOFT) — el usuario
        # pidió que esta franja (y la barra de cabecera de arriba) usen
        # el mismo fondo neutro del resto de la UI en vez de un color
        # de acento propio que desentona. Ronda siguiente: también BG en
        # vez de SURFACE, para que ni la cabecera ni esta franja tengan
        # un fondo "blanco" distinto del resto de la UI.
        self.setStyleSheet(f"background:{BG};border-bottom:1px solid {BORDER_SOFT};")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 0, 16, 0)
        lay.setSpacing(8)

        self._step_lbl = QLabel("")
        self._step_lbl.setStyleSheet(f"color:{TEXT_MUTE};font-size:11px;background:transparent;")
        self._name_lbl = QLabel("")
        self._name_lbl.setStyleSheet(f"color:{ACCENT_STRONG};font-size:12.5px;font-weight:700;background:transparent;")
        self._desc_lbl = QLabel("")
        self._desc_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;background:transparent;")

        lay.addWidget(self._step_lbl)
        lay.addWidget(self._sep())
        lay.addWidget(self._name_lbl)
        lay.addWidget(self._desc_lbl)
        lay.addStretch(1)

    def _sep(self):
        s = QLabel("│"); s.setStyleSheet(f"color:{BORDER};background:transparent;"); return s

    def set_step(self, n: int, name: str, desc: str) -> None:
        self._step_lbl.setText(f"Paso {n} de {self.N_STEPS}")
        self._name_lbl.setText(name)
        self._desc_lbl.setText(f"— {desc}")


# ── Riel de navegación (pasos + capas) ─────────────────────────────────────────

class _RailItem(QWidget):
    """Una fila del riel: icono + texto + check (si está completado)."""
    clicked = pyqtSignal()

    def __init__(self, icon_name: str, label: str, parent=None):
        super().__init__(parent)
        self._icon_name = icon_name
        self._state = "pending"   # pending | active | done
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(38)
        self.setAttribute(Qt.WA_StyledBackground, True)
        # ID-scoped selector: un setStyleSheet() sin selector en este
        # QWidget cascadearía su `border` a los QLabel hijos que no lo
        # sobrescriben (_text_lbl solo pone `background:transparent`,
        # no `border:none`) — cada hijo dibujaba su propio borde
        # heredado alrededor de su propio cuadro, dando el efecto de
        # "doble caja" reportado por el usuario. Mismo bug ya visto en
        # `_ArchButton` (export_dialog.py) y `_StepCard`
        # (welcome_dialog.py) — ver NOTES_CLAUDE.md.
        self.setObjectName("railItem")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 10, 0)
        lay.setSpacing(10)

        self._icon_lbl = QLabel()
        self._icon_lbl.setFixedSize(18, 18)
        self._text_lbl = QLabel(label)
        self._check_lbl = QLabel()
        self._check_lbl.setFixedSize(13, 13)
        self._check_lbl.setPixmap(qpixmap("check2", OK, 13))

        lay.addWidget(self._icon_lbl)
        lay.addWidget(self._text_lbl, 1)
        lay.addWidget(self._check_lbl)
        self._apply_state()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit()

    def set_state(self, state: str) -> None:
        if state == self._state: return
        self._state = state
        self._apply_state()

    def _apply_state(self) -> None:
        active = self._state == "active"
        done   = self._state == "done"
        icon_color = ACCENT_STRONG if active else (OK if done else TEXT_MUTE)
        text_color = ACCENT_STRONG if active else (TEXT if done else TEXT_MUTE)
        weight     = 700 if active else 600
        # `:hover` en los pasos no-activos: antes solo el paso activo daba
        # cualquier feedback visual — pasar el mouse sobre "Pre-clasificar",
        # "Etiquetar", etc. no comunicaba que son clicables. Mejora de
        # diseño añadida sin que se pidiera explícitamente.
        hover_bg = "transparent" if active else SURFACE_2
        self.setStyleSheet(
            f"QWidget#railItem{{background:{ACCENT_SOFT if active else 'transparent'};"
            f"border:1px solid {ACCENT_BORDER if active else 'transparent'};border-radius:4px;}}"
            f"QWidget#railItem:hover{{background:{ACCENT_SOFT if active else hover_bg};}}")
        self._icon_lbl.setPixmap(qpixmap(self._icon_name, icon_color, 18))
        self._text_lbl.setStyleSheet(f"color:{text_color};font-size:12px;font-weight:{weight};background:transparent;")
        self._check_lbl.setVisible(done)


class _StepRail(QWidget):
    """
    Riel vertical de navegación — reemplaza la barra de pasos horizontal.

    6 pasos (icono + texto real, no solo icono — para que sea inmediatamente
    claro sin adivinar) + un separador + "Capas de referencia" (deliberadamente
    aparte: no es un paso del flujo, es una vista auxiliar).
    """
    step_clicked  = pyqtSignal(int)
    capas_clicked = pyqtSignal()

    STEP_DEFS = [
        (1, "cloud-arrow-up", "Nube"),
        (2, "magic",          "Pre-clasificar"),
        (3, "tag",            "Etiquetar"),
        (4, "box-arrow-up",   "Exportar"),
        (5, "cpu",            "Entrenar"),
        (6, "bullseye",       "Inferir"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(176)
        # Sin border-right: el usuario pidió quitar la línea vertical
        # delgada que separaba el riel del panel de contenido — la
        # diferencia de fondo entre SURFACE (riel) y SURFACE_2 (panel)
        # ya basta para distinguirlos sin necesidad de una línea.
        self.setStyleSheet(f"background:{SURFACE};")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 12, 10, 12)
        lay.setSpacing(2)

        self._items = {}
        for step_num, icon_name, label in self.STEP_DEFS:
            item = _RailItem(icon_name, label)
            item.clicked.connect(lambda s=step_num: self.step_clicked.emit(s))
            self._items[step_num] = item
            lay.addWidget(item)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"background:{BORDER_SOFT};max-height:1px;margin:8px 4px;")
        lay.addWidget(sep)

        self._capas_item = _RailItem("layers", "Capas de referencia")
        self._capas_item.clicked.connect(self.capas_clicked.emit)
        lay.addWidget(self._capas_item)

        lay.addStretch(1)

    def set_active_step(self, step: int) -> None:
        for n, item in self._items.items():
            item.set_state("active" if n == step else ("done" if n < step else "pending"))
        self._capas_item.set_state("pending")

    def set_capas_active(self, active: bool) -> None:
        self._capas_item.set_state("active" if active else "pending")



class _OctreeBuilder(QThread):
    """Construye el octree de la nube en un hilo secundario."""
    ready = pyqtSignal(object)   # emite el Octree construido

    def __init__(self, pc):
        super().__init__()
        self._pc = pc

    def run(self):
        try:
            from core.octree import Octree
            import numpy as np
            xyz = self._pc.xyz
            if xyz is None or len(xyz) == 0:
                self.ready.emit(None)
                return
            # Limitar a 150M puntos para el octree de overview
            MAX_PTS = 150_000_000
            if len(xyz) > MAX_PTS:
                step = max(1, len(xyz) // MAX_PTS)
                xyz_sub = np.ascontiguousarray(xyz[::step])
            else:
                xyz_sub = xyz
            octree = Octree()
            octree.build(xyz_sub)
            self.ready.emit(octree)
        except Exception as e:
            print(f"[OctreeBuilder] Error: {e}")
            self.ready.emit(None)



class MainWindow(QMainWindow):
    """
    Ventana principal de GeoAnnotate3D.

    Layout:
        HeaderBar        — logo · archivo · CRS · modo · menú proyecto · timer
        Body
          ClassPanel     (izquierda, dock, tabbed con GeoPanel)
          AnnotationCanvas (centro)
          ToolPanel      (derecha, dock)
          TilePanel      (inferior izquierda, dock) ← NEW
        StatusBar        — conteo pts · balance clases
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GeoAnnotate3D")
        self.resize(1600, 950)
        self.setMinimumSize(1100, 650)
        self.setStyleSheet(QSS)

        self._project:  Optional[Project]       = None
        self._pc:       Optional[PointCloud]    = None
        self._pipeline: Optional[CloudPipeline] = None
        self._label_store  = LabelStore(self)
        self._active_tool: Optional[BaseTool]   = None
        self._terrain      = TerrainModel()
        self._rules_engine = RulesEngine()

        # Tile manager (NEW)
        self._tile_manager: Optional[TileManager]          = None
        self._tile_worker:  Optional[TileExtractionWorker] = None

        self._overlay_manager = None
        self._current_step    = 1
        self._build_ui()
        self._connect_signals()
        self._restore_window_state()

        self._setup_crash_handler()

    @staticmethod
    def _logs_dir() -> Path:
        d = Path.home() / ".geoannotate3d" / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _setup_crash_handler(self) -> None:
        """
        Activa faulthandler para que los SEGFAULT (crash del código C)
        escriban un traceback antes de morir, y un excepthook para
        errores Python no atrapados en el hilo principal de Qt.

        Antes ambos solo escribían a stderr — en el .exe empaquetado
        (ventana sin consola) eso es invisible: el programa "simplemente
        se cierra" sin dejar ningún rastro que el usuario pueda enviar
        como reporte de error. Ahora también se escribe a un archivo de
        log persistente (~/.geoannotate3d/logs/) y se deja un marcador
        de "sesión en curso" que se borra al cerrar limpiamente — si en
        el siguiente arranque el marcador sigue ahí, sabemos que la
        sesión anterior murió sin pasar por closeEvent (crash) y se lo
        avisamos al usuario con un botón para abrir la carpeta de logs.
        """
        import faulthandler, sys, io

        logs_dir = self._logs_dir()
        self._crash_log_path = logs_dir / "crash.log"
        self._session_marker = logs_dir / ".session_running"

        # faulthandler: además de stderr, duplicar a archivo (best-effort;
        # si falla abrir el archivo, seguir con stderr solamente).
        try:
            self._fault_log_fh = open(self._crash_log_path, "a", encoding="utf-8")
            faulthandler.enable(file=self._fault_log_fh)
        except Exception:
            try: faulthandler.enable(file=sys.stderr)
            except Exception: pass

        def _excepthook(exc_type, exc_val, exc_tb):
            import traceback, datetime
            msg = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            print(f"[CRASH] {msg}", file=sys.stderr)
            # NO mostrar QMessageBox aquí: si el crash ocurrió durante un
            # paintEvent o en un QThread, abrir un diálogo causa un crash
            # en cascada ("recursive repaint", "paint device destroyed").
            # Solo logear (stderr + archivo) es seguro siempre.
            try:
                with open(self._crash_log_path, "a", encoding="utf-8") as f:
                    f.write(f"\n--- {datetime.datetime.now().isoformat()} ---\n{msg}")
            except Exception:
                pass
        sys.excepthook = _excepthook

        # Marcador de sesión: si YA existía al llegar aquí, la sesión
        # anterior nunca pasó por closeEvent (crash / kill / corte de luz)
        # — lo recordamos y lo mostramos tras arrancar. Luego se
        # sobreescribe con la marca de ESTA sesión, y closeEvent la borra
        # en un cierre limpio, así el aviso no se repite en cada arranque.
        self._had_previous_crash = self._session_marker.exists()
        try:
            self._session_marker.write_text("running", encoding="utf-8")
        except Exception:
            pass

        self._setup_shortcuts()
        self.setAcceptDrops(True)

        self._session_timer = QTimer(self)
        self._session_timer.setInterval(1000)
        self._session_timer.timeout.connect(self._update_session_timer)

        if self._had_previous_crash:
            QTimer.singleShot(800, self._notify_previous_crash)

    def _notify_previous_crash(self) -> None:
        try:
            from PyQt5.QtWidgets import QMessageBox
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Cierre inesperado detectado")
            box.setText(
                "GeoAnnotate3D parece haberse cerrado inesperadamente la "
                "última vez (no se completó el cierre normal).")
            box.setInformativeText(
                "Si quieres reportar el problema, puedes abrir la carpeta "
                "de registros y adjuntar crash.log.")
            open_btn = box.addButton("Abrir carpeta de registros", QMessageBox.ActionRole)
            box.addButton("Cerrar", QMessageBox.RejectRole)
            box.exec_()
            if box.clickedButton() is open_btn:
                self.open_logs_folder()
        except Exception:
            pass

    def open_logs_folder(self) -> None:
        import subprocess
        try:
            subprocess.Popen(["explorer", str(self._logs_dir())])
        except Exception:
            pass

    # ── Persistencia del layout de ventana ───────────────────────────────────
    # Antes el tamaño/posición de la ventana y el ancho de los paneles
    # (splitter izquierdo/canvas/derecho) se reseteaban en cada arranque —
    # gap real reportado en la revisión previa de "qué falta para v1".
    # La app no usa QDockWidget (todo es QSplitter/QStackedWidget a mano),
    # así que no aplica QMainWindow.saveState() — se guarda la geometría de
    # la ventana + el estado del splitter principal directamente.

    def _restore_window_state(self) -> None:
        settings = QSettings("GeoAnnotate3D", "GeoAnnotate3D")
        geo = settings.value("window/geometry")
        if geo is not None:
            try: self.restoreGeometry(geo)
            except Exception: pass
        splitter_state = settings.value("window/main_splitter")
        if splitter_state is not None and hasattr(self, "_main_splitter"):
            try: self._main_splitter.restoreState(splitter_state)
            except Exception: pass

    def _save_window_state(self) -> None:
        settings = QSettings("GeoAnnotate3D", "GeoAnnotate3D")
        try:
            settings.setValue("window/geometry", self.saveGeometry())
            if hasattr(self, "_main_splitter"):
                settings.setValue("window/main_splitter", self._main_splitter.saveState())
        except Exception:
            pass

    # ── Construcción de UI ────────────────────────────────────────────────────

    # Descripciones del paso activo, mostradas en la franja de breadcrumb
    STEP_DESCRIPTIONS = {
        1: "cargar nube · configurar grid de tiles",
        2: "AGL · Region Growing · reglas automáticas",
        3: "clases · pincel · polígono · caja",
        4: "generar dataset (RandLA-Net / PointNet++ / KPConv)",
        5: "entrenar un modelo sobre el dataset exportado",
        6: "clasificar la nube con un modelo entrenado",
    }
    STEP_NAMES = {1: "Nube", 2: "Pre-clasificar", 3: "Etiquetar",
                  4: "Exportar", 5: "Entrenar", 6: "Inferir"}

    def _build_ui(self) -> None:
        # ── Header bar (logo + archivo + botones proyecto) ─────────────────────
        self._header = _HeaderBar(self)
        self.setMenuWidget(self._header)

        # ── Widget central ────────────────────────────────────────────────────
        central = QWidget()
        central.setStyleSheet(f"background:{BG};")
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ══ Franja de paso — orientación explícita bajo la cabecera ════════════
        self._breadcrumb = _StepBreadcrumb()
        outer.addWidget(self._breadcrumb)

        # ══ FILA DEL CUERPO: riel de navegación + contenido ═════════════════════
        body_row = QWidget()
        body_lay = QHBoxLayout(body_row)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(0)

        # ── RIEL — navegación de pasos + capas, siempre visible ─────────────────
        self._rail = _StepRail()
        self._rail.step_clicked.connect(self._on_workflow_step_clicked)
        self._rail.capas_clicked.connect(self._on_layers_btn_clicked)
        body_lay.addWidget(self._rail)

        # ══ MAIN AREA: splitter (left resizable) + canvas + right panel ════════
        from PyQt5.QtWidgets import QSplitter
        self._main_splitter = QSplitter(Qt.Horizontal)
        self._main_splitter.setHandleWidth(4)
        self._main_splitter.setStyleSheet(
            f"QSplitter::handle{{background:{BORDER};}}"
            f"QSplitter::handle:hover{{background:{ACCENT};}}")

        # ── LEFT PANEL (QStackedWidget — cambia según el paso) ─────────────────
        self._left_stack = QStackedWidget()
        self._left_stack.setMinimumWidth(160)
        self._left_stack.setMaximumWidth(480)
        self._left_stack.setStyleSheet(f"background:{SURFACE};")

        # Página 0 — Paso 1: Nube (tile panel directo, sin reemplazar)
        self._tile_panel = TilePanel(self)
        self._left_stack.addWidget(self._tile_panel)

        # Página 1 — Paso 2: Pre-clasificar (AGL + Region Growing)
        self._geo_panel = GeoPanel(self)
        self._left_stack.addWidget(self._geo_panel)

        # Página 2 — Paso 3: Etiquetar (clases)
        self._class_panel = ClassPanel(self)
        self._left_stack.addWidget(self._class_panel)

        # Página 3 — Capas de referencia (accesible desde el riel, NO es un paso)
        self._overlay_panel = OverlayPanel(self)
        self._left_stack.addWidget(self._overlay_panel)

        self._main_splitter.addWidget(self._left_stack)

        # ── CANVAS ────────────────────────────────────────────────────────────
        self._canvas = AnnotationCanvas()
        self._main_splitter.addWidget(self._canvas)

        # ── RIGHT PANEL ───────────────────────────────────────────────────────
        self._tool_panel = ToolPanel(self)
        self._tool_panel.setMinimumWidth(200)
        self._tool_panel.setMaximumWidth(420)
        self._tool_panel.setStyleSheet(f"background:{SURFACE};border-left:1px solid {BORDER};")
        self._main_splitter.addWidget(self._tool_panel)

        # Proporciones iniciales — ensanchadas ligeramente (antes 215/900/210):
        # con iconos de barra de herramientas + tarjetas, 210px se sentía
        # apretado (títulos de sección recortados, filas de controles
        # amontonadas). 236/230 da más aire sin robarle mucho al canvas.
        self._main_splitter.setSizes([236, 860, 230])
        self._main_splitter.setStretchFactor(0, 0)
        self._main_splitter.setStretchFactor(1, 1)
        self._main_splitter.setStretchFactor(2, 0)

        # Training panel (paso 5)
        from ui.training_panel import TrainingPanel
        self._training_panel = TrainingPanel(self)
        self._training_panel.training_finished.connect(self._on_training_finished)

        # Inference panel (paso 6)
        from ui.infer_panel import InferPanel
        self._infer_panel = InferPanel(self)
        self._infer_panel.inference_done.connect(self._on_inference_done)

        # Body stack: 0=normal layout, 1=training, 2=inference
        # (el riel queda FUERA de este stack — sigue visible en los 3 casos)
        self._body_stack = QStackedWidget()
        self._body_stack.addWidget(self._main_splitter)
        self._body_stack.addWidget(self._training_panel)
        self._body_stack.addWidget(self._infer_panel)
        self._body_stack.setCurrentIndex(0)
        body_lay.addWidget(self._body_stack, 1)

        outer.addWidget(body_row, 1)

        # Status bar
        self._status = AnnotationStatusBar(self)
        self.setStatusBar(self._status)

        # Set step 1 as active initially
        self._set_workflow_step(1)

        # Old dock reference (some code may still reference it)
        self._tile_dock = None


    def _connect_signals(self) -> None:
        # Canvas → status bar
        self._canvas.sig.fps.connect(self._status.update_fps)
        self._canvas.sig.render_info.connect(self._status.update_render_info)

        # Canvas → tile mode (NEW)
        self._canvas.sig.tile_mode_changed.connect(self._on_tile_mode_changed)

        # Canvas → tool panel
        self._canvas.sig.point_picked_abs.connect(self._on_point_picked)
        self._canvas.sig.measure_segment.connect(self._on_measure_done)

        # Label store → paneles
        self._label_store.stats_changed.connect(self._on_stats_changed)
        self._label_store.labels_changed.connect(self._on_labels_changed)
        self._label_store.labels_changed.connect(self._canvas.refresh_colors)

        # Class panel
        self._class_panel.active_class_changed.connect(self._on_active_class_changed)
        self._class_panel.schema_changed.connect(self._on_schema_changed)

        # Tool panel
        self._tool_panel.tool_changed.connect(self._on_tool_changed)
        self._tool_panel.export_requested.connect(self._open_export_dialog)
        self._tool_panel.export_requested.connect(lambda: self._set_workflow_step(4))
        self._tool_panel.color_mode_changed.connect(self._on_color_mode_changed)
        self._tool_panel.brush_radius_changed.connect(self._on_brush_radius_changed)
        self._tool_panel.brush_overlap_changed.connect(self._on_brush_overlap_changed)
        self._tool_panel.brush_thickness_changed.connect(self._on_brush_thickness_changed)
        self._tool_panel.radius_changed.connect(self._on_radius_changed)
        self._tool_panel.lasso_close_requested.connect(self._on_lasso_close)
        self._tool_panel.point_size_changed.connect(self._canvas.set_point_size)
        self._tool_panel.grid_toggled.connect(self._canvas.toggle_grid)
        self._tool_panel.show_unlabeled_toggled.connect(self._on_show_unlabeled)
        # v2.0: nuevas herramientas
        self._tool_panel.erase_mode_changed.connect(self._on_erase_mode_changed)
        self._tool_panel.only_unlabeled_changed.connect(self._on_only_unlabeled_changed)

        # Geo panel
        self._overlay_panel.layer_added.connect(self._on_overlay_add)
        self._overlay_panel.layer_removed.connect(self._on_overlay_remove)
        self._overlay_panel.layer_toggled.connect(self._on_overlay_toggle)
        self._overlay_panel.layer_opacity.connect(self._on_overlay_opacity)
        self._overlay_panel.layer_z_offset.connect(self._on_overlay_z_offset)
        self._overlay_panel.top_view_clicked.connect(self._on_top_view)
        self._overlay_panel.bg_opacity_changed.connect(self._on_cloud_opacity)
        # Nota: la gestión de capas vive SOLO en OverlayPanel (riel → "Capas de
        # referencia"). ToolPanel ya no duplica botones de añadir/quitar capa.
        # (El paso "Detectar terreno" se quitó del panel — is_flat_mode()
        # siempre devolvía True, por lo que el MDT nunca se usaba realmente
        # en la clasificación AGL; ver _on_agl_auto_classify.)
        self._geo_panel.agl_auto_classify_requested.connect(self._on_agl_auto_classify)
        self._geo_panel.csf_classify_requested.connect(self._on_csf_classify)
        self._geo_panel.agl_select_requested.connect(self._on_agl_select)
        self._geo_panel.rules_apply_all_requested.connect(self._on_rules_apply_all)

        # Tile panel (NEW)
        self._tile_panel.tile_selected.connect(self._on_tile_selected)
        self._tile_panel.exit_requested.connect(self._on_tile_exit)
        self._tile_panel.view_mode_requested.connect(self._on_view_mode_requested)
        self._tile_panel.tile_size_changed.connect(self._on_tile_size_changed)
        self._tile_panel.transform_changed.connect(self._on_tile_transform_changed)
        self._tile_panel.grid_edit_mode_changed.connect(self._on_grid_edit_mode_changed)
        self._canvas.sig.tile_hovered.connect(self._on_canvas_tile_hover)
        # v3.0: nuevas señales
        # Feature 1: hover desde panel 2D → resaltar en canvas 3D
        # Feature 2: botones Mover/Rotar → modo de edición del grid en canvas
        # Drag del grid: preview (solo spinboxes) y set (rebuild completo)
        self._canvas.sig.grid_transform_preview.connect(self._on_grid_transform_preview)
        self._canvas.sig.grid_transform_set.connect(self._on_grid_transform_set)

    def _setup_shortcuts(self) -> None:
        pass   # atajos en keyPressEvent

    # ── Ayuda ─────────────────────────────────────────────────────────────────

    def show_shortcuts(self) -> None:
        from ui.shortcuts_dialog import ShortcutsDialog
        dlg = ShortcutsDialog(self)
        dlg.exec_()

    def show_welcome_dialog(self) -> None:
        from ui.welcome_dialog import WelcomeDialog
        WelcomeDialog.show_always(self)

    # ── Gestión de proyectos ──────────────────────────────────────────────────

    def new_project(self) -> None:
        if not self._confirm_unsaved_changes():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar nube de puntos", "",
            "GA3D-Bin (*.ga3d_bin);;Nubes de puntos (*.las *.laz *.e57 *.ply *.pcd *.xyz *.txt *.csv *.npy);;"
            "LAS/LAZ (*.las *.laz);;E57 (*.e57);;PLY (*.ply);;PCD (*.pcd);;"
            "XYZ/TXT (*.xyz *.txt *.csv *.asc *.pts);;NumPy (*.npy);;Todos (*)")
        if path:
            self._new_project_from_file(path)

    def open_file(self, path: str) -> None:
        if os.path.isfile(path):
            self._new_project_from_file(path)

    def _new_project_from_file(self, path: str) -> None:
        # Si es un .ga3d_bin, cargar directamente sin pasar por el pipeline LAS
        if Path(path).suffix.lower() == ".ga3d_bin":
            project = Project.new(path)
            self._project = project
            self._update_title()
            # Verificar que el archivo es compatible antes de cargar
            try:
                from core.heavy_cloud import _read_hdr
                _read_hdr(Path(path))   # lanza excepcion si es incompatible
                self._load_ga3d_bin(path)
            except Exception as exc:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(self, "GA3D-Bin incompatible",
                    f"{exc}\n\n"
                    "Elimina el archivo .ga3d_bin y abre el .las original para reconvertirlo.")
            return
        project = Project.new(path)
        self._load_cloud(path, project)

    def load_project(self) -> None:
        if not self._confirm_unsaved_changes():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Cargar proyecto", "",
            "Proyectos GeoAnnotate3D (*.geoa3d);;Todos (*)")
        if not path:
            return
        try:
            project = Project.load(path)
        except Exception as exc:
            QMessageBox.critical(self, "Error al cargar proyecto",
                                 f"{exc}\n\nEl archivo puede estar corrupto.")
            return

        ok, msg = project.verify_source()
        if not ok:
            reply = QMessageBox.warning(
                self, "Advertencia de integridad", msg,
                QMessageBox.Ok | QMessageBox.Cancel)
            if reply == QMessageBox.Cancel:
                return

        if not Path(project.source_file).exists():
            new_path, _ = QFileDialog.getOpenFileName(
                self, f"Localizar nube fuente — {Path(project.source_file).name}",
                str(Path(path).parent),
                "Nubes de puntos (*.las *.laz *.ply *.pcd *.xyz *.npy *.e57);;Todos (*)")
            if not new_path:
                return
            project.source_file = new_path

        self._project = project
        self._load_cloud(project.source_file, project)

    # ── Heavy Cloud ──────────────────────────────────────────────────────────

    def _on_heavy_needed(self, pc) -> None:
        from core.heavy_cloud import is_converted, ga3d_bin_path, HeavyCloudConverter
        src_path = getattr(pc, '_heavy_src', None)
        if not src_path: return

        # Inicializar el pc en la UI (antes cloud_ready lo hacía,
        # pero lo quitamos para evitar el error COM 0x80010100)
        self._pc = pc
        self._project.sync_from_cloud(pc)
        if self._project.labels is None or len(self._project.labels) != pc.n_points:
            self._project.init_labels(pc.n_points)
        autosave_p = None
        if self._project.source_file:
            from pathlib import Path as _P
            autosave_p = str(_P(self._project.source_file).parent /
                             (_P(self._project.source_file).stem + ".geoa3d_autosave"))
        self._label_store.attach(self._project.labels, autosave_path=autosave_p)
        self._header.set_file(pc.filename, pc.n_points)
        self._set_workflow_step(2)   # nube cargada → paso 2
        # Mostrar automáticamente la pestaña de Pre-clasificar
        if hasattr(self, '_left_stack'):
            self._left_stack.setCurrentIndex(1)

        # Si ya existe el .ga3d_bin, preguntar si usarlo (no cargarlo directamente
        # sin confirmacion para evitar errores con archivos incompatibles)
        if is_converted(src_path):
            from PyQt5.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self, "GA3D-Bin encontrado",
                f"Se encontró un archivo GA3D-Bin junto a esta nube:\n"
                f"{ga3d_bin_path(src_path)}\n\n"
                "¿Cargarlo ahora? (Si da error, di No para reconvertir)",
                QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self._load_ga3d_bin(str(ga3d_bin_path(src_path)))
                return
            # Si dice No: ofrecer reconvertir (borrando el existente)
            try:
                from core.heavy_cloud import _xyz_cache_path
                bp = ga3d_bin_path(src_path)
                if bp.exists():
                    try: bp.unlink()
                    except Exception: pass
            except Exception: pass

        from PyQt5.QtWidgets import QMessageBox
        mins = max(1, pc.n_points // 60_000_000)
        reply = QMessageBox.question(
            self, "Nube enorme detectada",
            f"Esta nube tiene {pc.n_points/1e6:.0f}M puntos.\n\n"
            "Para abrirla sin problemas se convierte UNA VEZ a formato GA3D-Bin "
            "(similar al .bin de Leica).\n\n"
            f"Tiempo estimado: ~{mins} min en SSD\n"
            "Las siguientes aperturas serán instantáneas.\n\n"
            "¿Convertir ahora?",
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            # El usuario no quiere convertir → cargar la nube normalmente
            self._load_heavy_as_normal(pc)
            return

        self._status.show_loading(
            f"Convirtiendo {pc.n_points/1e6:.0f}M pts a GA3D-Bin… (no cerrar el programa)")
        self._heavy_conv = HeavyCloudConverter(src_path, parent=self)
        self._heavy_conv.progress.connect(self._status.update_loading_progress)
        self._heavy_conv.done.connect(self._on_heavy_conv_done)
        self._heavy_conv.error.connect(self._on_heavy_conv_error)
        self._heavy_conv.start()

    def _load_heavy_as_normal(self, pc) -> None:
        """
        Carga la nube enorme con el flujo normal (octree sobre mmap xyz).
        El octree usa el cap de 150M pts para no crashear la GPU.
        El usuario eligio no convertir a GA3D-Bin.
        """
        project = self._project
        if project is None or pc is None: return
        if project.labels is None:
            project.init_labels(pc.n_points)
        self._pc = pc
        self._canvas.load_cloud(pc, project)
        self._header.set_file(pc.filename, pc.n_points)
        if pc.crs: self._header.set_crs(pc.crs)
        self._class_panel.set_project(project)
        self._tool_panel.set_cloud(pc)
        self._status.set_cloud(pc, project)
        self._status.show_loading("Construyendo LOD…")
        self._bin_octree_thread = _OctreeBuilder(pc)
        self._bin_octree_thread.ready.connect(self._on_bin_octree_ready)
        self._bin_octree_thread.start()

    def _on_heavy_conv_done(self, bin_path: str) -> None:
        self._status.hide_loading()
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(self, "Conversión completada",
            f"Archivo GA3D-Bin guardado junto a la nube original:\n\n"
            f"{bin_path}\n\n"
            "La próxima vez que abras el LAS/LAZ original se cargará "
            "automáticamente este archivo optimizado.\n\n"
            "También puedes abrir el .ga3d_bin directamente.")
        self._load_ga3d_bin(bin_path)

    def _on_heavy_conv_error(self, msg: str) -> None:
        self._status.hide_loading()
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Error de conversión", msg[:600])

    def _load_ga3d_bin(self, bin_path: str) -> None:
        """
        Carga el .ga3d_bin como un PointCloud normal.
        Usa el mismo pipeline que cualquier otra nube — 
        el octree se construye normalmente con el cap de 150M pts.
        """
        from core.heavy_cloud import load_ga3d_bin
        from core.octree import Octree
        from render.colors import set_z_range_cache
        try:
            self._status.show_loading("Cargando GA3D-Bin…")
            pc = load_ga3d_bin(bin_path)
            pc.filename = self._pc.filename if self._pc else "nube"
            # Reutilizar el proyecto actual
            project = self._project
            if project is None: return
            if (project.labels is None or
                    len(project.labels) != pc.n_points):
                project.init_labels(pc.n_points)
            self._pc = pc
            self._canvas.load_cloud(pc, project)

            # CRÍTICO: conectar label_store al array de labels.
            # Sin esto label_store._labels queda None y annotate() no hace nada.
            autosave_p = str(Path(bin_path).with_suffix(".geoa3d_autosave"))
            self._label_store.attach(project.labels, autosave_path=autosave_p)
            self._label_store._autosave_t.setInterval(5 * 60_000)

            self._header.set_file(pc.filename, pc.n_points)
            if pc.crs: self._header.set_crs(pc.crs)
            self._class_panel.set_project(project)
            self._tool_panel.set_cloud(pc)
            self._status.set_cloud(pc, project)

            # Construir octree en background
            self._status.show_loading("Construyendo LOD…")
            from core.pointcloud import CloudPipeline
            self._bin_octree_thread = _OctreeBuilder(pc)
            self._bin_octree_thread.ready.connect(self._on_bin_octree_ready)
            self._bin_octree_thread.start()

        except Exception as exc:
            self._status.hide_loading()
            from PyQt5.QtWidgets import QMessageBox
            msg = str(exc)
            # Si es un archivo incompatible, ofrecer borrarlo y reconvertir
            if "Elimina" in msg or "incompatible" in msg or "non-negative" in msg or "WinError 8" in msg:
                reply = QMessageBox.question(
                    self, "GA3D-Bin incompatible",
                    f"{msg}\n\n"
                    "¿Quieres eliminar el archivo .ga3d_bin para volver a convertirlo?",
                    QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes and hasattr(self, '_pc') and self._pc:
                    src_path = getattr(self._pc, '_heavy_src', None)
                    if src_path:
                        from core.heavy_cloud import ga3d_bin_path
                        bp = ga3d_bin_path(src_path)
                        try:
                            if bp.exists(): bp.unlink()
                        except Exception: pass
                        QMessageBox.information(self, "Listo",
                            "Archivos eliminados. Vuelve a abrir el .las original para reconvertir.")
            else:
                QMessageBox.critical(self, "Error cargando GA3D-Bin", msg)

    def _on_bin_octree_ready(self, octree) -> None:
        self._status.hide_loading()
        if self._pc and octree:
            self._pc.octree = octree
            self._canvas.first_display(self._pc)
            self._canvas._req_worker()
            self._status.set_octree_ready(True)
            saved_size = getattr(self._project, 'tile_size_m', 50.0) or 50.0
        ox  = getattr(self._project, 'grid_offset_x', 0.0)
        oy  = getattr(self._project, 'grid_offset_y', 0.0)
        rot = getattr(self._project, 'grid_rotation',  0.0)
        # Restaurar tamaño en el panel
        if hasattr(self._tile_panel, '_current_size_m'):
            self._tile_panel._current_size_m = saved_size
        if hasattr(self._tile_panel, '_custom_size'):
            self._tile_panel._custom_size.setValue(saved_size)
        self._build_tile_manager(saved_size, ox, oy, rot)

    def save_project(self) -> None:
        if self._project is None:
            return
        # Sincronizar tile size y transform antes de guardar
        if hasattr(self._tile_panel, 'get_tile_size'):
            self._project.tile_size_m = self._tile_panel.get_tile_size()
        if hasattr(self._tile_panel, 'get_transform'):
            ox, oy, rot = self._tile_panel.get_transform()
            self._project.grid_offset_x = ox
            self._project.grid_offset_y = oy
            self._project.grid_rotation  = rot
        self._project.stop_session()
        default_name = f"{self._project.name}.geoa3d"
        default_dir  = (str(Path(self._project.source_file).parent)
                        if self._project.source_file else "")
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar proyecto",
            str(Path(default_dir) / default_name) if default_dir else default_name,
            "Proyectos GeoAnnotate3D (*.geoa3d);;Todos (*)")
        if path:
            try:
                self._project.save(path)
                self._project._has_saved_path = True   # permite anotar
                self._header.set_saved(True)
                self._update_title()
                # Actualizar autosave path con el path definitivo
                autosave_p = str(Path(path).with_suffix(".geoa3d_autosave"))
                self._label_store._autosave_path = autosave_p
                self._label_store._autosave_t.start()
            except Exception as exc:
                QMessageBox.critical(self, "Error al guardar", str(exc))
        self._project.start_session()

    def replace_cloud(self) -> None:
        if self._project is None:
            return
        reply = QMessageBox.question(
            self, "Reemplazar nube",
            "Las etiquetas actuales se perderán.\n¿Guardar proyecto antes de continuar?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if reply == QMessageBox.Save:
            self.save_project()
        elif reply == QMessageBox.Cancel:
            return
        self.new_project()

    # ── Carga de nube ─────────────────────────────────────────────────────────

    def _load_cloud(self, path: str, project: Project) -> None:
        if self._pipeline and self._pipeline.isRunning():
            self._pipeline.terminate()
            self._pipeline.wait()

        # Resetear tile manager al cargar nueva nube
        self._tile_manager = None
        self._tile_panel.set_tile_manager(None)

        self._project = project
        # Si el proyecto ya viene de un archivo guardado, marcarlo
        if getattr(project, 'path', None) or getattr(project, '_saved_path', None):
            project._has_saved_path = True
        self._update_title()
        self._status.show_loading(Path(path).name)

        self._pipeline = CloudPipeline(path, self)
        self._pipeline.progress.connect(self._status.update_loading_progress)
        self._pipeline.cloud_ready.connect(self._on_cloud_ready)
        self._pipeline.octree_ready.connect(self._on_octree_ready)
        self._pipeline.error.connect(self._on_load_error)
        self._pipeline.heavy_needed.connect(self._on_heavy_needed)
        self._pipeline.start()

        self._project.start_session()
        self._session_timer.start()

    def _on_cloud_ready(self, pc: PointCloud) -> None:
        self._pc = pc
        self._project.sync_from_cloud(pc)

        # Inicializar labels si no existen O si tienen tamaño incorrecto
        # (puede ocurrir al cargar un proyecto .geoa3d con nube diferente)
        if (self._project.labels is None or
                len(self._project.labels) != pc.n_points):
            if self._project.labels is not None:
                print(f"[Project] labels size mismatch: "
                      f"{len(self._project.labels)} vs pc.n_points={pc.n_points}. "
                      f"Re-inicializando.")
            self._project.init_labels(pc.n_points)
        # Autosave: guardar etiquetas en .geoa3d_autosave cada 5 minutos
        autosave_path = None
        if self._project.source_file:
            autosave_path = str(
                Path(self._project.source_file).parent /
                (Path(self._project.source_file).stem + ".geoa3d_autosave"))
        self._label_store.attach(self._project.labels, autosave_path=autosave_path)
        if autosave_path:
            self._label_store._autosave_t.setInterval(5 * 60_000)  # 5 min

        # Detectar clasificación previa en la nube (solo si no hay anotaciones)
        if self._project.labels is not None and (self._project.labels > 0).sum() == 0:
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(300, lambda: self._check_and_import_classification(pc))

        # Verificar integridad si el proyecto ya tenía datos
        if (self._project.n_total > 0 and
                self._project.n_total != pc.n_points):
            issues = self._validate_project_integrity(self._project, pc)
            if issues:
                if not self._show_integrity_warning(issues):
                    self._status.hide_loading()
                    return
                # Re-inicializar labels si tamaños no coinciden
                if (self._project.labels is None or
                        len(self._project.labels) != pc.n_points):
                    self._project.init_labels(pc.n_points)

        self._header.set_file(pc.filename, pc.n_points)
        if self._project.crs:
            self._header.set_crs(self._project.crs)
        self._header.set_saved(False)
        self._header.set_mode(False)

        self._class_panel.set_project(self._project)
        self._geo_panel.set_schema(self._project.schema)
        self._terrain = TerrainModel()
        self._tool_panel.set_cloud(pc)
        self._status.set_cloud(pc, self._project)

        self._canvas.load_cloud(pc, self._project)

        # Auto-select best initial color mode based on cloud data
        n_labeled = int((self._project.labels > 0).sum()) if self._project.labels is not None else 0
        if n_labeled > 0:
            initial_mode = "Anotación"
        elif hasattr(pc, 'rgb') and pc.rgb is not None:
            initial_mode = "RGB"
        else:
            initial_mode = "Elevación"
        self._canvas.set_color_mode(initial_mode)
        self._tool_panel.set_color_mode(initial_mode)

        self._activate_tool("Pincel")

    def _on_octree_ready(self, octree, pc: PointCloud) -> None:
        if octree is None or pc is None: return   # heavy cloud no tiene octree
        pc.octree = octree
        self._canvas.first_display(pc)
        self._canvas._req_worker()
        self._status.set_octree_ready(True)

        # Sincronizar la barra de botones del panel con el modo real del canvas
        # (canvas arranca en "sparse" por defecto)
        self._tile_panel.set_view_mode(self._canvas.overview_mode)

        # Construir TileManager automáticamente al tener la nube lista
        self._build_tile_manager()

    def _on_load_error(self, msg: str) -> None:
        # Si el programa llegó aquí, Python pudo atrapar el error.
        # Si fue un SEGFAULT puro en C, este método nunca se ejecuta.
        # Detectar si fue un error de memoria y dar consejo específico.
        self._status.hide_loading()
        QMessageBox.critical(self, "Error de carga",
                             f"{msg[:600]}\n\npip install laspy[lazrs] open3d pye57")

    # ── Tile Manager (NEW) ────────────────────────────────────────────────────

    def _build_tile_manager(self, tile_size_m: float = 50.0,
                            offset_x: float = 0.0, offset_y: float = 0.0,
                            rotation_deg: float = 0.0) -> None:
        """Construye o reconstruye el TileManager con los parámetros dados."""
        if self._pc is None or self._project is None:
            return
        try:
            from core.tile_manager import TileManager
            tm = TileManager(self._pc, self._project, tile_size_m,
                             offset_x=offset_x, offset_y=offset_y,
                             rotation_deg=rotation_deg)
            self._tile_manager = tm
            self._tile_panel.set_tile_manager(tm)
            self._canvas.set_tile_manager(tm)   # actualiza overlay 3D
        except Exception as exc:
            print(f"[TileManager] Error construyendo: {exc}")
        # Pasar schema para colorear barras de clase por clase
        if (self._project and self._project.schema and
                hasattr(self._tile_panel, '_grid')):
            self._tile_panel.set_schema(self._project.schema)
        # Construir índice de tiles en background para acelerar extracciones futuras
        if self._tile_manager is not None:
            self._start_tile_index_build()

    def _start_tile_index_build(self) -> None:
        """Construye el tile index en un hilo de fondo. Solo una vez por sesion."""
        tm = self._tile_manager
        if tm is None or tm._tile_index_built: return
        # Evitar lanzar múltiples builds simultáneos
        if getattr(self, '_tile_index_builder', None) is not None:
            if self._tile_index_builder.isRunning(): return
        from PyQt5.QtCore import QThread, pyqtSignal

        class _IndexBuilder(QThread):
            done = pyqtSignal()
            def __init__(self, tm_, status_): super().__init__(); self._tm=tm_; self._st=status_
            def run(self):
                try:
                    self._tm.build_tile_index(
                        progress_cb=lambda p,m: self._st.show_loading(m) if p < 100 else None)
                    print(f"[TileIndex] Listo — {len(self._tm._tile_index)/1e6:.0f}M pts indexados")
                except Exception as e:
                    print(f"[TileIndex] Error: {e}")
                finally:
                    self.done.emit()

        self._tile_index_builder = _IndexBuilder(tm, self._status)
        self._tile_index_builder.done.connect(lambda: self._status.hide_loading())
        self._tile_index_builder.start()

    def _on_tile_size_changed(self, size_m: float) -> None:
        """El usuario cambió el tamaño de tile desde el panel."""
        if self._canvas.is_tile_mode:
            self._canvas.exit_tile_mode()
        ox, oy, rot = self._tile_panel.get_transform()
        self._build_tile_manager(size_m, ox, oy, rot)

    def _on_tile_transform_changed(self, offset_x: float,
                                   offset_y: float, rotation_deg: float) -> None:
        """Spinboxes cambiaron manualmente — reconstruir con transform."""
        if self._canvas.is_tile_mode:
            return
        size_m = self._tile_panel.get_tile_size()
        self._build_tile_manager(size_m, offset_x, offset_y, rotation_deg)

    def _on_view_mode_requested(self, mode: str) -> None:
        """Cambia el modo de vista: Sparse, Full o Volar."""
        if mode == "Volar":
            if hasattr(self._canvas, 'toggle_fly_mode'):
                self._canvas.toggle_fly_mode()
        elif mode == "Sparse":
            self._canvas.set_overview_mode("sparse")
        elif mode == "Full":
            self._canvas.set_overview_mode("full")
        self._tile_panel.set_view_mode(mode)

    def _on_grid_edit_mode_changed(self, mode: str) -> None:
        """Feature 2: activar/desactivar modo de edición del grid con mouse."""
        self._canvas.set_grid_edit_mode(mode if mode else None)

    def _on_grid_transform_preview(self, ox: float, oy: float, rot: float) -> None:
        """Durante drag: solo actualizar spinboxes (el canvas ya actualizó la geometría)."""
        self._tile_panel.update_transform_spinboxes(ox, oy, rot)

    def _on_grid_transform_set(self, ox: float, oy: float, rot: float) -> None:
        """Al soltar el mouse: rebuild completo con sampling, conservando el tile size."""
        # Leer el tile_size del TileManager activo en el canvas (fuente de verdad
        # durante el drag) y del panel — usar el mayor valor confiable
        canvas_tm = getattr(self._canvas, '_tile_manager', None)
        if canvas_tm is not None and hasattr(canvas_tm, 'tile_size_m'):
            size_m = canvas_tm.tile_size_m
        else:
            size_m = self._tile_panel.get_tile_size()
        print(f"[GridTransformSet] size={size_m:.1f}m  ox={ox:.1f}  oy={oy:.1f}  rot={rot:.1f}")
        self._build_tile_manager(size_m, ox, oy, rot)
        self._tile_panel.update_transform_spinboxes(ox, oy, rot)

    def _on_tile_selected(self, tile) -> None:
        """
        El usuario hizo click en un tile — puede venir desde overview O desde
        otro tile activo (tile-to-tile navigation directa).
        """
        if self._pc is None or self._tile_manager is None:
            return

        # ── Guardar progreso del tile actual antes de cambiar ────────────────
        if self._canvas.is_tile_mode and self._canvas._active_tile is not None:
            if self._tile_manager is not None and self._canvas._tile_indices is not None:
                self._tile_manager.update_tile_progress(
                    self._canvas._active_tile,
                    self._canvas._tile_indices)
            # Cancelar el loader del tile anterior sin volver al overview:
            # seteamos _tile_mode=False para que enter_tile_mode lo sobrescriba
            # sin disparar exit_tile_mode (que haría reset de cámara al overview)
            self._canvas._tile_mode = False
            self._canvas._active_tile = None
            self._canvas._tile_indices = None
            if (self._canvas._tile_loader is not None and
                    self._canvas._tile_loader.isRunning()):
                self._canvas._tile_loader.terminate()
                self._canvas._tile_loader.wait()
            self._canvas._tile_loader = None

        # ── Cancelar extracción anterior si estaba en curso ──────────────────
        if self._tile_worker is not None and self._tile_worker.isRunning():
            self._tile_worker.terminate()
            self._tile_worker.wait()

        # Mostrar tile como activo en el panel antes de que cargue
        self._tile_panel.set_active_tile(tile)

        n_str = f"{tile.n_points/1e6:.1f}M pts" if tile.n_points > 0 else "pts"

        # ── PATH RÁPIDO: usar tile index si ya está construido ───────────────
        fast_idx = self._tile_manager.get_tile_indices_fast(tile)
        if fast_idx is not None and len(fast_idx) > 0:
            self._status.show_loading(
                f"Tile ({tile.col},{tile.row}) — {len(fast_idx)/1e6:.1f}M pts ✓")
            self._on_tile_extracted(tile, fast_idx)
            return

        # ── PATH LENTO: worker que escanea toda la nube ──────────────────────
        self._status.show_loading(
            f"Extrayendo tile ({tile.col},{tile.row}) — {n_str} (construyendo índice…)")

        self._tile_worker = self._tile_manager.create_extraction_worker(tile)
        self._tile_worker.progress.connect(self._status.update_loading_progress)
        self._tile_worker.ready.connect(self._on_tile_extracted)
        self._tile_worker.error.connect(self._on_tile_extract_error)
        self._tile_worker.start()

    def _on_tile_extracted(self, tile, indices: np.ndarray) -> None:
        """Extracción completa — entrar en tile mode en el canvas."""
        self._status.hide_loading()

        if len(indices) == 0:
            QMessageBox.information(
                self, "Tile vacío",
                f"El tile ({tile.col},{tile.row}) no contiene puntos.\n"
                "Elige otro tile.")
            # Si veníamos de otro tile, el canvas ya salió; volver a overview
            self._tile_panel.set_active_tile(None)
            self._header.set_mode(False, None)
            if not self._canvas.is_tile_mode:
                self._canvas._upload_coarse(self._canvas.pc)
            return

        # Entrar en tile mode en el canvas (carga el tile en background)
        self._canvas.enter_tile_mode(tile, indices)
        # El badge del header se actualiza en _on_tile_mode_changed
        if self._tile_dock: self._tile_dock.raise_()

    def _on_tile_extract_error(self, msg: str) -> None:
        self._status.hide_loading()
        QMessageBox.critical(self, "Error extrayendo tile", msg[:400])
        self._tile_panel.set_active_tile(None)

    def _on_tile_exit(self) -> None:
        """El usuario pulsó '← Vista global' en el TilePanel."""
        # Actualizar progreso del tile antes de salir
        if self._canvas.is_tile_mode and self._canvas._active_tile is not None:
            if self._tile_manager is not None:
                self._tile_manager.update_tile_progress(
                    self._canvas._active_tile,
                    self._canvas._tile_indices)
            self._tile_panel.refresh_grid()

        self._canvas.exit_tile_mode()

    def _on_tile_mode_changed(self, in_tile_mode: bool, tile) -> None:
        """Canvas notificó cambio de modo tile."""
        self._header.set_mode(in_tile_mode, tile)
        self._tile_panel.set_active_tile(tile if in_tile_mode else None)

        if not in_tile_mode:
            # Al volver al overview, refrescar estadísticas del grid
            if self._tile_manager is not None:
                self._tile_manager.refresh_all_progress()
                self._tile_panel.refresh_grid()

    # ── Exportación ───────────────────────────────────────────────────────────

    def _open_export_dialog(self) -> None:
        if self._project is None or self._pc is None:
            return
        dlg = ExportDialog(self._pc, self._project, self)
        dlg.exec_()

    # ── Callbacks de estado ───────────────────────────────────────────────────

    def _on_active_class_changed(self, class_id: int) -> None:
        if self._active_tool is not None:
            self._active_tool.active_class_id = class_id

    def _on_schema_changed(self, new_schema: list) -> None:
        """Schema changed (color/name edit or import) → rebuild LUT and repaint."""
        # 1. Update project schema (canvas.project is the same object)
        if self._project:
            self._project.schema = new_schema

        # 2. Rebuild the annotation LUT arrays from the updated schema
        #    rebuild_annotation_lut reads self.project.schema which is now updated
        try:
            self._canvas.rebuild_annotation_lut()
        except Exception as e:
            print(f"[Schema] rebuild_annotation_lut: {e}")

        # 3. Force a full color repaint with the new LUT
        try:
            self._canvas.force_color_rebuild()
        except Exception as e:
            print(f"[Schema] force_color_rebuild: {e}")

        # 4. Update other panels
        self._geo_panel.set_schema(new_schema)
        self._on_stats_changed()
        self._annotations_saved = False


    def _on_tool_changed(self, tool_name: str) -> None:
        self._activate_tool(tool_name)

    def _activate_tool(self, tool_name: str) -> None:
        tool_cls = TOOL_BY_NAME.get(tool_name)
        if tool_cls is None:
            return

        tool = tool_cls()
        tool.label_store     = self._label_store
        tool.active_class_id = self._class_panel._active_id
        # Propagar opciones globales al nuevo tool
        if self._active_tool is not None:
            tool.erase_mode     = self._active_tool.erase_mode
            tool.only_unlabeled = self._active_tool.only_unlabeled
        self._active_tool = tool
        self._canvas.set_active_tool(tool)
        self._tool_panel.set_active_tool(tool_name)

    def _on_color_mode_changed(self, mode: str) -> None:
        self._canvas.set_color_mode(mode)

    def _on_brush_radius_changed(self, radius_m: float) -> None:
        from annotation.tools import BrushTool
        if isinstance(self._active_tool, BrushTool):
            self._active_tool.radius_m = radius_m

    def _on_brush_overlap_changed(self, pct: float) -> None:
        from annotation.tools import BrushTool
        if isinstance(self._active_tool, BrushTool):
            self._active_tool.overlap_pct = pct / 100.0

    def _on_brush_thickness_changed(self, pct: float) -> None:
        from annotation.tools import BrushTool
        if isinstance(self._active_tool, BrushTool):
            self._active_tool.thickness = pct / 100.0

    def _on_radius_changed(self, radius_m: float) -> None:
        from annotation.tools import SphereSelectTool
        if isinstance(self._active_tool, SphereSelectTool):
            self._active_tool.radius_m = radius_m

    def _on_lasso_close(self) -> None:
        from annotation.tools import PolygonTool
        if isinstance(self._active_tool, PolygonTool):
            if len(self._active_tool._pts) >= 3:
                self._active_tool._apply_and_reset()

    def _on_show_unlabeled(self, show: bool) -> None:
        if self._canvas._annotation_lut is not None:
            self._canvas._annotation_lut[0, 3] = 0.45 if show else 0.0
        if self._canvas._annotation_lut_u8 is not None:
            self._canvas._annotation_lut_u8[0, 3] = 115 if show else 0
        self._canvas.refresh_colors()

    # ── Nuevos handlers v2.0 ──────────────────────────────────────────────────

    def _on_erase_mode_changed(self, v: bool) -> None:
        if self._active_tool: self._active_tool.erase_mode = v

    def _on_only_unlabeled_changed(self, v: bool) -> None:
        if self._active_tool: self._active_tool.only_unlabeled = v




    def _on_slice_mode_changed(self, mode: str) -> None:
        from annotation.tools import SliceTool
        if isinstance(self._active_tool, SliceTool):
            self._active_tool.mode = mode

    # ── Geo handlers ──────────────────────────────────────────────────────────

    def _on_agl_select(self, min_agl: float, max_agl: float) -> None:
        if self._pc is None or self._project is None or not self._terrain.ready:
            QMessageBox.information(self, "AGL",
                "Calcula el MDT primero (pulsa Calcular MDT en el panel Geo).")
            return
        mask = self._terrain.agl_range_mask(self._pc.xyz, min_agl, max_agl)
        idx  = np.where(mask)[0].astype(np.int32)
        if len(idx) == 0:
            QMessageBox.information(self, "AGL",
                f"No hay puntos entre {min_agl:.1f} y {max_agl:.1f} m sobre el suelo.")
            return
        class_id = self._geo_panel.get_agl_class_id()
        self._label_store.annotate(idx, class_id)
        self._canvas.refresh_colors(idx, class_id)

    def _on_rules_apply_all(self) -> None:
        if self._pc is None or self._project is None or self._project.labels is None:
            return
        rules = self._geo_panel.get_active_rules()
        if not rules:
            QMessageBox.information(self, "Reglas",
                "No hay reglas activas.\nAñade reglas desde las plantillas del panel Geo.")
            return
        total   = 0
        terrain = self._terrain if self._terrain.ready else None
        for rule in rules:
            self._rules_engine.clear()
            self._rules_engine.add_rule(rule)
            n = self._rules_engine.apply(self._pc, self._project.labels, terrain)
            total += n
        self._rules_engine.clear()
        self._canvas.refresh_colors()
        self._on_stats_changed()
        QMessageBox.information(self, "Reglas aplicadas",
            f"Se clasificaron {total:,} puntos con {len(rules)} regla(s).")

    def _on_point_picked(self, utm_e: float, utm_n: float, utm_z: float) -> None:
        if self._pc is None or self._project is None:
            return
        from utils.spatial import sphere_query
        from utils.geo import format_utm_coords
        offset = self._pc.offset
        local  = np.array([utm_e - offset[0], utm_n - offset[1],
                           utm_z - offset[2]], np.float32)
        idx    = sphere_query(self._pc.xyz, local, 0.05)
        if len(idx) > 0 and self._project.labels is not None:
            cid   = int(self._project.labels[idx[0]])
            sc    = next((s for s in self._project.schema if s.id == cid), None)
            cname  = sc.name  if sc else f"clase {cid}"
            ccolor = sc.color if sc else "#55585c"
        else:
            cname, ccolor = "desconocido", "#55585c"
        es, ns, zs = format_utm_coords(local, offset)
        self._tool_panel.show_pick_result(cname, ccolor, es, ns, zs)

    def _on_measure_done(self, d3d: float, dh: float, dz: float) -> None:
        self._tool_panel.show_measure_result(d3d, dh, dz)

    def _on_labels_changed(self, *args) -> None:
        self._annotations_saved = False
        if hasattr(self, '_header') and hasattr(self._header, '_set_step'):
            self._set_workflow_step(3)
        # Cambiar al tab de Clases para ver el balance en tiempo real
        if hasattr(self, '_left_stack') and self._left_stack.currentIndex() != 2:
            self._left_stack.setCurrentIndex(2)

    def _on_stats_changed(self) -> None:
        if self._project and self._label_store:
            self._status.update_stats_with_schema(
                self._label_store, self._project.schema)
            self._class_panel.update_counts(self._label_store.per_class_counts())
            n = self._label_store.n_labeled
            self._class_panel.update_progress(n, self._label_store.n_total)
            self._tool_panel.update_export_note(n)

    # ── Helpers de UI ─────────────────────────────────────────────────────────

    def _update_title(self) -> None:
        if self._project and self._project.name:
            name = self._project.name
        elif self._project and getattr(self._project, 'source_path', None):
            from pathlib import Path
            name = Path(self._project.source_path).stem
        else:
            name = "sin proyecto"
        self.setWindowTitle(f"GeoAnnotate3D — {name}")

    def _update_session_timer(self) -> None:
        if self._project:
            total = self._project.annotation_time_s + 1
            self._project.annotation_time_s = total
            h, r = divmod(total, 3600)
            m, s = divmod(r, 60)
            self._header.set_session_time(h, m, s)
            self._status.update_session_time(h, m, s)

    def _confirm_unsaved_changes(self) -> bool:
        if self._project is None or self._label_store.n_labeled == 0:
            return True
        reply = QMessageBox.question(
            self, "Cambios sin guardar",
            "Hay anotaciones sin guardar. ¿Guardar antes de continuar?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if reply == QMessageBox.Save:
            self.save_project()
            return True
        return reply == QMessageBox.Discard

    # ── Eventos Qt ────────────────────────────────────────────────────────────

    def keyPressEvent(self, event) -> None:
        key     = event.key()
        mod     = event.modifiers()
        key_str = event.text().upper()

        # Cambio de clase 1–9
        if Qt.Key_1 <= key <= Qt.Key_9 and not mod:
            self._class_panel.set_active_class(key - Qt.Key_0)
            return

        # Undo / Redo
        if key == Qt.Key_Z and mod & Qt.ControlModifier:
            if mod & Qt.ShiftModifier:
                did = self._label_store.redo()
                op_type = "redo"
            else:
                did = self._label_store.undo()
                op_type = "undo"
            stack_size = len(self._label_store._undo_stack)
            print(f"[{op_type}] did={did}  undo_stack={stack_size}")
            if did:
                # Estrategia: si estamos en tile mode, re-entrar al tile
                # (garantizado correcto). Si no, force_color_rebuild.
                if (self._canvas.is_tile_mode and
                        self._canvas._active_tile is not None and
                        self._canvas._tile_indices is not None):
                    # Re-entrar al tile con fit_camera=False fuerza reload
                    # completo de colores desde los labels actuales (ya modificados
                    # por undo). Es la forma más robusta — el mismo path que
                    # carga el tile por primera vez.
                    self._canvas.enter_tile_mode(
                        self._canvas._active_tile,
                        self._canvas._tile_indices,
                        fit_camera=False)
                else:
                    self._canvas.force_color_rebuild()
                self._on_stats_changed()
            else:
                # Stack vacío — informar al usuario
                if op_type == "undo":
                    self._status.show_message("Nada que deshacer", timeout=1500)
                else:
                    self._status.show_message("Nada que rehacer", timeout=1500)
            return

        # Guardar
        if key == Qt.Key_S and mod & Qt.ControlModifier:
            self.save_project(); return

        # Reset vista
        if key == Qt.Key_Space:
            self._canvas.reset_view(); return

        # F — toggle fly mode
        if key == Qt.Key_F and not mod:
            if hasattr(self._canvas, 'toggle_fly_mode'):
                self._canvas.toggle_fly_mode()
            self._tile_panel.set_view_mode("Volar")
            return

        # E — toggle erase mode
        if key == Qt.Key_E and not mod:
            if self._active_tool is not None:
                new_val = not self._active_tool.erase_mode
                self._active_tool.erase_mode = new_val
                self._tool_panel.set_erase_mode(new_val)
            return

        # Ctrl+T — ir al tab de Tiles
        if key == Qt.Key_T and mod & Qt.ControlModifier:
            if hasattr(self, '_left_tabs'):
                self._left_stack.setCurrentIndex(1)  # index 2 = Tiles
            return

        # Escape — salir del tile mode (NEW)
        if key == Qt.Key_Escape and self._canvas.is_tile_mode:
            self._on_tile_exit()
            return

        # Vistas de cámara — ANTES de tool_key_map para que no las intercepte
        if key == Qt.Key_V and not mod:
            self._set_top_view(); return
        if key == Qt.Key_Y and not mod:
            self._set_side_view(); return
        if key == Qt.Key_R and not mod:
            self._reset_camera_view(); return

        # Atajos de herramientas
        if not mod:
            tool_key_map = {t.key.upper(): t.name
                            for t in TOOL_BY_NAME.values() if t.key}
            if key_str in tool_key_map:
                self._activate_tool(tool_key_map[key_str])
                return

        super().keyPressEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)

    def closeEvent(self, event) -> None:
        if self._confirm_unsaved_changes():
            if self._project:
                self._project.stop_session()
            if self._pipeline and self._pipeline.isRunning():
                self._pipeline.terminate()
                self._pipeline.wait()
            if self._tile_worker and self._tile_worker.isRunning():
                self._tile_worker.terminate()
                self._tile_worker.wait()
            self._save_window_state()
            # Cierre limpio: borrar el marcador de sesión para que el
            # próximo arranque no lo confunda con un crash (ver
            # _setup_crash_handler).
            try:
                if getattr(self, "_session_marker", None) and self._session_marker.exists():
                    self._session_marker.unlink()
            except Exception:
                pass
            event.accept()
        else:
            event.ignore()

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            valid_exts = {".las", ".laz", ".ply", ".pcd", ".e57",
                          ".xyz", ".txt", ".csv", ".asc", ".pts", ".npy",
                          ".ga3d_bin"}
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if Path(path).suffix.lower() in valid_exts:
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event) -> None:
        valid_exts = {".las", ".laz", ".ply", ".pcd", ".e57",
                      ".xyz", ".txt", ".csv", ".asc", ".pts", ".npy",
                      ".ga3d_bin"}
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and Path(path).suffix.lower() in valid_exts:
                self.open_file(path)
                break

    # ── Overlay handlers ──────────────────────────────────────────────────────

    def _ensure_overlay_manager(self):
        import numpy as np
        from core.overlay_manager import OverlayManager
        if self._overlay_manager is None:
            crs    = getattr(self._project, 'crs', '') if self._project else ''
            offset = self._pc.offset if self._pc else np.zeros(3, np.float64)
            bounds = (self._pc.bounds if self._pc
                      else np.array([[-500,-500,-10],[500,500,100]], np.float32))
            self._overlay_manager = OverlayManager(offset, bounds, crs)
            if self._canvas:
                self._overlay_manager.set_canvas(self._canvas)
        elif self._canvas and self._overlay_manager._canvas is None:
            self._overlay_manager.set_canvas(self._canvas)
        return self._overlay_manager

    def _on_overlay_add(self, kind_path) -> None:
        kind, path = kind_path
        om = self._ensure_overlay_manager()
        if om is None:
            QMessageBox.warning(self, "Overlay", "Carga una nube primero.")
            return
        self._status.show_loading(f"Cargando {Path(path).name}...")
        from PyQt5.QtWidgets import QApplication
        QApplication.processEvents()
        layer = om.load_vector(path) if kind == "vector" else om.load_raster(path)
        self._overlay_panel.add_layer_ui(layer)
        if self._canvas:
            try: self._canvas.request_render()
            except Exception: pass
        self._status.hide_loading()

    def _on_overlay_remove(self, layer) -> None:
        om = self._ensure_overlay_manager()
        if om: om.remove_layer(layer)
        if self._canvas:
            try: self._canvas.request_render()
            except Exception: pass

    def _on_overlay_toggle(self, layer, visible: bool) -> None:
        om = self._ensure_overlay_manager()
        if om: om.set_layer_visible(layer, visible)
        if self._canvas:
            try: self._canvas.request_render()
            except Exception: pass

    def _on_overlay_opacity(self, layer, opacity: float) -> None:
        om = self._ensure_overlay_manager()
        if om: om.set_layer_opacity(layer, opacity)
        if self._canvas:
            try: self._canvas.request_render()
            except Exception: pass

    def _on_overlay_z_offset(self, layer, z_offset: float) -> None:
        om = self._ensure_overlay_manager()
        if om: om.set_z_offset(layer, z_offset)

    def _on_top_view(self) -> None:
        if self._canvas and hasattr(self._canvas, 'top_view'):
            self._canvas.top_view()

    def _on_cloud_opacity(self, opacity: float) -> None:
        if self._canvas is None: return
        try:
            self._canvas.set_cloud_opacity(opacity)
        except Exception as e:
            print(f"[cloud opacity] {e}")

    def _prefs_path(self) -> "Path":
        import sys
        return Path(sys.argv[0]).resolve().parent / ".geoannotate_prefs.json"

    def _maybe_show_onboarding(self) -> None:
        """Muestra el diálogo de cargar nube. Se llama siempre al inicio."""
        # Solo mostrar si no hay proyecto cargado ya
        if self._pc is not None:
            return
        dlg = self._build_onboarding_dialog()
        dlg.exec_()

    def _build_onboarding_dialog(self):
        """Diálogo flotante de bienvenida, centrado, con opción 'no mostrar'."""
        import json
        from PyQt5.QtWidgets import QDialog, QCheckBox, QDialogButtonBox, QScrollArea as _SA
        from PyQt5.QtCore import Qt

        dlg = QDialog(self)
        dlg.setWindowTitle("Bienvenido a GeoAnnotate3D")
        dlg.setFixedWidth(420)
        dlg.setStyleSheet("QDialog{background:#e8e9eb;}")

        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(20, 20, 20, 16); lay.setSpacing(10)

        # Header
        logo = QLabel("GeoAnnotate3D")
        logo.setStyleSheet("color:#0e7c86;font-size:16px;font-weight:600;"
                           "letter-spacing:1px;")
        logo.setAlignment(Qt.AlignCenter)
        sub = QLabel("Etiquetado LiDAR para redes neuronales")
        sub.setStyleSheet("color:#55585c;font-size:10.5px;padding-bottom:8px;")
        sub.setAlignment(Qt.AlignCenter)
        lay.addWidget(logo); lay.addWidget(sub)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background:#f4f5f6;max-height:1px;")
        lay.addWidget(sep)

        # Open button
        btn_open = QPushButton("  Abrir nube de puntos…")
        btn_open.setStyleSheet(
            "QPushButton{background:#0e7c86;color:#55585c;border:none;"
            "border-radius:4px;padding:10px;font-size:11px;font-weight:600;}"
            "QPushButton:hover{background:#0a5f67;}")
        def open_and_close():
            dlg.accept(); self.new_project()
        btn_open.clicked.connect(open_and_close)
        lay.addWidget(btn_open)

        hint = QLabel("o arrastra un archivo .las .laz .e57 a la ventana")
        hint.setStyleSheet("color:#55585c;font-size:10.5px;")
        hint.setAlignment(Qt.AlignCenter)
        lay.addWidget(hint)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("background:#f4f5f6;max-height:1px;margin:4px 0;")
        lay.addWidget(sep2)

        # Steps
        def step_card(num, title, body, kbd=""):
            card = QWidget()
            card.setStyleSheet(
                "background:#e8e9eb;border:1px solid #f4f5f6;"
                "border-radius:4px;margin:1px 0;")
            cl = QHBoxLayout(card)
            cl.setContentsMargins(10,7,10,7); cl.setSpacing(8)
            n = QLabel(num); n.setFixedSize(18,18); n.setAlignment(Qt.AlignCenter)
            n.setStyleSheet("background:#0e7c86;color:#55585c;border-radius:3px;"
                            "font-size:10.5px;font-weight:bold;")
            tl = QVBoxLayout(); tl.setSpacing(0)
            t = QLabel(title)
            t.setStyleSheet("color:#55585c;font-size:10.5px;font-weight:600;")
            d = QLabel(body); d.setWordWrap(True)
            d.setStyleSheet("color:#55585c;font-size:10.5px;")
            tl.addWidget(t); tl.addWidget(d)
            cl.addWidget(n); cl.addLayout(tl,1)
            if kbd:
                k = QLabel(kbd)
                k.setStyleSheet("background:#e8e9eb;border:1px solid #eceded;"
                    "border-radius:3px;color:#55585c;font-size:10.5px;padding:2px 5px;")
                cl.addWidget(k)
            return card

        lay.addWidget(step_card("1","Cargar nube",
            "Abre un .las .laz .e57 y configura el grid de tiles.",""))
        lay.addWidget(step_card("2","Pre-clasificar",
            "AGL clasifica por altura · Region Growing expande desde un punto.",""))
        lay.addWidget(step_card("3","Etiquetar",
            "Pincel, polígono, caja. Teclas 1-9 cambian la clase activa.","1–9"))
        lay.addWidget(step_card("4","Exportar",
            "Genera el dataset para RandLA-Net, PointNet++ o KPConv.",""))

        sep3 = QFrame(); sep3.setFrameShape(QFrame.HLine)
        sep3.setStyleSheet("background:#f4f5f6;max-height:1px;margin:4px 0;")
        lay.addWidget(sep3)

        # Footer: no mostrar + cerrar
        footer = QHBoxLayout()
        chk = QCheckBox("No volver a mostrar")
        chk.setStyleSheet("QCheckBox{color:#55585c;font-size:10.5px;}"
                          "QCheckBox::indicator{width:13px;height:13px;}")
        close_btn = QPushButton("Empezar")
        close_btn.setStyleSheet(
            "QPushButton{background:#f4f5f6;border:1px solid #eceded;"
            "border-radius:4px;color:#55585c;padding:6px 16px;font-size:10.5px;}"
            "QPushButton:hover{border-color:#0e7c86;color:#0e7c86;}")

        def on_close():
            if chk.isChecked():
                try:
                    pp = self._prefs_path()
                    prefs = json.loads(pp.read_text()) if pp.exists() else {}
                    prefs["hide_onboarding"] = True
                    pp.write_text(json.dumps(prefs, indent=2))
                except Exception: pass
            dlg.accept()

        close_btn.clicked.connect(on_close)
        footer.addWidget(chk); footer.addStretch(); footer.addWidget(close_btn)
        lay.addLayout(footer)
        return dlg

    def _build_onboarding(self):
        """Widget de bienvenida cuando no hay nube cargada."""
        from PyQt5.QtWidgets import QScrollArea as _SA
        w = QWidget(); w.setStyleSheet("background:#e8e9eb;")
        scroll = _SA(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(0)  # QFrame.NoFrame = 0
        scroll.setStyleSheet("background:transparent;border:none;")
        content = QWidget(); content.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(content)
        lay.setContentsMargins(14, 18, 14, 14); lay.setSpacing(0)

        def _lbl(txt, sty):
            l = QLabel(txt); l.setWordWrap(True); l.setStyleSheet(sty); return l

        lay.addWidget(_lbl("GeoAnnotate3D",
            "color:#0e7c86;font-size:15px;font-weight:600;letter-spacing:1px;"
            "padding-bottom:2px;qproperty-alignment:AlignCenter;"))
        lay.addWidget(_lbl("Etiquetado LiDAR para redes neuronales",
            "color:#55585c;font-size:10.5px;padding-bottom:18px;"
            "qproperty-alignment:AlignCenter;"))

        # Open button
        btn = QPushButton("Abrir nube de puntos…")
        btn.setStyleSheet(
            "QPushButton{background:#0e7c86;color:#55585c;border:none;"
            "border-radius:4px;padding:9px;font-size:10.5px;font-weight:600;}"
            "QPushButton:hover{background:#0a5f67;}")
        btn.clicked.connect(self.new_project)
        lay.addWidget(btn); lay.addSpacing(5)

        drag_hint = _lbl("o arrastra un archivo .las .laz .e57 a la ventana",
            "color:#55585c;font-size:10.5px;qproperty-alignment:AlignCenter;"
            "padding-bottom:18px;")
        lay.addWidget(drag_hint)

        # Steps
        def step_card(num, title, body, kbd=""):
            card = QWidget()
            card.setStyleSheet(
                "background:#e8e9eb;border:1px solid #f4f5f6;border-radius:3px;"
                "margin-bottom:6px;")
            cl = QHBoxLayout(card)
            cl.setContentsMargins(10, 8, 10, 8); cl.setSpacing(8)
            n = QLabel(num); n.setFixedSize(20,20); n.setAlignment(Qt.AlignCenter)
            n.setStyleSheet(
                "background:#0e7c86;color:#55585c;border-radius:10px;"
                "font-size:10.5px;font-weight:bold;")
            tl = QVBoxLayout(); tl.setSpacing(1)
            t = QLabel(title)
            t.setStyleSheet("color:#55585c;font-size:10.5px;font-weight:600;")
            d = QLabel(body); d.setWordWrap(True)
            d.setStyleSheet("color:#55585c;font-size:10.5px;")
            tl.addWidget(t); tl.addWidget(d)
            cl.addWidget(n); cl.addLayout(tl, 1)
            if kbd:
                k = QLabel(kbd)
                k.setStyleSheet("background:#e8e9eb;border:1px solid #eceded;"
                    "border-radius:3px;color:#55585c;font-size:10.5px;padding:2px 5px;")
                cl.addWidget(k)
            return card

        lay.addWidget(step_card("1","Cargar nube",
            "Abre un archivo LiDAR en cualquier formato soportado.","Ctrl+O"))
        lay.addWidget(step_card("2","Pre-clasificar",
            "AGL clasifica por altura · Region Growing expande desde un punto.",""))
        lay.addWidget(step_card("3","Etiquetar",
            "Pincel, polígono, caja y borrador. Teclas 1-9 cambian la clase.","1–9"))
        lay.addWidget(step_card("4","Exportar dataset",
            "Genera el dataset para RandLA-Net, PointNet++ o KPConv.",""))

        lay.addSpacing(14)
        lay.addWidget(_lbl("Formatos soportados",
            "color:#55585c;font-size:10.5px;font-weight:600;"
            "qproperty-alignment:AlignCenter;"))
        lay.addWidget(_lbl(".las  ·  .laz  ·  .e57  ·  .ga3d_bin",
            "color:#55585c;font-size:10.5px;qproperty-alignment:AlignCenter;"
            "padding-bottom:4px;"))
        lay.addWidget(_lbl("Ctrl+Z deshacer  ·  Ctrl+Y rehacer  ·  Ctrl+S guardar",
            "color:#55585c;font-size:10.5px;qproperty-alignment:AlignCenter;"))

        lay.addStretch()
        scroll.setWidget(content)
        ol = QVBoxLayout(w); ol.setContentsMargins(0,0,0,0); ol.addWidget(scroll)
        return w

    def _set_workflow_step(self, step: int) -> None:
        """Actualiza el riel de navegación y la franja de breadcrumb."""
        self._current_step = step
        if hasattr(self, '_rail'):
            self._rail.set_active_step(step)
        if hasattr(self, '_breadcrumb'):
            self._breadcrumb.set_step(step, self.STEP_NAMES.get(step, ""),
                                      self.STEP_DESCRIPTIONS.get(step, ""))


    def _on_layers_btn_clicked(self) -> None:
        """Toggle 'Capas de referencia' en el stack izquierdo (riel → capas_clicked)."""
        showing_layers = self._left_stack.currentIndex() == 3
        if not showing_layers:
            self._left_stack.setCurrentIndex(3)  # Capas
            if hasattr(self, '_rail'):
                self._rail.set_capas_active(True)
        else:
            # Volver al paso actual
            step_to_page = {1: 0, 2: 1, 3: 2, 4: 2}
            self._left_stack.setCurrentIndex(step_to_page.get(self._current_step, 0))
            if hasattr(self, '_rail'):
                self._rail.set_capas_active(False)

    def _on_workflow_step_clicked(self, step: int) -> None:
        """Cambia el panel según el paso activo."""
        # Restaurar layout normal si veníamos del paso 5 o 6
        if hasattr(self, '_body_stack') and self._body_stack.currentIndex() in (1, 2):
            self._body_stack.setCurrentIndex(0)

        if step == 4:
            self._set_workflow_step(4)
            self._open_export_dialog()
            return

        if step == 5:
            self._set_workflow_step(5)
            if hasattr(self, '_body_stack'):
                self._body_stack.setCurrentIndex(1)
            if self._project and hasattr(self, '_training_panel'):
                self._training_panel.set_project_context(self._project)
            return

        if step == 6:
            self._set_workflow_step(6)
            if hasattr(self, '_body_stack'):
                self._body_stack.setCurrentIndex(2)
            if hasattr(self, '_infer_panel'):
                self._infer_panel.set_context(self._pc, self._project)
            return

        # Steps 1-3: cambiar panel izquierdo
        page_map = {1: 0, 2: 1, 3: 2}
        if hasattr(self, '_left_stack'):
            self._left_stack.setCurrentIndex(page_map.get(step, 0))
        self._set_workflow_step(step)

    def _activate_annotation_color_if_labeled(self) -> None:
        if self._project is None or self._project.labels is None:
            return
        import numpy as np
        n_labeled = int((self._project.labels > 0).sum())
        if n_labeled == 0:
            return
        print(f"[Project] Restaurando {n_labeled:,} etiquetas...")
        try:
            self._label_store.attach(self._project.labels, autosave_path=None)
            if self._project.schema:
                self._canvas.rebuild_annotation_lut()
            # Only switch if not in a visual mode chosen by the user
            if self._canvas._color_mode not in ("RGB", "Intensidad", "Clasificación"):
                self._canvas.set_color_mode("Anotación")
                self._tool_panel.set_color_mode("Anotación")
            self._canvas._req_worker()
            self._on_stats_changed()
            print(f"[Project] Modo Anotación activo ({n_labeled:,} pts) ✓")
        except Exception as e:
            print(f"[Project] Error: {e}")

    def _on_training_finished(self, model_path: str) -> None:
        from PyQt5.QtWidgets import QMessageBox
        if self._project:
            self._project.best_model_path = model_path
        QMessageBox.information(self, "Entrenamiento completado",
            f"Modelo guardado en:\n{model_path}\n\n"
            "Puedes usarlo en el Paso 6 para inferencia.")

    def _on_inference_done(self, predictions) -> None:
        import numpy as np
        from PyQt5.QtWidgets import QMessageBox
        if self._pc is None or self._project is None: return
        n_pts = self._pc.n_points
        preds = np.array(predictions, dtype=np.uint8)
        if len(preds) != n_pts:
            QMessageBox.warning(self, "Inferencia",
                f"Predicciones ({len(preds):,}) vs nube ({n_pts:,}) no coinciden."); return
        idx_all = np.arange(n_pts, dtype=np.int64)
        for c in range(1, int(preds.max())+1):
            ci = idx_all[preds == c]
            if len(ci): self._label_store.annotate(ci, int(c))
        self._canvas.set_color_mode("Anotación")
        self._tool_panel.set_color_mode("Anotación")
        self._canvas.force_color_rebuild()
        self._on_stats_changed()
        self._annotations_saved = False
        QMessageBox.information(self, "Inferencia aplicada",
            f"\u2713 {int((preds>0).sum()):,} pts clasificados.\nRevisa en el Paso 3.")
        self._on_workflow_step_clicked(3)

    def _on_agl_auto_classify(self) -> None:
        if self._pc is None or self._project is None: return
        from PyQt5.QtWidgets import QApplication, QMessageBox
        import numpy as np
        # AGL "modo plano": altura relativa al punto más bajo de la nube.
        # (El modo "avanzado" con MDT/self._terrain se quitó — is_flat_mode()
        # siempre devolvía True, así que ese MDT nunca se usaba realmente.)
        self._status.show_loading("Calculando AGL...")
        QApplication.processEvents()
        z_min = float(self._pc.xyz[:, 2].min())
        agl   = (self._pc.xyz[:, 2] - z_min).astype(np.float32)
        layers = self._geo_panel.get_agl_layers()
        if not layers:
            QMessageBox.information(self, "AGL",
                "Ninguna capa tiene clase asignada.\nAsigna una clase en cada fila.")
            self._status.hide_loading(); return
        labels = self._project.labels
        total = 0; report = []
        for class_id, agl_min, agl_max in sorted(layers, key=lambda x: x[1]):
            mask = (agl >= agl_min) & (agl <= agl_max) & (labels == 0)
            idx  = np.where(mask)[0].astype(np.int64)
            if len(idx) > 0:
                self._label_store.annotate(idx, class_id)
                total += len(idx)
                name = next((s.name for s in self._project.schema if s.id==class_id), str(class_id))
                report.append(f"  {name}: {len(idx):,} pts  ({agl_min:.1f}-{agl_max:.1f}m AGL)")
        self._canvas.force_color_rebuild()
        self._on_stats_changed()
        self._status.hide_loading()
        QMessageBox.information(self, "AGL completado",
            f"Clasificados: {total:,} puntos\n\n" + "\n".join(report))

    def _on_csf_classify(self) -> None:
        if self._pc is None or self._project is None: return
        from PyQt5.QtWidgets import QApplication, QMessageBox, QProgressDialog
        import numpy as np
        params = self._geo_panel.get_csf_params()
        class_id   = params["class_id"]
        resolution = params["resolution"]
        try:
            import CSF
        except ImportError:
            QMessageBox.information(self, "CSF", 
                "La librería CSF no está instalada.\n\npip install cloth-simulation-filter"); return
        progress = QProgressDialog("Ejecutando CSF...", "Cancelar", 0, 0, self)
        progress.setWindowTitle("Cloth Simulation Filter")
        progress.setMinimumDuration(0); progress.setValue(0)
        QApplication.processEvents()
        try:
            xyz = self._pc.xyz.astype(np.float64)
            csf = CSF.CSF()
            csf.params.bSloopSmooth   = False
            csf.params.cloth_resolution = resolution
            csf.params.rigidness      = 3
            csf.params.time_step      = 0.65
            csf.params.class_threshold = 0.5
            csf.params.interations    = 500
            csf.setPointCloud(xyz)
            ground_idx     = CSF.VecInt()
            non_ground_idx = CSF.VecInt()
            self._status.show_loading("CSF: simulando tela...")
            QApplication.processEvents()
            csf.do_filtering(ground_idx, non_ground_idx, exportCloth=False)
            ground_arr = np.array(ground_idx, dtype=np.int64)
            progress.close()
            if len(ground_arr) == 0:
                QMessageBox.warning(self, "CSF", "No se detectaron puntos de suelo.")
                self._status.hide_loading(); return
            self._label_store.annotate(ground_arr, class_id)
            self._canvas.force_color_rebuild()
            self._on_stats_changed()
            self._status.hide_loading()
            class_name = next((s.name for s in self._project.schema if s.id==class_id), str(class_id))
            QMessageBox.information(self, "CSF completado",
                f"Suelo: {len(ground_arr):,} puntos\nClase: {class_name}\nResolución: {resolution}m")
        except Exception as e:
            progress.close(); self._status.hide_loading()
            QMessageBox.critical(self, "CSF Error", str(e))

    def _set_top_view(self) -> None:
        try:
            cam = self._canvas._ren.GetActiveCamera()
            fp  = cam.GetFocalPoint(); d = cam.GetDistance()
            cam.SetPosition(fp[0], fp[1], fp[2] + d)
            cam.SetViewUp(0, 1, 0)
            self._canvas._ren.ResetCameraClippingRange()
            self._canvas._do_render()
        except Exception as e:
            print(f"[TopView] {e}")

    def _set_side_view(self) -> None:
        try:
            cam = self._canvas._ren.GetActiveCamera()
            fp  = cam.GetFocalPoint(); d = cam.GetDistance()
            cam.SetPosition(fp[0], fp[1] + d, fp[2])
            cam.SetViewUp(0, 0, 1)
            self._canvas._ren.ResetCameraClippingRange()
            self._canvas._do_render()
        except Exception as e:
            print(f"[SideView] {e}")

    def _reset_camera_view(self) -> None:
        """Encuadra la nube completa en la vista — tecla F."""
        try:
            if hasattr(self._canvas, 'reset_view'):
                self._canvas.reset_view()
            else:
                self._canvas._ren.ResetCamera()
                self._canvas._ren.ResetCameraClippingRange()
                self._canvas._do_render()
        except Exception as e:
            print(f"[ResetView] {e}")


    def _on_canvas_tile_hover(self, tile) -> None:
        if hasattr(self._tile_panel, 'set_hover_tile'):
            self._tile_panel.set_hover_tile(tile)

    def _check_and_import_classification(self, pc) -> None:
        """
        Si la nube tiene campo 'classification' con clases válidas (>0),
        ofrece importarlas como anotaciones mapeando a las clases del proyecto.
        """
        import numpy as np
        from PyQt5.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
            QTableWidget, QTableWidgetItem, QComboBox, QHeaderView, QCheckBox)
        from PyQt5.QtCore import Qt

        cls_field = getattr(pc, 'classification', None)
        if cls_field is None:
            return
        cls_arr = np.asarray(cls_field, dtype=np.uint8)
        unique_cls = sorted([int(c) for c in np.unique(cls_arr) if c > 0])
        if not unique_cls:
            return

        # ASPRS LAS standard class names
        ASPRS = {
            0: "Sin clasificar",      1: "Sin asignar",         2: "Suelo",
            3: "Veg. baja",           4: "Veg. media",          5: "Veg. alta",
            6: "Edificio",            7: "Ruido bajo",          9: "Agua",
            10: "Ferrocarril",        11: "Carretera",          13: "Protección metálica",
            14: "Puente",             17: "Cable (trenzado)",   18: "Cable (hilo)",
            19: "Torre alta tensión", 20: "Conector cable",
        }

        schema = self._project.schema or []

        dlg = QDialog(self)
        dlg.setWindowTitle("Importar clasificación de la nube")
        dlg.setMinimumWidth(600)
        dlg.setStyleSheet(
            "QDialog{background:#e8e9eb;}"
            "QLabel{color:#55585c;}"
            "QTableWidget{background:#e8e9eb;border:1px solid #f4f5f6;color:#55585c;"
            "  gridline-color:#f4f5f6;font-size:10.5px;}"
            "QTableWidget::item{padding:4px;}"
            "QHeaderView::section{background:#e8e9eb;color:#55585c;border:none;"
            "  padding:5px;font-size:10.5px;border-bottom:1px solid #f4f5f6;}"
            "QComboBox{background:#e8e9eb;border:1px solid #eceded;border-radius:3px;"
            "  color:#55585c;padding:2px 6px;font-size:10.5px;}"
            "QComboBox QAbstractItemView{background:#f4f5f6;color:#55585c;"
            "  selection-background-color:#f4f5f6;selection-color:#0e7c86;}")

        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(16, 16, 16, 16); lay.setSpacing(10)

        hdr = QLabel(
            f"Se detectaron <b style='color:#0e7c86'>{len(unique_cls)} clases</b> "
            f"en el campo de clasificación de la nube.<br>"
            f"Mapea cada clase a una clase del proyecto o crea nuevas:")
        hdr.setWordWrap(True)
        hdr.setStyleSheet("color:#84888c;font-size:10.5px;")
        lay.addWidget(hdr)

        tbl = QTableWidget(len(unique_cls), 3)
        tbl.setHorizontalHeaderLabels(["Clase en la nube", "Puntos", "Mapear a →"])
        tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        tbl.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        tbl.verticalHeader().setVisible(False)
        tbl.setSelectionMode(QTableWidget.NoSelection)
        tbl.setMinimumHeight(min(48 + len(unique_cls) * 32, 320))

        proj_opts = ["-- Ignorar --"] + [f"{s.id}: {s.name}" for s in schema if s.id > 0]
        proj_opts.append("++ Crear clase nueva")

        combos = []
        for row, cls_id in enumerate(unique_cls):
            n_pts = int((cls_arr == cls_id).sum())
            asprs_name = ASPRS.get(cls_id, f"Clase {cls_id}")

            ni = QTableWidgetItem(f"{cls_id}  —  {asprs_name}")
            ni.setFlags(Qt.ItemIsEnabled)
            pi = QTableWidgetItem(f"{n_pts:,}")
            pi.setFlags(Qt.ItemIsEnabled)
            pi.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

            cb = QComboBox()
            for opt in proj_opts:
                cb.addItem(opt)

            # Auto-match by keyword similarity
            asprs_low = asprs_name.lower()
            keywords = {
                "suelo":      ["suelo","ground","terreno","tierra"],
                "veg":        ["veg","plant","hierba","pasto","tree","arbol","vegetacion"],
                "edificio":   ["edificio","building","roof","techo","construc"],
                "agua":       ["agua","water","lago","rio"],
                "carretera":  ["carretera","road","calle","vial","asfalto"],
                "cable":      ["cable","wire","power","linea"],
            }
            for i, s in enumerate(schema):
                if s.id <= 0: continue
                s_low = s.name.lower()
                for cat, kws in keywords.items():
                    if any(k in s_low for k in kws) and any(k in asprs_low for k in kws):
                        cb.setCurrentIndex(i + 1); break
                else:
                    continue
                break

            tbl.setItem(row, 0, ni)
            tbl.setItem(row, 1, pi)
            tbl.setCellWidget(row, 2, cb)
            combos.append((cls_id, cb))

        tbl.setRowCount(len(unique_cls))
        lay.addWidget(tbl)

        chk = QCheckBox("Sobrescribir anotaciones existentes")
        chk.setStyleSheet("QCheckBox{color:#84888c;font-size:10.5px;}"
                          "QCheckBox::indicator{width:13px;height:13px;}")
        lay.addWidget(chk)

        btns = QHBoxLayout()
        btn_cancel = QPushButton("Cancelar")
        btn_ok     = QPushButton("Importar")
        btn_cancel.setStyleSheet(
            "QPushButton{background:#e8e9eb;border:1px solid #eceded;border-radius:4px;"
            "color:#84888c;padding:7px 16px;font-size:10.5px;}"
            "QPushButton:hover{border-color:#0e7c86;color:#0e7c86;}")
        btn_ok.setStyleSheet(
            "QPushButton{background:#0e7c86;color:#55585c;border:none;"
            "border-radius:4px;padding:7px 16px;font-size:10.5px;font-weight:600;}"
            "QPushButton:hover{background:#0a5f67;}")
        btn_cancel.clicked.connect(dlg.reject)
        btn_ok.clicked.connect(dlg.accept)
        btns.addWidget(btn_cancel); btns.addStretch(); btns.addWidget(btn_ok)
        lay.addLayout(btns)

        if dlg.exec_() != QDialog.Accepted:
            return

        # ── Apply mapping ────────────────────────────────────────────────────
        import numpy as np, colorsys
        schema_changed = False

        any_imported = False
        for cls_id, cb in combos:
            choice = cb.currentText()
            if choice.startswith("--"):
                continue

            idx_pts = np.where(cls_arr == cls_id)[0].astype(np.int64)
            if not chk.isChecked():
                idx_pts = idx_pts[self._project.labels[idx_pts] == 0]
            if len(idx_pts) == 0:
                continue

            if choice.startswith("++"):
                # Create new class with auto color
                new_id = max((s.id for s in self._project.schema), default=0) + 1
                h = (new_id * 0.618033988749895) % 1.0
                r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.9)
                hex_col  = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                asprs_nm = ASPRS.get(int(cls_id), f"Clase {cls_id}")
                from core.project import SemanticClass
                new_sc   = SemanticClass(id=new_id, name=asprs_nm, color=hex_col)
                self._project.schema.append(new_sc)
                schema_changed = True
                target_id = new_id
            else:
                target_id = int(choice.split(":")[0])

            self._label_store.annotate(idx_pts, target_id)
            any_imported = True

        if schema_changed:
            self._class_panel.set_project(self._project)
            self._geo_panel.set_schema(self._project.schema)

        if any_imported:
            self._canvas.rebuild_annotation_lut()
            # Switch to Anotación only if user was not in a visual mode (RGB, etc.)
            current_mode = self._canvas._color_mode
            if current_mode not in ("RGB", "Intensidad", "Intensidad Color", "Clasificación"):
                self._canvas.set_color_mode("Anotación")
                self._tool_panel.set_color_mode("Anotación")
            self._canvas.force_color_rebuild()
            self._on_stats_changed()
            self._annotations_saved = False
            print(f"[Import] Clasificación importada. Modo actual: {self._canvas._color_mode}")

    def _validate_project_integrity(self, project, pc) -> list:
        issues = []
        if project.n_total > 0 and pc.n_points > 0 and project.n_total != pc.n_points:
            diff = abs(project.n_total - pc.n_points)
            pct  = 100.0 * diff / max(project.n_total, 1)
            if pct > 0.5:
                issues.append(
                    f"La nube tiene {pc.n_points:,} pts pero el proyecto fue guardado "
                    f"con {project.n_total:,} pts (diferencia: {pct:.1f}%).\n"
                    "Las etiquetas pueden no corresponder a los puntos correctos.")
        if project.labels is not None and len(project.labels) != pc.n_points:
            issues.append(
                f"Labels: {len(project.labels):,} vs nube: {pc.n_points:,} puntos.\n"
                "Las etiquetas serán reinicializadas.")
        if project.source_file:
            from pathlib import Path as _P
            if not _P(project.source_file).exists():
                issues.append(
                    f"Archivo de nube no encontrado:\n{project.source_file}")
        if project.labels is not None and project.schema:
            valid_ids = {0} | {s.id for s in project.schema}
            unknown = set(np.unique(project.labels).tolist()) - valid_ids
            if unknown:
                issues.append(f"Clases desconocidas en etiquetas: {unknown}")
        return issues

    def _show_integrity_warning(self, issues: list) -> bool:
        from PyQt5.QtWidgets import QMessageBox
        text = "\n\n".join(f"• {i}" for i in issues)
        r = QMessageBox.warning(self, "Integridad del proyecto",
            f"Se encontraron {len(issues)} problema(s):\n\n{text}\n\n"
            "¿Abrir el proyecto de todas formas?",
            QMessageBox.Yes | QMessageBox.No)
        return r == QMessageBox.Yes


