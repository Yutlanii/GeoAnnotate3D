"""
render/canvas.py — Canvas 3D respaldado por VTK (C++ OpenGL) v5.0

Cambios vs v4.0:
  - TRES MODOS DE VISTA:
      "sparse"  — nube completa, budget capped en SPARSE_CAP (~3M pts).
                  No crece automáticamente. Ideal para nubes enormes.
      "full"    — nube completa, LOD progresivo hasta densidad máxima (comportamiento original).
      "tile"    — tile individual a densidad máxima.
    set_overview_mode("sparse" | "full") para cambiar entre los dos primeros.
    El modo "tile" se activa/desactiva via enter_tile_mode / exit_tile_mode.

  - TILE GRID OVERLAY: clase _TileGridOverlay gestiona actores VTK para:
      * Líneas del grid (thin, amber-gris).
      * Rectángulo de hover (amber, actualizado con el ratón).
      * Rectángulo del tile activo (naranja brillante).
    Solo visible en modo overview (sparse o full).

  - HOVER DETECTION: en overview, mouse move detecta el tile bajo el cursor
    via ray–plane intersection y emite sig.tile_hovered(TileInfo|None).

  - set_tile_manager(tm): registra el TileManager para el overlay y hover.
"""
from __future__ import annotations
import math
import time
from typing import Optional, TYPE_CHECKING

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal, QThread

try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    VTK_OK = True
except ImportError:
    VTK_OK = False

if TYPE_CHECKING:
    from core.pointcloud import PointCloud
    from core.project     import Project
    from annotation.tools import BaseTool
    from core.tile_manager import TileInfo, TileManager

BG_R, BG_G, BG_B = 0.047, 0.039, 0.031

# ── Tuning ───────────────────────────────────────────────────────────────────
COARSE_TARGET_PTS       = 500_000
DRAG_TARGET_PTS         = 300_000
LOD_MS                  = 25
INTERACTION_COOLDOWN_MS = 300
BUDGET_FLOOR            = 20_000_000
MAX_BUDGET              = 200_000_000
OVERVIEW_BUDGET_CAP     = 10_000_000
SPARSE_CAP              = 3_000_000    # budget fijo en modo sparse
FPS_LOW,  FPS_HIGH      = 8, 20
# Tiles densos: por encima de este umbral, mostrar primero una vista previa
# decimada (rápida de leer/subir a GPU) mientras el tile completo se carga
# en segundo plano. Evita el "congelamiento" al entrar a un tile de 20-30M+
# puntos — antes se esperaba a tener el tile ENTERO listo antes de mostrar
# nada. El pincel/esfera/disco ya no dependen de esta vista para su precisión
# (usan el índice espacial global de pc.octree), así que decimar la vista
# no afecta la exactitud de esas herramientas durante la vista previa.
TILE_COARSE_THRESHOLD   = 3_000_000
TILE_COARSE_TARGET_PTS  = 1_200_000
_CURSOR_THROTTLE        = 5
_RENDER_THROTTLE_MS     = 40
_HOVER_THROTTLE         = 2            # actualizar hover cada N mouse-moves


# ─────────────────────────────────────────────────────────────────────────────
# Señales
# ─────────────────────────────────────────────────────────────────────────────

class CanvasSignals(QObject):
    fps               = pyqtSignal(float)
    render_info       = pyqtSignal(int, int)
    point_picked      = pyqtSignal(float, float, float)
    point_picked_abs  = pyqtSignal(float, float, float)
    measure_segment   = pyqtSignal(float, float, float)
    cursor_utm        = pyqtSignal(float, float, float)
    tile_mode_changed = pyqtSignal(bool, object)   # (en_tile_mode, TileInfo|None)
    tile_hovered      = pyqtSignal(object)         # TileInfo | None
    view_mode_changed = pyqtSignal(str)            # "sparse" | "full" | "tile"
    grid_transform_preview = pyqtSignal(float, float, float)  # ox,oy,rot — durante drag
    grid_transform_set     = pyqtSignal(float, float, float)  # ox,oy,rot — al soltar

# ─────────────────────────────────────────────────────────────────────────────
# Tile grid overlay
# ─────────────────────────────────────────────────────────────────────────────

class _TileGridOverlay:
    """Gestiona los actores VTK del grid de tiles en el viewport 3D."""

    def __init__(self, renderer):
        self._ren = renderer
        self._tm  = None

        def _make_actor(color, line_width, opacity):
            poly   = vtk.vtkPolyData()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly)
            # Polygon offset: empuja las lineas hacia el visor
            mapper.SetResolveCoincidentTopologyToPolygonOffset()
            mapper.SetRelativeCoincidentTopologyLineOffsetParameters(-1.0, -66000.0)
            actor  = vtk.vtkActor()
            actor.SetMapper(mapper)
            p = actor.GetProperty()
            p.SetColor(*color)
            p.SetLineWidth(line_width)
            p.SetOpacity(opacity)
            p.LightingOff()
            p.SetRenderLinesAsTubes(False)
            # Deshabilitar depth test -> siempre encima de la nube (VTK >= 8.2)
            try:
                p.SetDepthTesting(False)
            except AttributeError:
                pass
            actor.SetVisibility(0)
            renderer.AddActor(actor)
            return actor, poly

        # Grid completo (líneas finas)
        self._grid_actor, self._grid_poly = _make_actor(
            (0.5, 0.38, 0.1), 1.0, 0.55)
        # Hover (rectángulo más grueso, ámbar)
        self._hover_actor, self._hover_poly = _make_actor(
            (0.95, 0.6, 0.1), 2.2, 0.9)
        # Tile activo (naranja brillante)
        self._active_actor, self._active_poly = _make_actor(
            (1.0, 0.45, 0.0), 2.8, 1.0)

        self._visible = False

    # ── API ──────────────────────────────────────────────────────────────────

    def set_tile_manager(self, tm, show: bool = True) -> None:
        self._tm = tm
        if tm is None:
            self._grid_actor.SetVisibility(0)
            self._hover_actor.SetVisibility(0)
            self._active_actor.SetVisibility(0)
            return
        self._rebuild_grid()
        self.set_visible(show)

    def set_visible(self, v: bool) -> None:
        self._visible = v
        self._grid_actor.SetVisibility(int(v) if self._tm else 0)
        if not v:
            self._hover_actor.SetVisibility(0)
            self._active_actor.SetVisibility(0)

    def update_hover(self, tile) -> None:
        if tile is None or not self._visible:
            self._hover_actor.SetVisibility(0); return
        self._fill_rect_poly(self._hover_poly, self._tm.world_corners(tile))
        self._hover_actor.SetVisibility(1)

    def update_active(self, tile) -> None:
        if tile is None:
            self._active_actor.SetVisibility(0); return
        self._fill_rect_poly(self._active_poly, self._tm.world_corners(tile))
        self._active_actor.SetVisibility(int(self._visible))

    def rebuild(self) -> None:
        if self._tm: self._rebuild_grid()

    # ── Interno ───────────────────────────────────────────────────────────────

    def _rebuild_grid(self) -> None:
        starts, ends = self._tm.grid_line_endpoints()
        n = len(starts)

        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(n * 2)
        for i in range(n):
            pts.SetPoint(2*i,   starts[i][0], starts[i][1], starts[i][2])
            pts.SetPoint(2*i+1, ends[i][0],   ends[i][1],   ends[i][2])

        lines = vtk.vtkCellArray()
        for i in range(n):
            lines.InsertNextCell(2)
            lines.InsertCellPoint(2*i)
            lines.InsertCellPoint(2*i+1)

        self._grid_poly.SetPoints(pts)
        self._grid_poly.SetLines(lines)
        self._grid_poly.Modified()

    def _fill_rect_poly(self, poly, corners) -> None:
        """Dibuja un rectángulo cerrado con 4 esquinas [(x,y,z)*4]."""
        pts = vtk.vtkPoints()
        for c in corners:
            pts.InsertNextPoint(c[0], c[1], c[2] + 0.5)  # +0.5m Z para que flote sobre la nube
        poly.SetPoints(pts)
        lines = vtk.vtkCellArray()
        for i in range(4):
            lines.InsertNextCell(2)
            lines.InsertCellPoint(i)
            lines.InsertCellPoint((i+1) % 4)
        poly.SetLines(lines)
        poly.Modified()


# ─────────────────────────────────────────────────────────────────────────────
# Tile Loader (background)
# ─────────────────────────────────────────────────────────────────────────────

class _TileLoader(QThread):
    ready        = pyqtSignal(object, object, object)
    coarse_ready = pyqtSignal(object, object, object)   # vista previa decimada
    progress     = pyqtSignal(int, str)

    def __init__(self, pc, indices: np.ndarray, project,
                 color_mode: str, cmap: str,
                 annotation_lut_u8: Optional[np.ndarray], parent=None):
        super().__init__(parent)
        self._pc   = pc
        self._idx  = indices
        self._project = project
        self._cm   = color_mode
        self._cmap = cmap
        self._lut  = annotation_lut_u8

    def _emit_coarse_preview(self, idx: np.ndarray, n_tile: int) -> None:
        """
        Vista previa rápida y decimada del tile, para que aparezca algo en
        pantalla casi de inmediato en tiles muy densos (>TILE_COARSE_THRESHOLD)
        mientras el tile completo se lee/colorea en segundo plano.
        No afecta la precisión de Pincel/Esfera/Disco (usan pc.octree global).
        """
        try:
            from render.colors import compute_colors_u8
            from core._fast import FC
            step = max(1, n_tile // TILE_COARSE_TARGET_PTS)
            coarse_idx = idx[::step]
            xyz_c = np.ascontiguousarray(self._pc.xyz[coarse_idx], dtype=np.float32)
            attrs = self._pc.get_attrs(coarse_idx)
            lbl   = (self._project.labels[coarse_idx]
                     if self._project and self._project.labels is not None else None)
            col_u8 = compute_colors_u8(xyz_c, attrs, self._cm, self._cmap,
                                       annotation_labels=lbl,
                                       annotation_lut_u8=self._lut)
            self.coarse_ready.emit(xyz_c, col_u8, coarse_idx)
        except Exception as exc:
            print(f"[TileLoader] preview: {exc}")

    def run(self):
        try:
            from render.colors import compute_colors_u8
            idx = self._idx
            n_tile = len(idx)
            self.progress.emit(10, f"Cargando tile ({n_tile/1e6:.1f}M pts)…")

            # Tile denso: mostrar antes una vista previa decimada, rápida de
            # leer y subir a GPU, para que la navegación/anotación puedan
            # empezar de inmediato en vez de esperar el tile completo.
            if n_tile > TILE_COARSE_THRESHOLD:
                self._emit_coarse_preview(idx, n_tile)
                self.progress.emit(25, f"Cargando tile completo ({n_tile/1e6:.1f}M pts)…")

            # CRÍTICO para mmaps grandes: ordenar índices antes de acceder.
            # Acceso aleatorio a mmap de 10+ GB = tormenta de page faults → crash.
            # Acceso ordenado = páginas contiguas = sin crash.
            is_huge = getattr(self._pc, 'is_mmap', False) and len(self._pc.xyz) > 200_000_000
            if is_huge:
                self.progress.emit(35, f"Ordenando {n_tile/1e6:.1f}M índices…")
                sort_order = np.argsort(idx, kind='stable')
                idx_sorted = idx[sort_order]
                self.progress.emit(45, f"Leyendo tile del disco…")
                xyz_sorted = np.array(self._pc.xyz[idx_sorted], dtype=np.float32)
                # Restaurar orden original
                inv_order = np.argsort(sort_order, kind='stable')
                xyz_c = xyz_sorted[inv_order]
                del xyz_sorted
            else:
                xyz_c = np.ascontiguousarray(self._pc.xyz[idx], dtype=np.float32)

            self.progress.emit(70, "Calculando colores…")
            attrs = self._pc.get_attrs(idx)
            lbl   = (self._project.labels[idx]
                     if self._project and self._project.labels is not None else None)
            col_u8 = compute_colors_u8(xyz_c, attrs, self._cm, self._cmap,
                                       annotation_labels=lbl,
                                       annotation_lut_u8=self._lut)
            self.progress.emit(95, "Tile listo ✓")
            self.ready.emit(xyz_c, col_u8, idx)
        except Exception as exc:
            import traceback
            print(f"[TileLoader] {exc}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────────────
# Proxies de overlay (sin cambios)
# ─────────────────────────────────────────────────────────────────────────────

class _LineProxy:
    def __init__(self, renderer, render_fn):
        self._r = renderer; self._rfn = render_fn
        self._poly = vtk.vtkPolyData()
        self._poly.SetPoints(vtk.vtkPoints())
        m = vtk.vtkPolyDataMapper(); m.SetInputData(self._poly)
        self._actor = vtk.vtkActor(); self._actor.SetMapper(m)
        self._actor.GetProperty().SetLineWidth(2.0)
        self._actor.GetProperty().SetRenderLinesAsTubes(False)
        self._actor.GetProperty().LightingOff()
        self._actor.SetVisibility(0); renderer.AddActor(self._actor)
        self._visible = False

    @property
    def visible(self): return self._visible
    @visible.setter
    def visible(self, v):
        self._visible = bool(v); self._actor.SetVisibility(int(v))

    def set_data(self, pts, color=None, connect="strip", width=None, **_):
        if pts is None or len(pts) < 2:
            self._actor.SetVisibility(0); return
        if width is not None:
            self._actor.GetProperty().SetLineWidth(float(width))
        n = len(pts); arr = pts.astype(np.float32)
        vtk_pts = vtk.vtkPoints(); vtk_pts.SetData(numpy_to_vtk(arr, deep=True))
        self._poly.SetPoints(vtk_pts)
        if connect == "segments":
            n_seg = n // 2
            segs = np.empty(3*n_seg, np.int64)
            segs[0::3]=2; segs[1::3]=np.arange(0,n,2)[:n_seg]; segs[2::3]=np.arange(1,n,2)[:n_seg]
            cells = vtk.vtkCellArray(); cells.SetCells(n_seg, numpy_to_vtkIdTypeArray(segs, deep=True))
        else:
            segs = np.empty(n+1, np.int64); segs[0]=n; segs[1:]=np.arange(n)
            cells = vtk.vtkCellArray(); cells.SetCells(1, numpy_to_vtkIdTypeArray(segs, deep=True))
        self._poly.SetLines(cells); self._poly.Modified()
        if color is not None:
            c = np.asarray(color, np.float32)
            self._actor.GetProperty().SetColor(float(c[0]),float(c[1]),float(c[2]))
            self._actor.GetProperty().SetOpacity(float(c[3]) if len(c)>3 else 1.0)
        self._actor.SetVisibility(int(self._visible))


class _MarkersProxy:
    def __init__(self, renderer, render_fn):
        self._r = renderer
        self._poly = vtk.vtkPolyData()
        m = vtk.vtkPolyDataMapper(); m.SetInputData(self._poly)
        self._actor = vtk.vtkActor(); self._actor.SetMapper(m)
        self._actor.GetProperty().SetPointSize(8.0)
        self._actor.GetProperty().RenderPointsAsSpheresOn()
        self._actor.GetProperty().LightingOff()
        self._actor.SetVisibility(0); renderer.AddActor(self._actor)
        self._visible = False

    @property
    def visible(self): return self._visible
    @visible.setter
    def visible(self, v):
        self._visible = bool(v); self._actor.SetVisibility(int(v))

    def set_data(self, pos=None, face_color=None, size=8, **_):
        if pos is None: return
        pts_arr = np.atleast_2d(pos).astype(np.float32); n = len(pts_arr)
        vtk_pts = vtk.vtkPoints(); vtk_pts.SetData(numpy_to_vtk(pts_arr, deep=True))
        self._poly.SetPoints(vtk_pts)
        segs = np.empty(2*n, np.int64); segs[0::2]=1; segs[1::2]=np.arange(n)
        cells = vtk.vtkCellArray(); cells.SetCells(n, numpy_to_vtkIdTypeArray(segs, deep=True))
        self._poly.SetVerts(cells); self._poly.Modified()
        if face_color is not None:
            c = np.asarray(face_color, np.float32)
            if c.ndim == 1:
                self._actor.GetProperty().SetColor(float(c[0]),float(c[1]),float(c[2]))
                self._actor.GetProperty().SetOpacity(float(c[3]) if len(c)>3 else 1.0)
        self._actor.GetProperty().SetPointSize(float(size))
        self._actor.SetVisibility(int(self._visible))


# ─────────────────────────────────────────────────────────────────────────────
# Estilo de interacción
# ─────────────────────────────────────────────────────────────────────────────

class _AnnotationStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, canvas):
        super().__init__()
        self._canvas       = canvas
        self._annotating   = False
        self._cam_dragging = False
        self._freeze_nav   = False
        self._hover_tick   = 0
        self.AddObserver("LeftButtonPressEvent",    self._on_ld)
        self.AddObserver("LeftButtonReleaseEvent",  self._on_lu)
        self.AddObserver("RightButtonPressEvent",   self._on_rd)
        self.AddObserver("RightButtonReleaseEvent", self._on_ru)
        self.AddObserver("MouseMoveEvent",          self._on_mm)
        self.AddObserver("MouseWheelForwardEvent",  self._on_wf)
        self.AddObserver("MouseWheelBackwardEvent", self._on_wb)
        self.AddObserver("KeyPressEvent",           self._on_key)
        self.AddObserver("KeyReleaseEvent",         self._on_key_up)

    def _ctrl(self):  return bool(self.GetInteractor().GetControlKey())
    def _shift(self): return bool(self.GetInteractor().GetShiftKey())

    def _pos(self):
        iren = self.GetInteractor(); x, y_vtk = iren.GetEventPosition()
        _, h = iren.GetRenderWindow().GetSize()
        return (x, h - y_vtk - 1)

    def _on_ld(self, obj, ev):
        canvas = self._canvas
        # Grid edit mode captura el click antes que la cámara o anotación
        if canvas._grid_edit_mode is not None:
            canvas._start_grid_drag(self._pos())
            return
        if self._ctrl():
            self._annotating=True; self._cam_dragging=False
            if canvas.active_tool:
                try:
                    canvas._dispatch_press(self._pos())
                except Exception as _e:
                    print(f"[tool] {type(_e).__name__}: {_e}")
        elif self._shift():
            self._annotating=False; self._cam_dragging=True
            self._freeze_nav = True; self.OnLeftButtonDown()
        else:
            self._annotating=False; self._cam_dragging=True
            self._freeze_nav = False
            canvas._on_interaction_start(); self.OnLeftButtonDown()

    def _on_lu(self, obj, ev):
        canvas = self._canvas
        # Finalizar drag del grid
        if canvas._grid_edit_mode is not None and canvas._grid_drag_active:
            canvas._end_grid_drag()
            return
        if self._annotating:
            self._annotating = False
            if canvas.active_tool: canvas._dispatch_release(self._pos())
        else:
            self._cam_dragging = False
            self._freeze_nav = False
            self.OnLeftButtonUp()

    def _on_mm(self, obj, ev):
        canvas = self._canvas; pos = self._pos(); ctrl = self._ctrl()
        # Grid edit mode: actualizar drag o hover, nunca rotar cámara
        if canvas._grid_edit_mode is not None:
            if canvas._grid_drag_active:
                canvas._update_grid_drag(pos)
            else:
                # Hover de tiles durante el modo (sin drag)
                self._hover_tick = (self._hover_tick + 1) % _HOVER_THROTTLE
                if self._hover_tick == 0:
                    canvas._update_tile_hover(pos)
            return   # no propagar a cámara
        if self._annotating and ctrl:
            if canvas.active_tool: canvas._dispatch_move(pos); return
        if ctrl:
            canvas._cursor_tick = (canvas._cursor_tick+1) % _CURSOR_THROTTLE
            if canvas._cursor_tick == 0: canvas._update_brush_cursor(pos)
            return
        if self._cam_dragging:
            if not self._freeze_nav: canvas._on_interaction_start()
            canvas._hide_brush_cursor()
        else:
            self._hover_tick = (self._hover_tick + 1) % _HOVER_THROTTLE
            if self._hover_tick == 0 and not canvas._tile_mode:
                try:
                    canvas._update_tile_hover(pos)
                except AttributeError:
                    pass
            # Cursor de radio para Esfera y Pincel sin necesidad de Ctrl
            if (canvas.active_tool is not None and
                    canvas.active_tool.__class__.__name__
                    in ("SphereSelectTool", "BrushTool")):
                canvas._cursor_tick = (canvas._cursor_tick + 1) % _CURSOR_THROTTLE
                if canvas._cursor_tick == 0:
                    canvas._update_brush_cursor(pos)
        # Fly mode: right-drag = look
        if canvas._fly_mode and canvas._fly_rb_down:
            lx, ly = canvas._fly_last_px
            cx, cy = pos
            canvas._fly_look(cx - lx, cy - ly)
            canvas._fly_last_px = pos
            return   # no propagar a camara
        # Fly mode: capturar keys WASD desde el key de VTK tambien via fly_keys
        self.OnMouseMove()

    def _on_rd(self, obj, ev):
        canvas = self._canvas
        if canvas._fly_mode:
            canvas._fly_rb_down = True
            canvas._fly_last_px = self._pos()
        else:
            self.OnRightButtonDown()

    def _on_ru(self, obj, ev):
        canvas = self._canvas
        if canvas._fly_mode:
            canvas._fly_rb_down = False
        else:
            self.OnRightButtonUp()

    def _on_key_up(self, obj, ev):
        key = self.GetInteractor().GetKeySym()
        self._canvas._fly_keys.discard(key)
        self._canvas._fly_keys.discard(key.lower())
        self._canvas._fly_keys.discard(key.upper())

    def _on_wf(self, obj, ev):
        canvas = self._canvas
        if canvas._fly_mode:
            canvas._fly_speed = min(canvas._fly_speed * 1.35, 2000.0)
            return
        if self._shift(): self.OnMouseWheelForward()
        else: canvas._on_interaction_start(); self.OnMouseWheelForward()

    def _on_wb(self, obj, ev):
        canvas = self._canvas
        if canvas._fly_mode:
            canvas._fly_speed = max(canvas._fly_speed / 1.35, 0.5)
            return
        if self._shift(): self.OnMouseWheelBackward()
        else: canvas._on_interaction_start(); self.OnMouseWheelBackward()
        # Limitar alejamiento excesivo
        canvas.clamp_camera_to_cloud()

    def _on_key(self, obj, ev):
        iren = self.GetInteractor()
        key  = iren.GetKeySym()
        ctrl = bool(iren.GetControlKey())
        canvas = self._canvas

        # ── Ctrl+Z / Ctrl+Shift+Z: undo/redo ─────────────────────────────────
        # DEBE estar aquí — VTK intercepta todos los eventos de teclado antes
        # de que lleguen al keyPressEvent de Qt. Sin esto Ctrl+Z nunca funciona.
        if ctrl:
            k = key.lower()
            if k == 'z':
                shift = bool(iren.GetShiftKey())
                canvas._route_key_to_main('CTRL+SHIFT+Z' if shift else 'CTRL+Z')
                return
            if k == 'y':   # Ctrl+Y = redo (convencion Windows)
                canvas._route_key_to_main('CTRL+SHIFT+Z')
                return
            if k == 's':
                canvas._route_key_to_main('CTRL+S')
                return

        # Bloquear teclas 1-9 que VTK usa para estéreo/render modes
        # y redirigirlas al MainWindow para cambiar clase activa
        if len(key) == 1 and key.isdigit() and not ctrl:
            # Bloquear SIEMPRE (impide ventanas VTK de estéreo)
            # Convertir: tecla "0" → slot 9, "1"→0, "2"→1, ..., "9"→8
            canvas._route_key_to_main(f'CLASS_{key}')
            return

        # Fly mode: registrar tecla y toggle con F
        if canvas._fly_mode:
            canvas._fly_keys.add(key)
            canvas._fly_keys.add(key.lower())
            canvas._fly_keys.add(key.upper())
            if key in ('f', 'F', 'Escape'):
                canvas.toggle_fly_mode()
            return
        if key in ('f', 'F'):
            canvas.toggle_fly_mode(); return
        # Vista cenital (V), lateral (Y), encuadrar (R), espacio (reset)
        if key in ('v', 'V'):
            canvas._route_key_to_main('VIEW_TOP'); return
        if key in ('y', 'Y'):
            canvas._route_key_to_main('VIEW_SIDE'); return
        if key in ('r', 'R'):
            canvas._route_key_to_main('VIEW_RESET'); return
        if key == 'space':
            canvas._route_key_to_main('VIEW_RESET'); return

        from annotation.tools import TOOL_BY_KEY
        if key.upper() in TOOL_BY_KEY:
            canvas._route_key_to_main(key.upper()); return
        if canvas.active_tool:
            class _E:
                def __init__(self, k): self.key = k
            canvas.active_tool.on_key_press(_E(key))


# ─────────────────────────────────────────────────────────────────────────────
# Canvas principal
# ─────────────────────────────────────────────────────────────────────────────

class AnnotationCanvas(QWidget):

    def __init__(self):
        super().__init__()
        if not VTK_OK: raise ImportError("pip install vtk")

        self.sig = CanvasSignals()

        self._vtkw = QVTKRenderWindowInteractor(self)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.addWidget(self._vtkw)

        self._ren = vtk.vtkRenderer()
        self._ren.SetBackground(BG_R, BG_G, BG_B)
        self._ren.GradientBackgroundOff()
        rw = self._vtkw.GetRenderWindow()
        # Layer 0 — nube de puntos
        rw.SetNumberOfLayers(2)
        rw.AddRenderer(self._ren); rw.SetMultiSamples(0); rw.SetAlphaBitPlanes(0)
        # Layer 1 — overlay del grid (siempre encima de la nube)
        self._ren_overlay = vtk.vtkRenderer()
        self._ren_overlay.SetLayer(1)
        self._ren_overlay.SetInteractive(0)       # no captura eventos
        self._ren_overlay.SetBackgroundAlpha(0.0)
        self._ren_overlay.SetBackground(0, 0, 0)
        self._ren_overlay.EraseOff()              # fondo transparente
        rw.AddRenderer(self._ren_overlay)
        self._vtkw.Initialize()
        self._style = _AnnotationStyle(self)
        self._vtkw.SetInteractorStyle(self._style)

        # Estado general
        self.pc: Optional["PointCloud"]   = None
        self.project: Optional["Project"] = None
        self.active_tool: Optional["BaseTool"] = None
        self._annotation_lut    = None
        self._annotation_lut_u8 = None
        self._cur_xyz    = None
        self._cur_idx    = None
        self._cur_colors = None
        self._render_state = "IDLE"
        self._interacting  = False
        self._color_mode = "Anotación"
        self._cmap       = "viridis"
        self._sz         = 2.0
        self._cursor_tick = 0
        self._refresh_pending = False
        self._current_n  = 0
        self._budget     = 5_000_000
        self._budget_auto = True
        self._done_ticks = 0

        # ── MODO DE VISTA ─────────────────────────────────────────────────────
        # "sparse" por defecto — no traba al abrir nubes grandes
        self._overview_mode: str = "sparse"

        # ── TILE MODE ─────────────────────────────────────────────────────────
        self._tile_mode: bool             = False
        self._fit_camera_on_load: bool    = True
        self._tile_indices: Optional[np.ndarray] = None
        self._active_tile                  = None
        self._tile_loader: Optional[_TileLoader] = None

        # ── TileManager + overlay ─────────────────────────────────────────────
        self._tile_manager = None
        self._hover_tile   = None
        self._tile_grid    = _TileGridOverlay(self._ren_overlay)
        # Sincronizar cámara del overlay con la cámara principal
        # (se actualiza automáticamente en cada render porque comparten el mismo objeto)
        self._ren_overlay.SetActiveCamera(self._ren.GetActiveCamera())

        # ── Grid edit mode (Mover / Rotar con mouse) ──────────────────────────
        self._grid_edit_mode: Optional[str] = None   # "move" | "rotate" | None
        self._grid_drag_active: bool = False
        self._grid_drag_world_start = (0.0, 0.0)
        self._grid_offset_start     = (0.0, 0.0)
        self._grid_drag_angle_start = 0.0
        self._grid_rotation_start   = 0.0

        # ── FLY MODE (Unity-style WASD + mouse look) ──────────────────────────
        self._fly_mode:    bool  = False
        self._fly_keys:    set   = set()
        self._fly_speed:   float = 10.0   # m/s base
        self._fly_sens:    float = 0.18   # grados/pixel
        self._fly_rb_down: bool  = False
        self._fly_last_px: tuple = (0, 0)
        self._fly_timer = QTimer(self)
        self._fly_timer.setInterval(16)   # ~60 fps
        self._fly_timer.timeout.connect(self._fly_tick)

        # ── GPU buffer ───────────────────────────────────────────────────────
        self._gpu_capacity   = 0
        self._cells_arange   = None
        self._vtk_actor      = None
        self._cloud_opacity  = 1.0
        self._vtk_mapper     = None
        self._vtk_poly       = None
        self._cells_n        = 0
        self._ref_xyz        = None
        self._ref_col        = None
        self._ref_cells_segs = None

        # ── Timers y throttle ────────────────────────────────────────────────
        self._last_render_t  = 0.0
        self._render_queued  = False
        self._render_timer   = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._flush_render)

        self._interaction_timer = QTimer(self)
        self._interaction_timer.setSingleShot(True)
        self._interaction_timer.setInterval(INTERACTION_COOLDOWN_MS)
        self._interaction_timer.timeout.connect(self._on_interaction_end)

        # Overlays de herramientas
        _r = self._ren; _rf = self._do_render
        self._mline        = _LineProxy(_r, _rf)
        self._mline_live   = _LineProxy(_r, _rf)
        self._p1_mk        = _MarkersProxy(_r, _rf)
        self._snap_mk      = _MarkersProxy(_r, _rf)
        self._brush_cursor = _LineProxy(_r, _rf)
        self._brush_center = _MarkersProxy(_r, _rf)

        # LOD Worker
        from render.lod_worker import LODWorker
        self._worker       = LODWorker()
        self._worker.coarse_ready.connect(self._on_coarse_ready)
        self._worker.batch_ready.connect(self._on_batch_ready)

        # Timers de LOD y FPS
        self._lod_t = QTimer(); self._lod_t.setInterval(LOD_MS)
        self._lod_t.timeout.connect(self._lod_tick); self._lod_t.start()
        self._fps_t = QTimer(); self._fps_t.setInterval(1000)
        self._fps_t.timeout.connect(self._fps_tick); self._fps_t.start()
        self._frame_count = 0
        rw.AddObserver("EndEvent",
                       lambda o, e: setattr(self,"_frame_count",self._frame_count+1))

        # Observer que fuerza rango de clip libre en fly mode.
        # Dispara después de que VTK reinicia el clipping range internamente.
        self._ren.AddObserver("EndEvent", self._fly_fix_clipping)

    # ── Render ────────────────────────────────────────────────────────────────

    # ── Overlay API ───────────────────────────────────────────────────────────

    def add_overlay_actor(self, actor) -> None:
        """Añade un actor VTK a la escena (para capas superpuestas)."""
        self._ren.AddActor(actor)

    def remove_overlay_actor(self, actor) -> None:
        """Elimina un actor VTK de la escena."""
        try: self._ren.RemoveActor(actor)
        except Exception: pass

    def request_render(self) -> None:
        """Solicita un render del canvas (API pública)."""
        self._request_render()

    def _do_render(self):
        self._vtkw.GetRenderWindow().Render()
        self._last_render_t = time.monotonic()

    def _request_render(self):
        now = time.monotonic(); elapsed_ms = (now - self._last_render_t)*1000
        if elapsed_ms >= _RENDER_THROTTLE_MS:
            self._do_render(); self._render_queued = False
        elif not self._render_queued:
            self._render_queued = True
            self._render_timer.start(max(1, int(_RENDER_THROTTLE_MS - elapsed_ms)))

    def _flush_render(self):
        self._render_queued = False; self._do_render()

    def update(self): self._do_render()

    # ── Interaction ───────────────────────────────────────────────────────────

    def _on_interaction_start(self):
        self._interacting = True
        if self._render_state != "DRAG":
            self._render_state = "DRAG"; self._worker.cancel()
        self._interaction_timer.start()

    def _on_interaction_end(self):
        self._interacting = False; self._done_ticks = 0
        if self._render_state == "DRAG":
            self._render_state = "IDLE"

    # ── API pública ───────────────────────────────────────────────────────────

    def load_cloud(self, pc, project):
        self._worker.cancel()
        self.pc = pc; self.project = project
        from render.colors import build_annotation_lut, build_annotation_lut_u8
        self._annotation_lut    = build_annotation_lut(project.schema)
        self._annotation_lut_u8 = build_annotation_lut_u8(project.schema)
        # Respetar el modo actual al determinar budget inicial
        init_cap = SPARSE_CAP if self._overview_mode == "sparse" else OVERVIEW_BUDGET_CAP
        self._budget       = min(pc.n_points, init_cap)
        self._render_state = "IDLE"
        self._cur_xyz = None; self._cur_idx = None; self._cur_colors = None
        if self._tile_mode:
            self._tile_mode = False; self._active_tile = None; self._tile_indices = None
        self._ensure_gpu_buffer(min(pc.n_points, self._budget))

    def first_display(self, pc):
        self.pc = pc
        if pc.octree is not None and pc.octree._lod_ready:
            self._upload_coarse(pc)

    def set_active_tool(self, tool):
        if self.active_tool: self.active_tool.deactivate()
        self.active_tool = tool
        if tool: tool.activate(self)

    # ── MODO DE VISTA ─────────────────────────────────────────────────────────

    @property
    def overview_mode(self) -> str:
        return self._overview_mode

    def set_overview_mode(self, mode: str) -> None:
        """
        Cambia el modo de overview ("sparse" | "full").
        Ignorado si estamos en tile mode.
        """
        if mode not in ("sparse", "full"):
            return
        if self._overview_mode == mode:
            return
        self._overview_mode = mode
        if self._tile_mode:
            return   # se aplicará cuando salgamos del tile

        if mode == "sparse":
            # Cap budget inmediatamente
            self._budget = min(self._current_n or SPARSE_CAP, SPARSE_CAP)
            # Si ya tenemos más puntos de los que queremos, recargar con nivel más bajo
            if self._current_n > SPARSE_CAP and self.pc and self.pc.octree:
                self._upload_coarse_at(SPARSE_CAP)
        else:
            # "full": soltar el cap y dejar que el LOD refine
            self._budget = min(OVERVIEW_BUDGET_CAP, self.pc.n_points if self.pc else OVERVIEW_BUDGET_CAP)
            self._render_state = "IDLE"

        self.sig.view_mode_changed.emit(mode)

    def _upload_coarse_at(self, target_pts: int) -> None:
        """Muestra una vista LOD con exactamente target_pts puntos."""
        pc = self.pc
        if pc is None or pc.octree is None or not pc.octree._lod_ready:
            return
        from render.colors import compute_colors_u8
        xyz, idx = pc.octree.get_coarse_view(target_pts)
        if len(xyz) == 0: return
        lbl = (self.project.labels[idx]
               if self.project and self.project.labels is not None else None)
        col_u8 = compute_colors_u8(xyz, pc.get_attrs(idx), self._color_mode, self._cmap,
                                   annotation_labels=lbl,
                                   annotation_lut_u8=self._annotation_lut_u8)
        self._cur_xyz=xyz; self._cur_idx=idx
        self._ensure_gpu_buffer(len(xyz))
        self._gpu_upload_zero_copy(xyz, col_u8)
        self._do_render()

    # ── TILE MANAGER y OVERLAY ────────────────────────────────────────────────

    def set_tile_manager(self, tm) -> None:
        """Registra el TileManager para el grid overlay y el hover detection."""
        self._tile_manager = tm
        self._hover_tile   = None
        self._grid_drag_active = False
        show_grid = (tm is not None) and not self._tile_mode
        self._tile_grid.set_tile_manager(tm, show=show_grid)
        self._do_render()

    # ── Grid edit mode (Mover / Rotar con mouse) ──────────────────────────────

    def set_grid_edit_mode(self, mode: Optional[str]) -> None:
        """
        Activa o desactiva el modo de edición del grid con mouse.
        mode: "move" | "rotate" | None
        """
        self._grid_edit_mode   = mode
        self._grid_drag_active = False
        if mode == "move":
            self._vtkw.setCursor(Qt.SizeAllCursor)
        elif mode == "rotate":
            self._vtkw.setCursor(Qt.CrossCursor)
        else:
            self._vtkw.setCursor(Qt.ArrowCursor)

    def set_panel_hover_tile(self, tile) -> None:
        """
        Feature 1: hover desde el panel 2D → resaltar tile en el overlay 3D.
        Llamado cuando el mouse pasa por encima de un tile en el TilePanel.
        """
        if tile is not self._hover_tile:
            self._hover_tile = tile
            self._tile_grid.update_hover(tile)
            self._request_render()
        # No emitimos sig.tile_hovered para evitar bucle feedback

    def _start_grid_drag(self, screen_pos) -> None:
        if self._tile_manager is None:
            return
        wx, wy = self._screen_to_world_xy(screen_pos)
        if wx is None:
            return
        self._grid_drag_active = True
        if self._grid_edit_mode == "move":
            self._grid_drag_world_start = (wx, wy)
            self._grid_offset_start = (self._tile_manager.offset_x,
                                        self._tile_manager.offset_y)
        elif self._grid_edit_mode == "rotate":
            cx, cy = self._tile_manager.grid_center_world()
            self._grid_drag_angle_start = math.atan2(wy - cy, wx - cx)
            self._grid_rotation_start   = self._tile_manager.rotation_deg

    def _update_grid_drag(self, screen_pos) -> None:
        if not self._grid_drag_active or self._tile_manager is None:
            return
        wx, wy = self._screen_to_world_xy(screen_pos)
        if wx is None:
            return
        tm = self._tile_manager

        if self._grid_edit_mode == "move":
            wx0, wy0 = self._grid_drag_world_start
            ox0, oy0 = self._grid_offset_start
            new_ox = ox0 + (wx - wx0)
            new_oy = oy0 + (wy - wy0)
            # Preview rápido: solo geometry, sin re-sampling
            tm.update_transform_preview(new_ox, new_oy, tm.rotation_deg)
            self._tile_grid.rebuild()
            self.sig.grid_transform_preview.emit(new_ox, new_oy, tm.rotation_deg)

        elif self._grid_edit_mode == "rotate":
            cx, cy = tm.grid_center_world()
            current_angle = math.atan2(wy - cy, wx - cx)
            delta = math.degrees(current_angle - self._grid_drag_angle_start)
            # Corregir wrap-around de atan2
            while delta >  90: delta -= 180
            while delta < -90: delta += 180
            new_rot = max(-90.0, min(90.0, self._grid_rotation_start + delta))
            tm.update_transform_preview(tm.offset_x, tm.offset_y, new_rot)
            self._tile_grid.rebuild()
            self.sig.grid_transform_preview.emit(tm.offset_x, tm.offset_y, new_rot)

        self._request_render()

    def _end_grid_drag(self) -> None:
        """Al soltar el mouse: emitir señal para que main_window haga el rebuild completo."""
        self._grid_drag_active = False
        if self._tile_manager is not None:
            self.sig.grid_transform_set.emit(
                self._tile_manager.offset_x,
                self._tile_manager.offset_y,
                self._tile_manager.rotation_deg)

    def _update_tile_hover(self, screen_pos) -> None:
        """Detecta qué tile está bajo el cursor y actualiza el overlay y el panel."""
        if self._tile_manager is None or self._tile_mode:
            return
        wx, wy = self._screen_to_world_xy(screen_pos)
        if wx is None:
            tile = None
        else:
            tile = self._tile_manager.tile_at_world_xy(float(wx), float(wy))

        if tile is not self._hover_tile:
            self._hover_tile = tile
            self._tile_grid.update_hover(tile)
            self.sig.tile_hovered.emit(tile)
            self._request_render()

    def _screen_to_world_xy(self, screen_pos):
        """
        Proyecta screen_pos al plano Z = mean_z de la nube usando ray–plane intersection.
        Devuelve (wx, wy) o (None, None).
        """
        if self.pc is None:
            return None, None
        try:
            rw = self._vtkw.GetRenderWindow()
            W, H = float(rw.GetSize()[0]), float(rw.GetSize()[1])
            if W == 0 or H == 0:
                return None, None

            sx, sy = float(screen_pos[0]), float(screen_pos[1])

            # NDC (Y invertido entre Qt y VTK ya corregido en _pos())
            ndc_x = 2.0 * sx / W - 1.0
            ndc_y = 1.0 - 2.0 * sy / H

            MVP     = self._get_mvp(W, H)
            inv_MVP = np.linalg.inv(MVP)

            def unproject(ndc_z):
                p = inv_MVP @ np.array([ndc_x, ndc_y, ndc_z, 1.0])
                return p[:3] / p[3]

            near = unproject(-1.0)
            far  = unproject( 1.0)
            ray_dir = far - near
            z_target = float(self.pc.bounds[0, 2] + self.pc.bounds[1, 2]) * 0.5

            dz = ray_dir[2]
            if abs(dz) < 1e-9:
                return None, None
            t = (z_target - near[2]) / dz
            if t < 0:
                return None, None
            hit = near + t * ray_dir
            return float(hit[0]), float(hit[1])
        except Exception:
            return None, None

    # ── TILE MODE ─────────────────────────────────────────────────────────────

    @property
    def is_tile_mode(self) -> bool:
        return self._tile_mode

    def enter_tile_mode(self, tile, indices: np.ndarray, fit_camera: bool = True) -> None:
        if self.pc is None: return
        self._worker.cancel()
        if self._tile_loader is not None and self._tile_loader.isRunning():
            self._tile_loader.terminate(); self._tile_loader.wait()

        self._tile_mode    = True
        self._active_tile  = tile
        self._tile_indices = indices
        self._render_state = "TILE_LOADING"
        tile.n_points = len(indices); tile.loaded = True

        # Ocultar grid en tile mode, marcar tile activo
        self._tile_grid.set_visible(False)
        self._tile_grid.update_active(tile)

        self._fit_camera_on_load = fit_camera   # guardado para _on_tile_loaded
        self._tile_loader = _TileLoader(
            self.pc, indices, self.project,
            self._color_mode, self._cmap, self._annotation_lut_u8)
        self._tile_loader.ready.connect(self._on_tile_loaded)
        self._tile_loader.coarse_ready.connect(self._on_tile_coarse_ready)
        self._tile_loader.progress.connect(self._on_tile_load_progress)
        self._tile_loader.start()
        self.sig.tile_mode_changed.emit(True, tile)
        self.sig.view_mode_changed.emit("tile")

    def exit_tile_mode(self) -> None:
        if not self._tile_mode: return
        if self._tile_loader is not None and self._tile_loader.isRunning():
            self._tile_loader.terminate(); self._tile_loader.wait()
        self._tile_loader = None

        self._tile_mode   = False
        self._active_tile = None
        self._tile_indices = None
        self._render_state = "IDLE"

        # Restaurar grid overlay
        self._tile_grid.update_active(None)
        self._tile_grid.set_visible(True)

        # Restaurar budget según modo overview
        if self._overview_mode == "sparse":
            self._budget = SPARSE_CAP
        else:
            self._budget = min(OVERVIEW_BUDGET_CAP, self.pc.n_points if self.pc else OVERVIEW_BUDGET_CAP)

        if self.pc is not None:
            self._upload_coarse(self.pc)
            self._fit_camera(self.pc)

        self.sig.tile_mode_changed.emit(False, None)
        self.sig.view_mode_changed.emit(self._overview_mode)

    def _on_tile_load_progress(self, pct: int, msg: str) -> None:
        if self.pc:
            self.sig.render_info.emit(pct, -1)

    def _on_tile_coarse_ready(self, xyz_c: np.ndarray, col_u8: np.ndarray,
                              idx: np.ndarray) -> None:
        """
        Vista previa decimada de un tile denso — llega antes que el tile
        completo (_on_tile_loaded) para que la navegación y el pincel puedan
        empezar de inmediato en vez de esperar a que termine de leerse y
        colorearse el tile entero (podía tardar varios segundos en tiles
        de 20-30M+ puntos).
        """
        if not self._tile_mode: return
        n = len(xyz_c)
        if n == 0: return
        self._cur_xyz    = xyz_c
        self._cur_idx    = idx
        self._cur_colors = None
        self._budget     = n
        self._ensure_gpu_buffer(n)
        self._gpu_upload_zero_copy(xyz_c, col_u8)
        if self._active_tile is not None and getattr(self, '_fit_camera_on_load', True):
            self._fit_camera_to_tile(self._active_tile)
        self._render_state = "TILE_REFINING"
        self._do_render()
        self.sig.render_info.emit(n, self.pc.n_points if self.pc else n)

    def _on_tile_loaded(self, xyz_c: np.ndarray, col_u8: np.ndarray,
                        idx: np.ndarray) -> None:
        if not self._tile_mode: return
        n = len(xyz_c)
        if n == 0: return
        self._cur_xyz    = xyz_c
        self._cur_idx    = idx
        self._cur_colors = None
        self._budget     = n
        self._ensure_gpu_buffer(n)
        self._gpu_upload_zero_copy(xyz_c, col_u8)
        if self._active_tile is not None and getattr(self, '_fit_camera_on_load', True):
            self._fit_camera_to_tile(self._active_tile)
        self._render_state = "DONE"
        self._do_render()
        self.sig.render_info.emit(n, self.pc.n_points if self.pc else n)

    def _fit_camera_to_tile(self, tile) -> None:
        try:
            if self._tile_manager is None: return
            corners = self._tile_manager.world_corners(tile)
            xs = [c[0] for c in corners]; ys = [c[1] for c in corners]
            pad = self._tile_manager.tile_size_m * 0.15
            self._ren.ResetCamera(
                min(xs)-pad, max(xs)+pad, min(ys)-pad, max(ys)+pad,
                float(self._ren.GetActiveCamera().GetFocalPoint()[2]) - 50,
                float(self._ren.GetActiveCamera().GetFocalPoint()[2]) + 200)
            self._do_render()
        except Exception:
            pass

    # ── refresh_colors ────────────────────────────────────────────────────────

    def refresh_colors(self, changed_idx=None, changed_class=None):
        if self.pc is None or self._cur_idx is None or self._cur_xyz is None:
            return

        class_id_int = None
        if changed_class is not None:
            if isinstance(changed_class, (int, np.integer)):
                class_id_int = int(changed_class)
            elif isinstance(changed_class, np.ndarray) and len(changed_class) > 0:
                if np.all(changed_class == changed_class[0]):
                    class_id_int = int(changed_class[0])

        # Vía rápida — funciona igual en overview y en tile mode: actualiza
        # SOLO los puntos afectados directamente en el buffer de color ya
        # subido a GPU (C, GIL liberado). Antes, en tile mode se forzaba
        # SIEMPRE el camino lento de abajo (recompute + reupload del tile
        # completo, hasta 20-30M pts) en cada trazo de pincel — esa era la
        # causa principal del freeze al anotar nubes densas.
        if (changed_idx is not None and class_id_int is not None
                and self._color_mode == "Anotación"
                and self._annotation_lut_u8 is not None
                and self._ref_col is not None and self._current_n > 0):
            try:
                from core._fast import FC
                FC.refresh_colors_partial(
                    self._ref_col, self._cur_idx,
                    np.asarray(changed_idx, np.int32),
                    self._annotation_lut_u8, class_id_int)
                if self._vtk_poly is not None:
                    s = self._vtk_poly.GetPointData().GetScalars()
                    if s: s.Modified()
                    self._vtk_poly.Modified()
                self._do_render(); return
            except Exception:
                pass

        # Vía lenta: recompute completo. Necesaria cuando no hay un único
        # class_id (undo/redo con etiquetas mixtas, import masivo) o el modo
        # de color activo no es "Anotación" (Elevación, RGB, etc. también
        # deben reflejar el cambio aunque el color en sí no dependa de la clase).
        if not self._refresh_pending:
            self._refresh_pending = True
            QTimer.singleShot(0, self._do_refresh_colors)

    def _do_refresh_colors(self):
        self._refresh_pending = False
        if self.pc is None or self._cur_xyz is None or self._cur_idx is None:
            return
        try:
            from render.colors import compute_colors_u8
            lbl = (self.project.labels[self._cur_idx]
                   if self.project and self.project.labels is not None else None)
            col_u8 = compute_colors_u8(
                self._cur_xyz, self.pc.get_attrs(self._cur_idx),
                self._color_mode, self._cmap,
                annotation_labels=lbl, annotation_lut_u8=self._annotation_lut_u8)
            if self._vtk_poly is None or self._current_n == 0: return
            n = min(len(col_u8), self._current_n)
            col_u8_c = np.ascontiguousarray(col_u8[:n])
            self._ref_col = col_u8_c
            col_vtk = numpy_to_vtk(col_u8_c, deep=False)
            col_vtk.SetNumberOfComponents(4); col_vtk.SetName("rgba")
            self._vtk_poly.GetPointData().SetScalars(col_vtk)
            self._vtk_poly.Modified(); self._do_render()
        except Exception:
            import traceback; traceback.print_exc()

    def force_color_rebuild(self) -> None:
        """
        Reconstruye completamente los colores y los sube a la GPU.
        Llamar después de undo/redo para garantizar que la vista es correcta.
        Usa _gpu_upload_zero_copy (el mismo path que el render normal).
        """
        if self.pc is None or self._cur_xyz is None or self._cur_idx is None:
            return
        if self._current_n == 0:
            return
        try:
            from render.colors import compute_colors_u8
            idx = self._cur_idx
            # Si hay más índices que puntos visibles, recortar
            n   = min(len(self._cur_xyz), self._current_n, len(idx))
            if n == 0:
                return
            xyz_v  = self._cur_xyz[:n]
            idx_v  = idx[:n]
            lbl = (self.project.labels[idx_v]
                   if self.project and self.project.labels is not None else None)
            col_u8 = compute_colors_u8(
                xyz_v, self.pc.get_attrs(idx_v),
                self._color_mode, self._cmap,
                annotation_labels=lbl,
                annotation_lut_u8=self._annotation_lut_u8)
            # Usar gpu_upload que es el path correcto y funciona siempre
            self._gpu_upload_zero_copy(xyz_v, col_u8)
            self._do_render()
        except Exception:
            import traceback; traceback.print_exc()

    def rebuild_annotation_lut(self):
        if self.project is None: return
        from render.colors import build_annotation_lut, build_annotation_lut_u8
        self._annotation_lut    = build_annotation_lut(self.project.schema)
        self._annotation_lut_u8 = build_annotation_lut_u8(self.project.schema)
        self.refresh_colors()

    @property
    def rendered_n(self): return self._current_n

    def set_color_mode(self, mode, cmap="viridis"):
        if self._color_mode == mode and self._cmap == cmap: return
        self._color_mode = mode; self._cmap = cmap; self._current_n = 0
        if self._tile_mode:
            if self._tile_indices is not None and self._active_tile is not None:
                # fit_camera=False: mantener posición de cámara al cambiar color
                self.enter_tile_mode(self._active_tile, self._tile_indices,
                                     fit_camera=False)
        else:
            self._render_state = "IDLE"; self._req_worker()

    def set_point_size(self, sz):
        self._sz = sz
        if self._vtk_actor:
            self._vtk_actor.GetProperty().SetPointSize(sz); self._do_render()

    def set_edl(self, v, strength=0.6): pass

    def set_budget(self, n):
        if self._tile_mode or self._overview_mode == "sparse": return
        self._budget = max(BUDGET_FLOOR, min(MAX_BUDGET, n))
        self._render_state = "IDLE"; self._req_worker()

    def set_budget_auto(self, v): self._budget_auto = v
    def toggle_grid(self, show): pass

    # ── Eye-Dome Lighting ─────────────────────────────────────────────────────
    # Sombreado por profundidad (sin necesitar normales) que hace mucho más
    # legible el relieve de una nube de puntos sin color — el mismo efecto
    # que CloudCompare/Potree activan por defecto. VTK 9.x trae esto nativo
    # (vtkEDLShading, backend OpenGL2) — no hace falta un shader propio.
    # render/edl.py queda como estaba (stub) porque el pase real vive aquí,
    # directamente sobre el pipeline de render del canvas (SetPass en el
    # vtkRenderer), que es donde VTK espera que se conecte.
    def set_edl_enabled(self, enabled: bool) -> None:
        try:
            if enabled:
                if getattr(self, '_edl_pass', None) is None:
                    basic_passes = vtk.vtkRenderStepsPass()
                    edl = vtk.vtkEDLShading()
                    edl.SetDelegatePass(basic_passes)
                    self._edl_pass = edl
                self._ren.SetPass(self._edl_pass)
            else:
                self._ren.SetPass(None)
            self._do_render()
        except Exception as e:
            print(f"[EDL] No se pudo {'activar' if enabled else 'desactivar'}: {e}")

    def clamp_camera_to_cloud(self) -> None:
        """
        Evita que la camara se aleje demasiado de la nube.
        Si la distancia al centro supera 5x la diagonal de la nube,
        hace un reset suave hacia el centro.
        """
        if self.pc is None: return
        try:
            b    = self.pc.bounds
            center = ((b[0] + b[1]) * 0.5).astype(np.float64)
            diag   = float(np.linalg.norm(b[1] - b[0]))
            if diag < 0.1: return
            cam  = self._ren.GetActiveCamera()
            pos  = np.array(cam.GetPosition(), np.float64)
            dist = float(np.linalg.norm(pos - center))
            max_dist = diag * 5.0
            if dist > max_dist:
                # Mover la camara hacia el centro manteniendo la direccion
                direction = center - pos
                dn = float(np.linalg.norm(direction))
                if dn > 1e-9:
                    direction /= dn
                new_pos = center - direction * max_dist * 0.8
                cam.SetPosition(*new_pos)
                cam.SetFocalPoint(*center)
                self._do_render()
        except Exception:
            pass

    def reset_view(self):
        if self._tile_mode and self._active_tile:
            self._fit_camera_to_tile(self._active_tile)
        elif self.pc:
            self._fit_camera(self.pc)

    # ── GPU pipeline ─────────────────────────────────────────────────────────

    def _ensure_gpu_buffer(self, capacity):
        capacity = max(capacity, 1_000_000)
        if capacity > self._gpu_capacity:
            self._cells_arange = np.arange(capacity, dtype=np.int64)
            self._gpu_capacity = capacity
        if self._vtk_actor is None:
            poly = vtk.vtkPolyData()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly); mapper.SetColorModeToDirectScalars()
            mapper.SetScalarModeToUsePointData(); mapper.ScalarVisibilityOn()
            actor = vtk.vtkActor(); actor.SetMapper(mapper)
            p = actor.GetProperty(); p.SetPointSize(self._sz)
            p.RenderPointsAsSpheresOff(); p.LightingOff()
            self._ren.AddActor(actor)
            self._vtk_poly=poly; self._vtk_mapper=mapper; self._vtk_actor=actor
            self._cells_n=0; self._current_n=0

    def _make_cells_fast(self, n):
        if n > self._gpu_capacity: self._ensure_gpu_buffer(int(n*1.3))
        segs = np.empty(n+1, np.int64); segs[0]=n; segs[1:]=self._cells_arange[:n]
        self._ref_cells_segs = segs
        cells = vtk.vtkCellArray(); cells.SetCells(1, numpy_to_vtkIdTypeArray(segs, deep=False))
        return cells

    def _gpu_upload_zero_copy(self, xyz, col_u8):
        n = len(xyz)
        if n == 0 or self._vtk_poly is None: return
        if n > self._gpu_capacity: self._ensure_gpu_buffer(int(n*1.3))
        self._ref_xyz=xyz; self._ref_col=col_u8
        vtk_pts = vtk.vtkPoints(); vtk_pts.SetData(numpy_to_vtk(xyz, deep=False))
        self._vtk_poly.SetPoints(vtk_pts)
        col_vtk = numpy_to_vtk(col_u8, deep=False)
        col_vtk.SetNumberOfComponents(4); col_vtk.SetName("rgba")
        self._vtk_poly.GetPointData().SetScalars(col_vtk)
        if n != self._cells_n:
            self._vtk_poly.SetVerts(self._make_cells_fast(n)); self._cells_n=n
        self._vtk_poly.Modified(); self._current_n=n

    def _gpu_upload_full(self, xyz, colors):
        n = len(xyz)
        if n == 0: return
        col_u8 = np.ascontiguousarray(np.clip(colors*255,0,255).astype(np.uint8))
        self._gpu_upload_zero_copy(np.ascontiguousarray(xyz), col_u8)

    def _init_actor(self, xyz, colors):
        self._ensure_gpu_buffer(len(xyz)); self._gpu_upload_full(xyz, colors); self._do_render()

    def _update_vtk_all(self, xyz, colors):
        self._ensure_gpu_buffer(len(xyz)); self._gpu_upload_full(xyz, colors); self._request_render()

    # ── Cámara ────────────────────────────────────────────────────────────────

    def _fit_camera(self, pc):
        try:
            b = pc.bounds
            self._ren.ResetCamera(float(b[0,0]),float(b[1,0]),float(b[0,1]),
                                  float(b[1,1]),float(b[0,2]),float(b[1,2]))
            self._ren.GetActiveCamera().SetParallelProjection(False)
            self._do_render()
        except Exception: pass

    def set_cloud_opacity(self, opacity: float) -> None:
        """
        Ajusta la opacidad global de la nube de puntos.
        Los colores son RGBA (4 canales) — multiplica el canal A.
        opacity: 0.0 = invisible, 1.0 = completamente opaco
        """
        self._cloud_opacity = max(0.0, min(1.0, float(opacity)))
        # Forzar reconstrucción de colores con la nueva opacidad
        if self._ref_col is not None and len(self._ref_col) > 0:
            col = self._ref_col.copy()
            if col.shape[1] == 4:
                col[:, 3] = np.clip(col[:, 3] * self._cloud_opacity, 0, 255).astype(np.uint8)
            vtk_col = numpy_to_vtk(col, deep=True)
            vtk_col.SetNumberOfComponents(4); vtk_col.SetName("rgba")
            self._vtk_poly.GetPointData().SetScalars(vtk_col)
            self._vtk_poly.Modified()
        elif self._vtk_actor:
            # Fallback: opacidad del actor (funciona solo con RGB)
            self._vtk_actor.GetProperty().SetOpacity(self._cloud_opacity)
        self._do_render()

    def top_view(self, tight: bool = True) -> None:
        """
        Vista cenital (top-down) alineada exactamente con el Norte/eje Y.
        La camara mira hacia abajo en -Z, up vector = +Y (norte).
        Si tight=True, ajusta la distancia para encuadrar toda la escena.
        """
        cam = self._ren.GetActiveCamera()

        # Usar bounds de la nube actual o el tile activo
        if self._cur_xyz is not None and len(self._cur_xyz) > 0:
            xyz = self._cur_xyz
            cx = float(xyz[:, 0].mean())
            cy = float(xyz[:, 1].mean())
            cz = float(xyz[:, 2].mean())
            extent_x = float(xyz[:, 0].max() - xyz[:, 0].min())
            extent_y = float(xyz[:, 1].max() - xyz[:, 1].min())
        elif self._pc is not None and self._pc.bounds is not None:
            b = self._pc.bounds
            cx = float((b[0,0]+b[1,0])/2); cy = float((b[0,1]+b[1,1])/2)
            cz = float(b[0,2])
            extent_x = float(b[1,0]-b[0,0]); extent_y = float(b[1,1]-b[0,1])
        else:
            return

        extent = max(extent_x, extent_y, 1.0)

        # Altura de cámara = suficiente para ver todo con algo de margen
        height = extent * 1.15

        cam.SetFocalPoint(cx, cy, cz)
        cam.SetPosition(cx, cy, cz + height)
        cam.SetViewUp(0.0, 1.0, 0.0)    # Norte = +Y arriba en pantalla
        cam.SetViewAngle(60.0)

        # Proyección paralela opcional para vista cenital exacta
        # cam.SetParallelProjection(True)
        # cam.SetParallelScale(extent * 0.6)

        self._ren.ResetCameraClippingRange()
        self._do_render()

    def _get_eye3d(self):
        try: return np.array(self._ren.GetActiveCamera().GetPosition(), np.float32)
        except: return np.zeros(3, np.float32)

    def _get_fov(self):
        try: return float(self._ren.GetActiveCamera().GetViewAngle())
        except: return 45.0

    def _get_frustum_planes(self, W, H):
        try:
            from utils.spatial import extract_frustum_planes
            MVP = self._get_mvp(W, H).astype(np.float32)
            if not np.any(np.isnan(MVP)): return extract_frustum_planes(MVP)
        except Exception: pass
        return None

    def map_to_screen(self, xyz):
        try:
            rw = self._vtkw.GetRenderWindow()
            W, H = float(rw.GetSize()[0]), float(rw.GetSize()[1])
            if W==0 or H==0: return np.zeros((len(xyz),2), np.float32)
            MVP = self._get_mvp(W, H); n = len(xyz)
            xyzw = np.column_stack([xyz.astype(np.float64), np.ones(n)])
            clip = (MVP @ xyzw.T).T
            wc = np.where(np.abs(clip[:,3:4])<1e-9, 1.0, clip[:,3:4])
            ndc = clip[:,:2]/wc
            return np.column_stack([(ndc[:,0]*0.5+0.5)*W, (0.5-ndc[:,1]*0.5)*H]).astype(np.float32)
        except Exception: return np.zeros((len(xyz),2), np.float32)

    def get_frozen_projector(self):
        rw = self._vtkw.GetRenderWindow(); W,H = float(rw.GetSize()[0]),float(rw.GetSize()[1])
        MVP = self._get_mvp(W,H).copy(); _W,_H = W,H
        def _project(xyz):
            n=len(xyz); xyzw=np.column_stack([xyz.astype(np.float64),np.ones(n)])
            clip=(MVP@xyzw.T).T; wc=np.where(np.abs(clip[:,3:4])<1e-9,1.0,clip[:,3:4]); ndc=clip[:,:2]/wc
            return np.column_stack([(ndc[:,0]*0.5+0.5)*_W,(0.5-ndc[:,1]*0.5)*_H]).astype(np.float32)
        return _project

    def _get_mvp(self, W, H):
        cam=self._ren.GetActiveCamera(); aspect=W/H if H>0 else 1.0
        def _m4(v):
            m=np.zeros((4,4),np.float64)
            for i in range(4):
                for j in range(4): m[i,j]=v.GetElement(i,j)
            return m
        return _m4(cam.GetProjectionTransformMatrix(aspect,-1.,1.)) @ _m4(cam.GetModelViewTransformMatrix())

    # ── LOD ───────────────────────────────────────────────────────────────────

    def _max_lod_pts(self):
        if self.pc and self.pc.octree and self.pc.octree._lod_levels:
            return max(len(lv) for lv in self.pc.octree._lod_levels)
        return self.pc.n_points if self.pc else 5_000_000

    def _lod_tick(self):
        if self._interacting or self.pc is None or self._tile_mode:
            return

        if self._render_state == "IDLE":
            self._done_ticks = 0; self._render_state = "REFINING"
            self._req_worker()

        elif self._render_state == "DONE":
            # En sparse mode: nunca crecer budget
            if self._overview_mode == "sparse":
                return
            max_lod = self._max_lod_pts()
            if self._budget < max_lod:
                self._done_ticks += 1
                if self._done_ticks >= 40:
                    self._done_ticks = 0
                    self._budget = min(max_lod, int(self._budget * 2.0))
                    self._render_state = "IDLE"

    def _fps_tick(self):
        fps = float(self._frame_count); self._frame_count = 0
        self.sig.fps.emit(fps)
        # No ajustar budget en tile mode ni en sparse mode
        if self._tile_mode or self._overview_mode == "sparse" \
                or not self._budget_auto or self.pc is None:
            return
        max_lod = self._max_lod_pts()
        if fps < FPS_LOW and self._budget > BUDGET_FLOOR:
            self._budget = max(BUDGET_FLOOR, max(int(self._budget*0.75), self._current_n))
        elif fps > FPS_HIGH and self._budget < max_lod:
            self._budget = min(max_lod, int(self._budget*1.5))

    def _show_drag_view(self):
        pc = self.pc
        if pc is None or pc.octree is None or not pc.octree.ready: return
        try:
            from render.colors import compute_colors_u8
            xyz, idx = pc.octree.get_coarse_view(DRAG_TARGET_PTS)
            if len(xyz) == 0: return
            lbl = (self.project.labels[idx]
                   if self.project and self.project.labels is not None else None)
            col_u8 = compute_colors_u8(xyz, pc.get_attrs(idx), self._color_mode,
                                       self._cmap, annotation_labels=lbl,
                                       annotation_lut_u8=self._annotation_lut_u8)
            self._cur_xyz=xyz; self._cur_idx=idx
            self._gpu_upload_zero_copy(xyz, col_u8); self._do_render()
        except Exception: pass

    def _upload_coarse(self, pc):
        if pc.octree is None or not pc.octree._lod_ready: return
        from render.colors import compute_colors_u8
        target = SPARSE_CAP if self._overview_mode == "sparse" else COARSE_TARGET_PTS
        xyz, idx = pc.octree.get_coarse_view(target)
        if len(xyz) == 0: return
        self._ensure_gpu_buffer(max(len(xyz), self._budget))
        lbl = (self.project.labels[idx]
               if self.project and self.project.labels is not None else None)
        col_u8 = compute_colors_u8(xyz, pc.get_attrs(idx), self._color_mode,
                                   self._cmap, annotation_labels=lbl,
                                   annotation_lut_u8=self._annotation_lut_u8)
        self._cur_xyz=xyz; self._cur_idx=idx
        self._gpu_upload_zero_copy(xyz, col_u8)
        self._fit_camera(pc); self._do_render()

    def _req_worker(self):
        if self._tile_mode: return
        pc = self.pc
        if pc is None: return

        if pc.octree is None or not pc.octree.ready: return
        try:
            rw = self._vtkw.GetRenderWindow()
            W, H = float(rw.GetSize()[0]), float(rw.GetSize()[1])
            if W==0 or H==0: return
            # En sparse mode el budget es fijo y pequeño
            budget = SPARSE_CAP if self._overview_mode == "sparse" else self._budget
            self._worker.request(
                pc, self.project, self._get_eye3d(),
                self._get_frustum_planes(W,H), W, H, budget,
                self._color_mode, self._cmap, self._get_fov(),
                False, None, None, self._annotation_lut, self._annotation_lut_u8)
        except Exception: pass

    def _on_coarse_ready(self, xyz, col_u8, idx, epoch):
        if self._tile_mode or self.pc is None or len(xyz)==0: return
        if self._interacting: return
        if epoch != self._worker.epoch: return
        if self._current_n > 0 and len(xyz) <= self._current_n: return
        self._cur_xyz=xyz; self._cur_idx=idx; self._cur_colors=None
        self._render_state = "REFINING"
        self._ensure_gpu_buffer(len(xyz))
        self._gpu_upload_zero_copy(xyz, col_u8)
        self._do_render()   # forzar render inmediato (no throttled)
        self.sig.render_info.emit(len(xyz), self.pc.n_points)

    def _on_batch_ready(self, xyz, col_u8, idx, epoch, is_done):
        if self._tile_mode or self.pc is None or len(xyz)==0: return
        if self._interacting: return
        if epoch != self._worker.epoch: return
        if self._current_n > 0 and len(xyz) <= self._current_n:
            if is_done: self._render_state = "DONE"
            return
        self._cur_xyz=xyz; self._cur_idx=idx; self._cur_colors=None
        if is_done: self._render_state = "DONE"
        self._ensure_gpu_buffer(len(xyz))
        self._gpu_upload_zero_copy(xyz, col_u8); self._request_render()
        self.sig.render_info.emit(len(xyz), self.pc.n_points)

    # ── Dispatch de eventos ───────────────────────────────────────────────────

    def _dispatch_press(self, pos):
        if self.active_tool is None: return
        class _E:
            button=1
            def __init__(self,p): self.pos=p
        self.active_tool.on_mouse_press(_E(pos))

    def _dispatch_move(self, pos):
        if self.active_tool is None: return
        class _E:
            buttons=(1,); is_dragging=True
            def __init__(self,p): self.pos=p
        self.active_tool.on_mouse_move(_E(pos))

    def _dispatch_release(self, pos):
        if self.active_tool is None: return
        class _E:
            button=1
            def __init__(self,p): self.pos=p
        self.active_tool.on_mouse_release(_E(pos))

    def _route_key_to_main(self, key):
        try:
            p = self.parent()
            while p:
                if hasattr(p,"_activate_tool"):
                    # Ctrl+Z / Ctrl+Shift+Z: undo/redo
                    if key == 'CTRL+Z':
                        label_store = getattr(p, '_label_store', None)
                        if label_store:
                            did = label_store.undo()
                            if did:
                                # self IS the AnnotationCanvas here
                                if (self.is_tile_mode and
                                        self._active_tile is not None and
                                        self._tile_indices is not None):
                                    self.enter_tile_mode(
                                        self._active_tile,
                                        self._tile_indices,
                                        fit_camera=False)
                                else:
                                    self.force_color_rebuild()
                                p._on_stats_changed()
                        return
                    if key == 'CTRL+SHIFT+Z':
                        label_store = getattr(p, '_label_store', None)
                        if label_store:
                            did = label_store.redo()
                            if did:
                                if (self.is_tile_mode and
                                        self._active_tile is not None and
                                        self._tile_indices is not None):
                                    self.enter_tile_mode(
                                        self._active_tile,
                                        self._tile_indices,
                                        fit_camera=False)
                                else:
                                    self.force_color_rebuild()
                                p._on_stats_changed()
                        return
                    if key == 'CTRL+S':
                        if hasattr(p, 'save_project'): p.save_project()
                        return
                    # Cambio de clase por tecla numérica (1-9)
                    if key.startswith('CLASS_') and hasattr(p, '_activate_class_by_slot'):
                        digit = int(key[6:])
                        slot  = 9 if digit == 0 else digit - 1  # 0→9, 1→0, ..., 9→8
                        p._activate_class_by_slot(slot)
                        return
                    # Vistas de cámara (enrutadas desde VTK)
                    if key == 'VIEW_TOP':
                        if hasattr(p, '_set_top_view'): p._set_top_view()
                        return
                    if key == 'VIEW_SIDE':
                        if hasattr(p, '_set_side_view'): p._set_side_view()
                        return
                    if key == 'VIEW_RESET':
                        if hasattr(p, '_reset_camera_view'): p._reset_camera_view()
                        return
                    # Tecla de herramienta
                    from annotation.tools import TOOL_BY_KEY
                    if key in TOOL_BY_KEY: p._activate_tool(TOOL_BY_KEY[key].name)
                    return
                p = p.parent()
        except Exception:
            import traceback; traceback.print_exc()

    # ── Fly mode (Unity-style WASD + mouse look) ─────────────────────────────

    def _fly_fix_clipping(self, obj, event) -> None:
        """
        Observer en el renderer: dispara después de cada render.
        Cuando estamos en fly mode, forzamos un rango de clip generoso
        para el SIGUIENTE render. Así la cámara nunca recorta puntos cercanos.
        VTK llama ResetCameraClippingRange() internamente durante el render;
        este observer lo sobreescribe inmediatamente después, de modo que
        el siguiente frame tiene el rango correcto.
        """
        if not self._fly_mode:
            return
        try:
            cam = self._ren.GetActiveCamera()
            # Near extremadamente pequeño, far astronómico → nunca clipa nada
            cam.SetClippingRange(0.001, 1.0e9)
        except Exception:
            pass

    def toggle_fly_mode(self) -> bool:
        """
        Toggle del modo de vuelo. Devuelve el nuevo estado.
        Atajo: tecla F (o Escape para salir).
        Controles:
          Click derecho + arrastrar  → mirar (yaw/pitch)
          W / S                      → adelante / atrás
          A / D                      → strafe izquierda / derecha
          Q / E                      → subir / bajar
          Shift                      → 4× velocidad
        """
        self._fly_mode = not self._fly_mode
        if self._fly_mode:
            self._fly_keys.clear()
            self._fly_rb_down = False
            # Capturar dirección de vista actual y guardarla en estado interno.
            # _fly_tick usará este vector para moverse, evitando recalcularlo
            # desde GetDirectionOfProjection() que puede tener drift si el
            # focal está cerca (causa el teleport).
            try:
                cam = self._ren.GetActiveCamera()
                fwd = np.array(cam.GetDirectionOfProjection(), np.float64)
                fn  = float(np.linalg.norm(fwd))
                self._fly_dir = fwd / fn if fn > 1e-9 else np.array([0,1,0],np.float64)
                # Focal a distancia segura para que los renders sean correctos
                pos = np.array(cam.GetPosition(), np.float64)
                cam.SetFocalPoint(*(pos + self._fly_dir * 50.0))
            except Exception:
                self._fly_dir = np.array([0, 1, 0], np.float64)
            self._fly_timer.start()
            self._vtkw.setCursor(Qt.CrossCursor)
        else:
            self._fly_timer.stop()
            self._fly_keys.clear()
            self._fly_rb_down = False
            self._vtkw.setCursor(Qt.ArrowCursor)
            # Restaurar el estado de la camara guardado al entrar al fly mode
            # para que el trackball no tenga un "salto" al retomar el control
            try:
                if hasattr(self, '_fly_dir') and self._fly_dir is not None:
                    cam  = self._ren.GetActiveCamera()
                    pos  = np.array(cam.GetPosition(), np.float64)
                    fwd  = self._fly_dir
                    # Poner focal a distancia razonable para el trackball
                    cam.SetFocalPoint(*(pos + fwd * 100.0))
                    self._ren.ResetCameraClippingRange()
            except Exception:
                pass
        self.sig.view_mode_changed.emit("fly" if self._fly_mode else self._overview_mode)
        return self._fly_mode

    def _fly_tick(self) -> None:
        """Timer a 60fps: aplica movimiento WASD mientras estén pulsadas."""
        if not self._fly_mode:
            return
        keys = self._fly_keys
        if not keys:
            return

        cam   = self._ren.GetActiveCamera()
        pos   = np.array(cam.GetPosition(), np.float64)
        # Usar dirección guardada (no GetDirectionOfProjection) para evitar drift/teleport
        fwd   = getattr(self, '_fly_dir', None)
        if fwd is None:
            fwd = np.array(cam.GetDirectionOfProjection(), np.float64)
            fn  = float(np.linalg.norm(fwd))
            fwd = fwd / fn if fn > 1e-9 else np.array([0,1,0], np.float64)
            self._fly_dir = fwd

        # Right perpendicular a fwd en el plano XY (sin depender de up_v para evitar roll)
        right = np.array([-fwd[1], fwd[0], 0.0], np.float64)
        rlen  = float(np.linalg.norm(right))
        if rlen < 1e-6:
            right = np.array([1.0, 0.0, 0.0], np.float64)
        else:
            right /= rlen

        has_shift = bool(keys & {'Shift_L', 'Shift_R', 'shift_l', 'shift_r'})
        speed = self._fly_speed * (4.0 if has_shift else 1.0)
        dt    = 0.016
        move  = np.zeros(3, np.float64)

        if keys & {'w', 'W'}:  move += fwd   * speed * dt
        if keys & {'s', 'S'}:  move -= fwd   * speed * dt
        if keys & {'a', 'A'}:  move += right * speed * dt  # A = izquierda
        if keys & {'d', 'D'}:  move -= right * speed * dt  # D = derecha
        if keys & {'q', 'Q'}:  move -= np.array([0,0,1], np.float64) * speed * dt
        if keys & {'e', 'E'}:  move += np.array([0,0,1], np.float64) * speed * dt

        if not np.any(move != 0):
            return

        new_pos = pos + move
        cam.SetPosition(*new_pos)
        cam.SetFocalPoint(*(new_pos + fwd * 50.0))
        self._request_render()

    def _fly_look(self, dx: int, dy: int) -> None:
        """
        Rota la dirección de vista (yaw + pitch) en respuesta al arrastre del ratón.
        dx, dy: delta de píxeles. Mantiene el roll en 0 forzando el up a Z.
        """
        if dx == 0 and dy == 0:
            return
        sens = self._fly_sens
        cam  = self._ren.GetActiveCamera()
        pos  = np.array(cam.GetPosition(),   np.float64)
        focal= np.array(cam.GetFocalPoint(), np.float64)
        view = focal - pos
        dist = float(np.linalg.norm(view))
        if dist < 1e-9:
            return
        view /= dist

        world_up = np.array([0.0, 0.0, 1.0])

        # ── Yaw: rotar alrededor del eje Z mundo ────────────────────────────
        yaw = math.radians(-dx * sens)
        c, s = math.cos(yaw), math.sin(yaw)
        vx = c*view[0] - s*view[1]
        vy = s*view[0] + c*view[1]
        view = np.array([vx, vy, view[2]])
        n = float(np.linalg.norm(view))
        if n > 1e-9: view /= n

        # ── Pitch: rotar alrededor del eje derecho de la cámara ─────────────
        right = np.cross(view, world_up)
        rn = float(np.linalg.norm(right))
        if rn < 1e-9:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right /= rn

        pitch = math.radians(-dy * sens)   # Y invertido: arrastrar arriba = mirar arriba
        cp, sp = math.cos(pitch), math.sin(pitch)
        # Rodrigues: rotar view alrededor de right
        view = view*cp + np.cross(right, view)*sp + right*np.dot(right, view)*(1-cp)
        n = float(np.linalg.norm(view))
        if n < 1e-9: return
        view /= n

        # Limitar pitch para no voltear (max 80°)
        pitch_cur = math.degrees(math.asin(max(-1.0, min(1.0, float(view[2])))))
        if abs(pitch_cur) > 80:
            # Cancelar el pitch de este frame
            view[2] = math.copysign(math.sin(math.radians(80)), view[2])
            n = float(np.linalg.norm(view[:2]))
            if n > 1e-9:
                scale = math.cos(math.radians(80)) / n
                view[0] *= scale; view[1] *= scale

        cam.SetFocalPoint(*(pos + view * 50.0))   # focal siempre a 50m

        # Guardar nueva dirección en estado interno para _fly_tick
        self._fly_dir = view.copy()

        # Recalcular up para evitar roll
        new_right = np.cross(view, world_up)
        nr = float(np.linalg.norm(new_right))
        if nr > 1e-9:
            new_right /= nr
            new_up = np.cross(new_right, view)
            nu = float(np.linalg.norm(new_up))
            if nu > 1e-9:
                cam.SetViewUp(*new_up / nu)

        self._request_render()

    # ── Brush cursor ─────────────────────────────────────────────────────────

    def _update_brush_cursor(self, screen_pos):
        if (self.active_tool is None or
                self.active_tool.__class__.__name__ not in ("BrushTool","RadiusTool","SphereSelectTool")):
            self._hide_brush_cursor(); return
        if self._cur_xyz is None or len(self._cur_xyz) == 0: return
        try:
            xyz=self._cur_xyz; n=len(xyz)
            step=max(1,n//8_000); sub=xyz[::step]
            sc=self.map_to_screen(sub)
            sx,sy=float(screen_pos[0]),float(screen_pos[1])
            d2=(sc[:,0]-sx)**2+(sc[:,1]-sy)**2
            ki=np.argpartition(d2,min(8,len(d2)-1))[:8]
            pos3d=sub[ki].mean(0).astype(np.float32)
            radius=getattr(self.active_tool,"radius_m",5.0)
            angles=np.linspace(0,2*np.pi,37,np.float32)
            pts=np.column_stack([pos3d[0]+radius*np.cos(angles),
                                 pos3d[1]+radius*np.sin(angles),
                                 np.full(37,float(pos3d[2]),np.float32)])
            color=(0.75,0.47,0.09,0.85)
            try:
                if self.project:
                    s=next((s for s in self.project.schema
                            if s.id==self.active_tool.active_class_id),None)
                    if s:
                        h=s.color.lstrip("#"); r,g,b=(int(h[i:i+2],16)/255 for i in (0,2,4))
                        color=(r,g,b,0.9)
            except Exception: pass
            self._brush_cursor.set_data(pts,color=color); self._brush_cursor.visible=True
            self._brush_center.set_data(pos=pos3d[None,:],face_color=color,size=10)
            self._brush_center.visible=True; self._do_render()
        except Exception: pass

    def _hide_brush_cursor(self):
        changed=self._brush_cursor.visible or self._brush_center.visible
        if self._brush_cursor._actor:
            self._brush_cursor._visible=False; self._brush_cursor._actor.SetVisibility(0)
        if self._brush_center._actor:
            self._brush_center._visible=False; self._brush_center._actor.SetVisibility(0)
        if changed: self._do_render()
