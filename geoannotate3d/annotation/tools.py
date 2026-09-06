"""
annotation/tools.py — Herramientas de anotación profesional v2.0

Herramientas implementadas:
  B  BrushTool        — Pincel 3D esférico continuo (Ctrl+arrastrar)
  L  PolygonTool      — Polígono de pantalla extruido (selecciona TODO en columna 3D)
  X  BoxSelectTool    — Rectángulo de pantalla extruido (más rápido que polígono)
  R  SphereSelectTool — Clic único para esfera 3D
  C  SliceTool        — Rango Z: selecciona capa horizontal de puntos
  G  FloodFillTool    — Relleno por región (BFS desde semilla, cKDTree)
  I  PickTool         — Inspección de punto
  M  MeasureTool      — Medición 3D/2D/Z

Principios de diseño:
  - TODAS las herramientas de selección trabajan sobre pc.xyz completo (no solo
    los puntos visibles). Esto es esencial: en tile mode tienes 5-30M pts completos.
  - Modo "borrar" (erase_mode=True) en todas las herramientas de selección:
    revierte puntos a clase 0 (sin etiquetar).
  - Modo "eliminar" (delete_mode=True) en todas las herramientas de selección:
    ELIMINA los puntos de la nube (no solo su clase) — para limpiar ruido.
    Manda sobre erase_mode. Ver DELETED_LABEL en label_store.py.
  - Filtro "solo sin etiquetar" (only_unlabeled): aplica solo a puntos vacíos.
  - PolygonTool y BoxSelectTool usan proyección matricial sobre TODOS los puntos,
    igual que el "Segment" de CloudCompare — el diferencial principal.
  - FloodFillTool usa BFS + cKDTree: captura objetos conectados en 1 clic.
"""
from __future__ import annotations

import math
import time
from typing import Optional, List, Tuple, TYPE_CHECKING

import numpy as np

from utils.spatial import sphere_query, sphere_query_octree

if TYPE_CHECKING:
    from render.canvas import AnnotationCanvas
    from annotation.label_store import LabelStore

# ── Constantes ────────────────────────────────────────────────────────────────
SNAP_K         = 6
SNAP_SCREEN_R  = 28
PROJ_CHUNK     = 2_000_000   # puntos por chunk en proyección masiva


# ─────────────────────────────────────────────────────────────────────────────
# Clases base
# ─────────────────────────────────────────────────────────────────────────────

class BaseTool:
    name:    str = "base"
    key:     str = ""
    icon:    str = ""          # nombre del icono en _ToolButton
    tooltip: str = ""

    def __init__(self):
        self.canvas:          Optional["AnnotationCanvas"] = None
        self.label_store:     Optional["LabelStore"]       = None
        self.active_class_id: int = 1
        self.erase_mode:      bool = False    # True → borra (clase 0)
        self.delete_mode:     bool = False    # True → ELIMINA el punto de la nube
                                              # (no una clase — ver DELETED_LABEL
                                              # en label_store.py). Manda sobre
                                              # erase_mode si ambos están activos.
        self.only_unlabeled:  bool = False    # True → solo puntos sin etiquetar

    def activate(self, canvas: "AnnotationCanvas") -> None:
        self.canvas = canvas

    def deactivate(self) -> None:
        self._clear_overlays()

    def on_mouse_press(self, event):   pass
    def on_mouse_move(self, event):    pass
    def on_mouse_release(self, event): pass
    def on_key_press(self, event):     pass

    # ── Utilidades comunes ────────────────────────────────────────────────────

    def _clear_overlays(self) -> None:
        c = self.canvas
        if c is None: return
        try:
            for attr in ("_mline", "_mline_live", "_p1_mk", "_snap_mk"):
                ov = getattr(c, attr, None)
                if ov: ov.visible = False
            c.update()
        except Exception:
            pass

    def _world_pos(self, screen_pos, fast: bool = True) -> Optional[np.ndarray]:
        """
        Posición 3D del punto más cercano al cursor.
        Para nubes grandes en overview (puntos dispersos), el cursor puede
        caer en zonas sin puntos → fallback al plano Z medio de la nube.
        """
        c = self.canvas
        if c is None or c._cur_xyz is None or len(c._cur_xyz) == 0:
            return self._screen_to_world_fallback(screen_pos)
        try:
            xyz  = c._cur_xyz
            n    = len(xyz)
            # Submuestreo más generoso para nubes grandes en overview
            step = max(1, n // 30_000) if fast else max(1, n // 200_000)
            sub  = xyz[::step]
            sc   = c.map_to_screen(sub)
            sx, sy = float(screen_pos[0]), float(screen_pos[1])
            d2   = (sc[:, 0] - sx) ** 2 + (sc[:, 1] - sy) ** 2
            min_d = float(d2.min())
            # Si el punto más cercano está muy lejos en pantalla (>200px),
            # la nube es demasiado sparse → proyectar al plano Z medio
            if min_d > 200.0 ** 2:
                fb = self._screen_to_world_fallback(screen_pos)
                if fb is not None:
                    return fb
            ki = np.argpartition(d2, min(SNAP_K, len(d2) - 1))[:SNAP_K]
            return sub[ki].mean(axis=0).astype(np.float32)
        except Exception:
            return self._screen_to_world_fallback(screen_pos)

    def _snap_to_point(self, screen_pos,
                       radius_px: float = SNAP_SCREEN_R) -> Optional[np.ndarray]:
        """Snap al punto real más cercano (submuestra para velocidad)."""
        c = self.canvas
        if c is None or c._cur_xyz is None:
            return None
        try:
            xyz  = c._cur_xyz
            n    = len(xyz)
            step = max(1, n // 200_000)
            sub  = xyz[::step]
            sc   = c.map_to_screen(sub)
            sx, sy = float(screen_pos[0]), float(screen_pos[1])
            d2 = (sc[:, 0] - sx) ** 2 + (sc[:, 1] - sy) ** 2
            bi = int(np.argmin(d2))
            if d2[bi] <= radius_px ** 2:
                return sub[bi].copy()
        except Exception:
            pass
        return None

    def _get_projection_matrix(self) -> Optional[np.ndarray]:
        """Devuelve MVP 4×4 float64 para proyectar puntos mundo → pantalla."""
        c = self.canvas
        if c is None: return None
        try:
            rw   = c._vtkw.GetRenderWindow()
            W, H = float(rw.GetSize()[0]), float(rw.GetSize()[1])
            if W == 0 or H == 0: return None
            return c._get_mvp(W, H), W, H
        except Exception:
            return None

    def _project_visible(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Proyecta SOLO los puntos actualmente visibles/en-tile → pantalla.
        Retorna (screen_xy (K,2) float32, global_indices (K,) int32, W, H).

        Usa canvas._cur_idx (el tile activo o la vista LOD actual).
        Esto es 10-50x más rápido que proyectar la nube entera porque:
          - Tile mode:   5-30M pts  → 0.3-1.5s
          - Sparse mode: 3M pts     → 0.1s
        Nunca se proyectan 100M+ pts para una sola operación de herramienta.
        """
        c  = self.canvas
        pc = c.pc
        rw = c._vtkw.GetRenderWindow()
        W, H = float(rw.GetSize()[0]), float(rw.GetSize()[1])
        MVP  = c._get_mvp(W, H)

        # Usar índices visibles actuales (tile o LOD)
        cur_idx = c._cur_idx
        if cur_idx is not None and len(cur_idx) > 0:
            xyz        = pc.xyz[cur_idx]
            global_idx = cur_idx
        else:
            # Fallback: muestra uniforme de máx 2M pts
            n    = len(pc.xyz)
            step = max(1, n // 2_000_000)
            global_idx = np.arange(0, n, step, np.int32)
            xyz  = pc.xyz[global_idx]

        n   = len(xyz)
        out = np.empty((n, 2), np.float32)
        for start in range(0, n, PROJ_CHUNK):
            end   = min(start + PROJ_CHUNK, n)
            chunk = xyz[start:end].astype(np.float64)
            ones  = np.ones(len(chunk), np.float64)
            xyzw  = np.column_stack([chunk, ones])
            clip  = (MVP @ xyzw.T).T
            wc    = np.where(np.abs(clip[:, 3:4]) < 1e-9, 1.0, clip[:, 3:4])
            ndc   = clip[:, :2] / wc
            out[start:end, 0] = (ndc[:, 0] * 0.5 + 0.5) * W
            out[start:end, 1] = (0.5 - ndc[:, 1] * 0.5) * H
        return out, global_idx.astype(np.int32), W, H

    def _project_all(self, xyz: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Legacy: proyecta xyz dado. Usar _project_visible() para herramientas."""
        c  = self.canvas
        rw = c._vtkw.GetRenderWindow()
        W, H = float(rw.GetSize()[0]), float(rw.GetSize()[1])
        MVP  = c._get_mvp(W, H)
        n    = len(xyz)
        out  = np.empty((n, 2), np.float32)
        for start in range(0, n, PROJ_CHUNK):
            end   = min(start + PROJ_CHUNK, n)
            chunk = xyz[start:end].astype(np.float64)
            ones  = np.ones(len(chunk), np.float64)
            xyzw  = np.column_stack([chunk, ones])
            clip  = (MVP @ xyzw.T).T
            wc    = np.where(np.abs(clip[:, 3:4]) < 1e-9, 1.0, clip[:, 3:4])
            ndc   = clip[:, :2] / wc
            out[start:end, 0] = (ndc[:, 0] * 0.5 + 0.5) * W
            out[start:end, 1] = (0.5 - ndc[:, 1] * 0.5) * H
        return out, W, H

    def _screen_to_world_fallback(self, screen_pos) -> "np.ndarray | None":
        """
        Proyecta la posición de pantalla al plano Z = media de la nube.
        Funciona en CUALQUIER punto de la pantalla, incluso sin puntos cerca.
        Siempre devuelve algo si hay una nube cargada.
        """
        c = self.canvas
        if c is None or c.pc is None:
            return None
        # Método 1: usar _screen_to_world_xy del canvas
        try:
            wx, wy = c._screen_to_world_xy(screen_pos)
            if wx is not None:
                b  = c.pc.bounds
                z  = float((b[0, 2] + b[1, 2]) * 0.5) if b is not None else 0.0
                return np.array([wx, wy, z], np.float32)
        except Exception:
            pass
        # Método 2: inversión de MVP
        try:
            rw   = c._vtkw.GetRenderWindow()
            W, H = float(rw.GetSize()[0]), float(rw.GetSize()[1])
            if W == 0 or H == 0: return None
            sx, sy = float(screen_pos[0]), float(screen_pos[1])
            inv_mvp = np.linalg.inv(c._get_mvp(W, H))
            ndc   = np.array([2*sx/W-1, 1-2*sy/H, 0.0, 1.0])
            world = inv_mvp @ ndc
            w4    = world[3]
            if abs(w4) < 1e-12: w4 = 1.0
            world = world[:3] / w4
            b = c.pc.bounds
            if b is not None:
                z = float(np.clip(world[2], b[0,2], b[1,2]))
            else:
                z = float(world[2])
            return np.array([float(world[0]), float(world[1]), z], np.float32)
        except Exception:
            pass
        # Método 3: usar _cur_xyz media como punto de referencia
        try:
            if c._cur_xyz is not None and len(c._cur_xyz) > 0:
                return c._cur_xyz.mean(axis=0).astype(np.float32)
        except Exception:
            pass
        return None

    def _octree_sphere_query(self, center: np.ndarray,
                             radius_m: float) -> Optional[np.ndarray]:
        """
        Búsqueda esférica RÁPIDA usando el índice espacial global de la nube
        (grid en C / cKDTree, construido una vez en pc.octree).

        Esto evita escanear millones de puntos del tile activo en cada trazo
        de pincel: en vez de O(N_tile) con N_tile = 5-30M, es O(pocos cientos)
        gracias al grid hash de core/_fastcore.c (mismo índice que ya usa el
        render para LOD y que SphereSelectTool usaba solo como fallback).

        Retorna índices GLOBALES en pc.xyz (int64), o None si el octree
        todavía no está listo (justo después de cargar la nube) — en ese
        caso el llamador debe usar un fallback más lento sobre _cur_xyz.
        """
        c = self.canvas
        if c is None or c.pc is None:
            return None
        octree = getattr(c.pc, 'octree', None)
        if octree is None or not getattr(octree, 'ready', False):
            return None
        try:
            idx = sphere_query_octree(octree, center.astype(np.float32), float(radius_m))
            if idx is None:
                return np.zeros(0, np.int64)
            return np.asarray(idx, dtype=np.int64)
        except Exception:
            return None

    def _sphere_in_visible(self, center: np.ndarray, radius_m: float) -> np.ndarray:
        """
        Búsqueda esférica sobre los puntos visibles.
        Si _cur_idx es None (vista inicial), usa pc.xyz directamente con stride.
        Retorna índices GLOBALES en pc.xyz.
        """
        c = self.canvas
        if c is None or c._cur_xyz is None or len(c._cur_xyz) == 0:
            return np.zeros(0, np.int32)
        xyz  = c._cur_xyz
        r2   = float(radius_m) ** 2
        diff = xyz - center.astype(xyz.dtype)
        bbox_mask = ((np.abs(diff[:, 0]) <= radius_m) &
                     (np.abs(diff[:, 1]) <= radius_m) &
                     (np.abs(diff[:, 2]) <= radius_m))
        cands = np.where(bbox_mask)[0]
        if len(cands) == 0:
            return np.zeros(0, np.int32)
        d = diff[cands]
        inside = (d[:, 0]**2 + d[:, 1]**2 + d[:, 2]**2) <= r2
        local_idx = cands[inside]
        if len(local_idx) == 0:
            return np.zeros(0, np.int32)
        # Mapear a índices globales
        cur_idx = c._cur_idx
        if cur_idx is not None and len(cur_idx) >= len(xyz):
            return cur_idx[local_idx].astype(np.int32)
        elif cur_idx is not None and len(cur_idx) > 0:
            # cur_idx puede ser más pequeño si se actualizó — usar los que caben
            valid = local_idx[local_idx < len(cur_idx)]
            if len(valid) == 0:
                return local_idx.astype(np.int64)  # fallback: usar índices locales
            return cur_idx[valid].astype(np.int32)
        else:
            # Sin cur_idx: los índices locales son los globales (nube pequeña sin LOD)
            return local_idx.astype(np.int64)

    def _apply_selection(self, indices: np.ndarray) -> None:
        """Aplica la clase activa (o borra) a los índices seleccionados."""
        if self.label_store is None or len(indices) == 0:
            return
        indices = np.asarray(indices, dtype=np.int32)

        # Filtrar índices fuera de rango
        lbl = self.label_store._labels
        if lbl is None:
            print("[tools] WARN: label_store._labels es None — attach() no fue llamado")
            return
        if len(lbl) == 0:
            print("[tools] WARN: label_store._labels está vacío")
            return
        n_before = len(indices)
        indices = indices[(indices >= 0) & (indices < len(lbl))]
        if len(indices) == 0:
            print(f"[tools] WARN: todos los índices ({n_before}) fuera de rango "
                  f"(labels.size={len(lbl)})")
            return

        # Filtro: solo puntos sin etiquetar
        if self.only_unlabeled and self.canvas is not None:
            lbls = self.canvas.project.labels
            if lbls is not None and len(lbls) > 0:
                valid_mask = indices < len(lbls)
                indices = indices[valid_mask]
                if len(indices) == 0:
                    return
                mask    = lbls[indices] == 0
                indices = indices[mask]
        if len(indices) == 0:
            return
        if self.delete_mode:
            from annotation.label_store import DELETED_LABEL
            cid = DELETED_LABEL
        else:
            cid = 0 if self.erase_mode else self.active_class_id
        self.label_store.annotate(indices, cid)
        if self.canvas:
            self.canvas.refresh_colors(indices, cid)


# ─────────────────────────────────────────────────────────────────────────────
# 1. BrushTool — Pincel 3D esférico
# ─────────────────────────────────────────────────────────────────────────────

class BrushTool(BaseTool):
    """
    Pincel 3D que sigue la superficie local.

    En vez de una esfera perfecta, selecciona los puntos dentro de un disco
    orientado según la normal local estimada — así pinta solo los puntos
    de la superficie bajo el cursor sin "atravesar" estructuras delgadas
    (vallas, fachadas, bordes de tejados).

    Ctrl+arrastrar pinta de forma continua.
    Modo esfera (fallback): si no hay suficientes vecinos para estimar la normal,
    usa la esfera completa igual que antes.
    """
    name    = "Pincel"
    key     = "B"
    icon    = "brush"
    tooltip = "Ctrl+arrastrar · pinta siguiendo la superficie"

    def __init__(self):
        super().__init__()
        self.radius_m:    float = 5.0
        self.thickness:   float = 0.4   # fracción del radio para el grosor del disco
        self.overlap_pct: float = 0.8
        self._painting:   bool  = False
        self._last_pos:   Optional[np.ndarray] = None

    def on_mouse_press(self, event):
        if event.button != 1: return
        self._painting = True; self._last_pos = None
        p = self._world_pos(event.pos)
        if p is not None:
            self._paint_at(p); self._last_pos = p

    def on_mouse_move(self, event):
        if not self._painting: return
        p = self._world_pos(event.pos)
        if p is None: return
        if self._last_pos is not None:
            if float(np.linalg.norm(p - self._last_pos)) < self.radius_m * (1.0 - self.overlap_pct):
                return
        self._paint_at(p); self._last_pos = p

    def on_mouse_release(self, event):
        if event.button == 1:
            self._painting = False; self._last_pos = None

    def _paint_at(self, center: np.ndarray) -> None:
        """
        Pinta puntos dentro de la esfera (o disco si hay normal).

        Vía rápida (nube ya indexada): usa el grid espacial en C de
        pc.octree (_octree_sphere_query) — examina unos cientos de puntos
        sin importar si el tile activo tiene 5M o 30M puntos, en vez de
        escanear el tile entero en cada trazo. Esto es lo que evita el
        freeze al pintar con Pincel sobre nubes densas.

        Fallback (nube recién cargada, octree aún construyéndose): escanea
        los puntos actualmente cargados en el canvas (_cur_xyz/_cur_idx),
        que YA son el tile completo cuando estamos en tile mode — no hace
        falta volver a indexar pc.xyz con los índices del tile.
        """
        c = self.canvas
        if c is None or c.pc is None:
            return

        r  = self.radius_m
        r2 = r * r

        # ── Vía rápida: índice espacial global (grid en C / cKDTree) ───────
        fast_idx = self._octree_sphere_query(center, r)

        if fast_idx is not None:
            sphere_idx = fast_idx
            if len(sphere_idx) == 0:
                return
            cand_xyz = c.pc.xyz[sphere_idx]
        else:
            # ── Fallback: escanear los puntos ya cargados en el canvas ──────
            xyz = c._cur_xyz
            if xyz is None or len(xyz) == 0:
                return
            global_ref = c._cur_idx
            diff  = xyz - center.astype(xyz.dtype)
            bbox  = ((np.abs(diff[:,0]) <= r) &
                     (np.abs(diff[:,1]) <= r) &
                     (np.abs(diff[:,2]) <= r))
            cands_local = np.where(bbox)[0]
            if len(cands_local) == 0:
                return
            d      = diff[cands_local]
            inside = (d[:,0]**2 + d[:,1]**2 + d[:,2]**2) <= r2
            sphere_local = cands_local[inside]
            if len(sphere_local) == 0:
                return
            cand_xyz   = xyz[sphere_local]
            sphere_idx = (global_ref[sphere_local].astype(np.int64)
                          if global_ref is not None else sphere_local.astype(np.int64))

        normal = self._estimate_normal(cand_xyz, center)
        if normal is None:
            self._apply_selection(sphere_idx); return

        # Filtrar por disco (grosor a lo largo de la normal local)
        thickness_m = r * self.thickness
        proj      = np.abs(np.dot(cand_xyz - center.astype(np.float32), normal))
        disk_mask = proj <= thickness_m
        disk_idx  = sphere_idx[disk_mask]
        if len(disk_idx) == 0:
            disk_idx = sphere_idx

        self._apply_selection(disk_idx)

    @staticmethod
    def _estimate_normal(pts: np.ndarray,
                         center: np.ndarray) -> Optional[np.ndarray]:
        """
        Estima la normal de la superficie local usando PCA (SVD).
        La normal es el vector singular más pequeño (el eje de menor varianza).
        Retorna el vector normalizado float32, o None si hay muy pocos puntos.
        """
        if len(pts) < 6:
            return None
        try:
            # Usar hasta 2000 pts más cercanos para velocidad
            if len(pts) > 2000:
                idx = np.random.choice(len(pts), 2000, replace=False)
                pts = pts[idx]
            centered = pts - pts.mean(axis=0)
            # SVD económica
            _, _, Vt = np.linalg.svd(centered.astype(np.float64), full_matrices=False)
            normal = Vt[-1].astype(np.float32)   # último vector = menor varianza
            n = float(np.linalg.norm(normal))
            if n < 1e-9:
                return None
            return normal / n
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. PolygonTool — Polígono extruido (CloudCompare "Segment")
# ─────────────────────────────────────────────────────────────────────────────

class PolygonTool(BaseTool):
    """
    Herramienta estrella: dibuja un polígono en pantalla y selecciona TODOS
    los puntos que proyectan dentro de él, a cualquier profundidad.

    Flujo:
      Ctrl+clic → añade vértice
      Ctrl+clic cerca del primer vértice (radio 20px) → cierra y aplica
      Enter → cierra y aplica
      Escape → cancela
      Z → invertir selección (selecciona los de fuera)
    """
    name    = "Polígono"
    key     = "L"
    icon    = "polygon"
    tooltip = "Ctrl+clic para añadir vértices · Enter o clic inicio para cerrar"

    CLOSE_RADIUS_PX = 22

    def __init__(self):
        super().__init__()
        self._pts:     List[np.ndarray] = []    # puntos 3D del polígono
        self._drawing: bool = False
        self._proj:    Optional[callable] = None
        self._invert_selection: bool = False

    def activate(self, canvas):
        super().activate(canvas)
        self._reset()

    def deactivate(self):
        self._reset()
        super().deactivate()

    def _reset(self):
        self._pts    = []
        self._drawing = False
        self._proj   = None
        if self.canvas:
            self._clear_overlays()

    def on_mouse_press(self, event):
        if event.button != 1 or self.canvas is None:
            return
        # Prioridad 1: snap a punto 3D muy cercano (radio pequeño)
        pt = self._snap_to_point(event.pos, radius_px=20)
        if pt is None:
            # Prioridad 2: proyección directa al plano Z medio de la nube.
            # Funciona en CUALQUIER zona de la pantalla, dentro o fuera de la nube.
            pt = self._screen_to_world_fallback(event.pos)
        if pt is None:
            return   # solo falla si no hay nube cargada

        # Primer punto
        if not self._drawing:
            self._drawing = True
            self._proj = self.canvas.get_frozen_projector()
            self._pts  = [pt]
            self._update_overlay()
            return

        # ¿Cerrar polígono? (clic cerca del primer punto en pantalla)
        sc_first = self._proj(np.array(self._pts[:1], np.float32))
        sc_cur   = self._proj(pt[None, :])
        if sc_first.shape[0] > 0 and sc_cur.shape[0] > 0:
            d = math.hypot(float(sc_cur[0, 0] - sc_first[0, 0]),
                           float(sc_cur[0, 1] - sc_first[0, 1]))
            if d <= self.CLOSE_RADIUS_PX and len(self._pts) >= 3:
                self._apply_and_reset()
                return

        self._pts.append(pt)
        self._update_overlay()

    def on_mouse_move(self, event):
        if not self._drawing or not self._pts or self.canvas is None:
            return
        # Snap solo si hay punto muy cercano, si no proyectar directamente
        p = self._snap_to_point(event.pos, radius_px=20)
        if p is None:
            p = self._screen_to_world_fallback(event.pos)
        if p is None: return
        seg = np.array([self._pts[-1], p], np.float32)
        self.canvas._mline_live.set_data(seg, color=(0.2, 1.0, 0.35, 0.75), width=2.0)
        self.canvas._mline_live.visible = True
        self.canvas.update()

    def on_key_press(self, event):
        key = event.key
        if key in ("Return", "Enter"):
            if len(self._pts) >= 3:
                self._apply_and_reset()
        elif key == "Escape":
            self._reset()
        elif key in ("z", "Z"):
            self._invert_selection = not self._invert_selection

    def _update_overlay(self):
        if len(self._pts) < 2 or self.canvas is None: return
        poly = np.array(self._pts, np.float32)
        self.canvas._mline.set_data(poly, color=(0.2, 1.0, 0.35, 1.0), width=2.8)
        self.canvas._mline.visible = True
        self.canvas._p1_mk.set_data(pos=self._pts[0][None, :],
                                     face_color=(1.0, 0.35, 0.1, 1.0), size=12)
        self.canvas._p1_mk.visible = True
        self.canvas.update()

    def _apply_and_reset(self):
        if len(self._pts) < 3 or self.canvas is None: return
        c  = self.canvas
        pc = c.pc
        if pc is None or self.label_store is None: return

        # Proyectar solo puntos visibles/tile (10-50x más rápido que nube completa)
        try:
            sc, global_idx, W, H = self._project_visible()
        except Exception:
            self._reset(); return

        # Polígono 2D en pantalla desde los vértices 3D congelados
        poly_pts = np.array(self._pts, np.float32)
        proj = self._proj or c.get_frozen_projector()
        sc_poly = proj(poly_pts)[:, :2].astype(np.float64)

        try:
            from matplotlib.path import Path as MplPath
            mpl_poly = MplPath(sc_poly)
        except ImportError:
            mpl_poly = None

        n      = len(sc)
        inside = np.zeros(n, bool)
        chunk  = 1_000_000
        for start in range(0, n, chunk):
            end   = min(start + chunk, n)
            pts_c = sc[start:end].astype(np.float64)
            if mpl_poly is not None:
                inside[start:end] = mpl_poly.contains_points(pts_c)
            else:
                inside[start:end] = self._ray_cast_poly(pts_c, sc_poly)

        if self._invert_selection:
            inside = ~inside

        sel = global_idx[inside]
        if len(sel) > 0:
            self._apply_selection(sel)

        self._reset()

    @staticmethod
    def _ray_cast_poly(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
        """Fallback point-in-polygon O(N·V) vectorizado."""
        n  = len(pts)
        nv = len(poly)
        inside = np.zeros(n, bool)
        px, py = pts[:, 0], pts[:, 1]
        for i in range(nv):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % nv]
            cond = ((y1 <= py) & (py < y2)) | ((y2 <= py) & (py < y1))
            x_int = (x2 - x1) * (py - y1) / (y2 - y1 + 1e-9) + x1
            inside ^= cond & (px < x_int)
        return inside


# ─────────────────────────────────────────────────────────────────────────────
# 3. BoxSelectTool — Rectángulo de pantalla extruido
# ─────────────────────────────────────────────────────────────────────────────

class BoxSelectTool(BaseTool):
    """
    Ctrl+arrastrar → rectángulo de pantalla.
    Selecciona TODOS los puntos que proyectan dentro del rectángulo (cualquier Z).
    Más rápido que el polígono para selecciones rectangulares.
    """
    name    = "Caja"
    key     = "X"
    icon    = "box"
    tooltip = "Ctrl+arrastrar para rectángulo de selección · todo a cualquier profundidad"

    def __init__(self):
        super().__init__()
        self._p1_screen: Optional[Tuple[float, float]] = None
        self._dragging:  bool = False

    def on_mouse_press(self, event):
        if event.button != 1: return
        pos = event.pos
        self._p1_screen = (float(pos[0]), float(pos[1]))
        self._dragging  = True

    def on_mouse_move(self, event):
        if not self._dragging or self.canvas is None: return
        pos = event.pos
        x1, y1 = self._p1_screen
        x2, y2 = float(pos[0]), float(pos[1])
        xmn, xmx = min(x1, x2), max(x1, x2)
        ymn, ymx = min(y1, y2), max(y1, y2)
        # Dibujar rectángulo 2D en pantalla como polígono 3D aproximado
        if self.canvas._cur_xyz is not None and len(self.canvas._cur_xyz) > 0:
            # Mostrar rectángulo como overlay (en 3D usamos puntos de pantalla proyectados)
            try:
                inv_mvp = np.linalg.inv(self.canvas._get_mvp(
                    *[float(s) for s in self.canvas._vtkw.GetRenderWindow().GetSize()]))
                def unproj(sx, sy):
                    rw = self.canvas._vtkw.GetRenderWindow()
                    W, H = rw.GetSize()
                    ndc = np.array([2*sx/W-1, 1-2*sy/H, 0.0, 1.0])
                    p   = inv_mvp @ ndc
                    return p[:3] / p[3]
                corners = np.array([
                    unproj(xmn, ymn), unproj(xmx, ymn),
                    unproj(xmx, ymx), unproj(xmn, ymx),
                    unproj(xmn, ymn),
                ], np.float32)
                self.canvas._mline.set_data(corners, color=(1.0, 0.7, 0.1, 0.85))
                self.canvas._mline.visible = True
                self.canvas.update()
            except Exception:
                pass

    def on_mouse_release(self, event):
        if event.button != 1 or not self._dragging:
            return
        self._dragging = False
        if self.canvas is None or self._p1_screen is None:
            return
        pos = event.pos
        x1, y1 = self._p1_screen
        x2, y2 = float(pos[0]), float(pos[1])
        xmn, xmx = min(x1, x2), max(x1, x2)
        ymn, ymx = min(y1, y2), max(y1, y2)
        if abs(xmx - xmn) < 3 or abs(ymx - ymn) < 3:
            self._clear_overlays(); return

        c  = self.canvas
        pc = c.pc
        if pc is None or self.label_store is None:
            self._clear_overlays(); return

        try:
            sc, global_idx, W, H = self._project_visible()
            sx, sy = sc[:, 0], sc[:, 1]
            inside = ((sx >= xmn) & (sx <= xmx) &
                      (sy >= ymn) & (sy <= ymx))
            sel = global_idx[inside]
            if len(sel) > 0:
                self._apply_selection(sel)
        except Exception:
            pass

        self._clear_overlays()
        self._p1_screen = None

    def on_key_press(self, event):
        if event.key == "Escape":
            self._dragging = False
            self._p1_screen = None
            self._clear_overlays()


# ─────────────────────────────────────────────────────────────────────────────
# 4. SphereSelectTool — Esfera única por clic
# ─────────────────────────────────────────────────────────────────────────────

class SphereSelectTool(BaseTool):
    """
    ESFERA: un solo Ctrl+clic selecciona todos los puntos dentro de la esfera.
    No hay que arrastrar. Ideal para seleccionar objetos concretos de un toque.

    Diferencia vs Pincel:
      - Pincel:  Ctrl+ARRASTRAR → pinta una trayectoria continua
      - Esfera:  Ctrl+CLIC      → una bola completa en un punto, instantáneo
    """
    name    = "Esfera"
    # "R" colisionaba con el atajo de resetear cámara en main_window.py
    # (revisado antes que el mapa de herramientas) — nunca se activaba.
    key     = "H"
    icon    = "sphere"
    tooltip = "Ctrl+clic para seleccionar esfera 3D"

    def __init__(self):
        super().__init__()
        self.radius_m: float = 3.0

    def on_mouse_press(self, event):
        if event.button != 1: return
        p = self._world_pos(event.pos)
        if p is None: return
        # Vía rápida: grid espacial en C sobre pc.octree (cientos de puntos
        # examinados en vez de escanear el tile/vista visible entera).
        idx = self._octree_sphere_query(p, self.radius_m)
        if idx is None:
            # Octree aún no listo (nube recién cargada) — usar lo visible
            idx = self._sphere_in_visible(p, self.radius_m)
        if len(idx) > 0:
            self._apply_selection(idx)


# ─────────────────────────────────────────────────────────────────────────────
# 5. SliceTool — Rango de altura Z
# ─────────────────────────────────────────────────────────────────────────────

class SliceTool(BaseTool):
    """
    Selecciona puntos en un rango Z [z_min, z_max].

    Flujo:
      Ctrl+clic → establece z_min en la posición del cursor
      Ctrl+arrastrar arriba/abajo → ajusta z_max
      Soltar → aplica
      T → alternar: etiquetar por encima / por debajo / entre

    Modos:
      "between" → entre z_min y z_max (default)
      "above"   → encima de z_min
      "below"   → debajo de z_min
    """
    name    = "Corte Z"
    key     = "C"
    icon    = "slice"
    tooltip = "Ctrl+clic → base Z · arrastrar → altura · soltar → aplica"

    def __init__(self):
        super().__init__()
        self.z_min:  float = 0.0
        self.z_max:  float = 5.0
        self.mode:   str   = "between"   # "between" | "above" | "below"
        self._setting: bool = False
        self._y_start: float = 0.0
        self._z_start: float = 0.0

    def on_mouse_press(self, event):
        if event.button != 1: return
        p = self._world_pos(event.pos)
        if p is None: return
        self.z_min   = float(p[2])
        self.z_max   = float(p[2]) + 2.0
        self._setting = True
        self._y_start = float(event.pos[1])
        self._z_start = self.z_min
        self._draw_planes()

    def on_mouse_move(self, event):
        if not self._setting: return
        dy = float(event.pos[1]) - self._y_start
        # Escala: 1px = 0.05m (ajustable con shift)
        scale = 0.05
        self.z_max = self._z_start + max(0.01, -dy * scale)
        self._draw_planes()

    def on_mouse_release(self, event):
        if event.button != 1 or not self._setting: return
        self._setting = False
        self._apply_slice()
        self._clear_overlays()

    def on_key_press(self, event):
        key = event.key
        if key in ("t", "T"):
            modes = ["between", "above", "below"]
            idx   = modes.index(self.mode)
            self.mode = modes[(idx + 1) % 3]
        elif key == "Escape":
            self._setting = False
            self._clear_overlays()

    def _draw_planes(self):
        c = self.canvas
        if c is None or c.pc is None: return
        try:
            b = c.pc.bounds
            xmn, xmx = float(b[0, 0]), float(b[1, 0])
            ymn, ymx = float(b[0, 1]), float(b[1, 1])
            # Cuadrilátero en z_min y en z_max
            pts_low  = np.array([[xmn,ymn,self.z_min],[xmx,ymn,self.z_min],
                                  [xmx,ymx,self.z_min],[xmn,ymx,self.z_min],
                                  [xmn,ymn,self.z_min]], np.float32)
            pts_high = np.array([[xmn,ymn,self.z_max],[xmx,ymn,self.z_max],
                                  [xmx,ymx,self.z_max],[xmn,ymx,self.z_max],
                                  [xmn,ymn,self.z_max]], np.float32)
            combined = np.vstack([pts_low, pts_high])
            c._mline.set_data(combined, color=(0.9, 0.6, 0.1, 0.7))
            c._mline.visible = True
            c.update()
        except Exception:
            pass

    def _apply_slice(self):
        c = self.canvas
        if c is None or c.pc is None or self.label_store is None: return
        # Trabajar sobre puntos visibles/tile para velocidad
        cur_idx = c._cur_idx
        if cur_idx is not None and len(cur_idx) > 0:
            z = c._cur_xyz[:, 2]
        else:
            z = c.pc.xyz[:, 2]
            cur_idx = np.arange(len(z), dtype=np.int32)
        if self.mode == "between":
            mask = (z >= self.z_min) & (z <= self.z_max)
        elif self.mode == "above":
            mask = z >= self.z_min
        else:
            mask = z <= self.z_min
        sel = cur_idx[np.where(mask)[0]]
        if len(sel) > 0:
            self._apply_selection(sel.astype(np.int32))


# ─────────────────────────────────────────────────────────────────────────────
# 6. FloodFillTool — Relleno por región (BFS + cKDTree)
# ─────────────────────────────────────────────────────────────────────────────

class FloodFillTool(BaseTool):
    """
    Rellena una región conectada espacialmente desde un punto semilla.

    Algoritmo BFS + cKDTree:
      1. Click → punto semilla
      2. Construir cKDTree en los puntos del tile (o muestra de 1M)
      3. BFS: expandir desde semilla buscando vecinos en radio step_m
      4. Cada vecino encontrado se añade a la frontera
      5. Parar cuando: frontera vacía, max_points alcanzado, timeout

    Opciones:
      step_m:     radio de búsqueda en cada paso (0.1–10m)
      max_points: límite de puntos seleccionados
      z_tol:      máx. diferencia de altura permitida para expandir (0=sin límite)
    """
    name    = "Relleno"
    key     = "G"
    icon    = "fill"
    tooltip = "Ctrl+clic → rellena región conectada desde semilla (BFS)"

    def __init__(self):
        super().__init__()
        self.step_m:    float = 0.5
        self.max_pts:   int   = 500_000
        self.z_tol:     float = 0.0     # 0 = sin límite de altura
        self._kdtree          = None
        self._xyz_ref         = None

    def activate(self, canvas):
        super().activate(canvas)
        self._kdtree  = None
        self._xyz_ref = None

    def on_mouse_press(self, event):
        if event.button != 1: return
        seed = self._world_pos(event.pos, fast=False)
        if seed is None: return
        c  = self.canvas
        if c is None or c.pc is None or self.label_store is None: return
        self._ensure_kdtree(c)
        if self._kdtree is None: return
        idx = self._bfs(seed)
        if len(idx) > 0:
            self._apply_selection(idx)

    def _ensure_kdtree(self, c) -> None:
        """Construye cKDTree sobre _cur_xyz (tile activo). Mucho más rápido."""
        cur_xyz = c._cur_xyz
        if cur_xyz is None or len(cur_xyz) == 0:
            self._kdtree = None; return
        # Reconstruir solo si los puntos visibles cambiaron
        if self._xyz_ref is cur_xyz:
            return
        try:
            from scipy.spatial import cKDTree
            # _cur_xyz ya es el tile (5-30M pts) — construir KDTree directo
            # Para >5M pts, submuestra al 20% para velocidad de construcción
            n    = len(cur_xyz)
            step = max(1, n // 3_000_000)
            sub  = cur_xyz[::step]
            self._kdtree   = cKDTree(sub)
            self._xyz_ref  = cur_xyz
            self._step_idx = step
            # Guardamos _cur_idx para mapear a índices globales
            self._cur_idx_ref = c._cur_idx
        except Exception:
            self._kdtree = None

    def _bfs(self, seed: np.ndarray) -> np.ndarray:
        """BFS desde seed, devuelve índices globales en pc.xyz."""
        kd   = self._kdtree
        step = getattr(self, '_step_idx', 1)
        xyz  = self._xyz_ref  # esto es _cur_xyz del tile
        r    = self.step_m
        z_t  = self.z_tol
        maxp = self.max_pts

        # Índice de la semilla en el árbol subsampled
        _, seed_idx = kd.query(seed.astype(np.float64))

        visited  = set()
        frontier = [seed_idx]
        visited.add(seed_idx)

        while frontier and len(visited) < maxp:
            batch  = frontier[:512]
            frontier = frontier[512:]
            pts    = np.array([kd.data[i] for i in batch], np.float64)
            nbrs   = kd.query_ball_point(pts, r)
            for i_list in nbrs:
                for ni in i_list:
                    if ni in visited: continue
                    if z_t > 0:
                        # Filtrar por diferencia de altura
                        if abs(float(kd.data[ni][2]) - float(seed[2])) > z_t:
                            continue
                    visited.add(ni)
                    frontier.append(ni)
                if len(visited) >= maxp:
                    break

        # Mapear índices subsampled → índices en _cur_xyz, luego a globales
        local_sampled = np.array(sorted(visited), np.int32)
        local_in_cur  = (local_sampled * step).clip(0, len(xyz) - 1)
        # Mapear a índices globales via _cur_idx_ref
        cur_idx_ref = getattr(self, '_cur_idx_ref', None)
        if cur_idx_ref is not None and len(cur_idx_ref) > len(local_in_cur):
            global_idx = cur_idx_ref[local_in_cur]
        else:
            global_idx = local_in_cur
        return global_idx.astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# 7. PickTool — Inspección de punto
# ─────────────────────────────────────────────────────────────────────────────

class PickTool(BaseTool):
    """Inspecciona coordenadas y clase del punto bajo el cursor."""
    name    = "Pick"
    key     = "I"
    icon    = "pick"
    tooltip = "Ctrl+clic para inspeccionar punto"

    def on_mouse_press(self, event):
        if event.button != 1: return
        c = self.canvas
        if c is None or c.pc is None: return
        p = self._world_pos(event.pos, fast=False)
        if p is None: return
        abs_xyz = p.astype(np.float64) + c.pc.offset
        c.sig.point_picked.emit(float(p[0]), float(p[1]), float(p[2]))
        c.sig.point_picked_abs.emit(
            float(abs_xyz[0]), float(abs_xyz[1]), float(abs_xyz[2]))


# ─────────────────────────────────────────────────────────────────────────────
# 8. MeasureTool — Medición 3D / horizontal / dZ
# ─────────────────────────────────────────────────────────────────────────────

class MeasureTool(BaseTool):
    """
    Mide distancia 3D, distancia horizontal y diferencia de altura.
    Clic A → clic B → muestra resultado.
    """
    name    = "Medir"
    key     = "M"
    icon    = "ruler"
    tooltip = "Ctrl+clic A · clic B para medir distancia"

    def __init__(self):
        super().__init__()
        self._p1: Optional[np.ndarray] = None

    def deactivate(self):
        self._p1 = None
        super().deactivate()

    def on_mouse_press(self, event):
        if event.button != 1: return
        c = self.canvas
        if c is None: return
        p = self._snap_to_point(event.pos)
        if p is None:
            p = self._world_pos(event.pos, fast=False)
        if p is None:
            p = self._screen_to_world_fallback(event.pos)
        if p is None: return

        if self._p1 is None:
            self._p1 = p
            c._p1_mk.set_data(pos=p[None, :], face_color=(1.0, 0.45, 0.1, 1.0), size=12)
            c._p1_mk.visible = True
            c.update()
        else:
            d3d = float(np.linalg.norm(p - self._p1))
            dh  = float(np.linalg.norm((p - self._p1)[:2]))
            dz  = float(abs(p[2] - self._p1[2]))
            c.sig.measure_segment.emit(d3d, dh, dz)
            seg = np.array([self._p1, p], np.float32)
            c._mline.set_data(seg, color=(1.0, 0.9, 0.2, 1.0))
            c._mline.visible = True
            c.update()
            self._p1 = None

    def on_mouse_move(self, event):
        c = self.canvas
        if c is None or self._p1 is None: return
        p = self._snap_to_point(event.pos)
        if p is None:
            p = self._world_pos(event.pos)
        if p is None:
            p = self._screen_to_world_fallback(event.pos)
        if p is None: return
        seg = np.array([self._p1, p], np.float32)
        c._mline_live.set_data(seg, color=(1.0, 0.9, 0.2, 0.4))
        c._mline_live.visible = True
        c.update()

    def on_key_press(self, event):
        if event.key == "Escape":
            self._p1 = None
            self._clear_overlays()



class DiscTool(BaseTool):
    """Disco 3D orientado a la superficie. Ctrl+clic coloca el disco."""
    name    = "Disco"
    # "V" colisionaba con el atajo de vista cenital en main_window.py
    # (keyPressEvent revisa V/Y/R para cámara ANTES que el mapa de
    # herramientas), así que esta tecla nunca llegaba a activarse — bug
    # real encontrado al construir la referencia de atajos in-app.
    key     = "D"
    icon    = "disc"
    tooltip = "Ctrl+clic -> disco 3D inclinado segun la superficie"

    def __init__(self):
        super().__init__()
        self.radius_m:  float = 5.0
        self.thickness: float = 0.3

    def on_mouse_press(self, event):
        if event.button != 1: return
        p = self._world_pos(event.pos)
        if p is not None:
            self._apply_disc(p)

    def on_mouse_move(self, event):
        c = self.canvas
        if c is None or c._cur_xyz is None: return
        p = self._world_pos(event.pos)
        if p is None: return
        normal = self._get_normal_at(p)
        self._draw_disc_cursor(c, p, normal)

    def deactivate(self):
        self._clear_overlays()
        super().deactivate()

    def _get_normal_at(self, center):
        c = self.canvas
        if c is None:
            return np.array([0,0,1], np.float32)
        r = self.radius_m
        # Vía rápida: grid espacial en C (evita escanear el tile entero
        # en cada movimiento del mouse mientras se previsualiza el disco).
        fast_idx = self._octree_sphere_query(center, r)
        if fast_idx is not None and len(fast_idx) >= 6:
            cands = c.pc.xyz[fast_idx]
        else:
            if c._cur_xyz is None:
                return np.array([0,0,1], np.float32)
            xyz = c._cur_xyz
            diff = xyz - center.astype(xyz.dtype)
            bbox = ((np.abs(diff[:,0])<=r)&(np.abs(diff[:,1])<=r)&(np.abs(diff[:,2])<=r))
            cands = xyz[bbox]
        if len(cands) < 6:
            return np.array([0,0,1], np.float32)
        try:
            if len(cands) > 2000:
                cands = cands[np.random.choice(len(cands), 2000, replace=False)]
            centered = cands - cands.mean(0)
            _, _, Vt = np.linalg.svd(centered.astype(np.float64), full_matrices=False)
            n = Vt[-1].astype(np.float32)
            nm = float(np.linalg.norm(n))
            return n/nm if nm>1e-9 else np.array([0,0,1],np.float32)
        except Exception:
            return np.array([0,0,1], np.float32)

    def _apply_disc(self, center):
        c = self.canvas
        if c is None or c.pc is None: return
        r = self.radius_m
        normal = self._get_normal_at(center)

        # Vía rápida: grid espacial en C — no escanea el tile entero.
        fast_idx = self._octree_sphere_query(center, r)
        if fast_idx is not None:
            if len(fast_idx) == 0: return
            cand_xyz = c.pc.xyz[fast_idx]
            proj = np.abs(np.dot(cand_xyz - center.astype(cand_xyz.dtype), normal.astype(cand_xyz.dtype)))
            disk = fast_idx[proj <= r*self.thickness]
            if len(disk) == 0: disk = fast_idx
            self._apply_selection(disk.astype(np.int32))
            return

        # Fallback: nube recién cargada, octree aún no listo
        if c._cur_xyz is None: return
        xyz = c._cur_xyz
        diff = xyz - center.astype(xyz.dtype)
        bbox = ((np.abs(diff[:,0])<=r)&(np.abs(diff[:,1])<=r)&(np.abs(diff[:,2])<=r))
        cands_l = np.where(bbox)[0]
        if len(cands_l)==0: return
        d = diff[cands_l]
        sphere = cands_l[(d[:,0]**2+d[:,1]**2+d[:,2]**2)<=r*r]
        if len(sphere)==0: return
        proj = np.abs(np.dot(diff[sphere], normal.astype(diff.dtype)))
        disk = sphere[proj <= r*self.thickness]
        if len(disk)==0: disk = sphere
        idx = c._cur_idx[disk].astype(np.int32) if c._cur_idx is not None else disk.astype(np.int32)
        self._apply_selection(idx)

    def _draw_disc_cursor(self, c, center, normal):
        try:
            r = self.radius_m
            nf = normal.astype(np.float64)
            up = np.array([0,0,1.0],np.float64)
            if abs(np.dot(nf,up))>0.9: up=np.array([1,0,0.0],np.float64)
            u = np.cross(nf,up); u/=np.linalg.norm(u)
            v = np.cross(nf,u);  v/=np.linalg.norm(v)
            angles = np.linspace(0,2*np.pi,37,dtype=np.float32)
            pts = np.array([center.astype(np.float64)+r*(np.cos(a)*u+np.sin(a)*v)
                            for a in angles],np.float32)
            color = (0.2,0.9,1.0,0.9)
            try:
                if c.project:
                    s = next((s for s in c.project.schema if s.id==self.active_class_id),None)
                    if s:
                        h=s.color.lstrip('#')
                        color=(int(h[0:2],16)/255,int(h[2:4],16)/255,int(h[4:6],16)/255,1.0)
            except Exception: pass
            c._brush_cursor.set_data(pts,color=color,width=2.5)
            c._brush_cursor.visible=True
            n_line=np.array([center.astype(np.float32),(center+normal*r*0.5).astype(np.float32)],np.float32)
            c._mline_live.set_data(n_line,color=(*color[:3],0.5),width=1.5)
            c._mline_live.visible=True
            c.update()
        except Exception: pass


# ─────────────────────────────────────────────────────────────────────────────
# Import RegionGrowingTool
from annotation.region_growing import RegionGrowingTool

# ─────────────────────────────────────────────────────────────────────────────
# Registro
# ─────────────────────────────────────────────────────────────────────────────

TOOL_CLASSES = [
    BrushTool,
    DiscTool,
    RegionGrowingTool,
    PolygonTool,
    BoxSelectTool,
    SphereSelectTool,
    SliceTool,
    PickTool,
    MeasureTool,
]

TOOL_BY_KEY  = {t.key:  t for t in TOOL_CLASSES if t.key}
TOOL_BY_NAME = {t.name: t for t in TOOL_CLASSES}
ALL_TOOLS    = TOOL_CLASSES
