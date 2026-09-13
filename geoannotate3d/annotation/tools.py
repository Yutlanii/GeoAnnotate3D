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
        p, _idx = self._snap_to_point_idx(screen_pos, radius_px)
        return p

    def _snap_to_point_idx(self, screen_pos, radius_px: float = SNAP_SCREEN_R
                           ) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Igual que `_snap_to_point`, pero además devuelve el índice GLOBAL
        (dentro de pc.xyz / project.labels) del punto real encontrado —
        necesario para herramientas como Pick que necesitan leer atributos
        (clasificación, intensidad…) de exactamente ese punto, sin volver a
        buscarlo por coordenadas (frágil: la posición que se muestra puede
        ser un promedio de varios puntos, no la de uno solo).
        `c._cur_idx[i]` es el índice global correspondiente a `c._cur_xyz[i]`
        (ver render/canvas.py, siempre se asignan juntos) — por eso alcanza
        con submuestrear ambos arrays EN SINCRONÍA en vez de reconsultar
        pc.xyz.
        """
        c = self.canvas
        if c is None or c._cur_xyz is None:
            return None, None
        try:
            xyz = c._cur_xyz
            gi  = c._cur_idx
            n   = len(xyz)
            step = max(1, n // 200_000)
            sub  = xyz[::step]
            sub_idx = gi[::step] if gi is not None and len(gi) == n else None
            sc   = c.map_to_screen(sub)
            sx, sy = float(screen_pos[0]), float(screen_pos[1])
            d2 = (sc[:, 0] - sx) ** 2 + (sc[:, 1] - sy) ** 2
            bi = int(np.argmin(d2))
            if d2[bi] <= radius_px ** 2:
                pos = sub[bi].copy()
                idx = int(sub_idx[bi]) if sub_idx is not None else None
                return pos, idx
        except Exception:
            pass
        return None, None

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
        # Preferir el snap al punto REAL más cercano (con su índice global)
        # sobre el promedio de vecinos de _world_pos — así la clasificación
        # que se muestra corresponde exactamente al punto elegido, en vez
        # de tener que volver a buscarlo por coordenadas (ver point_picked_abs
        # en render/canvas.py para el porqué de este cambio).
        p, gidx = self._snap_to_point_idx(event.pos)
        if p is None:
            p = self._world_pos(event.pos, fast=False)
            gidx = None
        if p is None: return
        abs_xyz = p.astype(np.float64) + c.pc.offset
        c.sig.point_picked.emit(float(p[0]), float(p[1]), float(p[2]))
        c.sig.point_picked_abs.emit(
            float(abs_xyz[0]), float(abs_xyz[1]), float(abs_xyz[2]), gidx)


# ─────────────────────────────────────────────────────────────────────────────
# 8. MeasureTool — Medición 3D / horizontal / dZ
# ─────────────────────────────────────────────────────────────────────────────

class MeasureTool(BaseTool):
    """
    Mide distancia 3D, distancia horizontal y diferencia de altura.
    Clic A → clic B → muestra resultado.

    Ctrl+clic sobre una medida YA EXISTENTE (solo si no hay una en curso)
    la edita (color/grosor) o la elimina — antes una medida, una vez
    creada, no se podía borrar de ninguna forma.
    """
    name    = "Medir"
    key     = "M"
    icon    = "ruler"
    tooltip = "Ctrl+clic A · clic B mide · Ctrl+clic sobre una ya puesta la edita"

    _EDIT_RADIUS_PX = 14.0

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

        if self._p1 is None:
            existing = self._find_nearby_measure(event.pos)
            if existing is not None:
                self._edit_marker(existing)
                return

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
            # Persistir la medida (ver annotation/markers.py) — antes esto
            # se perdía en cuanto medías otra cosa o cambiabas de
            # herramienta; ahora queda anclada en 3D y se guarda con el
            # proyecto, igual que en CloudCompare/Cyclone 3DR.
            marker_store = getattr(c, "marker_store", None)
            if marker_store is not None:
                marker_store.add_measure(
                    self._p1, p, text=f"{d3d:.2f} m")
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

    def _find_nearby_measure(self, screen_pos):
        """Medida existente más cercana al clic en pantalla (distancia al
        SEGMENTO A-B, no solo a sus extremos — ver PolylineTool._find_nearby_polyline
        para el porqué)."""
        c = self.canvas
        store = getattr(c, "marker_store", None)
        if store is None or len(store) == 0:
            return None
        measures = [m for m in store.all() if m.kind == "measure" and m.pos_b is not None]
        if not measures:
            return None
        sx, sy = float(screen_pos[0]), float(screen_pos[1])
        best, best_d2 = None, self._EDIT_RADIUS_PX ** 2
        for m in measures:
            try:
                sc = c.map_to_screen(np.array([m.pos, m.pos_b], np.float32))
            except Exception:
                continue
            d2 = _point_segment_dist2(sx, sy, sc[0, 0], sc[0, 1], sc[1, 0], sc[1, 1])
            if d2 < best_d2:
                best, best_d2 = m, d2
        return best

    def _edit_marker(self, marker) -> None:
        c = self.canvas
        try:
            dlg = _LineMarkerEditDialog(c, "Editar medida", color=tuple(marker.color),
                                       line_width=getattr(marker, "line_width", 2.5),
                                       delete_label="Eliminar medida",
                                       color_title="Color de la medida")
            accepted, deleted, color, line_width = dlg.exec()
        except Exception:
            return
        if not accepted:
            return
        if deleted:
            c.marker_store.remove(marker.id)
            return
        c.marker_store.update(marker.id, color=list(color), line_width=line_width)


# ─────────────────────────────────────────────────────────────────────────────
# LabelMarkerTool — etiqueta de texto persistente en 3D
# ─────────────────────────────────────────────────────────────────────────────

class _LabelMarkerDialog:
    """
    Diálogo combinado (texto + tamaño de letra + color) usado tanto al
    CREAR una etiqueta nueva como al EDITAR una ya existente — mismo
    formulario en los dos casos, para que el tamaño/color se puedan fijar
    desde antes de crearla o cambiar después sin duplicar UI.
    """
    def __init__(self, parent, title, text="", color=(1.0, 0.85, 0.2), font_size=14.0):
        from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                     QLineEdit, QSpinBox, QPushButton, QDialogButtonBox)
        from PyQt5.QtGui import QColor

        self._color = tuple(color)
        self._QColor = QColor

        dlg = QDialog(parent)
        dlg.setWindowTitle(title)
        lay = QVBoxLayout(dlg)

        lay.addWidget(QLabel("Texto:"))
        self._text_edit = QLineEdit(text)
        lay.addWidget(self._text_edit)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Tamaño de letra:"))
        self._size_spin = QSpinBox()
        self._size_spin.setRange(8, 48)
        self._size_spin.setValue(int(font_size))
        size_row.addWidget(self._size_spin)
        lay.addLayout(size_row)

        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Color:"))
        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(50, 22)
        self._update_color_btn()
        self._color_btn.clicked.connect(self._pick_color)
        color_row.addWidget(self._color_btn)
        color_row.addStretch()
        lay.addLayout(color_row)

        del_btn = QPushButton("Eliminar etiqueta")
        del_btn.setStyleSheet("QPushButton{color:#c0392b;}")
        del_btn.clicked.connect(lambda: self._mark_deleted(dlg))
        lay.addWidget(del_btn)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        self._dlg = dlg
        self._deleted = False

    def _mark_deleted(self, dlg):
        self._deleted = True
        dlg.accept()

    def _update_color_btn(self):
        r, g, b = (int(max(0, min(1, c)) * 255) for c in self._color)
        self._color_btn.setStyleSheet(
            f"background: rgb({r},{g},{b}); border: 1px solid #888; border-radius: 3px;")

    def _pick_color(self):
        from PyQt5.QtWidgets import QColorDialog
        r, g, b = (int(max(0, min(1, c)) * 255) for c in self._color)
        picked = QColorDialog.getColor(self._QColor(r, g, b), self._dlg, "Color de la etiqueta")
        if picked.isValid():
            self._color = (picked.redF(), picked.greenF(), picked.blueF())
            self._update_color_btn()

    def exec(self):
        """Devuelve (accepted, deleted, text, color, font_size)."""
        accepted = bool(self._dlg.exec_())
        return (accepted, self._deleted, self._text_edit.text().strip(), self._color,
                float(self._size_spin.value()))


class LabelMarkerTool(BaseTool):
    """
    Ctrl+clic: coloca una etiqueta de texto persistente en ese punto —
    visible permanentemente (hasta que se borre) y guardada con el
    proyecto (ver annotation/markers.py::MarkerStore). A diferencia de
    PickTool (que solo inspecciona), esto deja una marca anotada en el
    espacio, igual que las "text labels" de CloudCompare o las
    anotaciones de Cyclone 3DR.

    Ctrl+clic sobre una etiqueta YA EXISTENTE la EDITA en vez de crear
    una nueva (mismo diálogo, precargado con su texto/tamaño/color).
    El tamaño y color elegidos se recuerdan para la siguiente etiqueta
    nueva que coloques con esta misma herramienta.
    """
    name    = "Etiqueta 3D"
    key     = "N"
    icon    = "tag"
    tooltip = ("Ctrl+clic → coloca una etiqueta de texto persistente en 3D · "
               "Ctrl+clic sobre una etiqueta existente → la edita")

    _EDIT_RADIUS_PX = 26.0

    def __init__(self):
        super().__init__()
        self.color:     tuple = (1.0, 0.85, 0.2)
        self.font_size: float = 14.0

    def on_mouse_press(self, event):
        if event.button != 1:
            return
        c = self.canvas
        if c is None:
            return
        marker_store = getattr(c, "marker_store", None)
        if marker_store is None:
            return

        existing = self._find_nearby_label(event.pos)
        if existing is not None:
            self._edit_marker(existing)
            return

        p = self._snap_to_point(event.pos)
        if p is None:
            p = self._world_pos(event.pos, fast=False)
        if p is None:
            return
        try:
            dlg = _LabelMarkerDialog(c, "Etiqueta 3D", text="",
                                     color=self.color, font_size=self.font_size)
            accepted, _deleted, text, color, font_size = dlg.exec()
        except Exception:
            return
        if accepted and text:
            marker_store.add_label(p, text, color=color, font_size=font_size)
            # Recordar tamaño/color para la próxima etiqueta que coloques.
            self.color, self.font_size = color, font_size
            if hasattr(c, "update"):
                c.update()

    def _find_nearby_label(self, screen_pos):
        """Etiqueta existente más cercana al clic en pantalla, si hay
        alguna dentro de _EDIT_RADIUS_PX — para editar en vez de crear."""
        c = self.canvas
        store = getattr(c, "marker_store", None)
        if store is None or len(store) == 0:
            return None
        labels = [m for m in store.all() if m.kind == "label"]
        if not labels:
            return None
        try:
            positions = np.array([m.pos for m in labels], np.float32)
            sc = c.map_to_screen(positions)
            sx, sy = float(screen_pos[0]), float(screen_pos[1])
            d2 = (sc[:, 0] - sx) ** 2 + (sc[:, 1] - sy) ** 2
            bi = int(np.argmin(d2))
            if d2[bi] <= self._EDIT_RADIUS_PX ** 2:
                return labels[bi]
        except Exception:
            pass
        return None

    def _edit_marker(self, marker) -> None:
        c = self.canvas
        try:
            dlg = _LabelMarkerDialog(c, "Editar etiqueta 3D", text=marker.text,
                                     color=tuple(marker.color), font_size=marker.font_size)
            accepted, deleted, text, color, font_size = dlg.exec()
        except Exception:
            return
        if not accepted:
            return
        if deleted:
            c.marker_store.remove(marker.id)
            return
        if text:
            c.marker_store.update(marker.id, text=text, color=list(color), font_size=font_size)
            self.color, self.font_size = color, font_size


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
# PolylineTool — polilínea persistente en 3D (centerlines: caminos, líneas
# eléctricas, taludes...)
# ─────────────────────────────────────────────────────────────────────────────

class _LineMarkerEditDialog:
    """
    Diálogo para editar el color y grosor de una marca "de línea" YA
    creada — polilínea O medida — (o fijarlos de antemano para la
    próxima). Mismo patrón que _LabelMarkerDialog, pero con un spinbox
    de grosor de línea en vez de tamaño de letra, y sin campo de texto
    (ninguna de las dos necesita uno tan prominente como una etiqueta;
    su descripción — longitud/vértices, o distancia — se recalcula
    sola). Incluye un botón "Eliminar" para borrarla del todo en el
    mismo diálogo, en vez de tener que buscar otra forma — antes NINGÚN
    marcador de línea (ni medida ni polilínea) se podía borrar una vez
    creado, solo las etiquetas de texto.
    """
    def __init__(self, parent, title, color=(0.2, 0.85, 1.0), line_width=2.5,
                allow_delete=True, delete_label="Eliminar", color_title="Color"):
        from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                     QDoubleSpinBox, QPushButton, QDialogButtonBox)
        from PyQt5.QtGui import QColor

        self._color = tuple(color)
        self._QColor = QColor
        self._deleted = False
        self._color_title = color_title

        dlg = QDialog(parent)
        dlg.setWindowTitle(title)
        lay = QVBoxLayout(dlg)

        width_row = QHBoxLayout()
        width_row.addWidget(QLabel("Grosor de línea:"))
        self._width_spin = QDoubleSpinBox()
        self._width_spin.setRange(0.5, 12.0)
        self._width_spin.setSingleStep(0.5)
        self._width_spin.setValue(float(line_width))
        width_row.addWidget(self._width_spin)
        lay.addLayout(width_row)

        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Color:"))
        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(50, 22)
        self._update_color_btn()
        self._color_btn.clicked.connect(self._pick_color)
        color_row.addWidget(self._color_btn)
        color_row.addStretch()
        lay.addLayout(color_row)

        if allow_delete:
            del_btn = QPushButton(delete_label)
            del_btn.setStyleSheet("QPushButton{color:#c0392b;}")
            del_btn.clicked.connect(lambda: self._mark_deleted(dlg))
            lay.addWidget(del_btn)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        self._dlg = dlg

    def _mark_deleted(self, dlg):
        self._deleted = True
        dlg.accept()

    def _update_color_btn(self):
        r, g, b = (int(max(0, min(1, c)) * 255) for c in self._color)
        self._color_btn.setStyleSheet(
            f"background: rgb({r},{g},{b}); border: 1px solid #888; border-radius: 3px;")

    def _pick_color(self):
        from PyQt5.QtWidgets import QColorDialog
        r, g, b = (int(max(0, min(1, c)) * 255) for c in self._color)
        picked = QColorDialog.getColor(self._QColor(r, g, b), self._dlg, self._color_title)
        if picked.isValid():
            self._color = (picked.redF(), picked.greenF(), picked.blueF())
            self._update_color_btn()

    def exec(self):
        """Devuelve (accepted, deleted, color, line_width)."""
        accepted = bool(self._dlg.exec_())
        return (accepted, self._deleted, self._color, float(self._width_spin.value()))


class PolylineTool(BaseTool):
    """
    Digitaliza una polilínea abierta de varios vértices sobre la nube —
    a diferencia de PolygonTool (cierra un área para SELECCIONAR puntos
    dentro), esto no selecciona nada: deja una marca persistente (ver
    annotation/markers.py) para digitalizar centerlines — el eje de un
    camino, el trazo de una línea eléctrica, el borde de un talud — que
    con solo un segmento (MeasureTool) o un punto (LabelMarkerTool) no
    se puede representar bien.

    Ctrl+clic (nube vacía): agrega un vértice. Enter: termina y la deja
    guardada. Escape: cancela la polilínea en curso. Backspace: quita el
    último vértice puesto (sin cancelar todo). Ctrl+clic sobre una
    polilínea YA EXISTENTE (solo si no hay una en curso) la edita —
    color, grosor, o eliminarla — en vez de empezar una nueva.
    """
    name    = "Polilínea"
    # Letra libre — B/L/X/H/C/G/I/M/N/D/P ya están tomadas por otras
    # herramientas, y F/V/Y/R/espacio/1-9 los intercepta la cámara antes
    # de llegar al mapa de herramientas (ver _AnnotationStyle._on_key en
    # render/canvas.py) — no hay una letra "mnemónica" libre para esto.
    key     = "K"
    icon    = "diagram-3"
    tooltip = ("Ctrl+clic agrega un vértice · Enter termina · Backspace quita "
              "el último · Escape cancela · Ctrl+clic sobre una ya puesta la edita")

    _EDIT_RADIUS_PX = 14.0   # distancia máx. a un SEGMENTO (no solo vértice)

    def __init__(self):
        super().__init__()
        self._pts: list = []
        self.color:      tuple = (0.2, 0.85, 1.0)
        self.line_width: float = 2.5

    def on_mouse_press(self, event):
        if event.button != 1:
            return
        c = self.canvas
        if c is None:
            return

        if not self._pts:
            existing = self._find_nearby_polyline(event.pos)
            if existing is not None:
                self._edit_marker(existing)
                return

        p = self._snap_to_point(event.pos)
        if p is None:
            p = self._world_pos(event.pos, fast=False)
        if p is None:
            p = self._screen_to_world_fallback(event.pos)
        if p is None:
            return
        self._pts.append(p.astype(np.float32))
        self._redraw_preview()

    def on_mouse_move(self, event):
        c = self.canvas
        if c is None or not self._pts:
            return
        p = self._snap_to_point(event.pos)
        if p is None:
            p = self._world_pos(event.pos)
        if p is None:
            p = self._screen_to_world_fallback(event.pos)
        if p is None:
            return
        seg = np.array([self._pts[-1], p], np.float32)
        c._mline_live.set_data(seg, color=(*self.color, 0.5), width=self.line_width)
        c._mline_live.visible = True
        c.update()

    def _redraw_preview(self):
        c = self.canvas
        if c is None:
            return
        if len(self._pts) >= 2:
            c._mline.set_data(np.array(self._pts, np.float32),
                              color=(*self.color, 0.9), width=self.line_width)
            c._mline.visible = True
        else:
            # Un solo vértice: nada que conectar todavía, pero limpiar
            # cualquier trazo previo (p.ej. tras un Backspace que dejó
            # solo 1 punto) para que no quede una línea vieja fantasma.
            c._mline.visible = False
        c.update()

    def on_key_press(self, event):
        key = getattr(event, "key", "")
        if key == "Escape":
            self._pts = []
            self._clear_overlays()
            return
        if key in ("Return", "Enter"):
            self._finish()
            return
        if key == "BackSpace" and self._pts:
            self._pts.pop()
            if not self._pts:
                self._clear_overlays()
            else:
                self._redraw_preview()
                c = self.canvas
                if c is not None:
                    c._mline_live.visible = False
                    c.update()

    def _finish(self) -> None:
        c = self.canvas
        if c is None or len(self._pts) < 2:
            self._pts = []
            self._clear_overlays()
            return
        marker_store = getattr(c, "marker_store", None)
        if marker_store is not None:
            length = float(sum(
                np.linalg.norm(self._pts[i + 1] - self._pts[i])
                for i in range(len(self._pts) - 1)))
            marker_store.add_polyline(
                self._pts, text=f"{length:.2f} m, {len(self._pts)} vértices",
                color=self.color, line_width=self.line_width)
        self._pts = []
        self._clear_overlays()

    def _find_nearby_polyline(self, screen_pos):
        """
        Polilínea existente más cercana al clic en pantalla — a
        diferencia de LabelMarkerTool._find_nearby_label (que solo mira
        UN punto por marcador), aquí hay que revisar la distancia del
        clic a cada SEGMENTO de cada polilínea, no solo a sus vértices,
        porque el punto medio de un segmento largo puede estar lejos de
        ambos vértices y aun así ser "sobre la línea" a ojo.
        """
        c = self.canvas
        store = getattr(c, "marker_store", None)
        if store is None or len(store) == 0:
            return None
        polylines = [m for m in store.all() if m.kind == "polyline" and m.points]
        if not polylines:
            return None
        sx, sy = float(screen_pos[0]), float(screen_pos[1])
        best, best_d2 = None, self._EDIT_RADIUS_PX ** 2
        for m in polylines:
            try:
                sc = c.map_to_screen(np.array(m.points, np.float32))
            except Exception:
                continue
            for i in range(len(sc) - 1):
                d2 = _point_segment_dist2(sx, sy, sc[i, 0], sc[i, 1], sc[i+1, 0], sc[i+1, 1])
                if d2 < best_d2:
                    best, best_d2 = m, d2
        return best

    def _edit_marker(self, marker) -> None:
        c = self.canvas
        try:
            dlg = _LineMarkerEditDialog(c, "Editar polilínea", color=tuple(marker.color),
                                      line_width=getattr(marker, "line_width", 2.5),
                                      delete_label="Eliminar polilínea",
                                      color_title="Color de la polilínea")
            accepted, deleted, color, line_width = dlg.exec()
        except Exception:
            return
        if not accepted:
            return
        if deleted:
            c.marker_store.remove(marker.id)
            return
        c.marker_store.update(marker.id, color=list(color), line_width=line_width)
        self.color, self.line_width = color, line_width

    def deactivate(self):
        # Terminar/descartar como Escape si se cambia de herramienta con
        # una polilínea a medio trazar, en vez de dejarla "colgada" sin
        # guardar y sin overlay visible tampoco (ninguna de las dos cosas
        # sería lo que el usuario esperaría).
        self._pts = []
        super().deactivate()


def _point_segment_dist2(px, py, ax, ay, bx, by) -> float:
    """Distancia² (en pantalla) de (px,py) al segmento A-B — usado para
    encontrar a qué polilínea corresponde un clic sobre cualquier parte
    de su trazo, no solo sus vértices."""
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    if seg_len2 < 1e-9:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len2))
    cx, cy = ax + t * dx, ay + t * dy
    return (px - cx) ** 2 + (py - cy) ** 2


# ─────────────────────────────────────────────────────────────────────────────
# ProfileTool — vista de perfil / corte vertical
# ─────────────────────────────────────────────────────────────────────────────

class ProfileTool(BaseTool):
    """
    Traza una línea A→B (Ctrl+clic A, clic B — igual que MeasureTool) y
    abre una ventana con el corte vertical de la franja de la nube
    alrededor de esa línea: distancia a lo largo de la línea en el eje
    X, altura (Z) en el eje Y. Útil para revisar taludes, líneas
    eléctricas, secciones de vía, perfiles de terreno, sin tener que
    girar la cámara 3D a un ángulo lateral incómodo para "ver de canto".

    El ancho de la franja (buffer perpendicular a la línea) es ajustable
    desde el panel de herramientas (ver ui/tool_panel.py, sección de
    contexto "Perfil"), igual que el radio del Pincel/Disco.
    """
    name    = "Perfil"
    key     = "O"   # letra libre — ver comentario de PolylineTool arriba
    icon    = "graph-up"
    tooltip = "Ctrl+clic A · clic B → abre el corte vertical de la franja entre A y B"

    def __init__(self):
        super().__init__()
        self._p1: Optional[np.ndarray] = None
        self.buffer_m: float = 2.0
        self._dialog = None   # se reusa entre perfiles sucesivos

    def deactivate(self):
        self._p1 = None
        super().deactivate()

    def on_mouse_press(self, event):
        if event.button != 1:
            return
        c = self.canvas
        if c is None:
            return
        p = self._snap_to_point(event.pos)
        if p is None:
            p = self._world_pos(event.pos, fast=False)
        if p is None:
            p = self._screen_to_world_fallback(event.pos)
        if p is None:
            return
        if self._p1 is None:
            self._p1 = p
            c._p1_mk.set_data(pos=p[None, :], face_color=(0.2, 0.85, 1.0, 1.0), size=12)
            c._p1_mk.visible = True
            c.update()
        else:
            seg = np.array([self._p1, p], np.float32)
            c._mline.set_data(seg, color=(0.2, 0.85, 1.0, 0.9))
            c._mline.visible = True
            c.update()
            self._show_profile(self._p1, p)
            self._p1 = None

    def on_mouse_move(self, event):
        c = self.canvas
        if c is None or self._p1 is None:
            return
        p = self._snap_to_point(event.pos)
        if p is None:
            p = self._world_pos(event.pos)
        if p is None:
            p = self._screen_to_world_fallback(event.pos)
        if p is None:
            return
        seg = np.array([self._p1, p], np.float32)
        c._mline_live.set_data(seg, color=(1.0, 0.9, 0.2, 0.4))
        c._mline_live.visible = True
        c.update()

    def on_key_press(self, event):
        if getattr(event, "key", "") == "Escape":
            self._p1 = None
            self._clear_overlays()

    def _show_profile(self, p1: np.ndarray, p2: np.ndarray) -> None:
        from annotation.profile import extract_profile_slice
        c = self.canvas
        if c is None or c._cur_xyz is None or len(c._cur_xyz) == 0:
            return
        xyz = c._cur_xyz
        gidx = c._cur_idx if (c._cur_idx is not None and len(c._cur_idx) == len(xyz)) else None

        t, z, idx_out, length = extract_profile_slice(xyz, p1, p2, self.buffer_m, gidx)
        colors, legend = self._colors_for(idx_out, z)

        try:
            from ui.profile_view import ProfileDialog
            if self._dialog is None:
                self._dialog = ProfileDialog(c.window() if hasattr(c, "window") else c)
            self._dialog.update_data(t, z, colors, length, self.buffer_m, legend=legend)
        except Exception:
            import traceback; traceback.print_exc()

    def _colors_for(self, idx_out: np.ndarray, z: np.ndarray):
        """
        Colorea el perfil por clasificación (modo anotación) si hay
        proyecto/labels disponibles y `idx_out` son índices GLOBALES;
        si no, cae a un degradado por altura — siempre se ve algo
        coherente, nunca un scatter monocolor por defecto sin razón.
        Devuelve (colors, legend) — legend es [(nombre, (r,g,b)), ...]
        con solo las clases REALMENTE presentes en esta franja (no todo
        el schema del proyecto), o [] si se cayó al degradado por altura.
        """
        c = self.canvas
        try:
            project = getattr(c, "project", None)
            if project is not None and project.labels is not None and len(idx_out) > 0:
                labels = project.labels[idx_out]
                lut = {0: (130, 130, 130)}
                names = {0: "sin etiquetar"}
                for sc in project.schema:
                    h = sc.color.lstrip("#")
                    if len(h) >= 6:
                        lut[sc.id] = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
                        names[sc.id] = sc.name
                out = np.empty((len(labels), 3), np.uint8)
                for i, lb in enumerate(labels):
                    out[i] = lut.get(int(lb), (90, 200, 230))
                present = sorted(set(int(v) for v in labels))
                legend = [(names.get(cid, f"clase {cid}"), lut.get(cid, (90, 200, 230)))
                         for cid in present]
                return out, legend
        except Exception:
            pass
        # Fallback: degradado por altura (azul=bajo, rojo=alto)
        if len(z) == 0:
            return None, []
        zmin, zmax = float(z.min()), float(z.max())
        span = max(zmax - zmin, 1e-6)
        f = np.clip((z - zmin) / span, 0.0, 1.0)
        out = np.empty((len(z), 3), np.uint8)
        out[:, 0] = (f * 255).astype(np.uint8)
        out[:, 1] = 60
        out[:, 2] = ((1 - f) * 255).astype(np.uint8)
        return out, []


# ─────────────────────────────────────────────────────────────────────────────
# Import RegionGrowingTool / PlaneFitTool
# (LabelMarkerTool vive arriba, en este mismo archivo — a diferencia de
# RegionGrowingTool/PlaneFitTool, necesita cero dependencias nuevas de
# annotation.markers, así que se evita el ciclo de imports que supondría
# que annotation/markers.py importara BaseTool desde aquí.)
from annotation.region_growing import RegionGrowingTool
from annotation.plane_fit import PlaneFitTool

# ─────────────────────────────────────────────────────────────────────────────
# Registro
# ─────────────────────────────────────────────────────────────────────────────

TOOL_CLASSES = [
    BrushTool,
    DiscTool,
    RegionGrowingTool,
    PlaneFitTool,
    PolygonTool,
    BoxSelectTool,
    SphereSelectTool,
    SliceTool,
    PickTool,
    MeasureTool,
    LabelMarkerTool,
    PolylineTool,
    ProfileTool,
]

TOOL_BY_KEY  = {t.key:  t for t in TOOL_CLASSES if t.key}
TOOL_BY_NAME = {t.name: t for t in TOOL_CLASSES}
ALL_TOOLS    = TOOL_CLASSES
