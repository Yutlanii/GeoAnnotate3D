"""
annotation/region_growing.py — Region Growing BFS real y conectado

DIFERENCIA CON LA VERSIÓN ANTERIOR:
  Antes: seleccionaba TODOS los puntos similares dentro de una esfera,
         sin importar si están conectados. Cruzaba paredes y vacíos.

  Ahora: BFS topológicamente conectado.
         Solo llega a un punto si existe una CADENA CONTINUA de puntos
         similares desde la semilla. Si hay una brecha, se detiene.

ALGORITMO:
  1. Semilla → obtener atributos (xyz, rgb, intensidad)
  2. Construir hash grid espacial de la zona
  3. BFS: en cada paso buscar vecinos dentro de step_dist_m
  4. Cada vecino se acepta solo si es similar a la SEMILLA ORIGINAL
  5. Aceptado → añadir a cola y selección. Rechazado → barrera.
"""
from __future__ import annotations
from collections import deque, defaultdict
import numpy as np
from annotation.tools import BaseTool


class RegionGrowingTool(BaseTool):
    name    = "Region Growing"
    key     = "G"
    icon    = "region"
    tooltip = "Ctrl+clic → crece conectadamente desde el punto semilla"

    def __init__(self):
        super().__init__()
        self.step_dist_m:  float = 0.5    # radio de propagación local (m)
        self.max_dist_m:   float = 30.0   # radio global máximo (m)
        self.tol_rgb:      float = 30.0   # tolerancia de color 0-255
        self.tol_z:        float = 1.0    # tolerancia de altura (m)
        self.tol_intensity:float = 0.15   # tolerancia de intensidad 0-1
        self.use_rgb:      bool  = True
        self.use_z:        bool  = True
        self.use_intensity:bool  = False
        self.max_points:   int   = 300_000

    def on_mouse_press(self, event):
        if event.button != 1: return
        p = self._world_pos(event.pos)
        if p is not None: self._grow(p)

    def on_mouse_move(self, event): pass
    def on_mouse_release(self, event): pass

    def _grow(self, seed_pos: np.ndarray) -> None:
        c = self.canvas
        if c is None or c.pc is None: return
        pc = c.pc

        # Usar tile activo si existe (densidad máxima)
        if c._tile_mode and hasattr(c, '_tile_indices') and c._tile_indices is not None:
            xyz_local       = np.array(pc.xyz[c._tile_indices], dtype=np.float32)
            local_to_global = c._tile_indices
        elif c._cur_xyz is not None and len(c._cur_xyz) > 0:
            xyz_local       = c._cur_xyz
            local_to_global = c._cur_idx
        else:
            return

        n = len(xyz_local)
        if n == 0: return

        # ── 1. Punto semilla más cercano ──────────────────────────────────────
        diff     = xyz_local - seed_pos.astype(xyz_local.dtype)
        seed_li  = int(np.argmin((diff**2).sum(1)))
        seed_gi  = int(local_to_global[seed_li]) if local_to_global is not None else seed_li
        seed_xyz = xyz_local[seed_li].copy()

        # Atributos de referencia (fijos — siempre comparamos contra la semilla)
        seed_rgb = (pc.rgb[seed_gi].astype(np.float32)
                    if self.use_rgb and pc.rgb is not None else None)
        seed_int = (float(pc.intensity[seed_gi])
                    if self.use_intensity and pc.intensity is not None else None)

        # ── 2. Hash grid espacial para búsqueda rápida de vecinos ────────────
        step = max(0.05, self.step_dist_m)
        mn   = xyz_local[:, :2].min(0)
        ci   = ((xyz_local[:, 0] - mn[0]) / step).astype(np.int32)
        ri   = ((xyz_local[:, 1] - mn[1]) / step).astype(np.int32)
        cell_map: dict = defaultdict(list)
        for li in range(n):
            cell_map[(int(ci[li]), int(ri[li]))].append(li)

        # ── 3. BFS conectado ──────────────────────────────────────────────────
        max_r2_global = self.max_dist_m ** 2
        step_r2       = (step * 1.6) ** 2   # vecinos dentro de 1.6× el paso

        visited  = np.zeros(n, dtype=bool)
        selected = []
        queue    = deque()

        visited[seed_li] = True
        queue.append(seed_li)
        selected.append(seed_li)

        while queue and len(selected) < self.max_points:
            curr_li = queue.popleft()
            curr_xy = xyz_local[curr_li, :2]

            # Límite global: no crecer más allá de max_dist_m desde la semilla
            ds = xyz_local[curr_li] - seed_xyz
            if (ds**2).sum() > max_r2_global:
                continue

            # Buscar vecinos en las 9 celdas adyacentes
            cc, cr = int(ci[curr_li]), int(ri[curr_li])
            for dc in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    neighbors = cell_map.get((cc+dc, cr+dr))
                    if not neighbors: continue
                    for nb_li in neighbors:
                        if visited[nb_li]: continue

                        # Distancia al punto actual (propagación local)
                        dxy = xyz_local[nb_li, :2] - curr_xy
                        if (dxy**2).sum() > step_r2: continue

                        # Test de similitud vs SEMILLA (no vs vecino actual)
                        similar = True

                        if self.use_z and similar:
                            if abs(float(xyz_local[nb_li, 2]) -
                                   float(seed_xyz[2])) > self.tol_z:
                                similar = False

                        if self.use_rgb and similar and seed_rgb is not None:
                            if pc.rgb is not None:
                                nb_gi = int(local_to_global[nb_li]) if local_to_global is not None else nb_li
                                if np.abs(pc.rgb[nb_gi].astype(np.float32)
                                          - seed_rgb).max() > self.tol_rgb:
                                    similar = False

                        if self.use_intensity and similar and seed_int is not None:
                            if pc.intensity is not None:
                                nb_gi = int(local_to_global[nb_li]) if local_to_global is not None else nb_li
                                if abs(float(pc.intensity[nb_gi]) -
                                       seed_int) > self.tol_intensity:
                                    similar = False

                        if similar:
                            visited[nb_li] = True
                            queue.append(nb_li)
                            selected.append(nb_li)

        if not selected:
            print("[RegionGrowing] Sin puntos similares conectados — "
                  "aumenta step_dist o tolerancias.")
            return

        sel_arr = np.array(selected, dtype=np.int64)
        global_idx = (local_to_global[sel_arr].astype(np.int64)
                      if local_to_global is not None else sel_arr)

        print(f"[RegionGrowing] {len(global_idx):,} pts "
              f"(BFS, step={step}m, tol_z={self.tol_z}m, tol_rgb={self.tol_rgb})")
        self._apply_selection(global_idx)
