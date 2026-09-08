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
from collections import deque
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

        # ── 1b. Recortar al vecindario local ANTES de todo lo demás ──────────
        # Antes se construía la grilla espacial sobre TODO el tile/nube
        # visible (hasta 8M+ puntos) sin importar qué tan chico fuera
        # max_dist_m — un clic siempre pagaba el costo completo del tile.
        # Region Growing por definición nunca sale de max_dist_m de la
        # semilla, así que basta indexar esa caja.
        margin = self.max_dist_m
        box = ((np.abs(xyz_local[:, 0] - seed_xyz[0]) <= margin) &
               (np.abs(xyz_local[:, 1] - seed_xyz[1]) <= margin))
        box_local_idx = np.nonzero(box)[0]
        xyz_box = xyz_local[box_local_idx]
        n_box = len(xyz_box)
        seed_bi = int(np.searchsorted(box_local_idx, seed_li))
        box_to_global = (local_to_global[box_local_idx]
                          if local_to_global is not None else box_local_idx)

        # ── 1c. Test de similitud vs semilla — VECTORIZADO, sobre la caja ────
        # El test de similitud (z/rgb/intensidad) siempre compara contra la
        # SEMILLA, nunca contra el vecino actual — es decir, NO depende de
        # quién lo esté consultando durante el BFS. Antes se recalculaba en
        # Python puro por cada candidato a vecino, una y otra vez, cada vez
        # que un punto del cluster lo encontraba en su vecindario 3×3 (un
        # fondo denso alrededor del cluster generaba así cientos de miles
        # de comprobaciones redundantes) — perfilado por separado, esto era
        # el verdadero costo (~4.4s de los ~4.5s totales con 8M pts / un
        # cluster de 4000), no la construcción de la grilla (~0.1s). Ahora
        # se calcula UNA vez, vectorizado sobre toda la caja.
        similar_mask = np.ones(n_box, dtype=bool)
        if self.use_z:
            similar_mask &= np.abs(xyz_box[:, 2] - seed_xyz[2]) <= self.tol_z
        if self.use_rgb and seed_rgb is not None and pc.rgb is not None:
            box_rgb = pc.rgb[box_to_global].astype(np.float32)
            similar_mask &= np.abs(box_rgb - seed_rgb).max(axis=1) <= self.tol_rgb
        if self.use_intensity and seed_int is not None and pc.intensity is not None:
            box_int = pc.intensity[box_to_global].astype(np.float32)
            similar_mask &= np.abs(box_int - seed_int) <= self.tol_intensity
        similar_mask[seed_bi] = True

        # ── 1d. Restringir a solo los candidatos similares ────────────────────
        # Un punto que no pasa similar_mask NUNCA puede terminar
        # seleccionado (sea cual sea su vecino de consulta) — así que ni
        # siquiera necesita existir en la grilla espacial del BFS. Esto
        # reduce el conjunto de trabajo de "toda la caja" (cientos de miles
        # de puntos de fondo incluidos) a solo los candidatos reales
        # (típicamente el tamaño del cluster), antes de construir nada.
        sim_idx  = np.nonzero(similar_mask)[0]
        xyz_local = xyz_box[sim_idx]
        n = len(sim_idx)
        local_to_global = box_to_global[sim_idx]
        seed_li = int(np.searchsorted(sim_idx, seed_bi))

        # ── 2. Hash grid espacial — construcción vectorizada ──────────────────
        # Antes: `for li in range(n): cell_map[(ci[li],ri[li])].append(li)`
        # — un bucle Python puro por cada punto. Ahora: agrupamos por celda
        # con argsort/unique (vectorizado en C dentro de NumPy) y los
        # grupos se guardan ya como lista de Python (`.tolist()` una sola
        # vez aquí, no en cada consulta dentro del BFS).
        step = max(0.05, self.step_dist_m)
        mn   = xyz_local[:, :2].min(0) if n else np.zeros(2, np.float32)
        ci   = ((xyz_local[:, 0] - mn[0]) / step).astype(np.int64)
        ri   = ((xyz_local[:, 1] - mn[1]) / step).astype(np.int64)
        row_span  = int(ri.max()) + 2 if n else 1
        cell_key  = ci * np.int64(row_span) + ri

        order      = np.argsort(cell_key, kind='stable')
        sorted_key = cell_key[order]
        uniq_key, first_pos = np.unique(sorted_key, return_index=True)
        groups   = np.split(order, first_pos[1:])
        cell_map = {int(k): g.tolist() for k, g in zip(uniq_key, groups)}

        # ── 3. BFS conectado ──────────────────────────────────────────────────
        # Ya solo quedan candidatos que pasaron similar_mask, así que aquí
        # dentro basta comprobar distancia (propagación local + límite
        # global) — nada de rgb/z/intensidad de nuevo.
        max_r2_global = self.max_dist_m ** 2
        step_r2       = (step * 1.6) ** 2   # vecinos dentro de 1.6× el paso

        visited  = [False] * n   # lista Python, no array NumPy — indexar un
                                  # escalar en un bucle Python puro es más
                                  # rápido en una lista que en un ndarray.
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

            # Buscar vecinos en las 9 celdas adyacentes — clave codificada
            # (col*row_span + row), no tupla, para calzar con cell_map.
            cc, cr = int(ci[curr_li]), int(ri[curr_li])
            for dc in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    neighbors = cell_map.get((cc+dc) * row_span + (cr+dr))
                    if neighbors is None: continue
                    for nb_li in neighbors:
                        if visited[nb_li]: continue

                        dxy = xyz_local[nb_li, :2] - curr_xy
                        if (dxy**2).sum() > step_r2: continue

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
