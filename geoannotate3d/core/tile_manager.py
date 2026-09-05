"""
core/tile_manager.py — Tile Manager v2.0

Cambios vs v1.0:
  - Transform: offset_x, offset_y, rotation_deg.
    La cuadrícula puede trasladarse y rotarse respecto a los bounds de la nube.
  - TileInfo almacena bounds en el marco rotado local (min_xr, max_xr, min_yr, max_yr)
    y TileManager expone world_corners(tile) para dibujar en 3D.
  - TileExtractionWorker usa rotación para filtrar puntos correctamente.
  - tile_at_world_xy(wx, wy): dado un punto mundo devuelve el TileInfo que lo contiene.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


# ─────────────────────────────────────────────────────────────────────────────
# TileInfo
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TileInfo:
    id:  int
    row: int
    col: int
    # Bounds en marco ROTADO (col*s, (col+1)*s, row*s, (row+1)*s)
    min_xr: float
    max_xr: float
    min_yr: float
    max_yr: float
    n_points:    int   = 0
    labeled_pct: float = 0.0
    loaded:      bool  = False
    class_counts: dict = field(default_factory=dict)  # {class_id: n_pts}

    # Compatibilidad: devuelve centro en marco local
    @property
    def center_xr(self): return (self.min_xr + self.max_xr) * 0.5
    @property
    def center_yr(self): return (self.min_yr + self.max_yr) * 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Worker de extracción con soporte de rotación
# ─────────────────────────────────────────────────────────────────────────────

class TileExtractionWorker(QThread):
    progress = pyqtSignal(int, str)
    ready    = pyqtSignal(object, object)   # (TileInfo, np.ndarray)
    error    = pyqtSignal(str)

    CHUNK = 5_000_000

    def __init__(self, xyz: np.ndarray, tile: TileInfo,
                 origin_x: float, origin_y: float,
                 cos_rot: float, sin_rot: float,
                 parent=None):
        super().__init__(parent)
        self._xyz     = xyz
        self._tile    = tile
        self._ox      = float(origin_x)
        self._oy      = float(origin_y)
        self._ca      = float(cos_rot)
        self._sa      = float(sin_rot)

    def run(self):
        try:
            tile  = self._tile
            xyz   = self._xyz
            n     = len(xyz)
            chunk = self.CHUNK
            ox, oy, ca, sa = self._ox, self._oy, self._ca, self._sa
            parts: List[np.ndarray] = []

            self.progress.emit(5, f"Extrayendo tile ({tile.row},{tile.col})…")

            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                xy  = xyz[start:end, :2]

                # Transformar al marco rotado del grid
                dx = xy[:, 0] - ox
                dy = xy[:, 1] - oy
                xr =  dx * ca + dy * sa
                yr = -dx * sa + dy * ca

                mask = (
                    (xr >= tile.min_xr) & (xr < tile.max_xr) &
                    (yr >= tile.min_yr) & (yr < tile.max_yr)
                )
                if mask.any():
                    parts.append(np.where(mask)[0].astype(np.int64) + start)

                pct = 10 + int(80 * end / max(n, 1))
                if end % (chunk * 4) < chunk:
                    self.progress.emit(
                        pct, f"Tile {end//1_000_000:.0f}M / {n//1_000_000:.0f}M pts…")

            indices = np.concatenate(parts).astype(np.int64) if parts \
                      else np.zeros(0, np.int64)

            self.progress.emit(95, f"Tile: {len(indices):,} puntos ✓")
            self.ready.emit(tile, indices)

        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────────────
# TileManager
# ─────────────────────────────────────────────────────────────────────────────

class TileManager:
    """
    Gestiona la cuadrícula de tiles con soporte de transform (offset + rotación).

    El marco local rotado tiene:
      - Origen en (origin_x, origin_y) en coordenadas mundo (locales de la nube).
      - Eje X rotado: u = (cos_rot, sin_rot)
      - Eje Y rotado: v = (-sin_rot, cos_rot)

    Para convertir un punto mundo (wx, wy) al marco local:
        dx = wx - origin_x
        dy = wy - origin_y
        xr =  dx*cos_rot + dy*sin_rot
        yr = -dx*sin_rot + dy*cos_rot

    Para convertir marco local (xr, yr) a mundo:
        wx = origin_x + xr*cos_rot - yr*sin_rot
        wy = origin_y + xr*sin_rot + yr*cos_rot
    """

    def __init__(self, pc, project, tile_size_m: float = 50.0,
                 offset_x: float = 0.0, offset_y: float = 0.0,
                 rotation_deg: float = 0.0):
        self._pc      = pc
        self._project = project

        bounds       = pc.bounds
        self._cloud_min_x = float(bounds[0, 0])
        self._cloud_min_y = float(bounds[0, 1])
        self._cloud_max_x = float(bounds[1, 0])
        self._cloud_max_y = float(bounds[1, 1])
        self._cloud_mean_z = float((bounds[0, 2] + bounds[1, 2]) * 0.5)

        self.tile_size_m  = float(tile_size_m)
        self.offset_x     = float(offset_x)
        self.offset_y     = float(offset_y)
        self.rotation_deg = float(rotation_deg)

        self.tiles: List[TileInfo] = []
        self.n_rows = 0
        self.n_cols = 0
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.cos_rot  = 1.0
        self.sin_rot  = 0.0
        self._adj_x   = 0.0    # offset del grid en el marco rotado
        self._adj_y   = 0.0

        self._rebuild_tiles()
        self._estimate_from_sample()

    # ── Transform API ─────────────────────────────────────────────────────────

    def set_transform(self, offset_x: float, offset_y: float,
                      rotation_deg: float) -> None:
        """Actualiza el transform y reconstruye la cuadrícula (incluye sampling)."""
        self.offset_x     = float(offset_x)
        self.offset_y     = float(offset_y)
        self.rotation_deg = float(rotation_deg)
        old_progress = {t.id: (t.labeled_pct, t.n_points, t.loaded)
                        for t in self.tiles}
        self._rebuild_tiles()
        for t in self.tiles:
            if t.id in old_progress:
                t.labeled_pct, t.n_points, t.loaded = old_progress[t.id]
        self._estimate_from_sample()

    def update_transform_preview(self, offset_x: float, offset_y: float,
                                  rotation_deg: float) -> None:
        """
        Actualiza solo la geometría del grid (sin sampling).
        Para drag en tiempo real: ~1ms vs ~100ms del rebuild completo.
        """
        self.offset_x     = float(offset_x)
        self.offset_y     = float(offset_y)
        self.rotation_deg = float(rotation_deg)
        old_progress = {t.id: (t.labeled_pct, t.n_points, t.loaded)
                        for t in self.tiles}
        self._rebuild_tiles()
        for t in self.tiles:
            if t.id in old_progress:
                t.labeled_pct, t.n_points, t.loaded = old_progress[t.id]
        # NO _estimate_from_sample — los labeled_pct existentes se preservan

    # ── Propiedades ───────────────────────────────────────────────────────────

    @property
    def n_tiles(self) -> int:
        return len(self.tiles)

    @property
    def n_tiles_complete(self) -> int:
        return sum(1 for t in self.tiles if t.labeled_pct >= 95.0)

    @property
    def n_tiles_started(self) -> int:
        return sum(1 for t in self.tiles if 0 < t.labeled_pct < 95.0)

    @property
    def overall_labeled_pct(self) -> float:
        if not self.tiles:
            return 0.0
        return float(np.mean([t.labeled_pct for t in self.tiles]))

    @property
    def mean_z(self) -> float:
        return self._cloud_mean_z

    # ── Geometría ─────────────────────────────────────────────────────────────

    def grid_center_world(self) -> Tuple[float, float]:
        """Centro del grid en coordenadas mundo (útil para rotación con mouse)."""
        cx_r = self._adj_x + self.n_cols * self.tile_size_m / 2.0
        cy_r = self._adj_y + self.n_rows * self.tile_size_m / 2.0
        ox, oy = self.origin_x, self.origin_y
        ca, sa = self.cos_rot, self.sin_rot
        return (ox + cx_r*ca - cy_r*sa, oy + cx_r*sa + cy_r*ca)

    def world_corners(self, tile: TileInfo) -> List[Tuple[float, float, float]]:
        """4 esquinas mundo (+ z) de un tile, en orden anti-horario."""
        ox, oy = self.origin_x, self.origin_y
        ca, sa = self.cos_rot, self.sin_rot
        z = self._cloud_mean_z

        def to_world(xr, yr):
            return (ox + xr*ca - yr*sa,
                    oy + xr*sa + yr*ca,
                    z)

        return [
            to_world(tile.min_xr, tile.min_yr),
            to_world(tile.max_xr, tile.min_yr),
            to_world(tile.max_xr, tile.max_yr),
            to_world(tile.min_xr, tile.max_yr),
        ]

    def tile_at_world_xy(self, wx: float, wy: float) -> Optional[TileInfo]:
        """Devuelve el TileInfo que contiene el punto mundo (wx, wy), o None."""
        dx = wx - self.origin_x
        dy = wy - self.origin_y
        xr =  dx * self.cos_rot + dy * self.sin_rot
        yr = -dx * self.sin_rot + dy * self.cos_rot

        # BUG FIX: tiles empiezan en _adj_x, _adj_y (puede ser negativo con rotación)
        col = int((xr - self._adj_x) / self.tile_size_m)
        row = int((yr - self._adj_y) / self.tile_size_m)
        if 0 <= row < self.n_rows and 0 <= col < self.n_cols:
            return self.tiles[row * self.n_cols + col]
        return None

    def grid_line_endpoints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Devuelve (starts, ends) arrays de float32 (N,3) para dibujar
        las líneas del grid en VTK. Z = mean_z de la nube.
        """
        ox, oy  = self.origin_x, self.origin_y
        ca, sa  = self.cos_rot, self.sin_rot
        s       = self.tile_size_m
        nr, nc  = self.n_rows, self.n_cols
        z       = self._cloud_mean_z

        def rot(xr, yr):
            return [ox + xr*ca - yr*sa, oy + xr*sa + yr*ca, z]

        starts, ends = [], []

        # BUG FIX: las líneas deben empezar en _adj_x, _adj_y (igual que los tiles)
        ax, ay = self._adj_x, self._adj_y

        # Líneas a lo largo de v (columnas del grid)
        for c in range(nc + 1):
            xr = ax + c * s
            starts.append(rot(xr, ay))
            ends.append(rot(xr, ay + nr * s))

        # Líneas a lo largo de u (filas del grid)
        for r in range(nr + 1):
            yr = ay + r * s
            starts.append(rot(ax, yr))
            ends.append(rot(ax + nc * s, yr))

        return (np.array(starts, np.float32),
                np.array(ends,   np.float32))

    # ── Extracción ────────────────────────────────────────────────────────────


    # ── Tile Index ─────────────────────────────────────────────────────────────
    _tile_index: object = None
    _tile_index_built: bool = False

    def _tile_index_path(self):
        """
        Ruta del tile index. El nombre encoda los parametros del grid con
        precision de centimetros para evitar falsos misses por floating point.
        Al reabrir el mismo proyecto (mismos params guardados) = mismo nombre = cache hit.

        BUG CORREGIDO: antes se leían `project.tile_rotation_deg` /
        `project.tile_origin_x` / `project.tile_origin_y` — campos "legacy"
        que NUNCA se actualizan en ningún lugar del código al mover la
        grilla (solo `project.grid_offset_x/grid_offset_y/grid_rotation` se
        mantienen al día, y son campos DISTINTOS). Resultado: tras mover la
        grilla con el mouse o los spinboxes, el nombre del archivo de caché
        no cambiaba (siempre rot=0, ox=0, oy=0), así que se reutilizaba el
        índice punto→tile calculado con la grilla en su posición ANTERIOR —
        el tile que se abría correspondía a la grilla de antes de moverla.
        Ahora se usan siempre los parámetros VIVOS de este TileManager
        (self.tile_size_m / self.rotation_deg / self.offset_x / self.offset_y),
        que sí reflejan la posición actual de la grilla en todo momento.
        """
        try:
            src = getattr(self._pc, '_cache_path', None) or getattr(self._pc, 'source_file', None)
            if src is None: return None
            from pathlib import Path
            ts  = int(round(self.tile_size_m  * 100))  # cm
            rot = int(round(self.rotation_deg * 100))  # 1/100 grado
            ox  = int(round(self.offset_x     * 100))  # cm
            oy  = int(round(self.offset_y     * 100))  # cm
            tag = f"ts{ts}_r{rot}_ox{ox}_oy{oy}"
            p   = Path(src)
            return p.parent / (p.stem + f".ga3d_tileindex_{tag}")
        except Exception:
            return None

    def build_tile_index(self, progress_cb=None) -> bool:
        if self._tile_index_built and self._tile_index is not None:
            return True
        idx_path = self._tile_index_path()
        n = len(self._pc.xyz)
        if idx_path and idx_path.exists():
            try:
                mm = np.memmap(str(idx_path), dtype=np.uint32, mode='r')
                if len(mm) == n:
                    self._tile_index = mm
                    self._tile_index_built = True
                    print(f"[TileIndex] Cargado desde disco — {n/1e6:.0f}M pts")
                    return True
                del mm; idx_path.unlink()
            except Exception:
                pass
        n = len(self._pc.xyz)
        # Pre-alocar el archivo explicitamente (mismo patron que heavy_cloud.py)
        # np.memmap(mode='w+') puede fallar silenciosamente en Windows para archivos grandes
        assignments = None
        if idx_path:
            try:
                size_bytes = int(n) * 4  # uint32 = 4 bytes
                with open(str(idx_path), 'wb') as _f:
                    _f.seek(size_bytes - 1)
                    _f.write(b'\x00')
                # Ahora abrir como mmap r+ (el archivo ya tiene el tamano correcto)
                assignments = np.memmap(str(idx_path), dtype=np.uint32,
                                        mode='r+', shape=(n,))
            except Exception as e:
                print(f"[TileIndex] No se pudo crear archivo en disco: {e}")
                # Limpiar archivo vacio si quedo
                try:
                    if idx_path.exists(): idx_path.unlink()
                except Exception: pass
                assignments = None
        if assignments is None:
            assignments = np.empty(n, dtype=np.uint32)

        ox, oy  = self.origin_x, self.origin_y
        ca, sa   = self.cos_rot, self.sin_rot
        ax, ay   = self._adj_x, self._adj_y
        ts       = self.tile_size_m
        nc, nr   = self.n_cols, self.n_rows
        xyz      = self._pc.xyz
        CHUNK    = 2_000_000
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            xy  = np.array(xyz[start:end, :2], dtype=np.float64)
            dx  = xy[:,0] - ox; dy = xy[:,1] - oy
            xr  =  dx*ca + dy*sa
            yr  = -dx*sa + dy*ca
            col = np.clip(((xr-ax)/ts).astype(np.int32), 0, nc-1)
            row = np.clip(((yr-ay)/ts).astype(np.int32), 0, nr-1)
            assignments[start:end] = (row * nc + col).astype(np.uint32)
            if progress_cb and end % (CHUNK*4) < CHUNK:
                progress_cb(int(90*end/n), f"Índice {end//1_000_000:.0f}M/{n//1_000_000:.0f}M pts…")
        if hasattr(assignments, 'flush'):
            assignments.flush()
            assignments.flush()   # doble flush para garantizar escritura en Windows
        self._tile_index = assignments
        self._tile_index_built = True
        if progress_cb: progress_cb(100, "Índice de tiles listo ✓")
        return True

    def get_tile_indices_fast(self, tile) -> object:
        if not self._tile_index_built or self._tile_index is None:
            return None
        return np.where(self._tile_index == np.uint32(tile.id))[0].astype(np.int64)

    def create_extraction_worker(self, tile: TileInfo) -> TileExtractionWorker:
        return TileExtractionWorker(
            self._pc.xyz, tile,
            self.origin_x, self.origin_y,
            self.cos_rot,  self.sin_rot)

    def update_tile_progress(self, tile: TileInfo,
                              indices: np.ndarray) -> None:
        labels = self._project.labels
        if labels is None or len(indices) == 0:
            tile.labeled_pct = 0.0; return
        lbl_tile = labels[indices]
        n_labeled = int(np.count_nonzero(lbl_tile))
        tile.labeled_pct = 100.0 * n_labeled / len(indices)
        tile.n_points    = len(indices)
        tile.loaded      = True
        # Contar puntos por clase (excluye clase 0 = sin etiquetar)
        unique, counts = np.unique(lbl_tile[lbl_tile > 0], return_counts=True)
        tile.class_counts = {int(u): int(c) for u, c in zip(unique, counts)}

    def refresh_all_progress(self) -> None:
        labels = self._project.labels
        if labels is None: return
        xyz = self._pc.xyz; n = len(xyz)
        step       = max(1, n // 1_000_000)
        sxy        = xyz[::step, :2]
        slbl       = labels[::step]
        dx = sxy[:, 0] - self.origin_x
        dy = sxy[:, 1] - self.origin_y
        xr =  dx * self.cos_rot + dy * self.sin_rot
        yr = -dx * self.sin_rot + dy * self.cos_rot
        col_idx = np.clip(((xr - self._adj_x) / self.tile_size_m).astype(int), 0, self.n_cols-1)
        row_idx = np.clip(((yr - self._adj_y) / self.tile_size_m).astype(int), 0, self.n_rows-1)
        tile_ids = row_idx * self.n_cols + col_idx
        for tile in self.tiles:
            mask = tile_ids == tile.id
            if mask.any():
                t_lbl = slbl[mask]
                tile.labeled_pct = 100.0 * int(np.count_nonzero(t_lbl)) / int(mask.sum())
                unique, counts = np.unique(t_lbl[t_lbl > 0], return_counts=True)
                tile.class_counts = {int(u): int(c) for u, c in zip(unique, counts)}
            else:
                tile.labeled_pct = 0.0
                tile.class_counts = {}

    # ── Privado ───────────────────────────────────────────────────────────────

    def _rebuild_tiles(self) -> None:
        rad = math.radians(self.rotation_deg)
        self.cos_rot = math.cos(rad)
        self.sin_rot = math.sin(rad)

        # Origen: esquina min de la nube + offset del usuario
        self.origin_x = self._cloud_min_x + self.offset_x
        self.origin_y = self._cloud_min_y + self.offset_y

        # Calcular cuántos tiles se necesitan para cubrir la nube
        # proyectando las 4 esquinas al marco rotado
        cloud_corners_w = [
            (self._cloud_min_x, self._cloud_min_y),
            (self._cloud_max_x, self._cloud_min_y),
            (self._cloud_max_x, self._cloud_max_y),
            (self._cloud_min_x, self._cloud_max_y),
        ]
        ca, sa = self.cos_rot, self.sin_rot
        ox, oy = self.origin_x, self.origin_y

        xrs, yrs = [], []
        for wx, wy in cloud_corners_w:
            dx = wx - ox; dy = wy - oy
            xrs.append( dx*ca + dy*sa)
            yrs.append(-dx*sa + dy*ca)

        min_xr = min(xrs); max_xr = max(xrs)
        min_yr = min(yrs); max_yr = max(yrs)

        # Ajustar origen al borde mínimo si está fuera
        adj_x = math.floor(min_xr / self.tile_size_m) * self.tile_size_m
        adj_y = math.floor(min_yr / self.tile_size_m) * self.tile_size_m

        self.n_cols = max(1, math.ceil((max_xr - adj_x) / self.tile_size_m))
        self.n_rows = max(1, math.ceil((max_yr - adj_y) / self.tile_size_m))

        # Guardar adj para usarlo en tile_at_world_xy, grid_line_endpoints, refresh_all_progress
        self._adj_x = adj_x
        self._adj_y = adj_y

        self.tiles = []
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                self.tiles.append(TileInfo(
                    id     = row * self.n_cols + col,
                    row    = row,
                    col    = col,
                    min_xr = adj_x + col * self.tile_size_m,
                    max_xr = adj_x + (col + 1) * self.tile_size_m,
                    min_yr = adj_y + row * self.tile_size_m,
                    max_yr = adj_y + (row + 1) * self.tile_size_m,
                ))

    def _estimate_from_sample(self) -> None:
        try:
            xyz    = self._pc.xyz
            labels = self._project.labels
            n      = len(xyz)
            is_mmap = getattr(self._pc, 'is_mmap', False)
            # Para mmaps grandes: reducir mucho el sample Y leer en chunks
            # para no competir con el LOD worker por las páginas del mmap.
            if is_mmap and n > 200_000_000:
                step = max(1, n // 100_000)   # solo 100K puntos
            elif n > 500_000_000:
                step = max(1, n // 200_000)
            else:
                step = max(1, n // 500_000)
            # Leer en chunks pequeños para no generar demasiados page faults
            # de una vez (compite con el LOD worker)
            indices = np.arange(0, n, step, dtype=np.int64)
            chunk_sz = 10_000
            xy_parts = []
            lbl_parts = []
            for i in range(0, len(indices), chunk_sz):
                chunk_idx = indices[i:i+chunk_sz]
                xy_parts.append(np.array(xyz[chunk_idx, :2], dtype=np.float32))
                if labels is not None:
                    lbl_parts.append(np.array(labels[chunk_idx], dtype=np.uint8))
            sxy  = np.vstack(xy_parts) if xy_parts else np.zeros((0,2), np.float32)
            slbl = np.concatenate(lbl_parts) if lbl_parts else None
        except Exception as e:
            print(f"[TileManager] _estimate_from_sample error: {e}")
            return
        ca, sa = self.cos_rot, self.sin_rot
        ox, oy = self.origin_x, self.origin_y

        dx = sxy[:, 0] - ox; dy = sxy[:, 1] - oy
        xr =  dx*ca + dy*sa
        yr = -dx*sa + dy*ca

        # Ajuste por adj offset (mismo que en _rebuild_tiles)
        if self.tiles:
            adj_x = self.tiles[0].min_xr
            adj_y = self.tiles[0].min_yr
        else:
            adj_x = adj_y = 0.0

        col_idx = np.clip(((xr - adj_x) / self.tile_size_m).astype(int),
                          0, self.n_cols - 1)
        row_idx = np.clip(((yr - adj_y) / self.tile_size_m).astype(int),
                          0, self.n_rows - 1)
        tile_ids = row_idx * self.n_cols + col_idx

        uids, counts = np.unique(tile_ids, return_counts=True)
        for tid, count in zip(uids.tolist(), counts.tolist()):
            if 0 <= tid < len(self.tiles):
                t = self.tiles[tid]
                t.n_points = int(count) * step
                if slbl is not None:
                    mask = tile_ids == tid
                    t.labeled_pct = 100.0 * int(np.count_nonzero(slbl[mask])) / max(int(mask.sum()), 1)
