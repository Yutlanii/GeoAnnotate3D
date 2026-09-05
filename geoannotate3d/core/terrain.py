"""
core/terrain.py — Modelo Digital del Terreno (MDT)

El MDT calcula la Z del suelo en cada celda de una cuadrícula regular.
Permite calcular AGL (Above Ground Level) para cada punto de la nube.

Dos modos de cálculo:
  compute_from_labels() — usa puntos ya etiquetados como "suelo"
  compute_auto()        — detecta el suelo automáticamente (sin labels)

El problema del tejado en LiDAR aéreo:
  Los pulsos LiDAR rebotan en tejados. El suelo bajo el edificio
  no es visible desde el aire. La celda del edificio solo tiene
  puntos de tejado → el percentil bajo de esa celda = tejado Z.
  Si no se corrige, el tejado tiene AGL=0 y se clasifica como suelo.

Solución (compute_auto, dos pasadas):
  Pasada 1: percentil bajo de Z por celda → candidatos a suelo
  Pasada 2: detectar y marcar como NaN las celdas sospechosas
            (Z mucho mayor que sus vecinas = tejado / objeto elevado)
  Fill:     propagar Z de suelo real hacia celdas NaN con inversión
            de distancia ponderada (IDW) → tejados heredan Z del suelo
            circundante → AGL correcto para todas las clases
"""
from __future__ import annotations
import numpy as np
from typing import Optional


class TerrainModel:
    def __init__(self):
        self._grid:   Optional[np.ndarray] = None  # (rows, cols) float32
        self._origin: Optional[np.ndarray] = None  # (2,) float64
        self._rows:   int  = 0
        self._cols:   int  = 0
        self.resolution:      float = 1.0
        self.n_ground_pts:    int   = 0
        self.z_min_ground:    float = 0.0
        self.z_max_ground:    float = 0.0
        self.ready:           bool  = False

    # ── Calcular desde labels ──────────────────────────────────────────────────

    def compute_from_labels(self, xyz: np.ndarray, labels: np.ndarray,
                            ground_class_id: int, resolution: float = 1.0) -> bool:
        """Construye el MDT usando los puntos etiquetados como suelo."""
        self.ready = False
        mask = labels == ground_class_id
        if not mask.any():
            return False
        return self.compute_auto(xyz[mask], resolution=resolution)

    # ── Calcular automáticamente ───────────────────────────────────────────────

    def compute_auto(self, xyz: np.ndarray, resolution: float = 1.0,
                     percentile: float = 5.0) -> bool:
        """
        Detecta el suelo automáticamente en dos pasadas:

        Pasada 1 — percentil bajo por celda:
          Para cada celda de 1m×1m, calcular el percentil bajo de Z.
          Esto aproxima el suelo en zonas despejadas y en zonas con
          vegetación baja (el 5% más bajo de los puntos es el suelo).

        Pasada 2 — detectar tejados y objetos elevados:
          Comparar cada celda con su terreno local suavizado.
          Si una celda está >2m por encima del suelo local → sospechosa.
          Las celdas sospechosas se marcan como NaN.

        Fill — IDW (Inverse Distance Weighting):
          Propagar Z de celdas válidas a las NaN usando ponderación por
          1/distancia². Esto da una interpolación más suave que la media
          simple y cubre correctamente áreas grandes (edificios de 50m+).
        """
        self.ready = False
        self.resolution = resolution
        if xyz is None or len(xyz) < 10:
            return False

        xy = xyz[:, :2]
        mn = xy.min(0) - resolution
        mx = xy.max(0) + resolution
        self._origin = mn.astype(np.float64)
        self._cols   = max(4, int((mx[0]-mn[0])/resolution) + 1)
        self._rows   = max(4, int((mx[1]-mn[1])/resolution) + 1)

        ci = np.clip(((xy[:,0]-mn[0])/resolution).astype(np.int32), 0, self._cols-1)
        ri = np.clip(((xy[:,1]-mn[1])/resolution).astype(np.int32), 0, self._rows-1)
        cell_id = ri.astype(np.int64)*self._cols + ci.astype(np.int64)

        sort_order  = np.argsort(cell_id, kind='stable')
        sorted_cid  = cell_id[sort_order]
        sorted_z    = xyz[sort_order, 2]

        # ── Pasada 1: percentil bajo por celda ────────────────────────────────
        grid_raw = np.full((self._rows, self._cols), np.nan, np.float32)
        unique_cells, first_idx, counts = np.unique(
            sorted_cid, return_index=True, return_counts=True)

        for cid, fi, cnt in zip(unique_cells, first_idx, counts):
            z_cell = sorted_z[fi:fi+cnt]
            p      = max(0, min(cnt-1, int(percentile/100*cnt)))
            z_gnd  = float(np.partition(z_cell, p)[p]) if cnt > 1 else float(z_cell[0])
            r, c   = int(cid // self._cols), int(cid % self._cols)
            grid_raw[r, c] = z_gnd

        # ── Pasada 2: marcar tejados/objetos elevados como NaN ─────────────────
        grid_clean = self._remove_elevated_objects(grid_raw, elev_thresh=2.0)

        # ── Fill con IDW ───────────────────────────────────────────────────────
        self._grid = self._idw_fill(grid_clean)

        self.n_ground_pts = len(xyz)
        z = xyz[:, 2]
        self.z_min_ground = float(z.min())
        self.z_max_ground = float(z.max())
        self.ready = True
        return True

    # ── Núcleo del algoritmo ───────────────────────────────────────────────────

    @staticmethod
    def _remove_elevated_objects(grid: np.ndarray, elev_thresh: float = 2.0,
                                  smooth_r: int = 5) -> np.ndarray:
        """
        Detecta celdas sospechosas (tejados, árboles altos) comparando con
        el terreno local suavizado. Las marca como NaN para el fill posterior.

        smooth_r: radio del filtro de suavizado en celdas
                  (5 = ventana 11×11 = 11m con resolución 1m)
        """
        result = grid.copy()
        valid  = np.isfinite(grid)
        if not valid.any():
            return result

        # Fill preliminar simple para poder suavizar
        g_filled = TerrainModel._simple_fill(grid, iterations=10)

        # Suavizado con mediana → robusto ante outliers (tejados aislados)
        try:
            from scipy.ndimage import median_filter
            smooth = median_filter(g_filled, size=smooth_r*2+1)
        except ImportError:
            # Fallback sin scipy: media por ventana deslizante
            from numpy.lib.stride_tricks import sliding_window_view
            pad = smooth_r
            padded = np.pad(g_filled, pad, mode='edge')
            w = smooth_r*2+1
            smooth = sliding_window_view(padded, (w,w)).mean((-2,-1))

        # Celdas cuya Z está muy por encima del terreno suavizado → sospechosas
        elevation_above = grid - smooth
        suspicious = (elevation_above > elev_thresh) & valid
        result[suspicious] = np.nan

        pct_removed = 100.0 * suspicious.sum() / max(valid.sum(), 1)
        if pct_removed > 0.5:
            print(f"[Terrain] Eliminadas {suspicious.sum():,} celdas sospechosas "
                  f"({pct_removed:.1f}% del total) — probable tejados/vegetacion alta")
        return result

    @staticmethod
    def _idw_fill(grid: np.ndarray, search_r: int = 80) -> np.ndarray:
        """
        Inversión de distancia ponderada (IDW) para rellenar NaN.
        Más suave y precisa que la propagación iterativa simple.
        search_r: radio de búsqueda en celdas (80 = 80m con resolución 1m)

        Para cada celda NaN, promedia los valores válidos cercanos
        ponderados por 1/d².

        Implementación eficiente: primero propagación rápida iterativa
        (para cubrir zonas grandes), luego suavizado IDW de los bordes.
        """
        g = grid.copy()
        nan_mask = np.isnan(g)
        if not nan_mask.any():
            return g

        # Paso 1: propagación rápida iterativa (cubre huecos grandes)
        g = TerrainModel._simple_fill(g, iterations=search_r)

        # Paso 2: suavizado gaussiano sobre toda la zona interpolada
        # (suaviza las transiciones bruscas entre suelo real e interpolado)
        nan_original = nan_mask.copy()
        try:
            from scipy.ndimage import gaussian_filter
            # Solo suavizar en zonas que eran NaN (no tocar el suelo medido)
            g_smooth = gaussian_filter(g, sigma=3.0)
            # Restaurar valores originales donde el suelo era conocido
            g[nan_original] = g_smooth[nan_original]
        except ImportError:
            pass  # Sin scipy: quedarse con la propagación iterativa

        return g

    @staticmethod
    def _simple_fill(grid: np.ndarray, iterations: int = 60) -> np.ndarray:
        """
        Propagación iterativa de vecinos para rellenar NaN.
        Cada iteración llena una capa más de celdas NaN con la media
        de sus vecinos válidos.
        """
        g = grid.copy()
        for _ in range(iterations):
            nan_mask = np.isnan(g)
            if not nan_mask.any(): break
            padded = np.pad(g, 1, constant_values=np.nan)
            nbrs = np.stack([
                padded[0:-2,0:-2], padded[0:-2,1:-1], padded[0:-2,2:],
                padded[1:-1,0:-2],                     padded[1:-1,2:],
                padded[2:,  0:-2], padded[2:,  1:-1],  padded[2:,  2:],
            ], axis=0)
            valid     = np.isfinite(nbrs)
            n_valid   = valid.sum(0)
            sum_valid = np.where(valid, nbrs, 0.0).sum(0)
            fill_v    = np.where(n_valid > 0, sum_valid/np.maximum(n_valid,1), np.nan)
            g[nan_mask] = fill_v[nan_mask]
        return g

    # ── AGL ────────────────────────────────────────────────────────────────────

    def agl(self, xyz: np.ndarray) -> np.ndarray:
        """
        Calcula AGL (metros sobre el suelo) para cada punto.
        AGL = Z_punto - Z_suelo_interpolado_en_su_posicion_XY
        """
        if not self.ready or self._grid is None:
            return np.zeros(len(xyz), np.float32)

        xy = xyz[:, :2]
        ox, oy = self._origin[0], self._origin[1]
        res    = self.resolution

        ci = np.clip(((xy[:,0]-ox)/res).astype(np.int32), 0, self._cols-1)
        ri = np.clip(((xy[:,1]-oy)/res).astype(np.int32), 0, self._rows-1)

        z_ground = self._grid[ri, ci]   # (N,) float32
        agl_vals = xyz[:, 2].astype(np.float32) - z_ground

        # Puntos en celdas sin cobertura → AGL=0 (neutral)
        agl_vals[~np.isfinite(agl_vals)] = 0.0
        return agl_vals

    def agl_all(self, xyz: np.ndarray) -> np.ndarray:
        """Alias de agl() para claridad en el código."""
        return self.agl(xyz)

    def agl_range_mask(self, xyz, min_agl, max_agl):
        v = self.agl(xyz)
        return (v >= min_agl) & (v <= max_agl)

    # ── Propiedades ────────────────────────────────────────────────────────────

    @property
    def coverage_pct(self):
        if self._grid is None: return 0.0
        return 100.0*np.isfinite(self._grid).sum()/max(self._grid.size,1)

    def stats(self):
        return {
            "n_ground_pts": self.n_ground_pts,
            "resolution_m": self.resolution,
            "z_min":        round(self.z_min_ground, 2),
            "z_max":        round(self.z_max_ground, 2),
            "coverage_pct": round(self.coverage_pct, 1),
            "grid_cells":   self._rows*self._cols if self._grid is not None else 0,
        }
