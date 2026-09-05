"""
utils/spatial.py — Funciones geométricas optimizadas v2.0

Cambios vs v1.0:
  - sphere_query_octree → usa cKDTree del Octree (400-4000x más rápido)
  - sphere_query brute → usa C extension cuando disponible
  - frustum_cull_points → C extension (per-point filtering antes de GPU)
  - density_decimate → sin cambios (ya era vectorizado)
  - extract_frustum_planes → cache de matriz MVP para evitar 32 GetElement calls
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

# C fast module
try:
    from core._fast import sphere_query_fast as _c_sphere, frustum_cull_fast as _c_frustum
    _FAST = True
except ImportError:
    _FAST = False


# ── Frustum culling ───────────────────────────────────────────────────────────

def extract_frustum_planes(mvp: np.ndarray) -> np.ndarray:
    """Extrae 6 planos del frustum de la MVP (Gribb-Hartmann)."""
    planes = np.zeros((6,4), np.float32)
    planes[0] = mvp[3] + mvp[0]
    planes[1] = mvp[3] - mvp[0]
    planes[2] = mvp[3] + mvp[1]
    planes[3] = mvp[3] - mvp[1]
    planes[4] = mvp[3] + mvp[2]
    planes[5] = mvp[3] - mvp[2]
    norms = np.linalg.norm(planes[:,:3], axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    return planes / norms


def aabb_in_frustum(bb_min: np.ndarray, bb_max: np.ndarray,
                    planes: np.ndarray) -> bool:
    """Test AABB vs 6 planos — vectorizado, sin bucle Python."""
    nx = planes[:,0]; ny = planes[:,1]; nz = planes[:,2]; d = planes[:,3]
    px = np.where(nx>=0, bb_max[0], bb_min[0])
    py = np.where(ny>=0, bb_max[1], bb_min[1])
    pz = np.where(nz>=0, bb_max[2], bb_min[2])
    return bool(np.all(nx*px + ny*py + nz*pz + d >= 0))


def frustum_cull_points(xyz: np.ndarray, planes: np.ndarray) -> np.ndarray:
    """
    Filtra puntos fuera del frustum. Retorna máscara booleana (N,).
    Usa C extension cuando disponible.
    """
    if _FAST:
        from core._fast import frustum_cull_fast
        return frustum_cull_fast(xyz, planes)
    # Numpy fallback
    nx=planes[:,0]; ny=planes[:,1]; nz=planes[:,2]; d=planes[:,3]
    inside = np.ones(len(xyz), bool)
    for p in range(6):
        dot = xyz[:,0]*nx[p] + xyz[:,1]*ny[p] + xyz[:,2]*nz[p] + d[p]
        inside &= dot >= 0
    return inside


# ── Density decimation ────────────────────────────────────────────────────────

def density_decimate(xyz_all, transform, W, H, max_per_cell=4, grid_size=64):
    """Decimación screen-space vectorizada (sin cambios)."""
    n = len(xyz_all)
    if n == 0:
        return np.ones(0, bool)
    try:
        sc = transform.map(xyz_all)
        if sc.shape[1] == 4:
            w  = sc[:,3:4]; w = np.where(np.abs(w)<1e-9, 1.0, w)
            px = sc[:,0]/w[:,0]; py = sc[:,1]/w[:,0]
        else:
            px, py = sc[:,0], sc[:,1]
        gx = np.clip((px/W*grid_size).astype(np.int32), 0, grid_size-1)
        gy = np.clip((py/H*grid_size).astype(np.int32), 0, grid_size-1)
        cell_key = (gx*grid_size+gy).astype(np.int32)
        order   = np.random.permutation(n).astype(np.int32)
        ck_rand = cell_key[order]
        si      = np.argsort(ck_rand, kind='stable')
        ck_sorted = ck_rand[si]
        group_starts = np.zeros(n, dtype=np.int32)
        boundaries   = np.concatenate([[0], np.where(np.diff(ck_sorted)!=0)[0]+1])
        group_starts[boundaries] = boundaries
        np.maximum.accumulate(group_starts, out=group_starts)
        rank_in_group = np.arange(n, dtype=np.int32) - group_starts
        keep_sorted   = rank_in_group < max_per_cell
        keep = np.zeros(n, bool)
        keep[order[si[keep_sorted]]] = True
        return keep
    except Exception:
        return np.ones(n, bool)


# ── Sphere query ──────────────────────────────────────────────────────────────

def sphere_query(xyz: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Búsqueda radial bruta O(N) — usar sphere_query_octree cuando sea posible."""
    if _FAST:
        return _c_sphere(xyz, center, radius)
    diff  = xyz - center.astype(xyz.dtype)
    dist2 = (diff*diff).sum(1)
    return np.where(dist2 <= radius*radius)[0].astype(np.int32)


def sphere_query_octree(octree, center: np.ndarray, radius: float) -> np.ndarray:
    """
    Sphere query acelerado: grid en C → cKDTree → BFS octree → fuerza bruta.

    BUG CORREGIDO (causaba que el Pincel/Esfera/Disco no seleccionaran NADA
    en nubes grandes): antes esta función solo comprobaba `_kdtree_ready`
    para decidir si usar el índice rápido, pero Octree.build() apaga
    `_kdtree_ready` a propósito cuando construye el grid en C (>30M pts,
    "prefer grid"). Y el siguiente `if hasattr(octree, '_sphere_query_octree_bfs')`
    es SIEMPRE verdadero (el método existe aunque no haya BFS construido),
    así que nunca llegaba ni al grid ni a la fuerza bruta real: para toda
    nube >=30M pts (grid preferido) y >=50M pts (sin BFS, ver
    BFS_SKIP_THRESHOLD en core/octree.py) devolvía SIEMPRE un array vacío.
    """
    if octree is None or not octree.ready or octree._xyz_ref is None:
        return np.zeros(0, np.int32)

    # 1) Grid en C (el más rápido para nubes grandes, construido en Octree.build)
    if getattr(octree, '_grid_ready', False) and octree._grid is not None:
        try:
            from core._fast import FC
            return FC.grid_sphere_query(
                octree._grid, octree._xyz_ref,
                float(center[0]), float(center[1]), float(center[2]), float(radius))
        except Exception:
            pass

    # 2) cKDTree
    if getattr(octree, '_kdtree_ready', False) and octree._kdtree is not None:
        try:
            local = octree._kdtree.query_ball_point(
                np.asarray(center, np.float64), float(radius))
            if not local:
                return np.zeros(0, np.int32)
            la = np.array(local, dtype=np.int32)
            return octree._kdtree_idx[la] if octree._kdtree_idx is not None else la
        except Exception:
            pass

    # 3) BFS octree — solo si realmente se construyó (nubes < 50M pts)
    if getattr(octree, 'root', None) is not None:
        return octree._sphere_query_octree_bfs(center, radius)

    # 4) Fuerza bruta como último recurso genuino
    return sphere_query(octree._xyz_ref, center, radius)


# ── Ray casting ───────────────────────────────────────────────────────────────

def screen_to_world_ray(px: float, py: float,
                         inv_mvp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convierte coordenadas de pantalla → rayo 3D."""
    near = inv_mvp @ np.array([px, py, -1.0, 1.0], np.float32)
    far  = inv_mvp @ np.array([px, py,  1.0, 1.0], np.float32)
    near = near[:3] / (near[3] + 1e-9)
    far  = far[:3]  / (far[3]  + 1e-9)
    direction = far - near
    norm = np.linalg.norm(direction)
    if norm > 1e-9:
        direction /= norm
    return near.astype(np.float32), direction.astype(np.float32)
