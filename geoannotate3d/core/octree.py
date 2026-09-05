"""
core/octree.py — Octree + LOD progresivo v9.0

Cambios vs v8.0:
  - EXTENDED LOD: 8 niveles [50K → 500K → 3M → 10M → 30M → 80M → 150M → N]
    Para 185M pts con budget 20M → LOD nivel 4 (30M), NO octree traversal.
  - BUILD OPTIMIZADO: para nubes >50M, salta octree BFS (ahorra 30-60s)
    Solo construye LOD + KDTree. El octree BFS solo es útil para
    refinamiento view-dependent que ya no es la ruta principal.
  - iter_lod_progression(): generador simple LOD0→LOD1→...→budget
    Sin heapq, sin Python loops sobre nodos. Cada yield = un LOD completo.
  - Octree BFS solo se construye si n < 50M (nubes pequeñas/medianas)
"""
from __future__ import annotations

import math
import time
from collections import deque
from typing import Optional, List, Tuple, Generator

import numpy as np

try:
    from core._fast import build_lod_levels, voxel_subsample_fast, _USE_C
    _FAST_AVAILABLE = _USE_C
except ImportError:
    _FAST_AVAILABLE = False

try:
    from scipy.spatial import cKDTree as _cKDTree
    _KDTREE_AVAILABLE = True
except ImportError:
    _KDTREE_AVAILABLE = False

OCTREE_MAX_DEPTH     = 10
OCTREE_MIN_POINTS    = 8
OCTREE_NODE_CAPACITY = 512
SSE_LOD_THRESHOLD    = 2.5

# Extended LOD targets — cubre desde 50K hasta N
LOD_TARGETS_BASE = [50_000, 500_000, 3_000_000, 15_000_000]
LOD_TARGETS_EXT  = [50_000, 500_000, 3_000_000, 10_000_000,
                    30_000_000, 80_000_000, 150_000_000]
# Backwards compat
LOD_TARGETS = LOD_TARGETS_BASE

# Umbral para saltar octree BFS (nubes muy grandes)
BFS_SKIP_THRESHOLD = 50_000_000


class OctreeNode:
    __slots__ = ('bb_min', 'bb_max', 'center', 'half_size',
                 'pts_idx', 'children', 'depth', 'n_total', 'is_leaf')

    def __init__(self, bb_min, bb_max, depth):
        self.bb_min    = bb_min.astype(np.float32)
        self.bb_max    = bb_max.astype(np.float32)
        self.center    = ((bb_min + bb_max) * 0.5).astype(np.float32)
        self.half_size = float(((bb_max - bb_min).max()) * 0.5)
        self.pts_idx   = None
        self.children  = [None] * 8
        self.depth     = depth
        self.n_total   = 0
        self.is_leaf   = False


class Octree:
    def __init__(self):
        self.root      = None
        self.ready     = False
        self.n_total   = 0
        self._xyz_ref  = None
        self._lod_levels  = []      # idx arrays, de grueso a fino
        self._lod_ready   = False
        self._kdtree      = None
        self._kdtree_idx  = None
        self._kdtree_ready = False
        self._grid        = None    # C grid index (for huge clouds)
        self._grid_ready  = False

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, xyz, cb=None):
        n = len(xyz)
        if n == 0:
            return
        self.n_total  = n
        self._xyz_ref = xyz

        # Activar faulthandler: si hay SEGFAULT, Python escribe un traceback
        # antes de morir en lugar de cerrar silenciosamente.
        # Esto no evita el crash pero da información de diagnóstico.
        try:
            import faulthandler, sys
            if not faulthandler.is_enabled():
                faulthandler.enable(file=sys.stderr)
        except Exception:
            pass

        # Fase 1: LOD extendido — 7-8 niveles para cubrir cualquier budget
        if cb: cb(5, "LOD progresivo…")
        t0 = time.perf_counter()
        self._build_extended_lod(xyz)
        t1 = time.perf_counter()
        if cb: cb(30, f"LOD listo ({t1-t0:.2f}s) — {len(self._lod_levels)} niveles ✓")

        # Fase 2: Octree BFS solo para nubes <50M (para SSE refinement)
        if n < BFS_SKIP_THRESHOLD:
            if cb: cb(35, "Octree BFS…")
            self._build_octree_bfs(xyz, cb)
        else:
            if cb: cb(35, f"Nube grande ({n/1e6:.0f}M) — usando LOD directo")

        # Fase 3: Spatial index for sphere queries
        # For very large clouds (>30M): use grid index in C (25x less memory than KDTree)
        # SAFETY: grid_build allocates ~N*8 bytes contiguously in C (hash table).
        # For 2000M points that's 16GB — skip if estimated RAM would be unsafe.
        grid_safe = True
        if n > 30_000_000:
            try:
                avail_gb = _available_ram_gb()
                needed_gb = n * 8 / 1e9   # ~8 bytes/pt for grid hash
                grid_safe = needed_gb < avail_gb * 0.40
            except Exception:
                grid_safe = n < 500_000_000  # conservative fallback

        if n > 30_000_000 and grid_safe:
            if cb: cb(92, "Grid index (C)…")
            try:
                from core._fast import FC
                # Cell size ~= average expected brush radius (5m typical)
                self._grid = FC.grid_build(xyz, 5.0)
                self._grid_ready = self._grid is not None
                if self._grid_ready:
                    self._kdtree_ready = False  # prefer grid
            except Exception as e:
                print(f"[Octree] Grid index: {e}")
        elif n > 30_000_000 and not grid_safe:
            print(f"[Octree] Saltando grid_build ({n/1e6:.0f}M pts, "                  f"necesita {n*8/1e9:.1f}GB — inseguro). "                  "Las herramientas de anotación usarán búsqueda directa.")

        if _KDTREE_AVAILABLE and not getattr(self, '_grid_ready', False):
            if cb: cb(92, "cKDTree…")
            try:
                if n > 10_000_000:
                    # Usar LOD más fino disponible que quepa en RAM
                    best_lod = self._lod_levels[-1] if self._lod_levels else None
                    for lv in reversed(self._lod_levels):
                        if len(lv) <= 15_000_000:
                            best_lod = lv; break
                    if best_lod is not None and len(best_lod) > 0:
                        self._kdtree     = _cKDTree(xyz[best_lod])
                        self._kdtree_idx = best_lod
                    else:
                        step = max(1, n // 10_000_000)
                        sub  = np.arange(0, n, step, dtype=np.int32)
                        self._kdtree     = _cKDTree(xyz[sub])
                        self._kdtree_idx = sub
                else:
                    self._kdtree     = _cKDTree(xyz)
                    self._kdtree_idx = None
                self._kdtree_ready = True
            except Exception as e:
                print(f"[Octree] cKDTree: {e}")

        self.ready = True
        if cb: cb(100, "Octree listo ✓")

    def _build_extended_lod(self, xyz):
        """
        Construye LOD levels via C (build_all_lod_levels).

        DISEÑO SEGURO PARA CUALQUIER MÁQUINA:
        ─────────────────────────────────────
        El crash sin aviso ocurre cuando FC.build_all_lod_levels intenta
        allocar bloques de memoria contigua muy grandes en C (via malloc).
        Cuando malloc() devuelve NULL, el código C hace SEGFAULT al escribir
        en ptr=NULL → el proceso termina sin que Python pueda atrapar nada.

        Solución: calcular cuánta RAM contigua puede tolerar la máquina
        ANTES de llamar al código C, y nunca pedir más de eso.
        La vista completa (LOD de N puntos) solo se construye si cabe.
        En cualquier otro caso el mayor nivel disponible (ej: 150M pts)
        es suficiente para llenar cualquier pantalla.

        Regla: para construir LOD de M puntos se necesitan ~M*12 bytes
        contiguos en RAM (hash int64 + índice int32 + trabajo interno).
        Nunca pedimos más del 60% de la RAM libre actual.
        """
        n = len(xyz)

        # ── Pre-check de memoria disponible ──────────────────────────────────
        # Estimar cuántos puntos podemos indexar de forma segura.
        # Fórmula: FC necesita ~12 bytes/punto contiguos (hash+idx+temporal)
        try:
            avail_gb = _available_ram_gb()
        except Exception:
            avail_gb = 8.0  # fallback: asumir 8GB disponibles
        try:
            # Usar máximo 60% de la RAM libre para el LOD build
            safe_bytes = avail_gb * 0.60 * 1e9
            max_safe_pts = int(safe_bytes / 12)
        except Exception:
            max_safe_pts = 100_000_000  # fallback conservador: 100M

        # ── Construir lista de targets ────────────────────────────────────────
        targets = [t for t in LOD_TARGETS_EXT if t < n]

        # Solo incluir el nivel completo (N puntos) si cabe en RAM segura
        if n <= max_safe_pts:
            targets.append(n)
        else:
            # No incluir LOD[N]: su construcción causaría SEGFAULT en C.
            # El nivel más fino de LOD_TARGETS_EXT (150M) ya llena cualquier pantalla.
            print(f"[Octree] Nube grande ({n/1e6:.0f}M pts): "
                  f"LOD max={targets[-1]/1e6:.0f}M (RAM segura: {avail_gb:.0f}GB × 60%)")

        if not targets:
            # Nube muy pequeña, ningún target < n
            targets = [n]

        # ── Intentar build en C — niveles de fallback ─────────────────────────
        self._lod_levels = []

        while targets:
            try:
                from core._fast import FC
                self._lod_levels = FC.build_all_lod_levels(xyz, targets)
                break  # éxito

            except MemoryError:
                # MemoryError de Python/NumPy: quitar el nivel más grande y reintentar
                if len(targets) > 1:
                    removed = targets.pop()
                    print(f"[Octree] MemoryError: quitando LOD {removed/1e6:.0f}M, reintentando")
                else:
                    break

            except Exception as e:
                # Cualquier otro error C: caer a Python puro
                print(f"[Octree] Error en build C ({e}), usando numpy fallback")
                for tgt in targets:
                    try:
                        self._lod_levels.append(_voxel_filter_numpy(xyz, tgt))
                    except Exception:
                        pass
                break

        # Si no se construyó nada: submuestreo básico como último recurso
        if not self._lod_levels:
            print("[Octree] Usando submuestreo de emergencia")
            for tgt in [500_000, 3_000_000, 10_000_000]:
                if tgt < n:
                    step = max(1, n // tgt)
                    self._lod_levels.append(
                        np.arange(0, n, step, dtype=np.int32))

        self._lod_ready = True

    def _build_octree_bfs(self, xyz, cb=None):
        n = len(xyz)
        bb_min = xyz.min(0).astype(np.float32)
        bb_max = xyz.max(0).astype(np.float32)
        ext    = (bb_max - bb_min).max()
        center = (bb_min + bb_max) * 0.5
        bb_min = center - ext * 0.501
        bb_max = center + ext * 0.501

        root    = OctreeNode(bb_min, bb_max, 0)
        all_idx = np.arange(n, dtype=np.int32)
        stack   = deque([(root, all_idx)])
        done    = 0
        rng     = np.random.default_rng(42)

        while stack:
            node, idx = stack.popleft()
            node.n_total = len(idx)
            if len(idx) <= OCTREE_MIN_POINTS or node.depth >= OCTREE_MAX_DEPTH:
                node.is_leaf = True; node.pts_idx = idx; done += 1; continue

            n_node       = min(OCTREE_NODE_CAPACITY, len(idx))
            node.pts_idx = _voxel_filter_numpy(xyz[idx], n_node, rng=rng, global_idx=idx)

            cx, cy, cz = node.center[0], node.center[1], node.center[2]
            pts = xyz[idx]
            oct_mask = (
                ((pts[:,0] >= cx).astype(np.uint8)      ) |
                ((pts[:,1] >= cy).astype(np.uint8) << 1 ) |
                ((pts[:,2] >= cz).astype(np.uint8) << 2 ))
            del pts

            for cid in range(8):
                m = oct_mask == cid
                if not m.any(): continue
                ci = idx[m]
                cmin = np.array([cx if (cid&1) else node.bb_min[0],
                                 cy if (cid&2) else node.bb_min[1],
                                 cz if (cid&4) else node.bb_min[2]], np.float32)
                cmax = np.array([node.bb_max[0] if (cid&1) else cx,
                                 node.bb_max[1] if (cid&2) else cy,
                                 node.bb_max[2] if (cid&4) else cz], np.float32)
                ch = OctreeNode(cmin, cmax, node.depth + 1)
                node.children[cid] = ch
                stack.append((ch, ci))

            done += 1
            if cb and done % 2000 == 0:
                cb(min(90, 35 + done // 200), f"Octree: {done} nodos…")

        self.root = root

    # ── LOD O(1) ──────────────────────────────────────────────────────────────

    def get_coarse_view(self, target_pts):
        if self._xyz_ref is None:
            return np.zeros((0,3), np.float32), np.zeros(0, np.int32)
        idx = self._select_lod_idx(target_pts)
        if len(idx) == 0:
            return np.zeros((0,3), np.float32), np.zeros(0, np.int32)
        return self._xyz_ref[idx], idx

    def _select_lod_idx(self, budget):
        if not self._lod_ready or not self._lod_levels:
            step = max(1, self.n_total // max(budget, 1))
            return np.arange(0, self.n_total, step, dtype=np.int32)
        # Buscar nivel con más puntos que quepa en budget
        for lv in reversed(self._lod_levels):
            if len(lv) <= budget:
                return lv
        # Si todos son > budget, sub-sample el más grueso
        coarsest = self._lod_levels[0]
        step = max(1, len(coarsest) // budget)
        return coarsest[::step].astype(np.int32)

    def select_lod(self, eye3d, frustum_planes, W, H, budget, fov_deg):
        return self.get_coarse_view(budget)

    # ── Progressive LOD iteration (reemplaza heapq) ──────────────────────────

    def iter_lod_progression(self, budget):
        """
        Generador que yield LOD niveles progresivamente de grueso a fino.

        is_done=True cuando:
          - Emitimos el último nivel completo
          - O cuando subsampling porque budget es menor que el nivel
            (signal al canvas para crecer budget → DONE → grow → IDLE)
        """
        if not self._lod_ready or not self._lod_levels:
            step = max(1, self.n_total // max(budget, 1))
            idx = np.arange(0, self.n_total, step, dtype=np.int32)
            yield idx[:min(len(idx), budget)], True
            return

        import math as _math
        last_n = 0
        n_levels = len(self._lod_levels)

        for i, lod_idx in enumerate(self._lod_levels):
            n = len(lod_idx)
            is_last_level = (i == n_levels - 1)

            if n > budget:
                # Subsample para caber en budget
                step = int(_math.ceil(n / budget))
                sub = lod_idx[::step].copy()
                # is_done=True: terminamos este ciclo. Si hay niveles más
                # finos, _lod_tick verá que budget < max_lod y crecerá budget.
                yield sub, True
                return

            if n > last_n * 1.15:
                yield lod_idx, is_last_level
                last_n = n

            if is_last_level:
                return

    # ── Legacy iter_refinement (para nubes <50M con octree BFS) ───────────────

    def iter_refinement(self, eye3d, frustum_planes, W, H, fov_deg, max_pts):
        """Refinamiento basado en octree — solo para nubes <50M."""
        if self._xyz_ref is None: return

        # Si no hay octree BFS (nube grande), usar LOD progresivo
        if self.root is None:
            for lod_idx, is_done in self.iter_lod_progression(max_pts):
                xyz_out = self._xyz_ref[lod_idx]
                yield xyz_out, lod_idx, is_done
                if is_done: return
            return

        from utils.spatial import aabb_in_frustum
        import heapq

        xyz     = self._xyz_ref
        fov_rad = math.radians(max(1.0, fov_deg))
        focal   = H / (2.0 * math.tan(fov_rad / 2.0) + 1e-9)
        eye     = eye3d.astype(np.float32)
        total   = 0

        def _priority(node):
            diff = node.center - eye
            dist = max(math.sqrt(float(diff[0]**2 + diff[1]**2 + diff[2]**2)), 1e-6)
            return (node.half_size * focal) / dist

        heap = []
        pri  = _priority(self.root)
        heapq.heappush(heap, (-pri, id(self.root), self.root))

        batch_parts = []
        batch_count = 0
        BATCH_EMIT  = 200_000

        while heap and total < max_pts:
            neg_pri, _, node = heapq.heappop(heap)
            sse = -neg_pri

            if frustum_planes is not None:
                if not aabb_in_frustum(node.bb_min, node.bb_max, frustum_planes):
                    continue

            if node.pts_idx is not None and len(node.pts_idx) > 0:
                take = min(len(node.pts_idx), max_pts - total)
                batch_parts.append(node.pts_idx[:take])
                total      += take
                batch_count += take

            if sse > SSE_LOD_THRESHOLD and not node.is_leaf:
                for ch in node.children:
                    if ch is not None:
                        cp = _priority(ch)
                        heapq.heappush(heap, (-cp, id(ch), ch))

            if batch_count >= BATCH_EMIT or (not heap and batch_parts):
                if batch_parts:
                    idx_b   = np.concatenate(batch_parts)
                    is_done = (not heap) or (total >= max_pts)
                    yield xyz[idx_b], idx_b, is_done
                    if is_done: return
                    batch_parts = []
                    batch_count = 0

        if batch_parts:
            idx_b = np.concatenate(batch_parts)
            yield xyz[idx_b], idx_b, True

    # ── Sphere query: cKDTree → octree → brute force ──────────────────────────

    def sphere_query_kdtree(self, center, radius):
        # Priority: C grid index (fastest for huge clouds)
        if self._grid_ready and self._grid is not None:
            try:
                from core._fast import FC
                return FC.grid_sphere_query(
                    self._grid, self._xyz_ref,
                    float(center[0]), float(center[1]), float(center[2]),
                    float(radius))
            except Exception as e:
                print(f"[Octree] grid query fallback: {e}")
        # Fallback: KDTree
        if self._kdtree_ready and self._kdtree is not None:
            try:
                local = self._kdtree.query_ball_point(
                    center.astype(np.float64), float(radius))
                if not local:
                    return np.zeros(0, np.int32)
                la = np.array(local, dtype=np.int32)
                return self._kdtree_idx[la] if self._kdtree_idx is not None else la
            except Exception:
                pass
        return self._sphere_query_octree_bfs(center, radius)

    def _sphere_query_octree_bfs(self, center, radius):
        if self.root is None or self._xyz_ref is None:
            return np.zeros(0, np.int32)
        xyz = self._xyz_ref; c = center.astype(np.float32)
        r2 = float(radius*radius); r = float(radius); res = []
        q = deque([self.root])
        while q:
            node = q.popleft()
            if node is None: continue
            dx = max(float(node.bb_min[0])-c[0], 0., c[0]-float(node.bb_max[0]))
            dy = max(float(node.bb_min[1])-c[1], 0., c[1]-float(node.bb_max[1]))
            dz = max(float(node.bb_min[2])-c[2], 0., c[2]-float(node.bb_max[2]))
            if dx*dx+dy*dy+dz*dz > r2: continue
            if node.is_leaf or all(ch is None for ch in node.children):
                if node.pts_idx is not None and len(node.pts_idx) > 0:
                    d = xyz[node.pts_idx] - c
                    mask = (d*d).sum(1) <= r2
                    if mask.any(): res.append(node.pts_idx[mask])
            else:
                inside = (float(node.bb_min[0])>=c[0]-r and float(node.bb_max[0])<=c[0]+r and
                          float(node.bb_min[1])>=c[1]-r and float(node.bb_max[1])<=c[1]+r and
                          float(node.bb_min[2])>=c[2]-r and float(node.bb_max[2])<=c[2]+r)
                if inside and node.pts_idx is not None:
                    res.append(node.pts_idx)
                for ch in node.children:
                    if ch: q.append(ch)
        if not res: return np.zeros(0, np.int32)
        return np.unique(np.concatenate(res)).astype(np.int32)

    def invalidate_colors(self): pass


# ── helpers ───────────────────────────────────────────────────────────────────

def _voxel_filter_numpy(pts, target, rng=None, global_idx=None):
    n = len(pts)
    if n <= target:
        if global_idx is not None: return global_idx.astype(np.int32)
        return np.arange(n, dtype=np.int32)
    bb  = pts.min(0); ext = float((pts.max(0)-bb).max())
    vox = ext * (target/n)**(1/3) * 0.5
    if vox <= 0:
        if rng is None: rng = np.random.default_rng(42)
        sel = rng.choice(n, target, replace=False)
        if global_idx is not None: return global_idx[sel].astype(np.int32)
        return sel.astype(np.int32)
    c  = ((pts-bb) / max(vox,1e-9)).astype(np.int32)
    hk = (c[:,0].astype(np.int64)*1_000_003*1_000_033 +
          c[:,1].astype(np.int64)*1_000_033 + c[:,2].astype(np.int64))
    si    = np.argsort(hk, kind='stable')
    _, fi = np.unique(hk[si], return_index=True)
    sel   = si[fi]
    if len(sel) > target:
        if rng is None: rng = np.random.default_rng(42)
        sel = rng.choice(sel, target, replace=False)
    sel = np.sort(sel)
    if global_idx is not None: return global_idx[sel].astype(np.int32)
    return sel.astype(np.int32)
