"""
render/lod_worker.py — Worker LOD v7.0

Key fixes:
  1. NO FRUSTUM CULLING in LOD loading.
     Frustum is for rendering, not data loading.
     All points in budget are loaded unconditionally.
     VTK shows only what's in view — we just need all points in GPU.
     This eliminates density inconsistency between clicks.

  2. ACCUMULATIVE RENDERING: Once a LOD level is loaded, it stays
     in GPU. New cycles only add more points, never replace with fewer.
     User never sees density drop on re-render.

  3. BUDGET GROWS MONOTONICALLY: never drops below current rendered pts.
"""
from __future__ import annotations
import threading
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

COARSE_TARGET_PTS = 3_000_000


class LODWorker(QThread):
    coarse_ready = pyqtSignal(object, object, object, int)
    batch_ready  = pyqtSignal(object, object, object, int, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lock = threading.Lock()
        self._req = False
        self._cancel = False
        self._epoch = 0
        self._pc = None
        self._project = None
        self._ann_lut_u8 = None
        self._eye3d = None
        self._W = self._H = 0.0
        self._max_pts = 0
        self._cm = "Anotación"
        self._cmap = "viridis"

    @property
    def epoch(self):
        with self._lock:
            return self._epoch

    def cancel(self):
        with self._lock:
            self._cancel = True
            self._epoch += 1

    def request(self, pc, project, eye3d, frustum_planes, W, H,
                max_pts, color_mode, cmap, fov_deg,
                clip_en=False, clip_min=None, clip_max=None,
                annotation_lut=None, annotation_lut_u8=None):
        with self._lock:
            self._pc = pc
            self._project = project
            self._ann_lut_u8 = annotation_lut_u8
            self._eye3d = eye3d.copy() if eye3d is not None else None
            # NOTE: frustum_planes intentionally NOT stored — no frustum in LOD
            self._W, self._H = W, H
            self._max_pts = max_pts
            self._cm, self._cmap = color_mode, cmap
            self._req = True
            self._cancel = True
            self._epoch += 1
        if not self.isRunning():
            self.start()

    def run(self):
        while True:
            with self._lock:
                if not self._req: return
                self._req = False
                self._cancel = False
                pc = self._pc
                project = self._project
                ann_lut_u8 = self._ann_lut_u8
                W, H = self._W, self._H
                max_pts = self._max_pts
                cm, cmap = self._cm, self._cmap
                epoch = self._epoch

            if pc is None or pc.octree is None or not pc.octree.ready:
                continue
            try:
                self._refine(pc, project, ann_lut_u8,
                              W, H, max_pts, cm, cmap, epoch)
            except Exception:
                import traceback; traceback.print_exc()

    def _is_stale(self, epoch):
        with self._lock:
            return self._cancel or self._req or self._epoch != epoch

    def _refine(self, pc, project, ann_lut_u8,
                W, H, max_pts, cm, cmap, epoch):
        from core._fast import FC
        from render.colors import get_z_range, _Z_RANGE_CACHE, _get_lut_u8

        octree = pc.octree
        xyz_all = octree._xyz_ref
        labels = project.labels if project is not None else None

        z_range = None
        if cm not in ("Anotación", "RGB", "Clasificación", "Intensidad",
                      "Intensidad Color", "Retorno", "Densidad", "Color único"):
            z_range = _Z_RANGE_CACHE.get(id(xyz_all)) or get_z_range(xyz_all)

        use_elevation = (z_range is not None)
        lut_u8 = ann_lut_u8 if cm == "Anotación" else None

        # Safety cap for gather_and_color on huge mmap clouds.
        # Random access into a multi-GB mmap with millions of scattered
        # indices causes excessive Windows page faults → access violation in C.
        # Cap: based on available RAM, never more than 50M pts per gather call.
        try:
            _avail = _available_ram_gb() if '_available_ram_gb' in dir() else 8.0
        except Exception:
            _avail = 8.0
        # 50M pts × 15 bytes (xyz+rgb) = 750MB — safe for any machine
        MAX_GATHER = min(50_000_000, max(5_000_000, int(_avail * 0.03 * 1e9 / 15)))

        first_emit = True
        for lod_idx, is_done in octree.iter_lod_progression(max_pts):
            if self._is_stale(epoch): return
            if len(lod_idx) == 0:
                if is_done: break
                continue

            # If lod_idx is larger than MAX_GATHER, stride it down.
            # This prevents page fault storms on huge mmap files.
            if len(lod_idx) > MAX_GATHER:
                step = max(1, len(lod_idx) // MAX_GATHER)
                lod_idx = np.asarray(lod_idx[::step], dtype=np.int32)

            # ── GATHER SEGURO PARA MMAP GRANDES ──────────────────────
            # PROBLEMA: accesos ALEATORIOS a un mmap de 10+ GB causan una
            # tormenta de page faults en Windows → ACCESS VIOLATION en C.
            # SOLUCIÓN: ordenar los índices antes de acceder al mmap.
            # Acceso ORDENADO = páginas contiguas = sin tormenta.
            # FC.gather_and_color hace acceso aleatorio internamente,
            # por lo que para nubes grandes usamos el path numpy.

            is_huge_mmap = getattr(pc, 'is_mmap', False) and len(xyz_all) > 200_000_000

            if is_huge_mmap:
                # Path seguro: ordenar índices → acceso casi secuencial al mmap
                sort_order = np.argsort(lod_idx, kind='stable')
                lod_sorted  = lod_idx[sort_order]
                # Leer del mmap en orden → eficiente, sin tormenta de page faults
                xyz_sorted  = np.array(xyz_all[lod_sorted], dtype=np.float32)
                if self._is_stale(epoch): return
                # Restaurar orden original para consistencia con idx_out
                inv_order   = np.argsort(sort_order, kind='stable')
                xyz_out     = xyz_sorted[inv_order]
                idx_out     = lod_idx
                lod_lbl     = labels[lod_idx] if labels is not None else None
                from render.colors import compute_colors_u8
                col_out = compute_colors_u8(
                    xyz_out, pc.get_attrs(lod_idx), cm, cmap, z_range,
                    annotation_labels=lod_lbl, annotation_lut_u8=ann_lut_u8)
            elif use_elevation and z_range:
                xyz_out, col_out, idx_out = FC.gather_and_color_elevation(
                    xyz_all, lod_idx, z_range[0], z_range[1],
                    _get_lut_u8(cmap), None)
            elif cm == "Anotación" and labels is not None and lut_u8 is not None:
                xyz_out, col_out, idx_out = FC.gather_and_color(
                    xyz_all, lod_idx, labels, lut_u8, None)
            else:
                from render.colors import compute_colors_u8
                lod_xyz = xyz_all[lod_idx]
                if self._is_stale(epoch): return
                lod_lbl = labels[lod_idx] if labels is not None else None
                col_out = compute_colors_u8(
                    lod_xyz, pc.get_attrs(lod_idx), cm, cmap, z_range,
                    annotation_labels=lod_lbl, annotation_lut_u8=ann_lut_u8)
                xyz_out, idx_out = lod_xyz, lod_idx

            if self._is_stale(epoch): return
            if len(xyz_out) == 0:
                if is_done: break
                continue

            if first_emit:
                self.coarse_ready.emit(xyz_out, col_out, idx_out, epoch)
                first_emit = False
            else:
                self.batch_ready.emit(xyz_out, col_out, idx_out, epoch, is_done)

            if is_done: return
