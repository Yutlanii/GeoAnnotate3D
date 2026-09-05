"""
core/_fast.py — Interface to _fastcore C extension v2.0

All heavy operations delegate to C with numpy fallback.

Usage:
    from core._fast import FC  # module-level access
    if FC.available:
        xyz, col, idx = FC.gather_and_color(xyz_all, lod_idx, labels, lut, planes)
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

HERE = Path(__file__).parent

# ── Load C extension ─────────────────────────────────────────────────────────
_fc = None

def _load():
    global _fc
    # Try to compile if needed
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "build_extension", HERE / "build_extension.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.build_fastcore()
    except Exception:
        pass

    # Find and load
    import sysconfig
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    for name in [f"_fastcore{suffix}", "_fastcore.so", "_fastcore.pyd"]:
        p = HERE / name
        if p.exists():
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("_fastcore", str(p))
                if spec and spec.loader:
                    m = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(m)
                    return m
            except Exception as e:
                print(f"[_fast] load failed: {e}")
    return None

_fc = _load()


class _FastCore:
    """Unified interface to C extension with numpy fallback."""

    def __init__(self, fc_module):
        self._fc = fc_module
        self.available = fc_module is not None
        if self.available:
            print("[GeoAnnotate3D] _fastcore C engine loaded ✓")
        else:
            print("[GeoAnnotate3D] _fastcore not available — numpy fallback")

    # ── LUT lookup ────────────────────────────────────────────────────────────

    def lut_lookup_u8(self, labels: np.ndarray, lut_u8: np.ndarray) -> np.ndarray:
        """labels (N,) u8 + lut (256,4) u8 → (N,4) u8"""
        labels = np.ascontiguousarray(labels, np.uint8)
        lut_u8 = np.ascontiguousarray(lut_u8, np.uint8)
        if self._fc:
            return self._fc.lut_lookup_u8(labels, lut_u8)
        return lut_u8[labels]

    # ── Elevation colors ──────────────────────────────────────────────────────

    def elevation_colors_u8(self, z: np.ndarray, z_min: float, z_max: float,
                            lut_u8: np.ndarray) -> np.ndarray:
        """z (N,) f32 → (N,4) u8 via LUT"""
        z = np.ascontiguousarray(z, np.float32)
        lut_u8 = np.ascontiguousarray(lut_u8, np.uint8)
        if self._fc:
            return self._fc.elevation_colors_u8(z, float(z_min), float(z_max), lut_u8)
        rng = max(z_max - z_min, 1e-6)
        idx = np.clip(((z - z_min) / rng * 255).astype(np.int32), 0, 255)
        return lut_u8[idx]

    # ── Sphere query ──────────────────────────────────────────────────────────

    def sphere_query(self, xyz: np.ndarray, cx: float, cy: float, cz: float,
                     radius: float) -> np.ndarray:
        """→ (K,) int32 indices"""
        xyz = np.ascontiguousarray(xyz, np.float32)
        if self._fc:
            return self._fc.sphere_query(xyz, float(cx), float(cy), float(cz), float(radius))
        diff = xyz - np.array([cx, cy, cz], np.float32)
        return np.where((diff*diff).sum(1) <= radius*radius)[0].astype(np.int32)

    # ── Frustum cull ──────────────────────────────────────────────────────────

    def frustum_cull(self, xyz: np.ndarray, planes: np.ndarray) -> np.ndarray:
        """→ (N,) bool mask"""
        xyz = np.ascontiguousarray(xyz, np.float32)
        planes = np.ascontiguousarray(planes, np.float32)
        if self._fc:
            return self._fc.frustum_cull(xyz, planes)
        nx,ny,nz,d = planes[:,0],planes[:,1],planes[:,2],planes[:,3]
        inside = np.ones(len(xyz), bool)
        for p in range(6):
            inside &= (xyz[:,0]*nx[p] + xyz[:,1]*ny[p] + xyz[:,2]*nz[p] + d[p]) >= 0
        return inside

    # ── Voxel subsample ───────────────────────────────────────────────────────

    def voxel_subsample(self, xyz: np.ndarray, target: int) -> np.ndarray:
        """→ (K,) int32 indices"""
        n = len(xyz)
        if n <= target:
            return np.arange(n, np.int32)
        xyz = np.ascontiguousarray(xyz, np.float32)
        if self._fc:
            return self._fc.voxel_subsample(xyz, int(target))
        # numpy fallback
        mn = xyz.min(0); mx = xyz.max(0)
        vol = max(float((mx[0]-mn[0])*(mx[1]-mn[1])*(mx[2]-mn[2])), 1e-9)
        vox = max(0.001, (vol/target)**(1/3))
        inv = 1.0/vox
        gx = np.floor((xyz[:,0]-mn[0])*inv).astype(np.int64)
        gy = np.floor((xyz[:,1]-mn[1])*inv).astype(np.int64)
        gz = np.floor((xyz[:,2]-mn[2])*inv).astype(np.int64)
        hk = gx*73856093 ^ gy*19349669 ^ gz*83492791
        _, fi = np.unique(hk, return_index=True)
        idx = fi.astype(np.int32)
        if len(idx) > target:
            idx = np.random.default_rng(42).choice(idx, target, replace=False).astype(np.int32)
        return idx

    # ── Build all LOD levels ──────────────────────────────────────────────────

    def build_all_lod_levels(self, xyz: np.ndarray, targets: list) -> list:
        """→ list of (K,) int32 index arrays"""
        n = len(xyz)
        xyz = np.ascontiguousarray(xyz, np.float32)
        if self._fc:
            return self._fc.build_all_lod_levels(xyz, targets)
        # numpy fallback
        result = []
        for t in targets:
            if t >= n:
                result.append(np.arange(n, dtype=np.int32))
            else:
                result.append(self.voxel_subsample(xyz, t))
        return result

    # ── FUSED gather + color + frustum ────────────────────────────────────────

    def gather_and_color(self, xyz_all: np.ndarray, lod_idx: np.ndarray,
                         labels: np.ndarray, lut_u8: np.ndarray,
                         planes: Optional[np.ndarray] = None
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fused: xyz[idx] + lut[labels[idx]] + frustum in ONE pass.
        Returns (xyz_out, col_u8_out, idx_out).
        """
        xyz_all = np.ascontiguousarray(xyz_all, np.float32)
        lod_idx = np.ascontiguousarray(lod_idx, np.int32)
        labels  = np.ascontiguousarray(labels, np.uint8)
        lut_u8  = np.ascontiguousarray(lut_u8, np.uint8)
        planes_arg = None if planes is None else np.ascontiguousarray(planes, np.float32)

        if self._fc:
            return self._fc.gather_and_color(xyz_all, lod_idx, labels, lut_u8,
                                              planes_arg if planes_arg is not None else None)
        # numpy fallback
        idx = lod_idx
        xyz = xyz_all[idx]
        col = lut_u8[labels[idx]]
        if planes is not None:
            from utils.spatial import frustum_cull_points
            mask = frustum_cull_points(xyz, planes)
            xyz, col, idx = xyz[mask], col[mask], idx[mask]
        return xyz, col, idx

    def gather_and_color_elevation(self, xyz_all: np.ndarray, lod_idx: np.ndarray,
                                    z_min: float, z_max: float,
                                    lut_u8: np.ndarray,
                                    planes: Optional[np.ndarray] = None
                                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fused gather + elevation color + frustum."""
        xyz_all = np.ascontiguousarray(xyz_all, np.float32)
        lod_idx = np.ascontiguousarray(lod_idx, np.int32)
        lut_u8  = np.ascontiguousarray(lut_u8, np.uint8)
        planes_arg = None if planes is None else np.ascontiguousarray(planes, np.float32)

        if self._fc:
            return self._fc.gather_and_color_elevation(
                xyz_all, lod_idx, float(z_min), float(z_max), lut_u8,
                planes_arg if planes_arg is not None else None)
        # numpy fallback
        idx = lod_idx
        xyz = xyz_all[idx]
        rng = max(z_max - z_min, 1e-6)
        ci = np.clip(((xyz[:,2] - z_min) / rng * 255).astype(np.int32), 0, 255)
        col = lut_u8[ci]
        if planes is not None:
            from utils.spatial import frustum_cull_points
            mask = frustum_cull_points(xyz, planes)
            xyz, col, idx = xyz[mask], col[mask], idx[mask]
        return xyz, col, idx

    # ── Grid index (replaces KDTree, 25x less memory) ─────────────────────────

    def grid_build(self, xyz: np.ndarray, cell_size: float):
        """Build uniform 3D grid index. Returns opaque capsule."""
        xyz = np.ascontiguousarray(xyz, np.float32)
        if self._fc:
            return self._fc.grid_build(xyz, float(cell_size))
        return None

    def grid_sphere_query(self, grid, xyz: np.ndarray,
                           cx: float, cy: float, cz: float,
                           radius: float) -> np.ndarray:
        """Sphere query using grid index. ~10x faster than KDTree brute."""
        xyz = np.ascontiguousarray(xyz, np.float32)
        if self._fc and grid is not None:
            return self._fc.grid_sphere_query(
                grid, xyz, float(cx), float(cy), float(cz), float(radius))
        return self.sphere_query(xyz, cx, cy, cz, radius)

    # ── Label writing ─────────────────────────────────────────────────────────

    def write_labels(self, labels: np.ndarray, idx: np.ndarray, class_id: int):
        """In-place: labels[idx] = class_id"""
        if self._fc:
            self._fc.write_labels(labels, np.ascontiguousarray(idx, np.int32), int(class_id))
        else:
            labels[idx] = np.uint8(class_id)

    # ── Per-class counts ──────────────────────────────────────────────────────

    def per_class_counts(self, labels: np.ndarray) -> dict:
        """Returns {class_id: count} dict. ~3x faster than np.unique."""
        if self._fc:
            return self._fc.per_class_counts(np.ascontiguousarray(labels, np.uint8))
        u, c = np.unique(labels, return_counts=True)
        return {int(k): int(v) for k, v in zip(u, c)}

    # ── Partial color refresh (for annotation, only affected pts) ────────────

    def refresh_colors_partial(self, col_buf: np.ndarray, idx_full: np.ndarray,
                                affected: np.ndarray, lut_u8: np.ndarray,
                                class_id: int):
        """In GPU color buffer, update only points in 'affected' to class_id color."""
        if self._fc:
            self._fc.refresh_colors_partial(
                col_buf,
                np.ascontiguousarray(idx_full, np.int32),
                np.ascontiguousarray(affected, np.int32),
                np.ascontiguousarray(lut_u8, np.uint8),
                int(class_id))
        else:
            # numpy fallback
            mask = np.isin(idx_full, affected, assume_unique=False)
            col_buf[mask] = lut_u8[class_id]


# ── Module-level singleton ────────────────────────────────────────────────────
FC = _FastCore(_fc)

# ── Legacy compatibility aliases ──────────────────────────────────────────────
FASTCORE_AVAILABLE = FC.available
_USE_C = FC.available

def apply_lut_annotation(labels, lut_f32):
    lut_u8 = np.clip(lut_f32 * 255, 0, 255).astype(np.uint8)
    return FC.lut_lookup_u8(labels.astype(np.uint8), lut_u8).astype(np.float32) / 255.0

def voxel_subsample_fast(xyz, target, vox_size=None):
    return FC.voxel_subsample(xyz, target)

def sphere_query_fast(xyz, center, radius):
    return FC.sphere_query(xyz, float(center[0]), float(center[1]), float(center[2]), float(radius))

def build_lod_levels(xyz):
    targets = [50_000, 500_000, 3_000_000, 15_000_000]
    return tuple(FC.build_all_lod_levels(xyz, targets))

def frustum_cull_fast(xyz, planes):
    return FC.frustum_cull(xyz, planes)

def compute_z_colors_fast(z, z_min, z_max, lut_f32):
    lut_u8 = np.clip(lut_f32 * 255, 0, 255).astype(np.uint8)
    return FC.elevation_colors_u8(z, z_min, z_max, lut_u8).astype(np.float32) / 255.0

def enhance_colors_fast(rgba, gamma, sat, bright):
    # No C version yet — numpy
    rgb = np.clip(rgba[:,:3], 0, 1)
    if gamma != 1.0: rgb = np.power(rgb + 1e-7, 1.0/gamma)
    if sat != 1.0:
        lum = (rgb[:,0]*0.299 + rgb[:,1]*0.587 + rgb[:,2]*0.114)[:,None]
        rgb = np.clip(lum + sat*(rgb-lum), 0, 1)
    if bright != 1.0: rgb = np.clip(rgb*bright, 0, 1)
    rgba[:,:3] = rgb
    return rgba

def float_to_u8_fast(rgba):
    return np.clip(rgba * 255, 0, 255).astype(np.uint8)
