"""
render/lod.py — Stub de compatibilidad.

El sistema LOD fue unificado en core/octree.py v8.0.
LODTree y LODWorker se delegan ahora al Octree directamente.
Este archivo existe solo para no romper imports existentes.
"""
# Re-exportar desde la nueva ubicación
from core.octree import Octree as LODTree, LOD_TARGETS

# Alias de compatibilidad
def voxel_subsample(xyz, target_n):
    from core._fast import voxel_subsample_fast
    return voxel_subsample_fast(xyz, target_n)
