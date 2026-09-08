"""
annotation/noise_filter.py — Statistical Outlier Removal (SOR)

Mismo algoritmo de detección de ruido que CloudCompare/PCL
(pcl::StatisticalOutlierRemoval): para cada punto se calcula la
distancia media a sus k vecinos más cercanos; un punto es candidato a
ruido si esa distancia media excede `mean_global + std_ratio *
std_global` — es decir, está anormalmente más aislado que el promedio
de la nube. No borra nada por sí solo: devuelve la máscara de
candidatos, quien llama decide qué hacer con ellos (en GeoAnnotate3D,
se ofrecen para eliminar con la herramienta de eliminar puntos ya
existente — ver annotation/label_store.py::delete_points).
"""
from __future__ import annotations
import numpy as np


def detect_outliers_sor(xyz: np.ndarray, k: int = 8,
                        std_ratio: float = 2.0) -> np.ndarray:
    """
    Devuelve una máscara booleana (N,) — True = candidato a ruido.

    xyz        — (N,3) float
    k          — número de vecinos más cercanos a considerar
    std_ratio  — multiplicador de desviación estándar (menor = más
                 agresivo, marca más puntos como ruido)
    """
    from scipy.spatial import cKDTree

    n = len(xyz)
    if n == 0:
        return np.zeros(0, dtype=bool)
    k = max(1, min(k, n - 1))
    if k < 1:
        return np.zeros(n, dtype=bool)

    tree = cKDTree(np.ascontiguousarray(xyz, dtype=np.float64))
    # k+1 porque el vecino más cercano de un punto es él mismo (distancia 0)
    dists, _ = tree.query(xyz, k=k + 1, workers=-1)
    mean_dist = dists[:, 1:].mean(axis=1)

    mu = float(mean_dist.mean())
    sigma = float(mean_dist.std())
    threshold = mu + std_ratio * sigma
    return mean_dist > threshold
