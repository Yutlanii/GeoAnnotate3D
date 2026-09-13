"""
annotation/label_smoothing.py — Suavizado de etiquetas por mayoría de
vecinos + detección de clusters aislados por clase (QA de anotación).

Post-proceso estándar en flujos de segmentación de nubes de puntos (a
veces llamado "majority filter" o "mode filter"): cada punto etiquetado
recalcula su clase como la más frecuente entre sus k vecinos más
cercanos, para limpiar bordes ruidosos entre clases (ej. un punto de
"edificio" aislado en medio de "vegetación" por un clic impreciso del
pincel). No toca puntos sin etiquetar (clase 0) ni eliminados
(DELETED_LABEL) — ni como candidatos a cambiar, ni como votantes: un
vecino sin etiquetar no debería poder "arrastrar" a un punto etiquetado
de vuelta a sin-etiquetar.

detect_isolated_clusters_per_class(): QA complementario — para cada
clase, agrupa sus puntos por conectividad espacial (grid hash, mismo
principio que RegionGrowing) y reporta clusters anormalmente pequeños,
candidatos a error de anotación (ej. 3 puntos sueltos de "vehículo" en
medio de "suelo" — probablemente un clic accidental).
"""
from __future__ import annotations
import numpy as np


def smooth_labels_majority(xyz: np.ndarray, labels: np.ndarray, k: int = 8,
                           restrict_idx: "np.ndarray | None" = None,
                           unlabeled_class: int = 0,
                           deleted_label: int = 255):
    """
    Suaviza etiquetas por voto de mayoría entre los k vecinos más cercanos.

    xyz            — (N,3) float, coordenadas de TODA la nube (o del tile)
    labels         — (N,) uint8, etiquetas actuales (mismo largo que xyz)
    k              — número de vecinos a considerar (sin contar el punto mismo)
    restrict_idx   — opcional: subconjunto de índices GLOBALES donde operar
                     (ej. el tile activo) — los vecinos también se buscan
                     solo dentro de ese subconjunto.
    unlabeled_class, deleted_label — se excluyen tanto de los candidatos a
                     cambiar (nunca se re-etiqueta un punto sin clase o
                     eliminado) como del voto de los vecinos.

    Retorna (changed_idx, new_labels) — SOLO los índices cuya clase
    cambió y su nueva clase, listos para LabelStore.annotate_bulk().
    Si nada cambió, ambos arrays vienen vacíos.
    """
    from scipy.spatial import cKDTree

    n = len(xyz)
    if n == 0 or len(labels) != n:
        return np.zeros(0, np.int64), np.zeros(0, np.uint8)

    universe = (np.asarray(restrict_idx, dtype=np.int64)
                if restrict_idx is not None else np.arange(n, dtype=np.int64))
    if len(universe) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.uint8)

    universe_labels = labels[universe]
    eligible_mask = (universe_labels != unlabeled_class) & (universe_labels != deleted_label)
    eligible = universe[eligible_mask]
    if len(eligible) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.uint8)

    k_eff = max(1, min(k, len(universe) - 1))
    tree = cKDTree(np.ascontiguousarray(xyz[universe], dtype=np.float64))
    _, q_local = tree.query(np.ascontiguousarray(xyz[eligible], dtype=np.float64),
                            k=k_eff + 1, workers=-1)
    if q_local.ndim == 1:
        q_local = q_local[:, None]

    neighbor_global = universe[q_local]              # (M, k+1) índices globales
    neighbor_labels = labels[neighbor_global]         # (M, k+1)
    is_self = neighbor_global == eligible[:, None]
    valid = (~is_self) & (neighbor_labels != unlabeled_class) & (neighbor_labels != deleted_label)

    own_labels = labels[eligible]
    best_class = own_labels.copy()
    best_count = np.zeros(len(eligible), dtype=np.int32)

    classes_present = np.unique(neighbor_labels[valid]) if valid.any() else np.zeros(0, labels.dtype)
    for c in classes_present:
        cnt = ((neighbor_labels == c) & valid).sum(axis=1)
        better = cnt > best_count
        best_count = np.where(better, cnt, best_count)
        best_class = np.where(better, c, best_class)

    changed_mask = (best_class != own_labels) & (best_count > 0)
    changed_idx = eligible[changed_mask]
    new_labels = best_class[changed_mask].astype(np.uint8)
    return changed_idx.astype(np.int64), new_labels


def detect_isolated_clusters_per_class(xyz: np.ndarray, labels: np.ndarray,
                                       max_cluster_size: int = 15,
                                       connect_dist_m: float = 1.0,
                                       restrict_idx: "np.ndarray | None" = None,
                                       unlabeled_class: int = 0,
                                       deleted_label: int = 255,
                                       max_points_per_class: int = 2_000_000):
    """
    QA de anotación: para cada clase, agrupa sus puntos por conectividad
    espacial (grid hash 3D, radio connect_dist_m) y reporta clusters con
    <= max_cluster_size puntos — candidatos a error de anotación (ruido,
    clic accidental).

    Retorna un dict {class_id: [(cluster_size, np.ndarray(global_idx)), ...]}
    ordenado de menor a mayor tamaño, SOLO con clases que tienen al menos
    un cluster sospechoso. No modifica nada — es puro diagnóstico; el
    llamador decide qué hacer con los índices reportados (ej. ofrecerlos
    para eliminar, igual que SOR).

    max_points_per_class: por seguridad, si una clase tiene más puntos que
    esto en el universo de búsqueda, se salta (evita construir un grid
    hash gigantesco por accidente sobre una clase con decenas de millones
    de puntos — no es lo que este chequeo está pensado para hacer).
    """
    n = len(xyz)
    if n == 0 or len(labels) != n:
        return {}
    universe = (np.asarray(restrict_idx, dtype=np.int64)
                if restrict_idx is not None else np.arange(n, dtype=np.int64))
    if len(universe) == 0:
        return {}

    universe_labels = labels[universe]
    result = {}
    for cid in np.unique(universe_labels):
        cid_int = int(cid)
        if cid_int == unlabeled_class or cid_int == deleted_label:
            continue
        cls_mask = universe_labels == cid
        cls_global = universe[cls_mask]
        if len(cls_global) == 0 or len(cls_global) > max_points_per_class:
            continue
        clusters = _connected_clusters(xyz[cls_global], connect_dist_m)
        small = [(len(c), cls_global[c]) for c in clusters if len(c) <= max_cluster_size]
        if small:
            small.sort(key=lambda t: t[0])
            result[cid_int] = small
    return result


def _connected_clusters(xyz: np.ndarray, connect_dist_m: float) -> list:
    """
    Agrupa xyz (ya filtrado a una sola clase) en componentes conectados
    espacialmente vía grid hash + BFS — mismo principio vectorizado que
    annotation/region_growing.py, aplicado aquí a TODOS los puntos de la
    clase (no desde una sola semilla): cada punto sin visitar arranca su
    propio BFS. Retorna lista de arrays de índices LOCALES (dentro de xyz).
    """
    from collections import deque
    n = len(xyz)
    if n == 0:
        return []
    if n == 1:
        return [np.array([0], dtype=np.int64)]

    step = max(0.05, connect_dist_m)
    mn = xyz[:, :2].min(0)
    ci = ((xyz[:, 0] - mn[0]) / step).astype(np.int64)
    ri = ((xyz[:, 1] - mn[1]) / step).astype(np.int64)
    row_span = int(ri.max()) + 2
    cell_key = ci * np.int64(row_span) + ri

    order = np.argsort(cell_key, kind='stable')
    sorted_key = cell_key[order]
    uniq_key, first_pos = np.unique(sorted_key, return_index=True)
    groups = np.split(order, first_pos[1:])
    cell_map = {int(k): g.tolist() for k, g in zip(uniq_key, groups)}

    r2 = step * step
    visited = [False] * n
    clusters = []
    for start in range(n):
        if visited[start]:
            continue
        visited[start] = True
        queue = deque([start])
        comp = [start]
        while queue:
            curr = queue.popleft()
            cc, cr = int(ci[curr]), int(ri[curr])
            cxy = xyz[curr, :2]
            for dc in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    neighbors = cell_map.get((cc + dc) * row_span + (cr + dr))
                    if neighbors is None:
                        continue
                    for nb in neighbors:
                        if visited[nb]:
                            continue
                        dxy = xyz[nb, :2] - cxy
                        if (dxy ** 2).sum() > r2:
                            continue
                        dz = abs(float(xyz[nb, 2] - xyz[curr, 2]))
                        if dz > step:
                            continue
                        visited[nb] = True
                        queue.append(nb)
                        comp.append(nb)
        clusters.append(np.array(comp, dtype=np.int64))
    return clusters
