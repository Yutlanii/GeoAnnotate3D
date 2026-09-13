"""
annotation/profile.py — Vista de perfil / corte vertical.

Extrae y proyecta los puntos dentro de una franja (buffer) alrededor de
una línea A→B trazada en planta, para verlos en 2D como (distancia a lo
largo de la línea, altura Z) — el mismo tipo de vista que CloudCompare
llama "Cross section" o "Profile view", útil para revisar líneas
eléctricas, taludes, secciones de vía, perfiles de terreno, etc. sin
tener que interpretar la nube en 3D desde un ángulo lateral incómodo.

Deliberadamente sin ninguna dependencia de VTK/Qt — solo numpy, así se
puede probar headless con datos sintéticos (ver tests/verify_fixes.py).
El widget que pinta esto (QPainter) vive en ui/profile_view.py.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def project_to_profile(xyz: np.ndarray, p1: np.ndarray, p2: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Proyecta `xyz` (N,3) sobre la línea A→B vista en planta (XY).

    Devuelve:
        t     (N,) float32 — distancia a lo largo de la línea (0 en A,
              `length` en B); puede ser negativa o mayor que `length`
              para puntos más allá de los extremos.
        perp  (N,) float32 — distancia perpendicular (con signo) a la
              línea, en XY — el ancho del "buffer" se aplica sobre esto.
        length float — distancia A→B en XY.

    La altura Z NO participa en la proyección — es una vista de perfil
    en planta con un eje de altura aparte, no una proyección 3D pura;
    así el eje horizontal del perfil sigue siendo una distancia real en
    el terreno sin importar cuánto suban o bajen A o B.
    """
    p1xy = np.asarray(p1, np.float64)[:2]
    p2xy = np.asarray(p2, np.float64)[:2]
    d    = p2xy - p1xy
    length = float(np.hypot(d[0], d[1]))
    if length < 1e-9:
        # A y B prácticamente en el mismo punto — no hay línea que trazar.
        n = len(xyz)
        return np.zeros(n, np.float32), np.zeros(n, np.float32), 0.0
    u = d / length
    v = np.asarray(xyz, np.float64)[:, :2] - p1xy
    t    = v[:, 0] * u[0] + v[:, 1] * u[1]
    perp = v[:, 0] * u[1] - v[:, 1] * u[0]
    return t.astype(np.float32), perp.astype(np.float32), length


def extract_profile_slice(xyz: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                          buffer_m: float,
                          global_idx: Optional[np.ndarray] = None
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Filtra `xyz` a la franja de ancho `2*buffer_m` centrada en la línea
    A→B, y dentro del rango [0, length] a lo largo de ella (con un
    margen extra de `buffer_m` en cada extremo, para no cortar en seco
    justo en A/B — el mismo criterio que usan las herramientas de
    "corte" de CloudCompare/Cyclone 3DR).

    Devuelve (t_sel, z_sel, idx_sel, length):
        t_sel   (K,) distancia a lo largo de la línea de cada punto que
                sobrevivió el filtro.
        z_sel   (K,) su altura Z.
        idx_sel (K,) su índice dentro de `xyz` (o el índice GLOBAL si se
                pasó `global_idx`, para poder leer clasificación/labels).
        length  distancia A→B.
    """
    xyz = np.asarray(xyz)
    if len(xyz) == 0:
        return (np.zeros(0, np.float32), np.zeros(0, np.float32),
                np.zeros(0, np.int64), 0.0)
    t, perp, length = project_to_profile(xyz, p1, p2)
    if length <= 1e-9:
        return (np.zeros(0, np.float32), np.zeros(0, np.float32),
                np.zeros(0, np.int64), 0.0)
    mask = (np.abs(perp) <= buffer_m) & (t >= -buffer_m) & (t <= length + buffer_m)
    idx = np.where(mask)[0]
    if global_idx is not None and len(global_idx) == len(xyz):
        idx_out = np.asarray(global_idx)[idx].astype(np.int64)
    else:
        idx_out = idx.astype(np.int64)
    return t[idx], xyz[idx, 2].astype(np.float32), idx_out, length
