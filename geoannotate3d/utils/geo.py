"""
utils/geo.py
============
Utilidades geoespaciales.

Contenido:
    parse_crs()             — parsea CRS de un header LAS
    offset_to_absolute()    — convierte coords locales → UTM absolutas
    absolute_to_offset()    — convierte UTM absolutas → coords locales
    bbox_to_geo()           — calcula bbox en coordenadas absolutas
    format_utm_coords()     — formatea coordenadas UTM para mostrar en UI
"""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


def parse_crs(las_header) -> str:
    """
    Extrae el CRS de un header laspy como string legible.
    Retorna "" si no disponible.
    """
    try:
        crs = las_header.parse_crs()
        if crs is not None:
            return str(getattr(crs, "name", crs))[:200]
    except Exception:
        pass
    return ""


def offset_to_absolute(xyz_local: np.ndarray,
                        offset: np.ndarray) -> np.ndarray:
    """
    Convierte coordenadas locales (centradas en el offset) a absolutas.

    xyz_local : (N, 3) float32 — centradas en 0,0,0
    offset    : (3,) float64   — traslación al origen del mundo

    Retorna (N, 3) float64 — coordenadas UTM absolutas.
    """
    return xyz_local.astype(np.float64) + offset.astype(np.float64)


def absolute_to_offset(xyz_abs: np.ndarray,
                        offset: np.ndarray) -> np.ndarray:
    """
    Convierte coordenadas absolutas (UTM) a locales centradas.

    xyz_abs : (N, 3) float64
    offset  : (3,) float64

    Retorna (N, 3) float32.
    """
    return (xyz_abs.astype(np.float64) - offset.astype(np.float64)).astype(np.float32)


def bbox_to_geo(bounds_local: np.ndarray,
                offset: np.ndarray) -> dict:
    """
    Calcula el bounding box en coordenadas absolutas.

    bounds_local : (2, 3) float32 — [min_xyz, max_xyz] locales
    offset       : (3,) float64

    Retorna dict {"min": [E, N, Z], "max": [E, N, Z]} como float64.
    """
    min_abs = (bounds_local[0].astype(np.float64) + offset).tolist()
    max_abs = (bounds_local[1].astype(np.float64) + offset).tolist()
    return {"min": min_abs, "max": max_abs}


def format_utm_coords(xyz_local: np.ndarray,
                       offset: np.ndarray,
                       crs: str = "") -> Tuple[str, str, str]:
    """
    Formatea un punto en coordenadas UTM absolutas para mostrar en la UI.

    Retorna (easting_str, northing_str, z_str).

    Ejemplo:
        ("397,241.8 E", "2,166,804.3 N", "1,421.5 m")
    """
    if xyz_local is None or offset is None:
        return ("—", "—", "—")
    abs_xyz = xyz_local.astype(np.float64) + offset.astype(np.float64)
    e_str = f"{abs_xyz[0]:,.1f} E"
    n_str = f"{abs_xyz[1]:,.1f} N"
    z_str = f"{abs_xyz[2]:.2f} m"
    return e_str, n_str, z_str


def estimate_utm_zone(easting: float, northing: float) -> Optional[str]:
    """
    Estima el código EPSG UTM a partir de coordenadas (si se desconoce el CRS).
    Solo funciona para coordenadas en el rango razonable de UTM.
    Retorna None si no se puede determinar.
    """
    # Las coordenadas E en UTM van de ~100,000 a ~900,000
    # Las coordenadas N en UTM van de 0 (ecuador) a ~10,000,000 (polo)
    if 100_000 <= easting <= 900_000 and 0 <= northing <= 10_000_000:
        # No podemos determinar la zona sin la longitud real
        return None
    return None
