"""
ui/icons.py — Iconos reales (Bootstrap Icons, licencia MIT) para toda la UI.

Antes las herramientas se dibujaban a mano con QPainter (líneas/arcos
primitivos en ui/tool_panel.py). Ahora se usan los SVG reales de
Bootstrap Icons, guardados en ui/icons/*.svg (ver LICENSE.txt ahí).

Los SVG originales usan fill="currentColor" (heredan el color del
contexto CSS); PyQt no interpreta currentColor, así que esta capa
sustituye el texto por el color pedido ANTES de pasarlo a QSvgRenderer.
Resultado cacheado por (nombre, color, tamaño) para no re-parsear el SVG
en cada repintado.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from PyQt5.QtCore import QByteArray, Qt
from PyQt5.QtGui import QIcon, QPixmap, QPainter
from PyQt5.QtSvg import QSvgRenderer

_ICON_DIR = Path(__file__).parent / "icons"
_CACHE: Dict[Tuple[str, str, int], QPixmap] = {}


def _raw_svg(name: str) -> str:
    path = _ICON_DIR / f"{name}.svg"
    return path.read_text(encoding="utf-8")


def pixmap(name: str, color: str = "#2b2d30", size: int = 16) -> QPixmap:
    """QPixmap del icono `name` (sin extensión) recoloreado a `color`."""
    key = (name, color, size)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    try:
        svg = _raw_svg(name).replace("currentColor", color)
    except FileNotFoundError:
        print(f"[icons] falta ui/icons/{name}.svg")
        _CACHE[key] = pm
        return pm

    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    painter = QPainter(pm)
    renderer.render(painter)
    painter.end()
    _CACHE[key] = pm
    return pm


def icon(name: str, color: str = "#2b2d30", size: int = 16) -> QIcon:
    """QIcon del icono `name` recoloreado a `color`."""
    return QIcon(pixmap(name, color, size))


def available(name: str) -> bool:
    return (_ICON_DIR / f"{name}.svg").exists()
