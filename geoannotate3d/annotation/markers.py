"""
annotation/markers.py — Etiquetas y medidas persistentes en 3D.

Antes, MeasureTool (annotation/tools.py) mostraba una medición
transitoria: se perdía en cuanto medías otra cosa o cambiabas de
herramienta. CloudCompare/Cyclone 3DR dejan marcas de texto y medidas
ANCLADAS en el espacio, visibles permanentemente y guardadas con el
proyecto — esto es el modelo de datos para eso.

MarkerStore es deliberadamente independiente de VTK/render — modelo de
datos puro, así se puede probar headless (ver tests/verify_fixes.py).
El render real de los actores 3D (texto flotante, esferas, líneas) vive
en render/canvas.py, que se suscribe a markers_changed para mantener
los actores sincronizados con la colección.

Coordenadas: pos/pos_b se guardan en el mismo sistema que pc.xyz
(relativas al offset del proyecto, NO coordenadas UTM absolutas) — igual
que todo lo demás en la app (ver core/pointcloud.py::PointCloud.offset).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal


@dataclass
class Marker:
    """Una marca persistente: etiqueta de texto en un punto, medida entre
    dos puntos (pos_b no-None solo para kind="measure"), o polilínea de
    varios vértices (points no-None solo para kind="polyline" — pos
    queda igual a points[0], para que el resto del código que ya asume
    "todo marcador tiene una posición de ancla" (la esfera + el texto en
    _MarkerOverlay) siga funcionando sin cambios para este kind nuevo)."""
    id:         str
    kind:       str                       # "label" | "measure" | "polyline"
    pos:        List[float]
    pos_b:      Optional[List[float]] = None
    points:     Optional[List[List[float]]] = None
    text:       str = ""
    color:      List[float] = field(default_factory=lambda: [1.0, 0.85, 0.2])
    font_size:  float = 14.0
    line_width: float = 2.5     # grosor de línea — "measure" y "polyline"
    created_at: str = ""

    def distance(self) -> Optional[float]:
        """Distancia 3D entre pos y pos_b — solo tiene sentido para "measure"."""
        if self.kind != "measure" or self.pos_b is None:
            return None
        a = np.asarray(self.pos, np.float64)
        b = np.asarray(self.pos_b, np.float64)
        return float(np.linalg.norm(b - a))

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind,
            "pos": [float(v) for v in self.pos],
            "pos_b": ([float(v) for v in self.pos_b] if self.pos_b is not None else None),
            "points": ([[float(v) for v in pt] for pt in self.points]
                      if self.points is not None else None),
            "text": self.text,
            "color": [float(v) for v in self.color],
            "font_size": float(self.font_size),
            "line_width": float(self.line_width),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Marker":
        pos_b = d.get("pos_b")
        points = d.get("points")
        return cls(
            id=str(d.get("id") or uuid.uuid4().hex),
            kind=str(d.get("kind", "label")),
            pos=[float(v) for v in d.get("pos", [0.0, 0.0, 0.0])],
            pos_b=([float(v) for v in pos_b] if pos_b is not None else None),
            points=([[float(v) for v in pt] for pt in points] if points is not None else None),
            text=str(d.get("text", "")),
            color=[float(v) for v in d.get("color", [1.0, 0.85, 0.2])],
            font_size=float(d.get("font_size", 14.0)),
            line_width=float(d.get("line_width", 2.5)),
            created_at=str(d.get("created_at", "")),
        )


class MarkerStore(QObject):
    """
    Colección de marcadores persistentes del proyecto activo.

    Uso típico:
        store.add_label(pos, "poste dañado")
        store.add_measure(p1, p2, "ancho de calle")
        ...
        project.markers = store.to_list()   # al guardar
        store.load_list(project.markers)    # al abrir
    """
    markers_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._markers: List[Marker] = []

    def add_label(self, pos, text: str, color=(1.0, 0.85, 0.2),
                 font_size: float = 14.0) -> Marker:
        m = Marker(id=uuid.uuid4().hex, kind="label",
                  pos=[float(pos[0]), float(pos[1]), float(pos[2])],
                  text=text, color=list(color), font_size=float(font_size),
                  created_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        self._markers.append(m)
        self.markers_changed.emit()
        return m

    def add_measure(self, pos_a, pos_b, text: str = "",
                    color=(1.0, 0.9, 0.2), font_size: float = 14.0,
                    line_width: float = 2.5) -> Marker:
        m = Marker(id=uuid.uuid4().hex, kind="measure",
                  pos=[float(pos_a[0]), float(pos_a[1]), float(pos_a[2])],
                  pos_b=[float(pos_b[0]), float(pos_b[1]), float(pos_b[2])],
                  text=text, color=list(color), font_size=float(font_size),
                  line_width=float(line_width),
                  created_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        self._markers.append(m)
        self.markers_changed.emit()
        return m

    def add_polyline(self, points, text: str = "",
                     color=(0.2, 0.85, 1.0), font_size: float = 14.0,
                     line_width: float = 2.5) -> Marker:
        """`points`: lista de (x,y,z), al menos 2 vértices."""
        pts = [[float(p[0]), float(p[1]), float(p[2])] for p in points]
        m = Marker(id=uuid.uuid4().hex, kind="polyline",
                  pos=list(pts[0]), points=pts,
                  text=text, color=list(color), font_size=float(font_size),
                  line_width=float(line_width),
                  created_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        self._markers.append(m)
        self.markers_changed.emit()
        return m

    def update(self, marker_id: str, **kwargs) -> bool:
        """
        Modifica un marcador existente in-place (texto/color/tamaño de
        letra/etc.) y notifica markers_changed para que el render se
        refresque. Usado por LabelMarkerTool al editar una etiqueta ya
        creada (Ctrl+clic sobre ella en vez de crear una nueva).
        """
        m = self.get(marker_id)
        if m is None:
            return False
        for k, v in kwargs.items():
            if hasattr(m, k):
                setattr(m, k, v)
        self.markers_changed.emit()
        return True

    def remove(self, marker_id: str) -> bool:
        before = len(self._markers)
        self._markers = [m for m in self._markers if m.id != marker_id]
        changed = len(self._markers) != before
        if changed:
            self.markers_changed.emit()
        return changed

    def clear(self) -> None:
        if not self._markers:
            return
        self._markers = []
        self.markers_changed.emit()

    def get(self, marker_id: str) -> Optional[Marker]:
        for m in self._markers:
            if m.id == marker_id:
                return m
        return None

    def all(self) -> List[Marker]:
        return list(self._markers)

    def __len__(self) -> int:
        return len(self._markers)

    # ── Persistencia (Project.markers) ────────────────────────────────────

    def to_list(self) -> list:
        return [m.to_dict() for m in self._markers]

    def load_list(self, data: Optional[list]) -> None:
        self._markers = [Marker.from_dict(d) for d in (data or [])]
        self.markers_changed.emit()
