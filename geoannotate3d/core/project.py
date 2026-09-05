"""
core/project.py
===============
Modelo de un proyecto de anotación GeoAnnotate3D.

Formato de archivo .geoa3d:
    Es un ZIP que contiene dos archivos:
        meta.json    — metadatos, schema de clases, estadísticas
        labels.npy   — array uint8 (N,) con una etiqueta por punto

Se usa ZIP en lugar de JSON+NPY separados para:
    - Mantener ambos archivos siempre sincronizados (una sola operación de escritura)
    - Portabilidad: mover un .geoa3d mueve todo junto
    - Posibilidad de añadir más archivos en el futuro (thumbnails, notas, etc.)

Estructura de meta.json:
{
  "version":          "0.1.0",
  "name":             "zona_norte_gdl",
  "created_at":       "2024-11-15T14:30:00",
  "modified_at":      "2024-11-15T16:45:22",
  "author":           "",
  "source_file":      "/ruta/absoluta/zona_norte.laz",
  "source_file_hash": "md5:abc123...",
  "crs":              "WGS 84 / UTM zone 14N",
  "offset_xyz":       [397000.0, 2166000.0, 1400.0],
  "bbox_geo":         {"min": [...], "max": [...]},
  "region_name":      "Guadalajara, Jalisco",
  "acquisition_date": "2024-03-10",
  "sensor_type":      "ALS",
  "annotation_time_s": 8073,
  "schema": [
    {"id": 0, "name": "sin etiquetar", "color": "#444444"},
    {"id": 1, "name": "suelo",         "color": "#9e7228"},
    ...
  ],
  "stats": {
    "n_labeled": 74618,
    "n_total":   1200000,
    "per_class_counts": {"1": 38420, ...}
  }
}
"""
from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

APP_VERSION = "0.1.0"
_META_FILENAME   = "meta.json"
_LABELS_FILENAME = "labels.npy"


# ---------------------------------------------------------------------------
# SemanticClass
# ---------------------------------------------------------------------------

@dataclass
class SemanticClass:
    """Una clase semántica del schema de anotación."""
    id:    int
    name:  str
    color: str   # hex "#rrggbb"

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "color": self.color}

    @classmethod
    def from_dict(cls, d: dict) -> "SemanticClass":
        return cls(id=int(d["id"]), name=str(d["name"]), color=str(d["color"]))


# ---------------------------------------------------------------------------
# Schema por defecto
# ---------------------------------------------------------------------------

DEFAULT_SCHEMA: List[SemanticClass] = [
    SemanticClass( 0, "sin etiquetar", "#444444"),
    # Geomática urbana
    SemanticClass( 1, "suelo",         "#9e7228"),
    SemanticClass( 2, "veg. baja",     "#5a8c18"),
    SemanticClass( 3, "árbol",         "#1a6c0e"),
    SemanticClass( 4, "edificio",      "#1e5ea0"),
    SemanticClass( 5, "vehículo",      "#b42c14"),
    SemanticClass( 6, "persona",       "#b82868"),
    SemanticClass( 7, "agua",          "#1070a0"),
    SemanticClass( 8, "estructura",    "#806018"),
    SemanticClass( 9, "ruido/error",   "#4c4440"),
    # Agricultura
    SemanticClass(10, "cultivo",       "#78b020"),
    SemanticClass(11, "suelo agrícola","#c8a060"),
    SemanticClass(12, "invernadero",   "#80c0c0"),
]


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------

class Project:
    """
    Modelo completo de un proyecto de anotación GeoAnnotate3D.

    Ciclo de vida básico:
        # Nuevo proyecto
        p = Project.new("nube.laz")
        p.sync_from_cloud(pc)        # copiar CRS, offset, bbox desde PointCloud
        p.init_labels(pc.n_points)
        p.save("proyecto.geoa3d")

        # Reanudar sesión
        p = Project.load("proyecto.geoa3d")
        # verificar que la nube fuente no cambió
        p.verify_source()
    """

    def __init__(self):
        self.version:     str  = APP_VERSION
        self.name:        str  = ""
        self.created_at:  str  = datetime.now().isoformat()
        self.modified_at: str  = datetime.now().isoformat()
        self.author:      str  = ""

        # Fuente
        self.source_file:      str = ""
        self.source_file_hash: str = ""

        # Geo
        self.crs:              str         = ""
        self.offset_xyz:       List[float] = [0.0, 0.0, 0.0]
        # Grid de tiles — se guardan para restaurar exactamente al reabrir
        self.tile_size_m:      float = 50.0
        self.grid_offset_x:    float = 0.0
        self.grid_offset_y:    float = 0.0
        self.grid_rotation:    float = 0.0
        self.tile_rotation_deg: float = 0.0
        self.tile_origin_x:    float = 0.0
        self.tile_origin_y:    float = 0.0
        self.bbox_geo:         Dict        = {}
        self.region_name:      str         = ""
        self.acquisition_date: str         = ""
        self.sensor_type:      str         = ""

        # Sesión
        self.annotation_time_s: int   = 0
        self._session_start:    float = 0.0

        # Schema
        self.schema: List[SemanticClass] = list(DEFAULT_SCHEMA)

        # Etiquetas
        self.labels:  Optional[np.ndarray] = None
        self.n_total: int                  = 0

    # ------------------------------------------------------------------
    # Construcción
    # ------------------------------------------------------------------

    @classmethod
    def new(cls, source_path: str, name: str = "") -> "Project":
        """Crea un proyecto nuevo vacío para la nube dada."""
        p = cls()
        p.source_file      = str(Path(source_path).resolve())
        p.name             = name or Path(source_path).stem
        p.source_file_hash = cls._hash_file(source_path)
        return p

    def sync_from_cloud(self, pc) -> None:
        """
        Copia metadatos geoespaciales desde un PointCloud recién cargado.
        Llamar después de cloud_ready para que el proyecto tenga CRS y offset.
        """
        if pc.crs:
            self.crs = pc.crs
        if pc.offset is not None:
            self.offset_xyz = [float(pc.offset[0]),
                               float(pc.offset[1]),
                               float(pc.offset[2])]
        if pc.bounds is not None:
            from utils.geo import bbox_to_geo
            self.bbox_geo = bbox_to_geo(pc.bounds, pc.offset)

    # ------------------------------------------------------------------
    # Serialización / Deserialización
    # ------------------------------------------------------------------

    def save(self, geoa3d_path: str) -> None:
        """
        Guarda el proyecto en un archivo .geoa3d (ZIP).

        Contenido del ZIP:
            meta.json   — metadatos JSON legibles
            labels.npy  — array uint8 de etiquetas (comprimido con ZIP deflate)

        El ZIP se escribe atómicamente: se construye en memoria y se escribe
        de una sola vez para evitar archivos corruptos si el proceso se interrumpe.
        """
        self.modified_at = datetime.now().isoformat()
        meta = self.to_dict()

        # Serializar labels.npy a bytes en memoria
        labels_bytes = io.BytesIO()
        lbl = self.labels if self.labels is not None else np.zeros(0, np.uint8)
        np.save(labels_bytes, lbl)
        labels_bytes.seek(0)

        # Construir ZIP en memoria para escritura atómica
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w",
                             compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6) as zf:
            # meta.json — legible y versionable
            zf.writestr(_META_FILENAME,
                        json.dumps(meta, indent=2, ensure_ascii=False))
            # labels.npy — comprimido (los arrays uint8 comprimen bien)
            zf.writestr(_LABELS_FILENAME, labels_bytes.read())

        # Escribir a disco de una sola vez (escritura atómica)
        path = Path(geoa3d_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(zip_buffer.getvalue())

    @classmethod
    def load(cls, geoa3d_path: str) -> "Project":
        """
        Carga un proyecto desde un archivo .geoa3d.

        Pasos:
            1. Abrir el ZIP
            2. Leer y parsear meta.json
            3. Leer labels.npy
            4. Reconstruir el objeto Project con todos sus campos

        Errores:
            FileNotFoundError  — el .geoa3d no existe
            ValueError         — ZIP corrupto o campos faltantes
            KeyError           — versión incompatible
        """
        path = Path(geoa3d_path)
        if not path.exists():
            raise FileNotFoundError(f"Proyecto no encontrado: {geoa3d_path}")

        try:
            with zipfile.ZipFile(str(path), mode="r") as zf:
                # Verificar contenido mínimo
                names = zf.namelist()
                if _META_FILENAME not in names:
                    raise ValueError(
                        f"Archivo .geoa3d inválido: falta {_META_FILENAME}")

                # Leer metadatos
                meta_bytes = zf.read(_META_FILENAME)
                meta       = json.loads(meta_bytes.decode("utf-8"))

                # Leer labels (puede no existir en proyectos muy nuevos)
                labels = None
                if _LABELS_FILENAME in names:
                    lbl_bytes = io.BytesIO(zf.read(_LABELS_FILENAME))
                    labels    = np.load(lbl_bytes)
                    if labels.dtype != np.uint8:
                        labels = labels.astype(np.uint8)

        except zipfile.BadZipFile as exc:
            raise ValueError(
                f"El archivo .geoa3d está corrupto: {exc}") from exc

        # Reconstruir Project desde el dict
        p = cls._from_dict(meta)
        if labels is not None and len(labels) > 0:
            p.attach_labels(labels)

        return p

    @classmethod
    def _from_dict(cls, meta: dict) -> "Project":
        """Reconstruye un Project desde el dict de meta.json."""
        p = cls()

        # Campos básicos con defaults seguros para compatibilidad futura
        p.version          = meta.get("version", APP_VERSION)
        p.name             = meta.get("name", "")
        p.created_at       = meta.get("created_at", datetime.now().isoformat())
        p.modified_at      = meta.get("modified_at", p.created_at)
        p.author           = meta.get("author", "")
        p.source_file      = meta.get("source_file", "")
        p.source_file_hash = meta.get("source_file_hash", "")
        p.crs              = meta.get("crs", "")
        p.offset_xyz       = meta.get("offset_xyz", [0.0, 0.0, 0.0])
        p.tile_size_m      = float(meta.get("tile_size_m", 50.0))
        p.grid_offset_x    = float(meta.get("grid_offset_x", 0.0))
        p.grid_offset_y    = float(meta.get("grid_offset_y", 0.0))
        p.grid_rotation    = float(meta.get("grid_rotation",  0.0))
        p.tile_rotation_deg= float(meta.get("tile_rotation_deg", 0.0))
        p.tile_origin_x    = float(meta.get("tile_origin_x", 0.0))
        p.tile_origin_y    = float(meta.get("tile_origin_y", 0.0))
        p.bbox_geo         = meta.get("bbox_geo", {})
        p.region_name      = meta.get("region_name", "")
        p.acquisition_date = meta.get("acquisition_date", "")
        p.sensor_type      = meta.get("sensor_type", "")
        p.annotation_time_s = int(meta.get("annotation_time_s", 0))

        # Schema de clases
        schema_raw = meta.get("schema", [])
        if schema_raw:
            p.schema = [SemanticClass.from_dict(d) for d in schema_raw]
        else:
            p.schema = list(DEFAULT_SCHEMA)

        # Stats (para restaurar n_total aunque los labels sean None)
        stats = meta.get("stats", {})
        p.n_total = int(stats.get("n_total", 0))

        return p

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    def init_labels(self, n_points: int) -> None:
        """Inicializa el array de etiquetas a 0 (sin etiquetar)."""
        self.n_total = n_points
        self.labels  = np.zeros(n_points, dtype=np.uint8)

    def attach_labels(self, labels: np.ndarray) -> None:
        """Adjunta un array de etiquetas existente."""
        if labels.dtype != np.uint8:
            labels = labels.astype(np.uint8)
        self.labels  = labels
        self.n_total = len(labels)

    # ------------------------------------------------------------------
    # Verificación de integridad
    # ------------------------------------------------------------------

    def verify_source(self) -> Tuple[bool, str]:
        """
        Verifica que el archivo fuente no haya cambiado desde la última sesión.

        Retorna (ok: bool, message: str).
        Útil para avisar al usuario si la nube fue modificada o movida.
        """
        if not self.source_file:
            return False, "No hay archivo fuente registrado."

        src = Path(self.source_file)
        if not src.exists():
            return False, (
                f"Archivo fuente no encontrado:\n{self.source_file}\n\n"
                "Usa 'Reemplazar nube' para seleccionar su nueva ubicación.")

        current_hash = self._hash_file(self.source_file)
        if current_hash != self.source_file_hash:
            return False, (
                f"El archivo fuente parece haber cambiado desde la última sesión.\n"
                f"Guardado: {self.source_file_hash}\n"
                f"Actual:   {current_hash}\n\n"
                "Las etiquetas pueden no corresponder con la nube actual.")

        return True, "OK"

    # ------------------------------------------------------------------
    # Estadísticas
    # ------------------------------------------------------------------

    @property
    def n_labeled(self) -> int:
        if self.labels is None:
            return 0
        return int(np.count_nonzero(self.labels))

    @property
    def coverage_pct(self) -> float:
        if self.n_total == 0:
            return 0.0
        return self.n_labeled / self.n_total * 100.0

    def per_class_counts(self) -> Dict[int, int]:
        if self.labels is None:
            return {}
        lbl = self.labels[self.labels > 0]
        if len(lbl) == 0:
            return {}
        unique, counts = np.unique(lbl, return_counts=True)
        return {int(u): int(c) for u, c in zip(unique, counts)}

    # ------------------------------------------------------------------
    # Sesión
    # ------------------------------------------------------------------

    def start_session(self) -> None:
        self._session_start = time.time()

    def stop_session(self) -> None:
        if self._session_start > 0:
            self.annotation_time_s += int(time.time() - self._session_start)
            self._session_start = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_file(path: str) -> str:
        """MD5 en chunks para no cargar el archivo completo en RAM."""
        try:
            h = hashlib.md5()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            return f"md5:{h.hexdigest()}"
        except Exception:
            return "md5:unknown"

    def to_dict(self) -> dict:
        """Serializa todos los metadatos a dict (sin labels)."""
        return {
            "version":           self.version,
            "name":              self.name,
            "created_at":        self.created_at,
            "modified_at":       self.modified_at,
            "author":            self.author,
            "source_file":       self.source_file,
            "source_file_hash":  self.source_file_hash,
            "crs":               self.crs,
            "offset_xyz":        self.offset_xyz,
            "tile_size_m":       self.tile_size_m,
            "grid_offset_x":     self.grid_offset_x,
            "grid_offset_y":     self.grid_offset_y,
            "grid_rotation":     self.grid_rotation,
            "tile_rotation_deg": self.tile_rotation_deg,
            "tile_origin_x":     self.tile_origin_x,
            "tile_origin_y":     self.tile_origin_y,
            "bbox_geo":          self.bbox_geo,
            "region_name":       self.region_name,
            "acquisition_date":  self.acquisition_date,
            "sensor_type":       self.sensor_type,
            "annotation_time_s": self.annotation_time_s,
            "schema":            [sc.to_dict() for sc in self.schema],
            "stats": {
                "n_labeled":        self.n_labeled,
                "n_total":          self.n_total,
                "per_class_counts": self.per_class_counts(),
                "coverage_pct":     round(self.coverage_pct, 2),
            },
        }

    def __repr__(self) -> str:
        return (f"Project({self.name!r}, "
                f"{self.n_labeled}/{self.n_total} pts etiquetados, "
                f"v{self.version})")
